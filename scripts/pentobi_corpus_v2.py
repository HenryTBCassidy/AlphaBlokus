"""Generate, inspect and validate a **v2** Pentobi distillation corpus.

The v2 generator (``docs/plans/pentobi-corpus-v2.md``) is two phases around one persistent
store (``docs/plans/corpus-search-space-store.md``):

1. ``plan`` — hand the whole game budget to the empty board and split it recursively in
   proportion to temperature-flattened Pentobi visit share, searching positions on demand
   as the split needs them. Depth is an *output*. The result is an allocation plan whose
   per-node game targets live in the store.
2. ``generate`` — fulfil that plan: take the lowest-fulfilment ``(node, replica)`` jobs,
   replay each start's opening prefix into the engine, and play at full strength to the
   end, harvesting the whole ``move_values`` distribution at **every** ply.

Usage::

    # Phase A: map the opening space a 10k-game plan needs (~1,600 L9 searches)
    uv run python -m scripts.pentobi_corpus_v2 plan --corpus temp/corpus_v2 \
        --budget 10000 --temperature 2 --min-replicas 2 --workers 12 \
        [--book ~/code/pentobi/opening_books/book_duo.blksgf]

    # Phase B: fulfil it (resumable; stop any time for a proportional slice of the plan)
    uv run python -m scripts.pentobi_corpus_v2 generate --corpus temp/corpus_v2 \
        --num-games 10000 --workers 12

    # Materialise the opening dataset, aggregate outcomes, report, validate
    uv run python -m scripts.pentobi_corpus_v2 link --corpus temp/corpus_v2
    uv run python -m scripts.pentobi_corpus_v2 export-opening --corpus temp/corpus_v2
    uv run python -m scripts.pentobi_corpus_v2 coverage --corpus temp/corpus_v2 [--json out.json]
    uv run python -m scripts.pentobi_corpus_v2 analyze  --corpus temp/corpus_v2 [--json out.json]
    uv run python -m scripts.pentobi_corpus_v2 validate --corpus temp/corpus_v2

A corpus directory holds ``store.sqlite`` (the DAG, the plans and the playout registry),
``games/`` and ``opening/``. **The DB is part of the corpus** — sync it with the shards;
it is the map of what they mean. Shards are self-describing enough to rebuild it
(``generate`` reconciles from footers on every start, which is also the crash repair).

Pentobi is CPU-only, so ``--workers`` should be about the machine's physical core count;
each worker owns one single-threaded ``pentobi-gtp`` process. The parent process owns the
database — workers only ever compute and return results.

The v1 corpus keeps its own script (``scripts/pentobi_corpus.py``): the two generators
have incompatible subcommand semantics, and v1's shards stay on disk as a mid-game
training supplement.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from alphablokus.games.blokusduo.game import BlokusDuoGame
from alphablokus.games.blokusduo.pentobi.corpus import CorpusGenerationError
from alphablokus.games.blokusduo.pentobi.corpus_v2 import (
    GameShardMeta,
    analyze_corpus,
    export_opening,
    game_shard_filename,
    game_shards,
    iter_shard_playouts,
    opening_shards,
    validate_game_shard,
    validate_opening_shard,
    write_game_shard,
)
from alphablokus.games.blokusduo.pentobi.gtp import find_pentobi_gtp
from alphablokus.games.blokusduo.pentobi.harvest import (
    PentobiSearchSource,
    map_plan,
    play_planned_game,
    read_book_lines,
    replay_witness,
)
from alphablokus.games.blokusduo.pentobi.store import (
    PlanParameters,
    PlayoutJob,
    SearchChild,
    SearchSpaceStore,
    StoreError,
)
from alphablokus.games.blokusduo.pieces import default_pieces_path

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Layout inside a corpus directory.
STORE_FILENAME = "store.sqlite"
GAMES_SUBDIR = "games"
OPENING_SUBDIR = "opening"

#: Nodes handed to one mapping worker at a time. Each worker keeps its engine process
#: alive across the batch, so bigger batches amortise process startup; small enough that
#: a killed run loses little.
_MAPPING_BATCH = 8

# A search returned across the process boundary: plain tuples only (spawn start method).
_WireChild = tuple[int, int, float]
_WireSearch = tuple[int, list[_WireChild], float | None, float]


def _open_store(corpus: Path, *, level: int | None = None, create: bool = False) -> SearchSpaceStore:
    """Open the corpus store, refusing to invent one unless asked.

    ``level = None`` adopts whatever the store already pins — the right thing for every
    command that does not drive the engine, which would otherwise impose the CLI default
    and refuse to open a corpus built at another level.
    """
    path = corpus / STORE_FILENAME
    if not path.exists() and not create:
        raise SystemExit(f"No store at {path} — run `plan` first.")
    return SearchSpaceStore(path, BlokusDuoGame(pieces_config_path=default_pieces_path()), level=level)


def _require_engine(binary: str | None) -> None:
    if find_pentobi_gtp() is None and binary is None:
        raise SystemExit("pentobi-gtp not found — build it or set $PENTOBI_GTP_PATH / pass --binary.")


# --------------------------------------------------------------------------- #
# Phase A: plan
# --------------------------------------------------------------------------- #


def _search_batch(
    requests: list[tuple[int, list[int], int]],
    level: int,
    binary: str | None,
) -> list[_WireSearch]:
    """Worker: search a batch of ``(node_id, witness_actions, seed)`` positions."""
    game = BlokusDuoGame(pieces_config_path=default_pieces_path())
    results: list[_WireSearch] = []
    with PentobiSearchSource(game, level, binary=binary) as source:
        for node_id, witness, seed in requests:
            board, player, prefix = replay_witness(game, witness)
            source.begin_position(seed, prefix)
            result = source.search(board, player)
            children = [(child.action, child.visits, child.value) for child in result.children]
            results.append((node_id, children, result.search_value, result.seconds))
    return results


def plan(args: argparse.Namespace) -> None:
    """Phase A: map the search space the allocation needs, then save the plan."""
    _require_engine(args.binary)
    corpus = Path(args.corpus)
    corpus.mkdir(parents=True, exist_ok=True)
    params = PlanParameters(args.budget, args.temperature, args.min_replicas)
    store = _open_store(corpus, level=args.level, create=True)
    try:
        if args.book:
            lines = read_book_lines(Path(args.book).expanduser(), store.game)
            terminals = store.insert_book_paths(lines)
            logger.info("Inserted {} book lines ({} terminal nodes) with a games floor.", len(lines), len(terminals))
        started = time.perf_counter()

        def search_nodes(node_ids: Sequence[int]) -> None:
            """Fan one mapping round out over the worker pool."""
            _run_searches(
                store,
                [
                    (node_id, list(store.node(node_id).witness_actions), int(store.node(node_id).engine_seed or 0))
                    for node_id in node_ids
                ],
                args,
            )

        # The mapping loop itself lives in the library, so the code that generates real
        # corpora and the code the tests exercise are the same code — only the way
        # searches are dispatched differs (a process pool here, in-process there).
        try:
            draft = map_plan(store, params, search_nodes, max_rounds=args.max_rounds)
        except CorpusGenerationError as error:
            raise SystemExit(str(error)) from error
        plan_id = store.save_plan(draft)
        logger.info(
            "Plan {} saved: {} openings, {} games in {:.1f} min (B={}, T={}, R={})",
            plan_id,
            len(draft.starts),
            draft.planned_games,
            (time.perf_counter() - started) / 60,
            params.budget,
            params.temperature,
            params.min_replicas,
        )
        _log_coverage(store)
    finally:
        store.close()


def _run_searches(
    store: SearchSpaceStore,
    requests: list[tuple[int, list[int], int]],
    args: argparse.Namespace,
) -> None:
    """Fan a mapping round out over the worker pool; the parent owns every DB write."""
    batches = [requests[i : i + _MAPPING_BATCH] for i in range(0, len(requests), _MAPPING_BATCH)]
    ctx = mp.get_context("spawn")
    done = 0
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futures = [pool.submit(_search_batch, batch, args.level, args.binary) for batch in batches]
        for future in as_completed(futures):
            for node_id, children, search_value, seconds in future.result():
                store.record_search(
                    node_id,
                    [SearchChild(action=action, visits=visits, value=value) for action, visits, value in children],
                    seconds=seconds,
                    search_value=search_value,
                )
                done += 1
            logger.info(
                "  searched {}/{} ({:.1f} searches/min)",
                done,
                len(requests),
                done / (time.perf_counter() - started) * 60,
            )


# --------------------------------------------------------------------------- #
# Phase B: generate
# --------------------------------------------------------------------------- #


def _generate_batch(
    jobs: list[tuple[int, int, int, int, bytes, list[int]]],
    shard_path: str,
    level: int,
    binary: str | None,
    top_k: int,
    shard_meta: dict[str, object],
) -> tuple[str, list[tuple[int, int, int, int]], int, float]:
    """Worker: play one shard's games and write it. Returns per-game results for the DB."""
    started = time.perf_counter()
    game = BlokusDuoGame(pieces_config_path=default_pieces_path())
    played = []
    with PentobiSearchSource(game, level, binary=binary) as source:
        for node_id, replica, game_id, seed, board_key, witness in jobs:
            job = PlayoutJob(
                node_id=node_id,
                replica=replica,
                game_id=game_id,
                engine_seed=seed,
                board_key=board_key,
                witness_actions=tuple(witness),
            )
            played.append(play_planned_game(game, source, job, top_k=top_k))
    meta = GameShardMeta(
        level=level,
        policy_size=game.get_action_size(),
        board_shape=(game.board_size, game.board_size),
        board_dtype="int8",
        dag_hash=str(shard_meta["dag_hash"]),
        plan_id=shard_meta["plan_id"],  # type: ignore[arg-type]
        budget=shard_meta["budget"],  # type: ignore[arg-type]
        temperature=shard_meta["temperature"],  # type: ignore[arg-type]
        min_replicas=shard_meta["min_replicas"],  # type: ignore[arg-type]
        game_sizes=tuple(len(g.plies) for g in played),
        games=(),
    )
    rows = write_game_shard(Path(shard_path), played, meta=meta)
    results = [(g.node_id, g.replica, g.white_margin, len(g.plies)) for g in played]
    return shard_path, results, rows, time.perf_counter() - started


def generate(args: argparse.Namespace) -> None:
    """Phase B: fulfil the active plan, lowest fulfilment first, resumably."""
    _require_engine(args.binary)
    corpus = Path(args.corpus)
    games_dir = corpus / GAMES_SUBDIR
    games_dir.mkdir(parents=True, exist_ok=True)
    for stale in games_dir.glob("*.tmp"):
        stale.unlink()  # torn shards from a killed run — regenerated below
    store = _open_store(corpus, level=args.level)  # generate drives the engine: the level must match
    try:
        active = store.active_plan()
        if active is None:
            raise SystemExit("No active plan — run `plan` first.")
        # Shards are the truth for games; repair the registry from their footers first,
        # so a run killed between a shard rename and its DB write does not replay games.
        repaired = store.reconcile(iter_shard_playouts(games_dir))
        if repaired.inserted or repaired.updated:
            logger.info("Reconciled from shard footers: +{} inserted, {} updated", repaired.inserted, repaired.updated)
        if repaired.unknown_nodes:
            logger.warning("{} shard games reference positions outside the DAG", len(repaired.unknown_nodes))

        jobs = store.pending_jobs()
        if jobs:
            logger.info("Resuming {} previously scheduled games", len(jobs))
        if len(jobs) < args.num_games:
            jobs += store.schedule(args.num_games - len(jobs))
        if not jobs:
            logger.info("Plan already fulfilled — nothing to do.")
            _log_coverage(store)
            return

        shard_meta = {
            "dag_hash": store.dag_hash(),
            "plan_id": active.plan_id,
            "budget": active.parameters.budget,
            "temperature": active.parameters.temperature,
            "min_replicas": active.parameters.min_replicas,
        }
        next_index = len(game_shards(games_dir))
        batches = [jobs[i : i + args.games_per_shard] for i in range(0, len(jobs), args.games_per_shard)]
        logger.info(
            "Generating {} games in {} shards (plan {}, level {}) over {} workers → {}",
            len(jobs),
            len(batches),
            active.plan_id,
            args.level,
            args.workers,
            games_dir,
        )
        started = time.perf_counter()
        total_games = total_rows = 0
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            futures = []
            for offset, batch in enumerate(batches):
                shard_path = games_dir / game_shard_filename(next_index + offset)
                futures.append(
                    pool.submit(
                        _generate_batch,
                        [
                            (j.node_id, j.replica, j.game_id, j.engine_seed, j.board_key, list(j.witness_actions))
                            for j in batch
                        ],
                        str(shard_path),
                        args.level,
                        args.binary,
                        args.top_k,
                        shard_meta,
                    ),
                )
            for future in as_completed(futures):
                shard_path, results, rows, seconds = future.result()
                for node_id, replica, margin, plies in results:
                    store.mark_done(node_id, replica, shard=Path(shard_path).name, white_margin=margin, plies=plies)
                total_games += len(results)
                total_rows += rows
                elapsed = time.perf_counter() - started
                logger.info(
                    "{}: {} games / {} positions in {:.0f}s | cumulative {:.1f} games/hour",
                    Path(shard_path).name,
                    len(results),
                    rows,
                    seconds,
                    total_games / elapsed * 3600,
                )
        elapsed = time.perf_counter() - started
        logger.info(
            "Done: {} games / {} positions in {:.1f} min — {:.1f} games/hour ({} workers)",
            total_games,
            total_rows,
            elapsed / 60,
            total_games / elapsed * 3600,
            args.workers,
        )
        _log_coverage(store)
    finally:
        store.close()


# --------------------------------------------------------------------------- #
# Store maintenance + reporting
# --------------------------------------------------------------------------- #


def link(args: argparse.Namespace) -> None:
    """Aggregate playout outcomes up the DAG into ``outcome_mean`` / ``outcome_count``."""
    store = _open_store(Path(args.corpus))
    try:
        logger.info("Linked outcomes into {} nodes.", store.link())
        root = store.node(store.root_node())
        logger.info("Root: {} games, mean outcome {}", root.outcome_count, root.outcome_mean)
    finally:
        store.close()


def export_opening_command(args: argparse.Namespace) -> None:
    """Materialise the opening dataset from the store (the DB is the source of truth)."""
    corpus = Path(args.corpus)
    store = _open_store(corpus)
    try:
        paths = export_opening(store, corpus / OPENING_SUBDIR, top_k=args.top_k)
        rows = len(store.nodes(status="searched"))
        logger.info("Exported {} opening rows to {} shard(s) at dag_hash {}", rows, len(paths), store.dag_hash()[:12])
    finally:
        store.close()


def coverage(args: argparse.Namespace) -> None:
    """Print the store's coverage report — plan fulfilment, mapping debt, opening fan."""
    store = _open_store(Path(args.corpus))
    try:
        report = store.coverage().to_dict()
        for key, value in report.items():
            logger.info("  {} = {}", key, value)
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2))
            logger.info("Coverage written to {}", args.json)
    finally:
        store.close()


def analyze(args: argparse.Namespace) -> None:
    """Print the corpus diagnostics: target richness, row mix, duplication, balance."""
    corpus = Path(args.corpus)
    report = analyze_corpus(corpus / GAMES_SUBDIR, corpus / OPENING_SUBDIR)
    logger.info("Corpus report for {}:", corpus)
    for key, value in report.to_dict().items():
        logger.info("  {} = {}", key, value)
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2))
        logger.info("Report written to {}", args.json)


def validate(args: argparse.Namespace) -> None:
    """Replay-validate every stored row of both datasets."""
    corpus = Path(args.corpus)
    game = BlokusDuoGame(pieces_config_path=default_pieces_path())
    shards = game_shards(corpus / GAMES_SUBDIR)
    openings = opening_shards(corpus / OPENING_SUBDIR)
    if not shards and not openings:
        raise SystemExit(f"No v2 shards found under {corpus}")
    total = 0
    for path in shards:
        checked = validate_game_shard(path, game)
        total += checked
        logger.info("{}: {} game positions OK", path.name, checked)
    store = _open_store(corpus) if (corpus / STORE_FILENAME).exists() else None
    try:
        for path in openings:
            checked = validate_opening_shard(path, game, store)
            total += checked
            logger.info("{}: {} opening rows OK", path.name, checked)
    finally:
        if store is not None:
            store.close()
    logger.info("Validation passed: {} rows across {} shards.", total, len(shards) + len(openings))


def _log_coverage(store: SearchSpaceStore) -> None:
    """One-line summary of where the active plan stands."""
    try:
        report = store.coverage()
    except StoreError as error:  # pragma: no cover — defensive: coverage must never block a run
        logger.warning("coverage report unavailable: {}", error)
        return
    logger.info(
        "Plan fulfilment: {}/{} games, mapping debt {}, {} distinct first moves ({} canonical positions)",
        report.actual_games,
        report.planned_games,
        report.mapping_debt,
        report.distinct_first_moves,
        report.distinct_first_positions,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Pentobi distillation corpus v2: plan / generate / inspect")
    sub = ap.add_subparsers(dest="command", required=True)

    def add_common(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--corpus", required=True, help="Corpus directory (store.sqlite + games/ + opening/)")

    planner = sub.add_parser("plan", help="Phase A: allocate a game budget over the search space")
    add_common(planner)
    planner.add_argument("--budget", type=int, required=True, help="B: total games to allocate")
    planner.add_argument("--temperature", type=float, default=2.0, help="T: weights are visit_share^(1/T)")
    planner.add_argument("--min-replicas", type=int, default=2, help="R: minimum games per opening; split floor 2R")
    planner.add_argument("--book", default=None, help="Opening book (.blksgf) to force-map with a games floor")
    planner.add_argument("--level", type=int, default=9, help="Pentobi level (pinned into the store)")
    planner.add_argument("--workers", type=int, default=4, help="Worker processes (~physical cores)")
    planner.add_argument("--max-rounds", type=int, default=64, help="Safety bound on the map/re-plan loop")
    planner.add_argument("--binary", default=None, help="pentobi-gtp path (default: $PENTOBI_GTP_PATH or the build)")
    planner.set_defaults(func=plan)

    gen = sub.add_parser("generate", help="Phase B: play games fulfilling the active plan")
    add_common(gen)
    gen.add_argument("--num-games", type=int, required=True, help="Games to generate this run (resumable)")
    gen.add_argument("--workers", type=int, default=4, help="Worker processes (~physical cores)")
    gen.add_argument("--level", type=int, default=9, help="Pentobi level for both sides")
    gen.add_argument("--games-per-shard", type=int, default=10, help="Games per shard — the resume unit")
    gen.add_argument("--top-k", type=int, default=32, help="Children kept in each stored soft target")
    gen.add_argument("--binary", default=None, help="pentobi-gtp path (default: $PENTOBI_GTP_PATH or the build)")
    gen.set_defaults(func=generate)

    linker = sub.add_parser("link", help="Aggregate playout outcomes up the DAG")
    add_common(linker)
    linker.set_defaults(func=link)

    exporter = sub.add_parser("export-opening", help="Materialise opening/*.parquet from the store")
    add_common(exporter)
    exporter.add_argument("--top-k", type=int, default=32, help="Children kept in each exported soft target")
    exporter.set_defaults(func=export_opening_command)

    cov = sub.add_parser("coverage", help="Plan fulfilment, mapping debt and opening-fan coverage")
    add_common(cov)
    cov.add_argument("--json", default=None, help="Also write the report as JSON to this path")
    cov.set_defaults(func=coverage)

    ana = sub.add_parser("analyze", help="Target richness, row mix, duplication and outcome balance")
    add_common(ana)
    ana.add_argument("--json", default=None, help="Also write the report as JSON to this path")
    ana.set_defaults(func=analyze)

    val = sub.add_parser("validate", help="Replay-validate every stored row of both datasets")
    add_common(val)
    val.set_defaults(func=validate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
