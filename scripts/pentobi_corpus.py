"""Generate, analyze, and validate a Pentobi distillation corpus (parquet shards).

The data half of the Pentobi-distillation framework (``docs/plans/pentobi-distillation.md``):
plays Pentobi-vs-Pentobi games at one level across worker processes, harvests one
training example per expert ply (canonical board, one-hot played move, outcome, score
margin, side-to-move), and writes resumable parquet shards via
:mod:`alphablokus.games.blokusduo.pentobi.corpus`.

Usage::

    # Generate (resumable: completed shards are skipped on rerun)
    uv run python -m scripts.pentobi_corpus generate --out temp/corpus_l9 \
        --num-games 200 --workers 16 --level 9 --seed 0 --opening-random-plies 4

    # Diversity report (the not-a-pile-of-clones proof)
    uv run python -m scripts.pentobi_corpus analyze --data temp/corpus_l9 [--json out.json]

    # Full-replay correctness validation of every stored row
    uv run python -m scripts.pentobi_corpus validate --data temp/corpus_l9

**Diversity mechanisms.** ``--seed`` gives every game a distinct engine seed
(``set_random_seed`` per game). ``--opening-random-plies k`` additionally plays a
**deterministic stratified opening key** of ``k`` plies before Pentobi takes over:
game ``i``'s first ply sweeps the legal first moves interleaved (``enum[i mod 414]``),
plies 2..k are keyed by ``(seed, game_id)``, repeats are impossible by construction,
and zero duplicate prefixes is asserted at end of run (``k=0`` disables — used for the
seed-variation-only A/B). Harvested examples come from Pentobi's plies only.

**Determinism/resume contract.** Game ``g`` always uses engine seed ``seed + g``, its
opening prefix is a pure function of ``(seed, g)`` (rebuilt every run by walking the
key builder over all game ids), and it always lands in shard ``g // games_per_shard``
— so a rerun of the same command regenerates exactly the missing shards and nothing else.

Pentobi is CPU-only: ``--workers`` should be ~the machine's physical core count
(each worker owns one single-threaded ``pentobi-gtp`` process).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

from alphablokus.games.blokusduo.game import BlokusDuoGame
from alphablokus.games.blokusduo.pentobi.corpus import (
    CorpusGame,
    OpeningPrefixBuilder,
    PentobiMoveSource,
    analyze_corpus,
    assert_unique_openings,
    collect_opening_prefixes,
    corpus_shards,
    play_corpus_game,
    shard_filename,
    validate_shard,
    write_shard,
)
from alphablokus.games.blokusduo.pentobi.gtp import find_pentobi_gtp
from alphablokus.games.blokusduo.pieces import default_pieces_path


def _generate_shard(
    out_dir: str,
    shard_index: int,
    game_ids: list[int],
    openings: list[tuple[int, ...]],
    level: int,
    base_seed: int,
    opening_random_plies: int,
    binary: str | None,
) -> tuple[int, int, int, float]:
    """Worker entry point: play one shard's games and write its parquet file.

    Plain picklable args only (spawn start method); ``openings`` carries each game's
    pre-built deterministic opening key, aligned with ``game_ids``. Returns
    ``(shard_index, games, positions, seconds)``.
    """
    t0 = time.perf_counter()
    game = BlokusDuoGame(pieces_config_path=default_pieces_path())
    games: list[CorpusGame] = []
    with PentobiMoveSource(game, level, binary=binary) as source:
        for game_id, opening in zip(game_ids, openings, strict=True):
            games.append(
                play_corpus_game(
                    game,
                    source,
                    game_id=game_id,
                    pentobi_seed=base_seed + game_id,
                    opening_actions=opening,
                ),
            )
    rows = write_shard(
        Path(out_dir) / shard_filename(shard_index),
        games,
        policy_size=game.get_action_size(),
        level=level,
        opening_random_plies=opening_random_plies,
    )
    return shard_index, len(games), rows, time.perf_counter() - t0


def generate(args: argparse.Namespace) -> None:
    """Fan shard generation out over a spawn pool, skipping completed shards."""
    if find_pentobi_gtp() is None and args.binary is None:
        raise SystemExit("pentobi-gtp not found — build it or set $PENTOBI_GTP_PATH / pass --binary.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.tmp"):
        stale.unlink()  # torn shards from a killed run — regenerated below

    game = BlokusDuoGame(pieces_config_path=default_pieces_path())
    builder = OpeningPrefixBuilder(game, base_seed=args.seed, num_plies=args.opening_random_plies)
    openings = [builder.prefix_for(game_id) for game_id in range(args.num_games)]
    logger.info(
        "Built {} deterministic opening keys (depth {}, interleaved sweep of {} legal first moves).",
        len(openings),
        args.opening_random_plies,
        len(builder.first_moves),
    )

    shards: dict[int, list[int]] = {}
    for game_id in range(args.num_games):
        shards.setdefault(game_id // args.games_per_shard, []).append(game_id)
    pending = {i: ids for i, ids in shards.items() if not (out_dir / shard_filename(i)).exists()}
    done = len(shards) - len(pending)
    if done:
        logger.info("Resume: {} of {} shards already complete — skipping them.", done, len(shards))
    if not pending:
        logger.info("Nothing to do.")
        _verify_opening_diversity(out_dir, args.opening_random_plies)
        return

    logger.info(
        "Generating {} games in {} shards (level {}, opening_random_plies {}, seed {}) over {} workers → {}",
        sum(len(v) for v in pending.values()),
        len(pending),
        args.level,
        args.opening_random_plies,
        args.seed,
        args.workers,
        out_dir,
    )
    t0 = time.perf_counter()
    total_games = 0
    total_rows = 0
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futures = [
            pool.submit(
                _generate_shard,
                str(out_dir),
                index,
                game_ids,
                [openings[g] for g in game_ids],
                args.level,
                args.seed,
                args.opening_random_plies,
                args.binary,
            )
            for index, game_ids in sorted(pending.items())
        ]
        for future in as_completed(futures):
            index, n_games, rows, seconds = future.result()
            total_games += n_games
            total_rows += rows
            elapsed = time.perf_counter() - t0
            logger.info(
                "shard {:05d}: {} games / {} positions in {:.0f}s | cumulative {:.1f} games/hour",
                index,
                n_games,
                rows,
                seconds,
                total_games / elapsed * 3600,
            )
    elapsed = time.perf_counter() - t0
    logger.info(
        "Done: {} games / {} positions in {:.1f} min — {:.1f} games/hour, {:.0f} positions/hour ({} workers)",
        total_games,
        total_rows,
        elapsed / 60,
        total_games / elapsed * 3600,
        total_rows / elapsed * 3600,
        args.workers,
    )
    _verify_opening_diversity(out_dir, args.opening_random_plies)


def _verify_opening_diversity(out_dir: Path, opening_random_plies: int) -> None:
    """End-of-run diversity guarantee: zero duplicate opening prefixes across the corpus.

    The hard assertion only applies at prefix depth >= 2 — at depth 1 the stratified
    sweep cycles by design past the 414 first moves, and at depth 0 every prefix is
    empty (the seed-variation-only A/B mode).
    """
    if opening_random_plies >= 2:
        distinct = assert_unique_openings(out_dir)
        logger.info("Opening keys verified: {} games, {} distinct prefixes, zero duplicates.", distinct, distinct)
    else:
        prefixes = collect_opening_prefixes(out_dir)
        logger.info(
            "Opening prefixes: {} distinct across {} games (no uniqueness guarantee at depth <2).",
            len(set(prefixes)),
            len(prefixes),
        )


def analyze(args: argparse.Namespace) -> None:
    """Compute and print the corpus diversity report."""
    report = analyze_corpus(Path(args.data))
    logger.info("Diversity report for {}:", args.data)
    for key, value in report.to_dict().items():
        logger.info("  {} = {}", key, value)
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2))
        logger.info("Report written to {}", args.json)


def validate(args: argparse.Namespace) -> None:
    """Replay-validate every shard in the corpus directory."""
    game = BlokusDuoGame(pieces_config_path=default_pieces_path())
    shards = corpus_shards(Path(args.data))
    if not shards:
        raise SystemExit(f"No corpus shards found in {args.data}")
    total = 0
    for path in shards:
        checked = validate_shard(path, game)
        total += checked
        logger.info("{}: {} positions OK", path.name, checked)
    logger.info("Validation passed: {} positions across {} shards.", total, len(shards))


def main() -> None:
    ap = argparse.ArgumentParser(description="Pentobi distillation corpus: generate / analyze / validate")
    sub = ap.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Play Pentobi-vs-Pentobi games and write parquet shards")
    gen.add_argument("--out", required=True, help="Output directory for corpus shards")
    gen.add_argument("--num-games", type=int, required=True, help="Total games (across all shards)")
    gen.add_argument("--workers", type=int, default=4, help="Worker processes (~physical cores; Pentobi is CPU-only)")
    gen.add_argument("--level", type=int, default=9, help="Pentobi level for both sides (default 9)")
    gen.add_argument(
        "--seed", type=int, default=0, help="Base seed: game g uses engine seed seed+g and opening key f(seed, g)"
    )
    gen.add_argument(
        "--opening-random-plies",
        type=int,
        default=4,
        help="Opening key depth k: deterministic stratified plies before Pentobi takes over — game i's first ply "
        "sweeps the legal first moves interleaved, plies 2..k are keyed by (seed, game_id) (0 = seed variation only)",
    )
    gen.add_argument(
        "--games-per-shard",
        type=int,
        default=10,
        help="Games per shard file — the resume and load-balancing unit (~1 h of L9 work per shard)",
    )
    gen.add_argument("--binary", default=None, help="pentobi-gtp path (default: $PENTOBI_GTP_PATH or the box build)")
    gen.set_defaults(func=generate)

    ana = sub.add_parser("analyze", help="Diversity report over an existing corpus directory")
    ana.add_argument("--data", required=True, help="Corpus directory")
    ana.add_argument("--json", default=None, help="Also write the report as JSON to this path")
    ana.set_defaults(func=analyze)

    val = sub.add_parser("validate", help="Replay-validate every stored row of a corpus")
    val.add_argument("--data", required=True, help="Corpus directory")
    val.set_defaults(func=validate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
