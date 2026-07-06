"""Post-hoc pool BayesElo tournament over a finished run's checkpoints.

The primary deliverable of ``docs/plans/archive/pool-based-elo.md``. It turns the
per-generation checkpoints a run already saved (``Nets/accepted_<N>.pth.tar`` +
the gen-0 ``elo_baseline.pth.tar``) into a proper, non-saturating Elo curve — the
way DeepMind measured strength: play a sparse round-robin *among the pool*, then
fit one consistent rating per checkpoint with BayesElo. No retraining; works on
any finished run.

Lives in the package (not just ``scripts/``) so ``cli.main`` can invoke it at
end-of-run when ``RunConfig.tournament.run_at_end`` is set — see
``docs/plans/archive/arena-derived-elo.md`` S8. ``scripts/tournament_elo.py`` is
the standalone CLI wrapper.

Outputs land in ``<run>/Tournament/``:

- ``tournament_ratings.parquet`` — one row per checkpoint (generation, rating,
  n_games, n_pairings), read by the HTML report's pool-Elo chart.
- ``tournament_raw.json`` — the raw pairing W/L/D matrix, so the fit can be
  re-run/audited without replaying games.
"""

from __future__ import annotations

import dataclasses
import json
import re
import time
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from alphablokus import registry
from alphablokus.evaluation.arena import Arena
from alphablokus.evaluation.players import NetworkPlayer
from alphablokus.evaluation.rating import RatingResult, fit_bayeselo
from alphablokus.evaluation.tournament import build_pairings

if TYPE_CHECKING:
    from pathlib import Path

    from alphablokus.config import MCTSConfig, RunConfig

# elo_baseline.pth.tar *is* the gen-0 network, so it anchors the pool at gen 0.
_ANCHOR_FILENAME = "elo_baseline.pth.tar"
_ANCHOR_ID = "gen0"
_ACCEPTED_RE = re.compile(r"^accepted_(\d+)\.pth\.tar$")


def run_tournament(config: RunConfig, *, dry_run: bool = False) -> RatingResult | None:
    """Run the pool tournament and write results. Returns the fit (None on dry-run/too-few).

    Args:
        config: The finished run's config (its ``net_directory`` holds the
            checkpoints; ``tournament`` block parameterises the schedule/fit).
        dry_run: If True, log the schedule and total game count, then return
            without playing anything.
    """
    tour = config.tournament
    checkpoints = _enumerate_checkpoints(config)
    if len(checkpoints) < 2:
        logger.warning(
            "Need >= 2 checkpoints for a tournament; found {} in {}. Nothing to do.",
            len(checkpoints),
            config.net_directory,
        )
        return None

    ids = [pid for pid, _ in checkpoints]
    path_by_id = dict(checkpoints)
    pairings = build_pairings(ids, tour.back_ref_offsets, tour.include_first_last)
    total_games = len(pairings) * tour.games_per_pairing

    logger.info(
        "Pool tournament: {} checkpoints, {} pairings, {} games/pairing = {} games ({} MCTS sims each, {} workers).",
        len(ids),
        len(pairings),
        tour.games_per_pairing,
        total_games,
        tour.num_mcts_sims,
        config.num_parallel_workers,
    )
    if dry_run:
        logger.info("Dry run — schedule only, playing nothing. Pairings:")
        for a, b in pairings:
            logger.info("  {} vs {}", a, b)
        return None

    wins: dict[tuple[str, str], int] = {}
    draws: dict[tuple[str, str], int] = {}
    games_played: dict[str, int] = {pid: 0 for pid in ids}
    pairings_played: dict[str, int] = {pid: 0 for pid in ids}

    start = time.perf_counter()
    for index, (a, b) in enumerate(pairings, start=1):
        a_wins, b_wins, drawn = _play_pairing(config, path_by_id[a], path_by_id[b], tour.games_per_pairing)
        wins[(a, b)] = wins.get((a, b), 0) + a_wins
        wins[(b, a)] = wins.get((b, a), 0) + b_wins
        draws[(a, b)] = draws.get((a, b), 0) + drawn
        played = a_wins + b_wins + drawn
        for pid in (a, b):
            games_played[pid] += played
            pairings_played[pid] += 1
        logger.info("[{}/{}] {} vs {}: {}-{}-{}", index, len(pairings), a, b, a_wins, b_wins, drawn)

    anchor = _ANCHOR_ID if _ANCHOR_ID in ids else ids[0]
    result = fit_bayeselo(
        ids,
        wins,
        draws,
        prior_games=tour.prior_games,
        anchor=anchor,
        anchor_rating=tour.anchor_rating,
    )
    elapsed = time.perf_counter() - start
    logger.info("Fitted BayesElo in {} iters (converged={}), {:.1f}s.", result.iterations, result.converged, elapsed)

    _write_results(config, ids, path_by_id, result, wins, draws, games_played, pairings_played)
    return result


def _enumerate_checkpoints(config: RunConfig) -> list[tuple[str, Path]]:
    """Ordered ``(player_id, absolute_path)`` for every rateable checkpoint.

    gen-0 (the frozen ``elo_baseline``) comes first if present, then every
    ``accepted_<N>`` in generation order. Subsampled per
    ``tournament.max_checkpoints`` (gen-0 and the final gen are always kept).
    """
    net_dir = config.net_directory
    entries: list[tuple[int, str, Path]] = []

    anchor = net_dir / _ANCHOR_FILENAME
    if anchor.exists():
        entries.append((0, _ANCHOR_ID, anchor.resolve()))

    accepted: list[tuple[int, Path]] = []
    for path in net_dir.glob("accepted_*.pth.tar"):
        match = _ACCEPTED_RE.match(path.name)
        if match:
            accepted.append((int(match.group(1)), path.resolve()))
    for gen, path in sorted(accepted, key=lambda t: t[0]):
        # An accepted_0 would collide with the gen-0 anchor id; skip the dup.
        if gen == 0 and any(e[0] == 0 for e in entries):
            continue
        entries.append((gen, f"gen{gen}", path))

    entries.sort(key=lambda t: t[0])
    subsampled = _subsample(entries, config.tournament.max_checkpoints)
    return [(pid, path) for _, pid, path in subsampled]


def _subsample(
    entries: list[tuple[int, str, Path]],
    max_checkpoints: int | None,
) -> list[tuple[int, str, Path]]:
    """Thin ``entries`` to at most ``max_checkpoints``, keeping first and last."""
    if max_checkpoints is None or len(entries) <= max_checkpoints:
        return entries
    step = -(-len(entries) // max_checkpoints)  # ceil division
    kept = entries[::step]
    if entries[-1] not in kept:
        kept.append(entries[-1])
    return kept


def _tournament_mcts_config(config: RunConfig) -> MCTSConfig:
    """Search config for tournament games: the run's ``mcts_config`` with the
    sim count overridden by the explicit (low) ``tournament.num_mcts_sims``.

    Ranking is robust to weak play, so we deliberately don't inherit the heavy
    training sim count — it would make the tournament take hours.
    """
    return dataclasses.replace(config.mcts_config, num_mcts_sims=config.tournament.num_mcts_sims)


def _play_pairing(
    config: RunConfig,
    path_a: Path,
    path_b: Path,
    num_games: int,
) -> tuple[int, int, int]:
    """Play ``num_games`` between two checkpoints. Returns ``(a_wins, b_wins, draws)``."""
    search_config = _tournament_mcts_config(config)
    if config.num_parallel_workers > 1:
        from alphablokus.parallel.pool import PHASE_ELO, run_two_player_games_parallel

        # The pool reads sims from the config, so hand it a copy carrying the
        # tournament's sim count. Checkpoints load relative to
        # ``config.net_directory``; both live there, so basenames resolve.
        a_wins, b_wins, draws, _ = run_two_player_games_parallel(
            config=dataclasses.replace(config, mcts_config=search_config),
            generation=0,
            checkpoint_a_path=path_a.name,
            checkpoint_b_path=path_b.name,
            num_games=num_games,
            num_workers=config.num_parallel_workers,
            phase=PHASE_ELO,
            record=False,
            top_k=0,
            desc="Tournament",
        )
        return a_wins, b_wins, draws

    game, nnet_a = registry.instantiate_game_and_network(config)
    _, nnet_b = registry.instantiate_game_and_network(config)
    if getattr(config, "use_optimised_movegen", False):
        enable = getattr(game, "enable_optimised_movegen", None)
        if enable is not None:
            enable()
    nnet_a.load_checkpoint(filename=str(path_a))
    nnet_b.load_checkpoint(filename=str(path_b))
    player_a = NetworkPlayer(game=game, nnet=nnet_a, mcts_config=search_config, temp=0.0)
    player_b = NetworkPlayer(game=game, nnet=nnet_b, mcts_config=search_config, temp=0.0)
    a_wins, b_wins, draws, _ = Arena(player_a, player_b, game).play_games(num_games)
    return a_wins, b_wins, draws


def _generation_of(player_id: str) -> int:
    """``"gen12" -> 12``."""
    return int(player_id.removeprefix("gen"))


def _write_results(
    config: RunConfig,
    ids: list[str],
    path_by_id: dict[str, Path],
    result: RatingResult,
    wins: dict[tuple[str, str], int],
    draws: dict[tuple[str, str], int],
    games_played: dict[str, int],
    pairings_played: dict[str, int],
) -> None:
    """Write the ratings parquet + raw W/L/D JSON to ``config.tournament_directory``."""
    out_dir = config.tournament_directory
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "generation": _generation_of(pid),
            "rating": result.ratings[pid],
            "n_games": games_played[pid],
            "n_pairings": pairings_played[pid],
        }
        for pid in ids
    ]
    ratings_df = pd.DataFrame(rows).sort_values("generation").reset_index(drop=True)
    ratings_path = out_dir / "tournament_ratings.parquet"
    ratings_df.to_parquet(ratings_path, index=False)

    raw = {
        "players": ids,
        "checkpoints": {pid: str(path_by_id[pid]) for pid in ids},
        "tournament_config": dataclasses.asdict(config.tournament),
        "mcts_sims": config.tournament.num_mcts_sims,
        "converged": result.converged,
        "iterations": result.iterations,
        "ratings": result.ratings,
        # Serialise the pair-keyed dicts as lists (JSON has no tuple keys).
        "wins": [{"a": a, "b": b, "count": c} for (a, b), c in wins.items()],
        "draws": [{"a": a, "b": b, "count": c} for (a, b), c in draws.items()],
    }
    raw_path = out_dir / "tournament_raw.json"
    raw_path.write_text(json.dumps(raw, indent=2))

    logger.info("Wrote {} ({} checkpoints) and {}.", ratings_path, len(ids), raw_path)
    final = ratings_df.iloc[-1]
    logger.info("Final generation {} pool Elo: {:.0f}.", int(final["generation"]), final["rating"])
