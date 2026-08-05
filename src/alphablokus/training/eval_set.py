"""Held-out eval set for per-epoch network diagnostics.

Rebuilt every ``RunConfig.eval_set_rebuild_every`` generations from the current
replay buffer, so the diagnostics measure the net against positions it is
actually meeting rather than against the run's weakest-ever data. Setting the
knob to 0 restores the historical behaviour (build once from generation 1 and
freeze it) — which is how a run's dashboards could read healthy while its real
strength fell: the frozen set got easier as the net improved, and every metric
computed from it drifted for that reason alone.

Every position records the **source game** it came from. Positions within a game
share one outcome label, so any interval over this set has to be a game-cluster
bootstrap (:mod:`alphablokus.bootstrap`).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from alphablokus.storage.metrics import EvalSet
from alphablokus.storage.sparse_policy import as_dense
from alphablokus.training.replay_buffer import game_fingerprint

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from alphablokus.config import RunConfig
    from alphablokus.interfaces import IGame, IOracle
    from alphablokus.selfplay.episode import GameExamples

# Metadata sidecar: records how the targets were made *and* which generation's
# buffer they came from. ``targets_kind.txt`` predates it and is still written
# so an older reader (and the existing on-disk sets) keep working.
METADATA_FILENAME = "metadata.json"

# Content hashes of the eval set's source games. Read back on resume so those
# games stay out of training even though buffer indices have not survived.
FINGERPRINTS_FILENAME = "source_fingerprints.json"

# Positions taken from any one source game. Low on purpose: the source games are
# withheld from training, so a set of ``size`` positions costs ``size /
# MAX_EVAL_POSITIONS_PER_GAME`` games of training data — cheap against a 60k-game
# buffer — and every extra distinct game is a genuine extra independent
# observation, which is what the confidence intervals depend on. Positions within
# one game share an outcome label and include symmetry twins, so the second
# position from a game adds much less than the first.
MAX_EVAL_POSITIONS_PER_GAME = 2

# Ceiling on the share of the buffer's games the eval set may claim. The source
# games are withheld from training, so without a cap a small buffer could hand its
# entire contents to the eval set and leave nothing to train on. At production
# scale the cap never binds — 200 positions needs ~100 games out of 60,000 (0.17%)
# — but it keeps short runs and tests honest, shrinking the eval set rather than
# starving training.
MAX_EVAL_GAME_FRACTION = 0.2


def _write_metadata(eval_dir: Path, *, targets_kind: str, generation: int, n_games: int, n_positions: int) -> None:
    """Record the eval set's provenance next to the arrays."""
    payload = {
        "targets_kind": targets_kind,
        "built_at_generation": generation,
        "n_source_games": n_games,
        "n_positions": n_positions,
    }
    (eval_dir / METADATA_FILENAME).write_text(json.dumps(payload, indent=2))


def _read_metadata(eval_dir: Path) -> dict[str, object]:
    """Read the provenance sidecar, tolerating its absence on older runs."""
    path = eval_dir / METADATA_FILENAME
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        # Provenance is a diagnostic aid; a corrupt sidecar must not sink a run.
        logger.warning("Could not read eval-set metadata at {} — treating as absent", path)
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _read_fingerprints(path: Path) -> tuple[str, ...]:
    """Read the source-game fingerprints, tolerating their absence.

    An eval set built before fingerprints existed has none. In that case the
    Coach cannot withhold its source games — the set is *not* held out — and
    ``Coach._ensure_eval_set`` warns and rebuilds rather than reporting in-sample
    numbers as held-out.
    """
    if not path.exists():
        return ()
    try:
        loaded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read eval-set fingerprints at {} — treating as absent", path)
        return ()
    if not isinstance(loaded, list):
        return ()
    return tuple(str(item) for item in loaded)


def should_rebuild(generation: int, rebuild_every: int) -> bool:
    """Whether generation ``generation`` should resample the eval set.

    Args:
        generation: 1-based generation about to train.
        rebuild_every: Cadence from ``RunConfig.eval_set_rebuild_every``; 0
            disables rebuilding (build once, then freeze).

    Returns:
        True on the generations whose buffer should become the new eval set.
        Generation 1 always builds (there is nothing to reuse yet); after that a
        rebuild happens every ``rebuild_every`` generations.
    """
    if rebuild_every <= 0:
        return False
    return generation > 1 and (generation - 1) % rebuild_every == 0


def build_or_load_eval_set(
    config: RunConfig,
    game: IGame,
    oracle: IOracle | None,
    games: Sequence[GameExamples],
    size: int,
    *,
    generation: int = 1,
    force_rebuild: bool = False,
) -> EvalSet | None:
    """Build (or load from disk) the held-out eval set.

    When the game has a perfect-play ``oracle``, the self-play-derived targets
    are replaced with oracle targets: each position's ``target_policies`` row
    is uniform over all game-theoretically optimal actions, and
    ``target_values`` is the true minimax value of the position. This makes
    the per-generation top-K agreement plot answer the right question ("does
    the net pick a truly optimal move?") rather than chasing gen-1's noisy
    MCTS targets, which is what was making that curve dip over training.

    Without an oracle the eval set targets are MCTS visit distributions and
    final game outcomes recorded during self-play.

    Positions are drawn uniformly at random from every position in ``games``, so
    the sampling distribution is unchanged from when this function took a flat
    list — what is new is that each drawn position keeps the index of the game it
    came from, and that the whole set can be resampled later in the run.

    Args:
        config: Run config; supplies ``eval_set_directory`` and ``seed``.
        game: Game implementation, for action-space size and board encoding.
        oracle: Perfect-play oracle when the game has one, else ``None``.
        games: The replay buffer's games, boundaries intact. Passing games
            rather than a flat list is what makes source-game provenance
            possible — see :attr:`EvalSet.source_game_ids`.
        size: Target number of positions.
        generation: Generation being trained, recorded as the set's vintage.
        force_rebuild: Resample even when a usable set exists on disk.

    Returns:
        The eval set, or ``None`` when there is nothing on disk and no games to
        sample from yet.
    """
    eval_dir = config.eval_set_directory
    boards_path = eval_dir / "boards.npy"
    policies_path = eval_dir / "target_policies.npy"
    values_path = eval_dir / "target_values.npy"
    # Compact (canonical int8) boards let the diagnostic rebuild playable
    # positions and search them with the current net's MCTS. Optional: absent
    # for eval sets built before this was persisted, so we load it only if there.
    compact_path = eval_dir / "compact_boards.npy"
    # Source game per position. Optional for the same reason — an eval set built
    # before provenance existed has no ids, and diagnostics that need intervals
    # skip rather than resample positions as if they were independent.
    game_ids_path = eval_dir / "source_game_ids.npy"
    # Content hashes of the source games, so training can withhold them across a
    # resume (buffer indices do not survive; content does).
    fingerprints_path = eval_dir / FINGERPRINTS_FILENAME
    # Marker file: tells us *how* the targets were generated. We refuse to
    # reuse an on-disk eval set whose targets don't match the current
    # scheme — otherwise an old "selfplay-targets" file would silently
    # poison the metrics on a run that now expects oracle targets.
    marker_path = eval_dir / "targets_kind.txt"
    expected_kind = "minimax_v1" if oracle is not None else "selfplay_v1"
    reusable = (
        boards_path.exists()
        and policies_path.exists()
        and values_path.exists()
        and marker_path.exists()
        and marker_path.read_text().strip() == expected_kind
    )
    if reusable and not force_rebuild:
        metadata = _read_metadata(eval_dir)
        built_at = metadata.get("built_at_generation")
        eval_set = EvalSet(
            boards=np.load(boards_path),
            target_policies=np.load(policies_path),
            target_values=np.load(values_path),
            compact_boards=np.load(compact_path) if compact_path.exists() else None,
            source_game_ids=np.load(game_ids_path) if game_ids_path.exists() else None,
            source_fingerprints=_read_fingerprints(fingerprints_path),
            built_at_generation=int(built_at) if isinstance(built_at, int) else None,
        )
        logger.info(
            "Loaded eval set ({} positions, {} source games, kind={}, built at gen {}) from {}",
            len(eval_set),
            eval_set.n_source_games,
            expected_kind,
            eval_set.built_at_generation,
            eval_dir,
        )
        return eval_set

    non_empty = [(index, examples) for index, examples in enumerate(games) if examples]
    if not non_empty:
        return None

    # Sample at **game** granularity, then take a few positions from each chosen
    # game. Two reasons, both load-bearing:
    #
    # 1. The chosen games are withheld from training entirely
    #    (``ReplayBuffer.exclude_games``). Holding out individual positions cannot
    #    work: their symmetry twins and same-game siblings carry the same outcome
    #    label, so the answer leaks anyway.
    # 2. Capping positions per game maximises the number of independent lineages
    #    for a given set size, which is what the confidence intervals actually
    #    depend on — 200 positions spread over 100 games carries far more
    #    information than 200 spread over 3.
    #
    # Vary the draw with the vintage, otherwise every rebuild at the same seed
    # would pick the same games. ``config.seed`` may legitimately be 0, so test
    # for None explicitly rather than using ``or``.
    base_seed = 0 if config.seed is None else config.seed
    rng = np.random.default_rng(seed=base_seed + generation)
    order = rng.permutation(len(non_empty))

    # Never claim more than a fixed share of the buffer, since the chosen games
    # leave training entirely. With a single game in the buffer there is no
    # holdout that leaves anything to train on, so build no eval set at all
    # rather than withholding the whole buffer — `max(1, ...)` would otherwise
    # claim the sole game and `flat_examples()` would return nothing, silently
    # skipping training on one-game runs.
    if len(non_empty) < 2:
        logger.warning(
            "Not building an eval set: the buffer holds {} non-empty game(s), and withholding any of "
            "them would leave no training data. Raise num_eps to get a held-out set.",
            len(non_empty),
        )
        return None

    max_games = max(1, int(len(non_empty) * MAX_EVAL_GAME_FRACTION))
    budget = min(size, max_games * MAX_EVAL_POSITIONS_PER_GAME)
    if budget < size:
        logger.warning(
            "Eval set capped at {} positions (asked for {}): only {} games in the buffer, and at most "
            "{:.0%} of them ({}) may be withheld from training. Raise num_eps or lower the eval-set "
            "size if you need a bigger holdout.",
            budget,
            size,
            len(non_empty),
            MAX_EVAL_GAME_FRACTION,
            max_games,
        )

    selected: list[tuple[int, int]] = []  # (game index, position index within game)
    for slot in order[:max_games]:
        game_index, examples = non_empty[slot]
        take = min(MAX_EVAL_POSITIONS_PER_GAME, len(examples), budget - len(selected))
        if take <= 0:
            break
        picks = rng.choice(len(examples), size=take, replace=False)
        selected.extend((game_index, int(pick)) for pick in picks)
        if len(selected) >= budget:
            break

    if not selected:
        return None

    games_by_index = dict(non_empty)
    sampled = [games_by_index[game_index][position] for game_index, position in selected]
    source_game_ids = np.array([game_index for game_index, _ in selected], dtype=np.int32)
    # Fingerprints of the source games, so training can withhold them for as long
    # as this eval set is in use — including across a resume, where buffer indices
    # are meaningless but content hashes still match.
    source_fingerprints = sorted({game_fingerprint(games_by_index[game_index]) for game_index, _ in selected})

    # Boards are stored **compact** (``to_compact()``); the eval set feeds
    # boards straight to the network, so encode the sampled compact boards
    # to dense planes here. Policies are stored sparse (indices, values) —
    # densify to the full action-space vector the eval set holds.
    action_size = game.get_action_size()
    n = len(sampled)
    sampled_compact = [ex[0] for ex in sampled]
    compact_boards = np.array(sampled_compact)
    sampled_boards = np.array([game.encode_compact(b) for b in sampled_compact])
    target_policies = np.array([as_dense(ex[1], action_size) for ex in sampled])
    target_values = np.array([ex[2] for ex in sampled])

    if oracle is not None:
        target_policies, target_values = oracle.eval_targets(
            sampled_compact,
            action_size=action_size,
        )

    eval_set = EvalSet(
        boards=sampled_boards,
        target_policies=target_policies,
        target_values=target_values,
        compact_boards=compact_boards,
        source_game_ids=source_game_ids,
        source_fingerprints=tuple(source_fingerprints),
        built_at_generation=generation,
    )

    eval_dir.mkdir(parents=True, exist_ok=True)
    np.save(boards_path, eval_set.boards)
    np.save(policies_path, eval_set.target_policies)
    np.save(values_path, eval_set.target_values)
    np.save(compact_path, compact_boards)
    np.save(game_ids_path, source_game_ids)
    fingerprints_path.write_text(json.dumps(source_fingerprints, indent=2))
    marker_path.write_text(expected_kind)
    n_games = eval_set.n_source_games or 0
    _write_metadata(
        eval_dir,
        targets_kind=expected_kind,
        generation=generation,
        n_games=n_games,
        n_positions=n,
    )
    logger.info(
        "Built eval set ({} positions from {} source games, {:.1f} positions/game, kind={}, gen {}) → {}. "
        "Those {} games are withheld from training.",
        n,
        n_games,
        n / n_games if n_games else 0.0,
        expected_kind,
        generation,
        eval_dir,
        len(source_fingerprints),
    )
    return eval_set
