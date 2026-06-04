"""Stratified random Blokus Duo positions for move-gen equivalence tests.

What this module provides
-------------------------

- :func:`generate_position_sequences` — yields action-sequence rows for
  N stratified random Blokus Duo positions. Each row is a sequence of
  legal action IDs that, when replayed starting from a fresh board,
  reaches a specific game state.
- :func:`build_cache` — runs the generator and writes a ``.npz`` cache.
- :func:`load_cache` — loads a cache and returns the action sequences.
- :func:`replay_to_board` — applies a sequence of actions to a fresh
  board, producing the position.
- :func:`replay_to_board_and_player` — same but also returns the
  player-to-move (some valid_moves consumers need it).

Why action sequences, not boards
--------------------------------

The equivalence test compares ``valid_moves()`` between two
implementations of the move generator on identical positions. The
cleanest way to guarantee both implementations see the *same* position
is to play the same action sequence through both implementations,
each starting from a fresh board.

Action sequences are also stable across implementation changes: a
sequence cached today still works after the table-driven generator
lands, because the action encoding doesn't change (only the move-gen
logic does).

Stratification
--------------

Bug surface is concentrated in early- and mid-game positions where
attach-point lists are long and the corner/edge rules are most active.
Late-game positions have few legal moves and few subtle constraints —
they're cheap to get right. So we deliberately oversample early/mid:

- Empty (0 moves)  : 5%   — tests the starting-cell rule
- Early (1-8)      : 35%  — wide attach-point fanout
- Mid (9-18)       : 40%  — the busiest constraint region
- Late (19-26)     : 15%  — sparse boards, but edge cases
- End (27+)        : 5%   — forced-pass and game-end conditions

If a position's game ends before reaching the target move count, we
yield the current state anyway — those forced-end positions are
themselves interesting test cases.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

    from games.blokusduo.board import BlokusDuoBoard
    from games.blokusduo.game import BlokusDuoGame


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Phase:
    """A game-phase bucket — range of move counts and sampling weight."""
    name: str
    min_moves: int
    max_moves: int
    weight: float


# Phase weights designed to oversample where the move-gen invariants are
# hardest. Sum to 1.0.
PHASES: tuple[Phase, ...] = (
    Phase("empty", 0, 0, 0.05),
    Phase("early", 1, 8, 0.35),
    Phase("mid", 9, 18, 0.40),
    Phase("late", 19, 26, 0.15),
    Phase("end", 27, 40, 0.05),
)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

# Sentinel value in the action-sequence array for "no action here" (padding).
# Chosen as -1 so any legitimate action ID (which are non-negative) is
# distinguishable.
PAD_ACTION: int = -1


def generate_position_sequences(
    game: BlokusDuoGame, n: int, *, seed: int = 42,
) -> Iterator[NDArray]:
    """Yield N action sequences that reach stratified random positions.

    Each yielded sequence is a 1D numpy ``int32`` array of action IDs,
    with no padding (use the array's length to know how many moves).
    Replay via :func:`replay_to_board` to get the position itself.

    The current ``BlokusDuoGame.valid_move_masking`` is used to pick
    each step's random legal move — so the new table-driven move-gen
    will be tested on positions IT didn't pick, which is what we want.

    Yields:
        numpy int32 arrays, one per requested position.
    """
    rng = np.random.default_rng(seed)
    phase_weights = np.array([p.weight for p in PHASES], dtype=float)
    phase_weights /= phase_weights.sum()  # normalise just in case

    for _ in range(n):
        phase_idx = int(rng.choice(len(PHASES), p=phase_weights))
        phase = PHASES[phase_idx]
        target_moves = int(rng.integers(phase.min_moves, phase.max_moves + 1))

        actions: list[int] = []
        board = game.initialise_board()
        player = 1
        for _ in range(target_moves):
            valids = game.valid_move_masking(board, player)
            legal_action_ids = np.where(valids > 0)[0]
            if len(legal_action_ids) == 0:
                # Game ended early; the partial position is still a
                # valid test case so we yield what we have.
                break
            action_id = int(rng.choice(legal_action_ids))
            board, player = game.get_next_state(board, player, action_id)
            actions.append(action_id)
            if game.get_game_ended(board, player) != 0:
                break

        yield np.array(actions, dtype=np.int32)


# ---------------------------------------------------------------------------
# Cache: build, save, load
# ---------------------------------------------------------------------------

def _default_cache_dir() -> Path:
    """Where cached fixtures live by default — alongside this module."""
    return Path(__file__).resolve().parent / "blokus_duo_positions"


def build_cache(
    game: BlokusDuoGame, n: int, *, seed: int = 42,
    cache_dir: Path | None = None, label: str | None = None,
) -> Path:
    """Generate N positions and write them to a .npz cache.

    Two arrays are stored:

    - ``actions`` — ``int32`` shape ``(N, max_moves)``, padded with
      :data:`PAD_ACTION` so the array is rectangular.
    - ``n_moves`` — ``int32`` shape ``(N,)``, the actual move count
      per position so we know how much of each row is real.

    The output filename is derived from ``label or f"{n}_seed{seed}"``,
    e.g. ``dev_5000_seed42.npz`` or ``gauntlet_50000_seed42.npz``.
    """
    cache_dir = cache_dir or _default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    label = label or f"{n}_seed{seed}"
    out_path = cache_dir / f"{label}.npz"

    start = time.perf_counter()
    sequences = list(generate_position_sequences(game, n, seed=seed))
    gen_elapsed = time.perf_counter() - start

    max_moves = max((len(s) for s in sequences), default=0)
    actions_array = np.full((n, max_moves), PAD_ACTION, dtype=np.int32)
    n_moves_array = np.zeros(n, dtype=np.int32)
    for i, seq in enumerate(sequences):
        actions_array[i, : len(seq)] = seq
        n_moves_array[i] = len(seq)

    np.savez_compressed(
        out_path,
        actions=actions_array,
        n_moves=n_moves_array,
        seed=np.array([seed], dtype=np.int32),
        n_positions=np.array([n], dtype=np.int32),
    )

    print(f"Generated {n} positions in {gen_elapsed:.1f}s "
          f"(avg {gen_elapsed * 1000 / max(n, 1):.1f}ms/position).")
    print(f"Cache size: {out_path.stat().st_size / 1024:.1f} KB at {out_path}")
    return out_path


def load_cache(path: Path) -> tuple[NDArray, NDArray]:
    """Load a cache file and return ``(actions, n_moves)``.

    Returns:
        - ``actions``: int32 ``(N, max_moves)`` with PAD_ACTION padding.
        - ``n_moves``: int32 ``(N,)`` actual move count per position.
    """
    with np.load(path) as data:
        return data["actions"].copy(), data["n_moves"].copy()


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def replay_to_board(
    game: BlokusDuoGame, actions: NDArray | list[int],
) -> BlokusDuoBoard:
    """Apply ``actions`` (a 1D array of action IDs) to a fresh board.

    PAD_ACTION entries are ignored so this works with rows pulled
    straight from the cache's actions array.
    """
    board = game.initialise_board()
    player = 1
    for action_id in actions:
        action_id = int(action_id)
        if action_id == PAD_ACTION:
            break
        board, player = game.get_next_state(board, player, action_id)
    return board


def replay_to_board_and_player(
    game: BlokusDuoGame, actions: NDArray | list[int],
) -> tuple[BlokusDuoBoard, int]:
    """Like :func:`replay_to_board` but also returns the player-to-move."""
    board = game.initialise_board()
    player = 1
    for action_id in actions:
        action_id = int(action_id)
        if action_id == PAD_ACTION:
            break
        board, player = game.get_next_state(board, player, action_id)
    return board, player


def iter_cached_positions(
    game: BlokusDuoGame, path: Path,
) -> Iterator[tuple[BlokusDuoBoard, int, NDArray]]:
    """Iterate ``(board, player_to_move, action_sequence)`` triples from a cache file.

    Convenience helper for tests — avoids each test having to handle the
    npz format directly.
    """
    actions_array, n_moves_array = load_cache(path)
    for i in range(len(n_moves_array)):
        n_moves = int(n_moves_array[i])
        sequence = actions_array[i, :n_moves]
        board, player = replay_to_board_and_player(game, sequence)
        yield board, player, sequence
