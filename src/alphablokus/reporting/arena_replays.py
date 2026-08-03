"""Arena replay payloads for the report's interactive game browser.

The browser renders boards client-side, so this module reduces each recorded
arena game to compact JSON: for every move, the cells the move added (diffed
from the board state, so it is game-agnostic), a human-readable caption, the
MCTS visit share, and the top alternative moves MCTS considered (as ghost-cell
overlays). The JS app replays those deltas onto a single board with a
move-by-move stepper — no per-turn HTML is embedded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from alphablokus.games.blokusduo.game import BlokusDuoGame
from alphablokus.registry import instantiate_game

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from alphablokus.config import RunConfig
    from alphablokus.interfaces import IBoard, IGame

_REPLAY_MAX_GENERATIONS = 16
_REPLAY_MAX_GAMES_PER_GEN = 6
_MAX_ALTERNATIVES = 3


def _evenly_sample(values: list[int], n: int) -> list[int]:
    """Pick up to ``n`` values spread evenly across ``values`` (sorted),
    always including the first and last. Returns all of them if ``len <= n``."""
    if n <= 0 or len(values) <= n:
        return list(values)
    idxs = {round(i * (len(values) - 1) / (n - 1)) for i in range(n)}
    return sorted(values[i] for i in idxs)


def load_sampled_replays(directory: Path) -> pd.DataFrame | None:
    """Read only the replay slice the viewer renders, not the whole history.

    The ArenaReplays store grows unbounded with generations, but the viewer
    shows at most ``_REPLAY_MAX_GENERATIONS × _REPLAY_MAX_GAMES_PER_GEN``
    games. Sample the generations from the hive partition names first (no row
    data read), then push both the generation sample and the games-per-gen cap
    into the parquet read via ``filters=`` — hive partitioning turns the
    generation filter into per-directory file pruning, so a long run's report
    reads ~100 games instead of everything (oom-hardening O7).

    Returns ``None`` when the directory holds no generation partitions.
    """
    import pandas as pd

    generations = sorted(int(p.name.split("=", 1)[1]) for p in directory.glob("generation=*") if p.is_dir())
    if not generations:
        return None
    sampled = _evenly_sample(generations, _REPLAY_MAX_GENERATIONS)
    return pd.read_parquet(
        directory,
        filters=[
            ("generation", "in", sampled),
            ("game_idx", "<", _REPLAY_MAX_GAMES_PER_GEN),
        ],
    )


def _move_cells(before: IBoard, after: IBoard) -> list[list[int]]:
    """Cells a move added, as ``[row, col, label]`` triples.

    Computed by diffing the 2D board views, so it works for any game. The
    label is the piece id for Blokus (from the signed placement grid) and the
    mover's sign for games without piece identity.
    """
    before_2d = np.asarray(before.as_2d)
    after_2d = np.asarray(after.as_2d)
    changed = np.argwhere(before_2d != after_2d)
    placement = getattr(after, "placement_grid", None)
    cells: list[list[int]] = []
    for row, col in changed:
        if placement is not None:
            label = int(abs(placement[row, col]))
        else:
            label = int(after_2d[row, col])
        cells.append([int(row), int(col), label])
    return cells


def _action_caption(game: IGame[Any], action: int, cells: list[list[int]]) -> str:
    """Human-readable move description; falls back to the raw action id."""
    if isinstance(game, BlokusDuoGame):
        if game.action_codec.is_pass(action):
            return "Pass"
        decoded = game.action_codec.decode(action)
        piece = game.piece_manager.pieces[decoded.piece_id]
        return (
            f"Piece {decoded.piece_id} ({piece.name}, {decoded.orientation.value}) "
            f"at ({decoded.x_coordinate}, {decoded.y_coordinate})"
        )
    if not cells:
        return "Pass"
    row, col = cells[0][0], cells[0][1]
    return f"({row}, {col})"


def _alternative_payloads(
    game: IGame[Any],
    board: IBoard,
    player: int,
    played_action: int,
    top_k_actions: list[int],
    top_k_probs: list[float],
) -> list[dict[str, Any]]:
    """The top alternative moves MCTS considered, as ghost-cell overlays.

    Alternatives are simulated on a copy of the pre-move board and diffed the
    same way as real moves, so the overlay works for any game. Zero-probability
    entries (older parquets persisted padding) are dropped.
    """
    alternatives: list[dict[str, Any]] = []
    for action, prob in sorted(zip(top_k_actions, top_k_probs, strict=False), key=lambda ap: -ap[1]):
        if prob <= 0 or action == played_action:
            continue
        try:
            hypothetical, _ = game.get_next_state(board, player, action)
        except (ValueError, IndexError, KeyError):  # defensive: malformed action in an old parquet
            continue
        cells = _move_cells(board, hypothetical)
        alternatives.append(
            {
                "cells": [[c[0], c[1]] for c in cells],
                "cap": _action_caption(game, action, cells),
                "prob": round(float(prob), 4),
            }
        )
        if len(alternatives) >= _MAX_ALTERNATIVES:
            break
    return alternatives


def build_replay_payload(df: pd.DataFrame, config: RunConfig) -> dict[str, Any] | None:
    """Reduce sampled arena replays to the JSON the browser renders from.

    Player 1 is the previous (incumbent) net and player 2 the new candidate;
    ``outcome`` is stored from Player 1's perspective. Colour (+1 = White,
    first mover) is per-game via ``player1_was_white``.
    """
    game = instantiate_game(config)
    rows, cols = game.get_board_size()

    df = df.copy()
    df["generation"] = df["generation"].astype(int)

    all_gens = sorted(df["generation"].unique())
    sampled_gens = _evenly_sample(all_gens, _REPLAY_MAX_GENERATIONS)
    df = df[df["generation"].isin(sampled_gens) & (df["game_idx"] < _REPLAY_MAX_GAMES_PER_GEN)]
    if df.empty:
        return None
    if len(sampled_gens) < len(all_gens):
        logger.info(
            "Arena replays: rendering {} of {} generations (evenly sampled) × up to {} games/gen to bound report size",
            len(sampled_gens),
            len(all_gens),
            _REPLAY_MAX_GAMES_PER_GEN,
        )
    df = df.sort_values(["generation", "game_idx", "move_idx"])

    games_by_gen: dict[str, list[dict[str, Any]]] = {}
    for (gen, game_idx), group in df.groupby(["generation", "game_idx"]):
        moves = group.sort_values("move_idx")
        first = moves.iloc[0]
        player1_was_white = bool(first["player1_was_white"])
        outcome = float(first["outcome"])

        board = game.initialise_board()
        move_payloads: list[dict[str, Any]] = []
        for _, move_row in moves.iterrows():
            action = int(move_row["action"])
            player = int(move_row["player"])
            top_k_actions = [int(a) for a in move_row["top_k_actions"]]
            top_k_probs = [float(p) for p in move_row["top_k_probs"]]
            if "played_prob" in move_row and move_row["played_prob"] is not None:
                played_prob = float(move_row["played_prob"])
            else:
                visited = {a: p for a, p in zip(top_k_actions, top_k_probs, strict=False) if p > 0}
                played_prob = visited.get(action, 0.0)

            alternatives = _alternative_payloads(game, board, player, action, top_k_actions, top_k_probs)
            next_board, _ = game.get_next_state(board, player, action)
            cells = _move_cells(board, next_board)
            move_payloads.append(
                {
                    "p": player,
                    "cells": cells,
                    "cap": _action_caption(game, action, cells),
                    "prob": round(played_prob, 4) if played_prob > 0 else None,
                    "alts": alternatives,
                }
            )
            board = next_board

        if outcome > 0.5:
            winner, label = "prev", f"{'White' if player1_was_white else 'Black'} wins — previous net"
        elif outcome < -0.5:
            winner, label = "new", f"{'Black' if player1_was_white else 'White'} wins — new net"
        else:
            winner, label = "draw", "Draw"

        games_by_gen.setdefault(str(int(gen)), []).append(
            {
                "idx": int(game_idx),
                "winner": winner,
                "label": label,
                "p1_white": player1_was_white,
                "moves": move_payloads,
            }
        )

    return {
        "game": config.game,
        "rows": rows,
        "cols": cols,
        "gens": games_by_gen,
    }
