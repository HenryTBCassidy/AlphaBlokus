"""Verify our action-space encoding agrees with Pentobi's move count.

The Blokus Duo action space has 17,837 entries: 14×14 = 196 anchor cells ×
91 piece-orientations + 1 pass action = 17,836 placement actions + 1 pass.

Many of those 17,836 placements are not physically possible because the
piece extends off the board (e.g. a 5-cell vertical piece anchored on row
13 would have cells at rows 13, 14, 15, 16, 17 — only the first is on the
board).

Pentobi's source (`libpentobi_base/Move.h:32`) reports **13,729** legal
on-board placements for Blokus Duo. If our count disagrees, our F2 plan
(which sizes precomputed tables to 13,729 entries) is broken from the
start and we need to investigate before doing any further F2 work.

Possible reasons for a mismatch:
- Our ``pieces.json`` definitions differ from Pentobi's piece shapes.
- Our anchor-cell convention differs (which cell of the piece is the
  "reference cell" that the action coordinate specifies).
- Pentobi excludes pass / starts at 1 / has some other off-by-one.

Usage::

    uv run python -m scripts.count_onboard_placements

Exits with status 0 if the count matches 13,729, status 1 with a diff
report otherwise. The diff report classifies mismatches per (piece, x-row,
y-row) so the cause is easy to localise.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from games.blokusduo.board import ActionCodec, CoordinateIndexDecoder
from games.blokusduo.pieces import pieces_loader

# Pentobi's hard-coded Duo move count, from libpentobi_base/Move.h:32.
PENTOBI_DUO_MOVE_COUNT = 13_729

PIECES_PATH = Path(__file__).resolve().parent.parent / "games" / "blokusduo" / "pieces.json"
BOARD_SIZE = 14


def count_onboard_placements() -> tuple[int, list[dict]]:
    """Enumerate every (cell × piece-orientation) action and count those
    whose piece footprint fits entirely within the 14×14 board.

    Returns:
        (onboard_count, offboard_samples) — the count, plus the first
        few off-board placements with details (for diff-report use).
    """
    piece_manager = pieces_loader(PIECES_PATH)
    codec = ActionCodec(board_size=BOARD_SIZE, piece_manager=piece_manager)
    decoder = CoordinateIndexDecoder(board_size=BOARD_SIZE)

    onboard = 0
    offboard_by_piece: dict[int, int] = defaultdict(int)
    offboard_samples: list[dict] = []

    # Skip the last action (the pass action). 0..17_835 are placement actions.
    for action_id in range(codec.pass_action_index):
        action = codec.decode(action_id)
        piece_orientation = piece_manager.get_piece_orientation_array(
            action.piece_id, action.orientation,
        )
        length_idx, width_idx = decoder.to_idx(
            coordinate=(action.x_coordinate, action.y_coordinate),
        )

        # Iterate the piece's grid; check that every filled cell is on the board.
        # Mirrors the loop inside BlokusDuoBoard.with_piece, but without
        # mutating any state.
        p_len, p_wid = piece_orientation.shape
        fits = True
        for i in range(p_len):
            for j in range(p_wid):
                if piece_orientation[i, j] == 0:
                    continue
                ri, ci = length_idx + i, width_idx + j
                if not (0 <= ri < BOARD_SIZE and 0 <= ci < BOARD_SIZE):
                    fits = False
                    break
            if not fits:
                break

        if fits:
            onboard += 1
        else:
            offboard_by_piece[action.piece_id] += 1
            if len(offboard_samples) < 5:
                offboard_samples.append({
                    "action_id": action_id,
                    "piece_id": action.piece_id,
                    "orientation": str(action.orientation),
                    "anchor": (action.x_coordinate, action.y_coordinate),
                    "piece_shape": piece_orientation.tolist(),
                })

    return onboard, offboard_samples


def main() -> int:
    print(f"Enumerating the {ActionCodec(BOARD_SIZE, pieces_loader(PIECES_PATH)).pass_action_index} "
          f"placement actions (skipping pass)...")
    print()

    onboard, offboard_samples = count_onboard_placements()
    pass_action_index = (
        BOARD_SIZE * BOARD_SIZE
        * pieces_loader(PIECES_PATH).num_entries
    )
    offboard = pass_action_index - onboard

    print(f"  On-board placements:   {onboard:>6,}")
    print(f"  Off-board placements:  {offboard:>6,}")
    print(f"  Total placement slots: {pass_action_index:>6,}  "
          f"(= {BOARD_SIZE}^2 × {pieces_loader(PIECES_PATH).num_entries} orientations)")
    print()
    print(f"  Pentobi (Move.h:32):   {PENTOBI_DUO_MOVE_COUNT:>6,}")
    print(f"  Difference:            {onboard - PENTOBI_DUO_MOVE_COUNT:>+6,}")
    print()

    if onboard == PENTOBI_DUO_MOVE_COUNT:
        print("MATCH. F2 can safely size its precomputed tables to 13,729 entries.")
        return 0

    print(f"MISMATCH. Our on-board count ({onboard:,}) disagrees with "
          f"Pentobi's ({PENTOBI_DUO_MOVE_COUNT:,}).")
    print()
    print("First few off-board samples (for diff investigation):")
    for sample in offboard_samples:
        print(f"  action_id={sample['action_id']:5d}  "
              f"piece={sample['piece_id']:2d}  "
              f"orient={sample['orientation']:8s}  "
              f"anchor=({sample['anchor'][0]:2d},{sample['anchor'][1]:2d})  "
              f"shape={sample['piece_shape']}")
    print()
    print("Investigate before proceeding with F2:")
    print(" - Compare pieces.json shapes against Pentobi's piece definitions.")
    print(" - Check our anchor-cell convention vs Pentobi's "
          "(libpentobi_base/PieceTransformsClassic.cpp).")
    print(" - Confirm Pentobi's 13,729 is the Duo count and not another variant.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
