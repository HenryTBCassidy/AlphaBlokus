"""Play TicTacToe interactively against a trained model.

A terminal UI for putting a checkpoint through its paces. Renders a 3×3 grid
with action indices on empty squares; you type the number of the square you
want to play. The model uses the same MCTS settings as its training config
(``config.mcts_config.num_mcts_sims``) and plays the deterministic best move
(``temp=0``).

Examples::

    # Play first (you're X) against a trained model.
    uv run python -m scripts.play_ttt \\
        --checkpoint temp/ttt_full/Nets/best.pth.tar \\
        --config run_configurations/ttt_full.json

    # Play second (you're O).
    uv run python -m scripts.play_ttt \\
        --checkpoint temp/ttt_full/Nets/best.pth.tar \\
        --config run_configurations/ttt_full.json \\
        --model-first

The grid layout matches the action index space. Action ``a`` places at
column ``a // 3``, row ``a % 3``::

     0 | 3 | 6
    ---+---+---
     1 | 4 | 7
    ---+---+---
     2 | 5 | 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from core.config import load_args
from core.players import NetworkPlayer
from games.tictactoe.game import TicTacToeGame


def _render_board(board, current_player: int) -> str:
    """Return a printable 3×3 grid showing X/O placements and action indices
    on empty cells. ``board`` is in raw (not canonical) form; we colour by the
    actual ``+1`` / ``-1`` cell value.
    """
    arr = board.as_2d  # shape (3, 3), indexed [col, row]
    cells = []
    for action in range(9):
        col, row = divmod(action, 3)
        v = int(arr[col, row])
        if v == 1:
            cells.append(" X ")
        elif v == -1:
            cells.append(" O ")
        else:
            cells.append(f" {action} ")
    rows = []
    for row in range(3):
        rows.append(cells[row] + "|" + cells[3 + row] + "|" + cells[6 + row])
    return ("\n" + "---+---+---\n").join(rows)


def _ask_human_for_action(valids) -> int:
    legal = [i for i in range(len(valids)) if valids[i] > 0]
    legal_str = ", ".join(str(i) for i in legal)
    while True:
        try:
            raw = input(f"Your move (legal: {legal_str}): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborting.")
            sys.exit(0)
        try:
            action = int(raw)
        except ValueError:
            print(f"Please enter an integer from: {legal_str}")
            continue
        if action not in legal:
            print(f"{action} isn't legal here. Legal: {legal_str}")
            continue
        return action


def main() -> None:
    parser = argparse.ArgumentParser(description="Play TicTacToe against a trained model.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pth.tar checkpoint")
    parser.add_argument(
        "--config", default="run_configurations/smoke_test.json",
        help="RunConfig JSON whose net architecture matches the checkpoint.",
    )
    parser.add_argument(
        "--model-first", action="store_true",
        help="Model plays X (first). Default: human plays X.",
    )
    args = parser.parse_args()

    config = load_args(args.config)
    if config.game != "tictactoe":
        raise SystemExit(f"play_ttt only supports tictactoe; config is {config.game!r}")

    game = TicTacToeGame()
    from games.tictactoe.neuralnets.wrapper import NNetWrapper
    wrapper = NNetWrapper(game, config)

    ckpt_path = Path(args.checkpoint)
    map_location = None if config.net_config.cuda else "cpu"
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    wrapper.nnet.load_state_dict(ckpt["state_dict"])

    model_player = NetworkPlayer(
        game=game, nnet=wrapper, mcts_config=config.mcts_config, temp=0.0,
    )

    human_plays_x = not args.model_first
    print(f"You play {'X (first)' if human_plays_x else 'O (second)'}.")
    print(f"Model: {ckpt_path.name} | MCTS sims: {config.mcts_config.num_mcts_sims}")

    board = game.initialise_board()
    current_player = 1  # X always starts

    while True:
        print(_render_board(board, current_player))
        outcome = game.get_game_ended(board, 1)
        if outcome != 0:
            if outcome == 1:
                print("\nX wins!" if human_plays_x else "\nModel (X) wins!")
            elif outcome == -1:
                print("\nO wins!" if not human_plays_x else "\nModel (O) wins!")
            else:
                print("\nDraw.")
            return

        is_human_turn = (current_player == 1) == human_plays_x
        canonical = game.get_canonical_form(board, current_player)
        valids = game.valid_move_masking(canonical, 1)

        if is_human_turn:
            action = _ask_human_for_action(valids)
        else:
            print("Model thinking...")
            action = model_player(canonical)
            print(f"Model plays {action}.")

        board, current_player = game.get_next_state(board, current_player, action)


if __name__ == "__main__":
    main()
