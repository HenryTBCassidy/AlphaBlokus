"""Print a recorded arena game in the terminal.

Loads a single recorded game from a run's ``ArenaReplays/`` parquet and
step-prints the board and top-K candidate moves at each turn. Useful for
ad-hoc inspection without opening the HTML report.

Example::

    uv run python -m scripts.replay \\
        --run temp/ttt_mac_demo \\
        --gen 11 \\
        --game 3 \\
        --top-k 3

(The companion interactive HTML widget lives in ``Reporting/report.html``
under the "Arena Game Replays" section.)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger


def _instantiate_game(game_name: str):
    if game_name == "tictactoe":
        from games.tictactoe.game import TicTacToeGame
        return TicTacToeGame()
    if game_name == "blokusduo":
        from games.blokusduo.game import BlokusDuoGame
        return BlokusDuoGame(Path("games/blokusduo/pieces.json"))
    raise ValueError(f"Unknown game: {game_name}")


def _infer_game_from_run(run_dir: Path) -> str:
    """Read the run's config to figure out which game was trained."""
    import json
    # Each run directory has a sibling JSON config — we don't strictly know its
    # name, so look for any file with a ``game`` key.
    config_dir = Path("run_configurations")
    candidate = config_dir / f"{run_dir.name}.json"
    if candidate.exists():
        return json.loads(candidate.read_text())["game"]
    # Fallback: scan the directory and pick whichever has the right run_name.
    for c in config_dir.glob("*.json"):
        try:
            obj = json.loads(c.read_text())
            if obj.get("run_name") == run_dir.name:
                return obj["game"]
        except Exception:
            continue
    raise SystemExit(
        f"Couldn't infer game for run {run_dir.name!r}; checked "
        f"{candidate} and all configs in {config_dir}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a recorded arena game in the terminal.")
    parser.add_argument("--run", required=True, help="Path to the run directory, e.g. temp/ttt_mac_demo")
    parser.add_argument("--gen", type=int, required=True, help="Generation index")
    parser.add_argument("--game", type=int, required=True, help="Game index within the gen (0-based)")
    parser.add_argument("--top-k", type=int, default=3, help="How many top moves to print per turn")
    args = parser.parse_args()

    run_dir = Path(args.run)
    replays_dir = run_dir / "ArenaReplays"
    if not replays_dir.exists():
        raise SystemExit(f"No ArenaReplays directory at {replays_dir}.")

    df = pd.read_parquet(replays_dir)
    df["generation"] = df["generation"].astype(int)
    df["game_idx"] = df["game_idx"].astype(int)

    sub = df[(df["generation"] == args.gen) & (df["game_idx"] == args.game)]
    if sub.empty:
        raise SystemExit(
            f"No game at gen={args.gen} game={args.game} in {replays_dir}."
        )
    sub = sub.sort_values("move_idx")

    game_name = _infer_game_from_run(run_dir)
    game = _instantiate_game(game_name)
    board = game.initialise_board()
    cur_player = 1

    outcome = float(sub["outcome"].iloc[0])
    p1_white = bool(sub["player1_was_white"].iloc[0])
    if outcome > 0.5:
        result_str = "Player 1 (previous best) won"
    elif outcome < -0.5:
        result_str = "Player 2 (new candidate) won"
    else:
        result_str = "Draw"

    print(f"Run: {run_dir.name} · Game: {game_name}")
    print(f"Generation {args.gen}, Game {args.game} — {result_str}")
    print(f"Player 1 played {'White' if p1_white else 'Black'}.")
    print("=" * 60)

    for _, m in sub.iterrows():
        action = int(m["action"])
        top_actions = [int(a) for a in m["top_k_actions"]][: args.top_k]
        top_probs = [float(p) for p in m["top_k_probs"]][: args.top_k]

        canonical = game.get_canonical_form(board, cur_player)
        print(f"\nMove {int(m['move_idx']) + 1} — player {int(m['player']):+d}:")
        print(f"  Top-{args.top_k}: " + ", ".join(
            f"action {a} ({p:.0%})" for a, p in zip(top_actions, top_probs, strict=False)
        ))
        print(f"  Played: action {action}")

        # Game-specific ASCII dump for context.
        if game_name == "tictactoe":
            _print_ttt(canonical.as_2d)

        board, cur_player = game.get_next_state(board, cur_player, action)

    final_canonical = game.get_canonical_form(board, cur_player)
    print("\nFinal position:")
    if game_name == "tictactoe":
        _print_ttt(final_canonical.as_2d)


def _print_ttt(arr) -> None:
    """Pretty-print a 3×3 board in ASCII."""
    glyphs = []
    for row in range(3):
        row_cells = []
        for col in range(3):
            v = int(arr[col, row])
            if v == 1:
                row_cells.append(" X ")
            elif v == -1:
                row_cells.append(" O ")
            else:
                action = col * 3 + row
                row_cells.append(f" {action} ")
        glyphs.append("|".join(row_cells))
    print("  " + "\n  ---+---+---\n  ".join(glyphs))


if __name__ == "__main__":
    main()
