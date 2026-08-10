"""C3/C6 probe: per-depth completed-Q perspective, checked against exhaustive negamax.

Builds real Blokus Duo endgame positions by random playout, solves them exactly
with negamax over the python rules (win/draw/loss for the player to move, and
the per-move class), then runs the REAL production JAX Gumbel search on the
same positions and checks:

  1. the Sequential-Halving winner (chosen_global) is an optimal move;
  2. the completed-Q policy target (action_weights) ranks every winning move
     above every losing move;
  3. the root value's sign agrees with the negamax value (decisive positions);
  4. single-legal-action positions put all target mass on that action.

A perspective/sign error at ANY tree depth corrupts completed-Q and makes the
search prefer moves that hand the opponent the win — positions are filtered to
those with mixed move classes and depth >= 2 so the checks discriminate.
"""

from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

import jax
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ab-audit"))

from alphablokus.games.blokusduo.game import BlokusDuoGame
from alphablokus.games.blokusduo.jax.bridge import numpy_state_from_board
from alphablokus.games.blokusduo.jax.checkpoint import convert_state_dict, params_to_device
from alphablokus.games.blokusduo.jax.kernels import GameState, make_kernels
from alphablokus.games.blokusduo.jax.search import SearchConfig, make_search
from alphablokus.games.blokusduo.jax.tables import build_jax_tables
from alphablokus.games.blokusduo.nn.wrapper import NNetWrapper
from alphablokus.games.blokusduo.pieces import default_pieces_path
from tests.games.blokusduo.jax.conftest import make_search_config

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


game = BlokusDuoGame(pieces_config_path=default_pieces_path())
game.enable_optimised_movegen()
rng = random.Random(1234)

NODE_CAP = 20_000


class Budget:
    def __init__(self) -> None:
        self.nodes = 0


def legal_actions(board, player) -> list[int]:
    mask = game.valid_move_masking(board, player)
    return [int(a) for a in np.nonzero(mask)[0]]


def negamax(board, player, budget: Budget) -> int:
    """Exact game value for the player to move: +1 win, 0 draw, -1 loss."""
    budget.nodes += 1
    if budget.nodes > NODE_CAP:
        raise RecursionError("node cap")
    r = game.get_game_ended(board, player)
    if r != 0:
        return 0 if abs(r) < 0.5 else int(np.sign(r))
    best = -2
    for a in legal_actions(board, player):
        nb, np_ = game.get_next_state(board, player, a)
        best = max(best, -negamax(nb, np_, budget))
        if best == 1:
            break  # alpha cut at proven win
    return best


def move_classes(board, player, budget: Budget) -> dict[int, int]:
    """Exact class of every legal move: value to the mover after playing it."""
    out = {}
    for a in legal_actions(board, player):
        nb, np_ = game.get_next_state(board, player, a)
        out[a] = -negamax(nb, np_, budget)
    return out


# ---------------------------------------------------------- collect positions
print("== collecting endgame positions by random playout", flush=True)
mixed_positions = []  # (board, player, classes, depth = plies-to-end in the playout line)
forced_positions = []  # (board, player, the single action)
playouts = 0
while (len(mixed_positions) < 12 or len(forced_positions) < 4) and playouts < 300:
    playouts += 1
    board, player = game.initialise_board(), 1
    history = []
    while game.get_game_ended(board, player) == 0:
        acts = legal_actions(board, player)
        history.append((board, player))
        board, player = game.get_next_state(board, player, rng.choice(acts))
    L = len(history)
    for pos_idx in range(max(0, L - 6), L):
        board_h, player_h = history[pos_idx]
        d = L - pos_idx  # plies to termination along this playout line
        acts = legal_actions(board_h, player_h)
        if len(acts) == 1 and len(forced_positions) < 4 and d >= 2:
            forced_positions.append((board_h, player_h, acts[0]))
        if d < 2 or not (2 <= len(acts) <= 24):
            continue
        try:
            budget = Budget()
            classes = move_classes(board_h, player_h, budget)
        except RecursionError:
            continue
        if len(set(classes.values())) < 2:
            continue  # all moves equivalent: not discriminating
        if len(mixed_positions) < 12:
            mixed_positions.append((board_h, player_h, classes, d))
            print(f"  playout {playouts}: kept mixed pos (d={d}, {len(acts)} moves, {budget.nodes} nodes)", flush=True)

print(f"  {len(mixed_positions)} mixed-class positions (depths {[d for *_, d in mixed_positions]}),")
print(f"  {len(forced_positions)} forced single-action positions, from {playouts} playouts")
assert len(mixed_positions) >= 8, "not enough discriminating endgames collected"

# ------------------------------------------------------------- run the search
torch.manual_seed(21)
nnet = NNetWrapper(game, make_search_config(Path(tempfile.mkdtemp())))
params = params_to_device(convert_state_dict(nnet.nnet.state_dict(), num_residual_blocks=1))
kernels = make_kernels(build_jax_tables(game))

all_pos = [(b, p) for b, p, *_ in mixed_positions] + [(b, p) for b, p, _ in forced_positions]
states = GameState(
    *(
        np.stack(rows)
        for rows in zip(*(numpy_state_from_board(b, p) for b, p in all_pos), strict=True)
    )
)
# production-shaped search: gumbel, top_k=64, considered=64, 256 sims
search = make_search(kernels, SearchConfig(num_simulations=256, top_k=64, policy="gumbel", gumbel_max_considered=64))
result = search(params, jax.random.PRNGKey(99), states)
chosen = np.asarray(result.chosen_global)
weights = np.asarray(result.action_weights)
topk_ids = np.asarray(result.topk_ids)
root_value = np.asarray(result.root_value)

print("== 1. Sequential-Halving winner is an exact-optimal move")
subopt = []
for i, (b, p, classes, d) in enumerate(mixed_positions):
    best = max(classes.values())
    optimal = {a for a, c in classes.items() if c == best}
    if int(chosen[i]) not in optimal:
        subopt.append((i, d, classes.get(int(chosen[i])), best))
check(
    "all mixed positions solved",
    not subopt,
    f"{len(subopt)}/{len(mixed_positions)} suboptimal: {subopt[:4]}",
)

print("== 2. completed-Q target ranks winning moves above losing moves")
rank_errs = 0
pairs = 0
for i, (b, p, classes, d) in enumerate(mixed_positions):
    id_to_w = {int(a): float(w) for a, w in zip(topk_ids[i], weights[i], strict=True)}
    for a_win, c_win in classes.items():
        for a_lose, c_lose in classes.items():
            if c_win > c_lose and a_win in id_to_w and a_lose in id_to_w:
                pairs += 1
                if id_to_w[a_win] <= id_to_w[a_lose]:
                    rank_errs += 1
check("better-class moves always weighted higher", rank_errs == 0, f"{rank_errs}/{pairs} inverted pairs")

print("== 3. root value sign matches negamax (decisive positions)")
sign_errs = []
for i, (b, p, classes, d) in enumerate(mixed_positions):
    v = max(classes.values())  # negamax value of the position for the mover
    if v == 0:
        continue
    if np.sign(root_value[i]) != v:
        sign_errs.append((i, float(root_value[i]), v, d))
check("signs agree", not sign_errs, f"{sign_errs[:4]}")

print("== 4. forced single-action positions")
off = len(mixed_positions)
errs = []
for j, (b, p, act) in enumerate(forced_positions):
    i = off + j
    id_to_w = {int(a): float(w) for a, w in zip(topk_ids[i], weights[i], strict=True)}
    mass_on_forced = id_to_w.get(act, 0.0)
    if int(chosen[i]) != act or mass_on_forced < 0.999:
        errs.append((j, int(chosen[i]), act, mass_on_forced))
check("all mass on the only legal action", not errs, f"{errs}")

print()
if FAILURES:
    print("FAILURES:", FAILURES)
    sys.exit(1)
print("C3 probe: all checks passed")
