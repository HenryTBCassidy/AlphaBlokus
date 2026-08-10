"""C3 probe, round 2: discriminate 'sign bug' from 'search budget'.

A per-depth completed-Q sign error does NOT improve as simulations grow — the
corrupted Q locks the search onto bad moves. A budget limitation does improve.
So: same exact-solved endgame positions, gumbel search at 128 / 512 / 2048
sims, plus the python full-action-space PUCT MCTS (independently implemented,
conventions already verified) as a reference at 512 sims.

Also re-checks the forced-pass positions with correct duplicate-id
aggregation (padded compact slots legitimately share the pass id).
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

from alphablokus.config import MCTSConfig
from alphablokus.games.blokusduo.game import BlokusDuoGame
from alphablokus.games.blokusduo.jax.bridge import numpy_state_from_board
from alphablokus.games.blokusduo.jax.checkpoint import convert_state_dict, params_to_device
from alphablokus.games.blokusduo.jax.kernels import GameState, make_kernels
from alphablokus.games.blokusduo.jax.search import SearchConfig, make_search
from alphablokus.games.blokusduo.jax.tables import build_jax_tables
from alphablokus.games.blokusduo.nn.wrapper import NNetWrapper
from alphablokus.games.blokusduo.pieces import default_pieces_path
from alphablokus.search.mcts import MCTS
from tests.games.blokusduo.jax.conftest import make_search_config

game = BlokusDuoGame(pieces_config_path=default_pieces_path())
game.enable_optimised_movegen()
rng = random.Random(1234)
NODE_CAP = 20_000


class Budget:
    def __init__(self) -> None:
        self.nodes = 0


def legal_actions(board, player):
    return [int(a) for a in np.nonzero(game.valid_move_masking(board, player))[0]]


def negamax(board, player, budget):
    budget.nodes += 1
    if budget.nodes > NODE_CAP:
        raise RecursionError
    r = game.get_game_ended(board, player)
    if r != 0:
        return 0 if abs(r) < 0.5 else int(np.sign(r))
    best = -2
    for a in legal_actions(board, player):
        nb, np_ = game.get_next_state(board, player, a)
        best = max(best, -negamax(nb, np_, budget))
        if best == 1:
            break
    return best


def move_classes(board, player):
    out = {}
    for a in legal_actions(board, player):
        nb, np_ = game.get_next_state(board, player, a)
        out[a] = -negamax(nb, np_, Budget())
    return out


print("== collecting positions (same seed as round 1)", flush=True)
mixed = []
forced = []
playouts = 0
while (len(mixed) < 12 or len(forced) < 4) and playouts < 300:
    playouts += 1
    board, player = game.initialise_board(), 1
    history = []
    while game.get_game_ended(board, player) == 0:
        history.append((board, player))
        board, player = game.get_next_state(board, player, rng.choice(legal_actions(board, player)))
    L = len(history)
    for pos_idx in range(max(0, L - 6), L):
        b, p = history[pos_idx]
        d = L - pos_idx
        acts = legal_actions(b, p)
        if len(acts) == 1 and len(forced) < 4 and d >= 2:
            forced.append((b, p, acts[0]))
        if d < 2 or not (2 <= len(acts) <= 24):
            continue
        try:
            classes = move_classes(b, p)
        except RecursionError:
            continue
        if len(set(classes.values())) >= 2 and len(mixed) < 12:
            mixed.append((b, p, classes, d))

print(f"  {len(mixed)} mixed positions, {len(forced)} forced", flush=True)

torch.manual_seed(21)
nnet = NNetWrapper(game, make_search_config(Path(tempfile.mkdtemp())))
params = params_to_device(convert_state_dict(nnet.nnet.state_dict(), num_residual_blocks=1))
kernels = make_kernels(build_jax_tables(game))

all_pos = [(b, p) for b, p, *_ in mixed] + [(b, p) for b, p, _ in forced]
states = GameState(
    *(np.stack(rows) for rows in zip(*(numpy_state_from_board(b, p) for b, p in all_pos), strict=True))
)


def weights_by_id(topk_ids_row, weights_row):
    out: dict[int, float] = {}
    for a, w in zip(topk_ids_row, weights_row, strict=True):
        out[int(a)] = out.get(int(a), 0.0) + float(w)
    return out


def evaluate(chosen, topk_ids, weights):
    solved, per_pos = 0, []
    inverted = pairs = 0
    for i, (b, p, classes, d) in enumerate(mixed):
        best = max(classes.values())
        optimal = {a for a, c in classes.items() if c == best}
        ok = int(chosen[i]) in optimal
        solved += ok
        idw = weights_by_id(topk_ids[i], weights[i])
        inv = tot = 0
        for aw, cw in classes.items():
            for al, cl in classes.items():
                if cw > cl and aw in idw and al in idw:
                    tot += 1
                    inv += idw[aw] <= idw[al]
        inverted += inv
        pairs += tot
        per_pos.append((d, ok, f"{inv}/{tot}"))
    return solved, per_pos, inverted, pairs


print("== gumbel sims ladder (solved / inverted-pairs; a sign bug would NOT improve with sims)")
for sims in (128, 512, 2048):
    search = make_search(kernels, SearchConfig(num_simulations=sims, top_k=64, policy="gumbel", gumbel_max_considered=64))
    res = search(params, jax.random.PRNGKey(99), states)
    solved, per_pos, inv, pairs = evaluate(np.asarray(res.chosen_global), np.asarray(res.topk_ids), np.asarray(res.action_weights))
    print(f"  sims={sims:5d}: solved {solved}/{len(mixed)}  inverted pairs {inv}/{pairs}  per-pos {per_pos}", flush=True)

print("== python full-action PUCT reference at 512 sims")
config = MCTSConfig(num_mcts_sims=512, cpuct=2.5, mcts_batch_size=16)
py_solved = 0
py_per_pos = []
for b, p, classes, d in mixed:
    mcts = MCTS(game, nnet, config)
    pi = np.asarray(mcts.get_action_prob(game.get_canonical_form(b, p), temp=0))
    a_py = int(pi.argmax())
    best = max(classes.values())
    ok = a_py in {a for a, c in classes.items() if c == best}
    py_solved += ok
    py_per_pos.append((d, ok))
print(f"  python: solved {py_solved}/{len(mixed)}  per-pos {py_per_pos}", flush=True)

print("== forced single-action positions (aggregated weights)")
search = make_search(kernels, SearchConfig(num_simulations=128, top_k=64, policy="gumbel", gumbel_max_considered=64))
res = search(params, jax.random.PRNGKey(5), states)
chosen = np.asarray(res.chosen_global)
off = len(mixed)
for j, (b, p, act) in enumerate(forced):
    i = off + j
    idw = weights_by_id(np.asarray(res.topk_ids)[i], np.asarray(res.action_weights)[i])
    print(f"  forced[{j}]: chosen={int(chosen[i])} expected={act} mass={idw.get(act, 0.0):.4f}")
