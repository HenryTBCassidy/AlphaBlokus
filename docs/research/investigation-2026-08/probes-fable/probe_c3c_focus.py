"""C3 probe, round 3: zoom on the two positions unresolved at 512 sims.

Same deterministic collection as round 2; keep positions 5 (d=4) and 9 (d=3).
For each: exact class per move, gumbel completed-Q target at 512/2048 sims,
and the python full-action-space PUCT search (independent implementation,
conventions verified clean) at 512 sims with the same net. If python fails the
same way, the failures are search-budget/prior effects, not a JAX sign bug.
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


def legal_actions(board, player):
    return [int(a) for a in np.nonzero(game.valid_move_masking(board, player))[0]]


def negamax(board, player, budget):
    budget[0] += 1
    if budget[0] > NODE_CAP:
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


print("== collecting (same seed)", flush=True)
mixed = []
playouts = 0
while len(mixed) < 12 and playouts < 300:
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
        if d < 2 or not (2 <= len(acts) <= 24):
            continue
        try:
            classes = {}
            for a in acts:
                nb, np_ = game.get_next_state(b, p, a)
                classes[a] = -negamax(nb, np_, [0])
        except RecursionError:
            continue
        if len(set(classes.values())) >= 2 and len(mixed) < 12:
            mixed.append((b, p, classes, d))

focus = [mixed[5], mixed[9]]
for label, (_b, _p, classes, d) in zip(["pos5", "pos9"], focus, strict=True):
    print(f"  {label}: d={d}, classes histogram {sorted(classes.values())}", flush=True)

torch.manual_seed(21)
nnet = NNetWrapper(game, make_search_config(Path(tempfile.mkdtemp())))
params = params_to_device(convert_state_dict(nnet.nnet.state_dict(), num_residual_blocks=1))
kernels = make_kernels(build_jax_tables(game))

states = GameState(
    *(np.stack(rows) for rows in zip(*(numpy_state_from_board(b, p) for b, p, _, _ in focus), strict=True))
)


def weights_by_id(ids_row, w_row):
    out = {}
    for a, w in zip(ids_row, w_row, strict=True):
        out[int(a)] = out.get(int(a), 0.0) + float(w)
    return out


for sims in (512, 2048):
    search = make_search(kernels, SearchConfig(num_simulations=sims, top_k=64, policy="gumbel", gumbel_max_considered=64))
    res = search(params, jax.random.PRNGKey(99), states)
    for i, (label, (b, p, classes, d)) in enumerate(zip(["pos5", "pos9"], focus, strict=True)):
        idw = weights_by_id(np.asarray(res.topk_ids)[i], np.asarray(res.action_weights)[i])
        best = max(classes.values())
        chosen = int(np.asarray(res.chosen_global)[i])
        ok = classes.get(chosen) == best
        by_class = {}
        for a, c in classes.items():
            by_class.setdefault(c, []).append(idw.get(a, 0.0))
        summary = {c: f"n={len(v)} mean_w={np.mean(v):.4f} max_w={max(v):.4f}" for c, v in sorted(by_class.items())}
        print(f"  gumbel sims={sims} {label}: chosen class={classes.get(chosen)} (best {best}, ok={ok}) weights by class: {summary}", flush=True)

print("== python full-action PUCT at 512 sims", flush=True)
for label, (b, p, classes, d) in zip(["pos5", "pos9"], focus, strict=True):
    mcts = MCTS(game, nnet, MCTSConfig(num_mcts_sims=512, cpuct=2.5, mcts_batch_size=16))
    pi = np.asarray(mcts.get_action_prob(game.get_canonical_form(b, p), temp=0))
    a_py = int(pi.argmax())
    best = max(classes.values())
    by_class = {}
    for a, c in classes.items():
        by_class.setdefault(c, []).append(float(pi[a]))
    summary = {c: f"n={len(v)} mean_pi={np.mean(v):.4f} max_pi={max(v):.4f}" for c, v in sorted(by_class.items())}
    print(f"  python {label}: chosen class={classes.get(a_py)} (best {best}, ok={classes.get(a_py) == best}) visit share by class: {summary}", flush=True)
print("done", flush=True)
