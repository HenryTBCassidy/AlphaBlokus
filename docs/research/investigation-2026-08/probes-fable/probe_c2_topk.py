"""C2 probe: is top-K applied AFTER illegal-move masking, at the root and at
every expanded child?

Method: run the REAL search body (make_search) with jax.jit patched to identity
so it executes eagerly, and a spy on mctx.gumbel_muzero_policy that captures the
concrete root passed in and the concrete search tree coming out. Then verify,
independently, for the root and for every expanded interior node:

  1. every compact slot with a finite prior logit maps to a LEGAL global action
     of that node's state (legality recomputed from the node's own embedding);
  2. the set of global ids in finite slots is EXACTLY the top-K of the
     legal-masked log-priors (recomputed from scratch with forward + mask) —
     i.e. no legal move was displaced by an illegal one, and the window is the
     one the config promises;
  3. every padded slot (-inf logit) has its id remapped to pass;
  4. cross-check the JAX legality mask against the python game's
     get_valid_moves on the root positions (independent rules implementation).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import mctx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ab-audit"))

from alphablokus.games.blokusduo.game import BlokusDuoGame
from alphablokus.games.blokusduo.jax.bridge import numpy_state_from_board
from alphablokus.games.blokusduo.jax.checkpoint import convert_state_dict, params_to_device
from alphablokus.games.blokusduo.jax.kernels import GameState, make_kernels
from alphablokus.games.blokusduo.jax.net import encode_states, forward
from alphablokus.games.blokusduo.jax.tables import build_jax_tables
from alphablokus.games.blokusduo.nn.wrapper import NNetWrapper
from alphablokus.games.blokusduo.pieces import default_pieces_path
from alphablokus.testing.positions import iter_cached_positions
from tests.games.blokusduo.conftest import DEV_CACHE_PATH
from tests.games.blokusduo.jax.conftest import make_search_config

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


torch.manual_seed(11)
game = BlokusDuoGame(pieces_config_path=default_pieces_path())
game.enable_optimised_movegen()
nnet = NNetWrapper(game, make_search_config(Path(tempfile.mkdtemp())))
params = params_to_device(convert_state_dict(nnet.nnet.state_dict(), num_residual_blocks=1))
kernels = make_kernels(build_jax_tables(game))

# Positions: early (huge branching, >K legal), mid, late (< K legal), + initial.
boards, players = [], []
for board, player, seq in iter_cached_positions(game, DEV_CACHE_PATH):
    if len(boards) < 8 and 0 <= len(seq) <= 4:
        boards.append(board), players.append(player)
    elif len(boards) < 16 and 8 <= len(seq) <= 14:
        boards.append(board), players.append(player)
    elif len(boards) < 24 and len(seq) >= 22:
        boards.append(board), players.append(player)
    if len(boards) >= 24:
        break
print(f"positions: {len(boards)} (mixed phases)")

states = GameState(
    *(
        np.stack(rows)
        for rows in zip(*(numpy_state_from_board(b, p) for b, p in zip(boards, players, strict=True)), strict=True)
    )
)
B = len(boards)
TOP_K, SIMS = 16, 48  # small K so early positions have far more legal moves than slots

# ---- spy on mctx + disable jit so concrete values are observable
captured: dict = {}
_orig_policy = mctx.gumbel_muzero_policy


def spy_policy(**kwargs):
    out = _orig_policy(**kwargs)
    captured["root"] = kwargs["root"]
    captured["invalid_actions"] = kwargs["invalid_actions"]
    captured["out"] = out
    return out


_orig_jit = jax.jit
jax.jit = lambda fn, **kw: fn  # identity: run the real search body eagerly
mctx.gumbel_muzero_policy = spy_policy
try:
    from alphablokus.games.blokusduo.jax.search import SearchConfig, make_search

    search = make_search(kernels, SearchConfig(num_simulations=SIMS, top_k=TOP_K, policy="gumbel", gumbel_max_considered=TOP_K))
    result = search(params, jax.random.PRNGKey(7), states)
finally:
    jax.jit = _orig_jit
    mctx.gumbel_muzero_policy = _orig_policy

root = captured["root"]
tree = captured["out"].search_tree

masks_full = np.asarray(kernels.legal_mask_batch(states))  # (B, A)
action_size, pass_index = kernels.action_size, kernels.pass_index

# ---- independent legality cross-check: python game vs jax kernels (root)
print("== 0. python get_valid_moves vs jax legal_mask on the root positions")
mismatch = 0
for i, (b, p) in enumerate(zip(boards, players, strict=True)):
    py_mask = np.asarray(game.valid_move_masking(b, p), dtype=bool)
    mismatch += int(not np.array_equal(py_mask, masks_full[i]))
check("legality masks identical", mismatch == 0, f"{mismatch} mismatching positions")

# ---- recompute the expected window from scratch
log_pi_all = np.asarray(forward(params, encode_states(states.ppb, states.current_player))[0])


def expected_window(log_pi_row: np.ndarray, mask_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    masked = np.where(mask_row, log_pi_row, -np.inf)
    ids = np.argsort(-masked, kind="stable")[:TOP_K]  # top-K by masked prior
    return masked, ids


def audit_node(tag: str, prior_logits: np.ndarray, ids: np.ndarray, mask_row: np.ndarray, masked_expected: np.ndarray, expected_ids: np.ndarray) -> list[str]:
    errs = []
    finite = np.isfinite(prior_logits)
    # 1. every finite slot is a legal action
    if not mask_row[ids[finite]].all():
        errs.append(f"{tag}: finite slot holds an ILLEGAL action")
    # 2. finite slots == top-K of masked priors (as a set; ties broken arbitrarily)
    n_legal = int(mask_row.sum())
    n_finite = int(finite.sum())
    if n_finite != min(n_legal, TOP_K):
        errs.append(f"{tag}: {n_finite} finite slots vs expected {min(n_legal, TOP_K)}")
    got, want = set(ids[finite].tolist()), set(expected_ids[: min(n_legal, TOP_K)].tolist())
    if got != want:
        # tolerate tie-order differences: compare the logit value sets instead
        got_vals = np.sort(masked_expected[list(got)])
        want_vals = np.sort(masked_expected[list(want)])
        if not np.allclose(got_vals, want_vals, atol=1e-6):
            errs.append(f"{tag}: window is NOT the top-{TOP_K} of the masked priors ({len(got - want)} displaced)")
    # 3. padded slots remapped to pass
    if not (ids[~finite] == pass_index).all():
        errs.append(f"{tag}: padded slot id not remapped to pass")
    return errs


print("== 1. root window exactness")
root_logits = np.asarray(root.prior_logits)  # (B, K)
_, root_ids_emb = root.embedding
root_ids = np.asarray(root_ids_emb)
errors: list[str] = []
for i in range(B):
    masked, exp_ids = expected_window(log_pi_all[i], masks_full[i])
    errors += audit_node(f"root[{i}]", root_logits[i], root_ids[i], masks_full[i], masked, exp_ids)
check("root windows exact for all positions", not errors, "; ".join(errors[:4]))
n_legal_per_pos = masks_full.sum(axis=1)
print(f"  legal-move counts ranged {n_legal_per_pos.min()}–{n_legal_per_pos.max()} (K={TOP_K})")

print("== 2. invalid_actions passed to mctx == non-finite root slots")
inv = np.asarray(captured["invalid_actions"])
check("invalid_actions mask correct", bool((inv == ~np.isfinite(root_logits)).all()))

print("== 3. every expanded child node's window")
node_states, node_topk_ids = tree.embeddings  # GameState leaves (B, N, ...), ids (B, N, K)
node_visits = np.asarray(tree.node_visits)  # (B, N)
child_logits = np.asarray(tree.children_prior_logits)  # (B, N, K)
node_ids_np = np.asarray(node_topk_ids)
N = node_visits.shape[1]

flat_states = GameState(*(np.asarray(leaf).reshape((-1,) + tuple(np.asarray(leaf).shape[2:])) for leaf in node_states))
flat_masks = np.asarray(kernels.legal_mask_batch(flat_states)).reshape(B, N, action_size)
flat_logpi = np.asarray(
    forward(
        params,
        encode_states(
            jnp.asarray(np.asarray(node_states.ppb).reshape(-1, 196)),
            jnp.asarray(np.asarray(node_states.current_player).reshape(-1)),
        ),
    )[0]
).reshape(B, N, action_size)

errors = []
audited = 0
for b in range(B):
    for n in range(1, N):  # skip root (already audited); unvisited nodes hold garbage
        if node_visits[b, n] <= 0:
            continue
        # a visited node was expanded by recurrent_fn: its children window must be exact
        # terminal nodes: recurrent_fn still emits a window (value forced to 0), audit anyway
        masked, exp_ids = expected_window(flat_logpi[b, n], flat_masks[b, n])
        errs = audit_node(f"node[{b},{n}]", child_logits[b, n], node_ids_np[b, n], flat_masks[b, n], masked, exp_ids)
        errors += errs
        audited += 1
check(f"child windows exact for all {audited} expanded nodes", not errors, "; ".join(errors[:4]))

print()
if FAILURES:
    print("FAILURES:", FAILURES)
    sys.exit(1)
print("C2 probe: all checks passed")
