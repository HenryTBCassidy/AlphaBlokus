"""C1 probe: PRNG key uniqueness and Gumbel-noise independence in lockstep self-play.

Checks, empirically, against the real production modules:

1. Key-chain uniqueness: reproduce backend.py's exact derivation
   (PRNGKey(seed) -> fold_in(generation) -> split per wave -> split(wave_key,
   wave_plies) -> split(step_key, 3)) and assert every derived key is unique
   across generations x waves x plies x roles.
2. No Gumbel broadcast: monkeypatch jax.random.gumbel to record every shape
   requested during a traced gumbel search; assert all shapes are batched
   (leading dim == B), never a single (K,) vector shared across games.
3. Behavioural diversity: B identical initial states, one search key (exactly
   how actors.py calls it) -> chosen actions must differ across slots.
4. Per-ply freshness: the same batch searched under two consecutive step keys
   picks a different pattern of actions.
5. Determinism sanity: same key -> identical output.
"""

from __future__ import annotations

import collections
import sys

import jax
import jax.numpy as jnp
import numpy as np
import torch

from alphablokus.games.blokusduo.game import BlokusDuoGame
import tempfile
from pathlib import Path

from alphablokus.games.blokusduo.jax.checkpoint import convert_state_dict, params_to_device
from alphablokus.games.blokusduo.jax.kernels import make_kernels
from alphablokus.games.blokusduo.jax.search import SearchConfig, make_search
from alphablokus.games.blokusduo.jax.tables import build_jax_tables
from alphablokus.games.blokusduo.pieces import default_pieces_path

SEED = 42  # production seed (config.resolved.json)
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------- 1. key chain
print("== 1. key-chain uniqueness (backend.py -> actors.py derivation)")
all_keys: list[tuple[int, ...]] = []
per_role: dict[str, list] = collections.defaultdict(list)
GENS, WAVES, WAVE_PLIES = 3, 6, 32
for gen in range(1, GENS + 1):
    rng_key = jax.random.fold_in(jax.random.PRNGKey(SEED), gen)  # backend.py:167
    for _wave in range(WAVES):
        rng_key, wave_key = jax.random.split(rng_key)  # backend.py:179
        step_keys = jax.random.split(wave_key, WAVE_PLIES)  # actors.py:137
        for step_key in step_keys:
            search_key, sample_key, tie_key = jax.random.split(step_key, 3)  # actors.py:99
            noise_key, mctx_key = jax.random.split(search_key)  # search.py:190
            for role, k in [
                ("search", search_key),
                ("sample", sample_key),
                ("tie", tie_key),
                ("noise", noise_key),
                ("mctx", mctx_key),
            ]:
                t = tuple(np.asarray(k).tolist())
                all_keys.append(t)
                per_role[role].append(t)

n, uniq = len(all_keys), len(set(all_keys))
check("all derived keys unique", n == uniq, f"{uniq}/{n} unique over {GENS} gens x {WAVES} waves x {WAVE_PLIES} plies x 5 roles")

# cross-generation: do generation streams ever collide?
gen_streams = []
for gen in range(1, 40 + 1):
    gen_streams.append(tuple(np.asarray(jax.random.fold_in(jax.random.PRNGKey(SEED), gen)).tolist()))
check("40 generation root keys unique", len(set(gen_streams)) == 40)

# ------------------------------------------------- 2. gumbel shape recording
print("== 2. recorded jax.random.gumbel shapes during a traced gumbel search")
torch.manual_seed(0)
game = BlokusDuoGame(pieces_config_path=default_pieces_path())
game.enable_optimised_movegen()
import sys as _sys  # noqa: E402

_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ab-audit"))
from alphablokus.games.blokusduo.nn.wrapper import NNetWrapper  # noqa: E402
from tests.games.blokusduo.jax.conftest import make_search_config  # noqa: E402

_tmp = tempfile.mkdtemp()
nnet = NNetWrapper(game, make_search_config(Path(_tmp)))
params = params_to_device(convert_state_dict(nnet.nnet.state_dict(), num_residual_blocks=1))
kernels = make_kernels(build_jax_tables(game))

B, TOP_K, SIMS, CONSIDERED = 64, 16, 32, 16
single = kernels.initial_state()
states = jax.tree.map(lambda x: jnp.broadcast_to(x, (B, *x.shape)), single)

recorded_shapes: list[tuple] = []
_orig_gumbel = jax.random.gumbel


def _spy_gumbel(key, shape=(), dtype=float):
    recorded_shapes.append(tuple(shape))
    return _orig_gumbel(key, shape=shape, dtype=dtype)


jax.random.gumbel = _spy_gumbel
try:
    cfg = SearchConfig(num_simulations=SIMS, top_k=TOP_K, policy="gumbel", gumbel_max_considered=CONSIDERED)
    search = make_search(kernels, cfg)
    res0 = search(params, jax.random.PRNGKey(123), states)  # traces + runs
    jax.block_until_ready(res0.chosen_global)
finally:
    jax.random.gumbel = _orig_gumbel

print(f"  recorded gumbel shapes: {sorted(set(recorded_shapes))}")
check(
    "every gumbel draw is batched with leading dim B",
    len(recorded_shapes) > 0 and all(len(s) >= 2 and s[0] == B for s in recorded_shapes),
    f"{len(recorded_shapes)} draw(s)",
)

# ------------------------------------------- 3. behavioural diversity, one key
print("== 3. identical states, one shared key -> per-slot action diversity")
chosen = np.asarray(res0.chosen_global)
n_distinct = len(set(chosen.tolist()))
check("chosen actions differ across slots", n_distinct > 1, f"{n_distinct} distinct actions over {B} identical slots")
counts = collections.Counter(chosen.tolist())
top_share = counts.most_common(1)[0][1] / B
print(f"  most common action share: {top_share:.2f} (1.0 would mean full lockstep correlation)")

# also check the policy TARGETS (action_weights) are identical for identical
# states (they should be: the target is noise-free completed-Q improved policy)
w = np.asarray(res0.action_weights)
target_spread = float(np.abs(w - w[0]).max())
print(f"  action_weights spread across identical slots: {target_spread:.4g}")
print("  (targets vary per slot only through gumbel-directed tree growth)")

# ------------------------------------------------- 4. per-ply key freshness
print("== 4. two consecutive step keys -> different action pattern")
wave_key = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(SEED), 1))[1]
k0, k1 = jax.random.split(wave_key, 2)
s0, _, _ = jax.random.split(k0, 3)
s1, _, _ = jax.random.split(k1, 3)
r0 = search(params, s0, states)
r1 = search(params, s1, states)
a0, a1 = np.asarray(r0.chosen_global), np.asarray(r1.chosen_global)
check("different plies pick different per-slot patterns", not np.array_equal(a0, a1), f"agreement {np.mean(a0 == a1):.2f}")

# ------------------------------------------------------- 5. determinism sanity
print("== 5. same key twice -> identical result")
r2 = search(params, s0, states)
check("determinism", np.array_equal(np.asarray(r2.chosen_global), a0))

print()
if FAILURES:
    print("FAILURES:", FAILURES)
    sys.exit(1)
print("C1 probe: all checks passed")
