"""Batched AlphaZero search: mctx ``muzero_policy`` over a top-K compact action space (G4).

mctx allocates six dense ``[B, S+1, A]`` tree arrays, which is untenable at
A=17,837 (~428 KB per node per game). This module searches a **compact** action
space instead: at every node expansion only the top-``K`` legal actions by
prior are kept; their global action ids ride along in the node embedding, and
``recurrent_fn`` maps compact→global before stepping the env. Root Dirichlet
noise is applied to the full legal distribution **before** top-K selection so
noise can promote low-prior moves into the searched set (matching the spirit
of ``mcts.py::_apply_root_dirichlet_noise``, which mixes over legal moves).

PUCT parity with ``search/mcts.py`` (see the plan's fidelity contract):

- ``pb_c_init = cpuct`` and ``pb_c_base = 1e9`` make mctx's
  ``pb_c = pb_c_init + log((N + pb_c_base + 1)/pb_c_base)`` ≈ ``cpuct``;
  mctx's ``policy_score = sqrt(N)·pb_c·P/(1+n)`` then equals the python
  exploration term exactly.
- ``qtransform_raw_value`` reproduces python Q semantics: raw (unnormalised)
  Q from the node player's perspective, 0 for unvisited children — vs mctx's
  default min-max-normalised transform (kept available via config for A/B).
- Rewards enter on terminal transitions from the mover's perspective with
  discount 0 (absorbing); non-terminal discount is −1 (two-player sign flip),
  so ``qvalues = reward + discount·child_value`` is parent-perspective, like
  python's backprop.

Compact slots beyond the number of legal actions carry ``-inf`` prior logits
(→ zero prior probability) and have their action ids remapped to **pass**, so
even if a slot is ever selected the env step is mechanically safe;
``qtransform_raw_value`` additionally scores them at −1e9 so they never win
selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import mctx

from alphablokus.games.blokusduo.jax.net import encode_states, forward

if TYPE_CHECKING:
    from collections.abc import Callable

    from alphablokus.games.blokusduo.jax.kernels import GameState, JaxKernels

#: Selection score for illegal/padded compact slots — beneath any real Q ∈ [-1, 1].
_ILLEGAL_SCORE = -1e9


@dataclass(frozen=True)
class SearchConfig:
    """Knobs for one search invocation (all static under jit).

    Attributes:
        num_simulations: Simulations per move (flat; the python branching taper
            does not translate to fixed-shape search — fidelity contract).
        top_k: Compact action-space size per node.
        cpuct: PUCT exploration constant, mapped onto ``pb_c_init``.
        dirichlet_epsilon: Root noise mix fraction (0 disables, e.g. for eval).
        dirichlet_alpha: Root noise concentration.
        qtransform: ``"raw"`` = python-parity Q semantics (default);
            ``"mctx"`` = library default (min-max normalised) for A/B runs.
        dtype: Net compute dtype, ``"float32"`` or ``"bfloat16"``.
    """

    num_simulations: int
    top_k: int = 256
    cpuct: float = 2.5
    dirichlet_epsilon: float = 0.0
    dirichlet_alpha: float = 0.03
    qtransform: str = "raw"
    dtype: str = "float32"
    policy: str = "puct"  # "puct" | "gumbel" (mctx gumbel_muzero_policy — G10)
    gumbel_max_considered: int = 16


class SearchResult(NamedTuple):
    """Per-game outputs of one batched search.

    Attributes:
        action_weights: ``(B, K)`` policy target over compact slots — the root
            visit distribution under PUCT, the completed-Q improved policy
            under Gumbel.
        topk_ids: ``(B, K)`` global action id per compact slot.
        root_value: ``(B,)`` tree value estimate of the root.
        visit_counts: ``(B, K)`` raw root visit counts.
        chosen_global: ``(B,)`` the search's own selected action as a global
            id. Under Gumbel this is the Sequential-Halving winner and is what
            self-play should play; under PUCT the actor's temperature sampling
            supersedes it.
    """

    action_weights: jnp.ndarray
    topk_ids: jnp.ndarray
    root_value: jnp.ndarray
    visit_counts: jnp.ndarray
    chosen_global: jnp.ndarray


def qtransform_raw_value(tree: mctx.Tree, node_index: jnp.ndarray) -> jnp.ndarray:
    """Python-parity value score: raw Q, unvisited → 0, illegal slots → −1e9."""
    qvalues = tree.qvalues(node_index)
    visits = tree.children_visits[node_index]
    prior_logits = tree.children_prior_logits[node_index]
    q = jnp.where(visits > 0, qvalues, 0.0)
    return jnp.where(jnp.isneginf(prior_logits), _ILLEGAL_SCORE, q)


def dense_policy(action_weights: jnp.ndarray, topk_ids: jnp.ndarray, action_size: int) -> jnp.ndarray:
    """Scatter compact root weights back to the full action space, ``(B, A)``."""

    def scatter(weights: jnp.ndarray, ids: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros(action_size, dtype=weights.dtype).at[ids].add(weights)

    return jax.vmap(scatter)(action_weights, topk_ids)


def make_search(kernels: JaxKernels, config: SearchConfig) -> Callable[..., SearchResult]:
    """Build a jitted ``search(params, rng_key, states) -> SearchResult``.

    ``states`` is a batched :class:`GameState`; ``params`` the converted net
    pytree. The returned callable runs ``config.num_simulations`` PUCT
    simulations for every game in the batch simultaneously.
    """
    top_k = config.top_k
    pass_index = kernels.pass_index
    dtype = jnp.float32 if config.dtype == "float32" else jnp.bfloat16
    game_result_per_state = jax.vmap(kernels.game_result)

    def policy_value(params: dict[str, Any], states: GameState) -> tuple[jnp.ndarray, jnp.ndarray]:
        planes = encode_states(states.ppb, states.current_player, dtype=dtype)
        return forward(params, planes)  # (B, A) log-probs fp32, (B,) value fp32

    def masked_root_logits(states: GameState, log_pi: jnp.ndarray, rng_key: jnp.ndarray) -> jnp.ndarray:
        """Legal-masked log-priors with Dirichlet noise mixed over legal moves."""
        masks = kernels.legal_mask_batch(states)  # (B, A) bool
        logits = jnp.where(masks, log_pi, -jnp.inf)
        if config.dirichlet_epsilon <= 0.0:
            return logits
        priors = jax.nn.softmax(logits, axis=-1)
        # Dirichlet over the legal subset == iid Gamma(alpha) on legal, normalised.
        gammas = jax.random.gamma(rng_key, config.dirichlet_alpha, shape=priors.shape)
        gammas = jnp.where(masks, gammas, 0.0)
        noise = gammas / jnp.maximum(gammas.sum(axis=-1, keepdims=True), 1e-30)
        mixed = (1.0 - config.dirichlet_epsilon) * priors + config.dirichlet_epsilon * noise
        return jnp.where(masks, jnp.log(jnp.maximum(mixed, 1e-30)), -jnp.inf)

    def topk_legal(logits: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Top-K logits + ids; padded (-inf) slots remapped to the safe pass action."""
        values, ids = jax.lax.top_k(logits, top_k)
        ids = jnp.where(jnp.isneginf(values), pass_index, ids)
        return values, ids.astype(jnp.int32)

    def recurrent_fn(
        params: dict[str, Any], rng_key: jnp.ndarray, action: jnp.ndarray, embedding: tuple,
    ) -> tuple[mctx.RecurrentFnOutput, tuple]:
        states, topk_ids = embedding
        batch = jnp.arange(action.shape[0])
        global_action = topk_ids[batch, action]
        movers = states.current_player
        new_states = kernels.step_batch(states, global_action)

        result_for_mover = game_result_per_state(new_states, movers)  # parent-perspective
        terminated = result_for_mover != 0.0
        reward = jnp.where(terminated, result_for_mover, 0.0)
        discount = jnp.where(terminated, 0.0, -1.0)

        log_pi, value = policy_value(params, new_states)
        value = jnp.where(terminated, 0.0, value)
        masks = kernels.legal_mask_batch(new_states)
        child_logits, child_ids = topk_legal(jnp.where(masks, log_pi, -jnp.inf))

        output = mctx.RecurrentFnOutput(
            reward=reward, discount=discount, prior_logits=child_logits, value=value,
        )
        return output, (new_states, child_ids)

    qtransform = (
        qtransform_raw_value if config.qtransform == "raw" else mctx.qtransform_by_parent_and_siblings
    )

    def search(params: dict[str, Any], rng_key: jnp.ndarray, states: GameState) -> SearchResult:
        noise_key, search_key = jax.random.split(rng_key)
        log_pi, root_value = policy_value(params, states)
        # Gumbel supplies its own root exploration (Gumbel noise + Sequential
        # Halving); Dirichlet pre-mixing is a PUCT-only concept.
        root_log_pi = log_pi if config.policy == "gumbel" else masked_root_logits(states, log_pi, noise_key)
        if config.policy == "gumbel":
            masks = kernels.legal_mask_batch(states)
            root_log_pi = jnp.where(masks, root_log_pi, -jnp.inf)
        root_logits, root_ids = topk_legal(root_log_pi)
        root = mctx.RootFnOutput(
            prior_logits=root_logits, value=root_value, embedding=(states, root_ids),
        )
        if config.policy == "gumbel":
            policy_output = mctx.gumbel_muzero_policy(
                params=params,
                rng_key=search_key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=config.num_simulations,
                invalid_actions=jnp.isneginf(root_logits),
                max_num_considered_actions=config.gumbel_max_considered,
            )
            # Gumbel's action_weights (softmax of prior + completed Q) is the
            # paper's policy-improvement target — not the visit distribution.
            action_weights = policy_output.action_weights
        else:
            policy_output = mctx.muzero_policy(
                params=params,
                rng_key=search_key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=config.num_simulations,
                invalid_actions=jnp.isneginf(root_logits),
                dirichlet_fraction=0.0,  # noise pre-mixed before top-K, see module docstring
                pb_c_init=config.cpuct,
                pb_c_base=1e9,
                qtransform=qtransform,
            )
            action_weights = None  # visit distribution, filled from the summary below
        summary = policy_output.search_tree.summary()
        batch_index = jnp.arange(root_ids.shape[0])
        return SearchResult(
            action_weights=summary.visit_probs if action_weights is None else action_weights,
            topk_ids=root_ids,
            root_value=summary.value,
            visit_counts=summary.visit_counts,
            chosen_global=root_ids[batch_index, policy_output.action],
        )

    return jax.jit(search)
