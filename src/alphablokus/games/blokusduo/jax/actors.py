"""Jitted batched self-play actor loop (plan step G5).

``B`` game slots are stepped in lockstep inside a ``lax.scan`` over a fixed
``wave_plies`` horizon (the pgx auto-reset pattern): every step runs one
batched mctx search, samples an action per slot under the python backend's
temperature semantics (sample from the visit distribution before
``temp_threshold`` plies, argmax with random tie-break after —
``selfplay/episode.py`` (temperature block) / ``search/mcts.py`` (``get_action_prob``)), steps the env, and
resets any finished slot to a fresh game in place. Each step emits one
fixed-shape trace row; the host (:mod:`games.blokusduo.jax.harvest`) assembles rows
into completed games between waves.

Draw-sign parity detail: the trace records the terminal state's player-to-move
(``end_player``) because the python path's value backfill gives the +1e-4 draw
value to the final ``current_player`` and −1e-4 to the opponent
(``self_play.py:109-116``); win/loss values only need ``result_white``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Callable

    from alphablokus.games.blokusduo.jax.kernels import GameState, JaxKernels
    from alphablokus.games.blokusduo.jax.search import SearchResult


class ActorCarry(NamedTuple):
    """Loop-carried state: the live games and their ply counters."""

    games: GameState
    move_counts: jnp.ndarray  # (B,) int32 — plies played in each slot's current game


class WaveTrace(NamedTuple):
    """One wave's per-step records, every leaf shaped ``(wave_plies, B, ...)``.

    ``ppb``/``player``/``move_count`` describe the position *before* the move;
    ``action_weights``/``topk_ids`` are the search's root visit distribution
    over compact slots and their global action ids; ``terminated``/
    ``result_white``/``end_player`` describe the state *after* the move.
    """

    ppb: jnp.ndarray  # int8 (T, B, 196)
    player: jnp.ndarray  # int8 (T, B)
    move_count: jnp.ndarray  # int32 (T, B)
    action_weights: jnp.ndarray  # float32 (T, B, K)
    topk_ids: jnp.ndarray  # int32 (T, B, K)
    terminated: jnp.ndarray  # bool (T, B)
    result_white: jnp.ndarray  # float32 (T, B) — game result from White's perspective
    end_player: jnp.ndarray  # int8 (T, B) — player-to-move in the terminal state


def _reset_where(terminated: jnp.ndarray, fresh: jnp.ndarray, current: jnp.ndarray) -> jnp.ndarray:
    """Per-slot select between a fresh-game leaf and the current leaf."""
    flags = terminated.reshape((-1,) + (1,) * (current.ndim - 1))
    return jnp.where(flags, fresh, current)


# Network weights as a jax pytree (nested dict of arrays from checkpoint.convert_state_dict).
Params = dict


def make_actor(
    kernels: JaxKernels,
    search: Callable[..., SearchResult],
    *,
    batch_size: int,
    temp_threshold: int,
    wave_plies: int,
    use_search_action: bool = False,
) -> tuple[Callable[[], ActorCarry], Callable[[Params, jnp.ndarray, ActorCarry], tuple[ActorCarry, WaveTrace]]]:
    """Build ``(initial_carry, run_wave)`` for a fixed batch size.

    Returns:
        ``initial_carry()`` — fresh games in every slot, counters zeroed.
        ``run_wave(params, rng_key, carry)`` — jitted; advances every slot by
        ``wave_plies`` moves and returns ``(carry, WaveTrace)``.
    """

    def _fresh_games() -> GameState:
        single = kernels.initial_state()
        return jax.tree.map(lambda x: jnp.broadcast_to(x, (batch_size, *x.shape)), single)

    def initial_carry() -> ActorCarry:
        return ActorCarry(games=_fresh_games(), move_counts=jnp.zeros(batch_size, dtype=jnp.int32))

    batch_index = jnp.arange(batch_size)

    def run_wave(params: Params, rng_key: jnp.ndarray, carry: ActorCarry) -> tuple[ActorCarry, WaveTrace]:
        fresh = _fresh_games()

        def step_fn(carry: ActorCarry, key: jnp.ndarray) -> tuple[ActorCarry, WaveTrace]:
            search_key, sample_key, tie_key = jax.random.split(key, 3)
            result = search(params, search_key, carry.games)
            weights = result.action_weights  # (B, K)

            if use_search_action:
                # Gumbel mode: play the Sequential-Halving winner — exploration
                # comes from the Gumbel noise inside the search itself.
                global_action = result.chosen_global
            else:
                sampled = jax.random.categorical(sample_key, jnp.log(jnp.maximum(weights, 1e-30)))
                is_max = weights == weights.max(axis=-1, keepdims=True)
                tie_noise = jax.random.uniform(tie_key, weights.shape)
                greedy = jnp.argmax(jnp.where(is_max, tie_noise, -1.0), axis=-1)
                compact = jnp.where(carry.move_counts < temp_threshold, sampled, greedy)
                global_action = result.topk_ids[batch_index, compact]

            new_games = kernels.step_batch(carry.games, global_action)
            result_white = kernels.game_result_batch(new_games, jnp.int8(1))
            terminated = result_white != 0.0

            row = WaveTrace(
                ppb=carry.games.ppb,
                player=carry.games.current_player,
                move_count=carry.move_counts,
                action_weights=weights,
                topk_ids=result.topk_ids,
                terminated=terminated,
                result_white=result_white,
                end_player=new_games.current_player,
            )
            next_games = jax.tree.map(
                lambda fresh_leaf, current_leaf: _reset_where(terminated, fresh_leaf, current_leaf),
                fresh,
                new_games,
            )
            next_counts = jnp.where(terminated, 0, carry.move_counts + 1)
            return ActorCarry(games=next_games, move_counts=next_counts), row

        keys = jax.random.split(rng_key, wave_plies)
        return jax.lax.scan(step_fn, carry, keys)

    return initial_carry, jax.jit(run_wave)
