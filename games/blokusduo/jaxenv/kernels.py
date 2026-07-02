"""JAX legality, step, and game-end kernels (plan steps J3/J4).

Pure fixed-shape JAX: no host callbacks, no per-state Python, `jit`/`vmap`
clean. The rule conditions, for player ``p`` with occupancy vectors derived
from the signed placement board:

- overlap-free:      ``cover @ occ == 0``
- no own-edge touch: ``edge @ own == 0``
- own-corner touch:  ``corner @ own > 0``
- first move:        footprint covers the player's start cell (replaces the
  corner rule)
- availability:      the placement's piece is still in the player's inventory
- pass:              legal iff no placement is legal

Semantics mirror ``games/blokusduo/game.py`` exactly (pass leaves the board
unchanged; the game ends when *neither* player has a legal placement; scoring
is -(remaining squares), +15 all-placed, +5 more for monomino-last; draw is
encoded as ``1e-4``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

from games.blokusduo.jaxenv.tables import NUM_PIECE_IDS

if TYPE_CHECKING:
    from collections.abc import Callable

    from games.blokusduo.jaxenv.tables import JaxTables

#: Result encoding shared with ``BlokusDuoGame.get_game_ended``.
DRAW_VALUE = 1e-4


class GameState(NamedTuple):
    """Batchable Blokus Duo state. All fixed-shape; a valid jax pytree.

    Attributes:
        ppb: int8 ``(196,)`` signed placement board — ``+piece_id`` White,
            ``-piece_id`` Black, 0 empty. Same convention (flattened) as
            ``BlokusDuoBoard._piece_placement_board``.
        remaining: bool ``(2, NUM_PIECE_IDS + 1)`` — piece inventory per player
            slot (0 = White, 1 = Black), indexed by piece id; column 0 unused
            and always False.
        last_piece: int8 ``(2,)`` — last piece id placed per slot, 0 = none yet.
        current_player: int8 scalar, +1 White / -1 Black.
    """

    ppb: jnp.ndarray
    remaining: jnp.ndarray
    last_piece: jnp.ndarray
    current_player: jnp.ndarray


def _player_slot(player: jnp.ndarray) -> jnp.ndarray:
    """+1 -> slot 0 (White), -1 -> slot 1 (Black)."""
    return (1 - player.astype(jnp.int32)) // 2


@dataclass(frozen=True)
class JaxKernels:
    """Jitted kernels closed over one set of static tables.

    Attributes:
        initial_state: () -> GameState — fresh empty-board state.
        legal_mask: (GameState) -> bool (action_size,) — full legal-move mask
            for the state's current player, pass bit included.
        legal_mask_batch: vmapped ``legal_mask``.
        has_any_placement: (GameState, player) -> bool scalar — does ``player``
            have any legal *placement* (pass excluded)?
        step: (GameState, action) -> GameState — apply one action (placement
            or pass) for the current player.
        step_batch: vmapped ``step``.
        game_result: (GameState, player) -> float32 scalar — 0 ongoing, ±1
            win/loss from ``player``'s perspective, ``DRAW_VALUE`` draw.
        game_result_batch: vmapped ``game_result``.
        score: (GameState, slot) -> int32 scalar — the player-slot's final
            score per ``BlokusDuoGame._calculate_score`` (exposed for tests).
        action_size: Flat action-space size.
        pass_index: Flat action id of pass.
    """

    initial_state: Callable[[], GameState]
    legal_mask: Callable[[GameState], jnp.ndarray]
    legal_mask_batch: Callable[[GameState], jnp.ndarray]
    has_any_placement: Callable[[GameState, jnp.ndarray], jnp.ndarray]
    step: Callable[[GameState, jnp.ndarray], GameState]
    step_batch: Callable[[GameState, jnp.ndarray], GameState]
    game_result: Callable[[GameState, jnp.ndarray], jnp.ndarray]
    game_result_batch: Callable[[GameState, jnp.ndarray], jnp.ndarray]
    score: Callable[[GameState, jnp.ndarray], jnp.ndarray]
    action_size: int
    pass_index: int


def make_kernels(tables: JaxTables) -> JaxKernels:
    """Build jitted kernels closed over ``tables`` as on-device constants."""
    cover = jnp.asarray(tables.cover)  # int8 (A, 196)
    edge = jnp.asarray(tables.edge)
    corner = jnp.asarray(tables.corner)
    piece_of_action = jnp.asarray(tables.piece_of_action)  # int8 (A,)
    placeable = jnp.asarray(tables.placeable)  # bool (A,)
    piece_sizes = jnp.asarray(tables.piece_sizes, dtype=jnp.int32)  # (22,)
    start_cell = jnp.asarray(tables.start_cell)  # int32 (2,)
    pass_index = tables.pass_index
    action_size = tables.action_size

    def initial_state() -> GameState:
        remaining = jnp.ones((2, NUM_PIECE_IDS + 1), dtype=jnp.bool_).at[:, 0].set(False)
        return GameState(
            ppb=jnp.zeros(tables.num_cells, dtype=jnp.int8),
            remaining=remaining,
            last_piece=jnp.zeros(2, dtype=jnp.int8),
            current_player=jnp.int8(1),
        )

    def _placement_mask(state: GameState, player: jnp.ndarray) -> jnp.ndarray:
        """Bool (A,): legal placements (pass excluded) for ``player``."""
        slot = _player_slot(player)
        own = (state.ppb * player > 0).astype(jnp.int8)  # (196,)
        occ = (state.ppb != 0).astype(jnp.int8)
        overlap = jnp.matmul(cover, occ, preferred_element_type=jnp.int32)
        edge_hits = jnp.matmul(edge, own, preferred_element_type=jnp.int32)
        corner_hits = jnp.matmul(corner, own, preferred_element_type=jnp.int32)
        remaining_row = state.remaining[slot]  # (22,) bool
        available = remaining_row[piece_of_action]
        first_move = remaining_row[1:].all()
        corner_ok = jnp.where(first_move, cover[:, start_cell[slot]] > 0, corner_hits > 0)
        return placeable & available & (overlap == 0) & (edge_hits == 0) & corner_ok

    @jax.jit
    def legal_mask(state: GameState) -> jnp.ndarray:
        mask = _placement_mask(state, state.current_player)
        return mask.at[pass_index].set(~mask.any())

    @jax.jit
    def has_any_placement(state: GameState, player: jnp.ndarray) -> jnp.ndarray:
        return _placement_mask(state, player).any()

    @jax.jit
    def step(state: GameState, action: jnp.ndarray) -> GameState:
        """Apply ``action`` for the current player.

        Pass needs no special-casing: its ``cover`` row is all-zero and its
        ``piece_of_action`` entry is 0, so the board, inventory (column 0 is
        already False) and last-piece (guarded below) are all unchanged.
        """
        player = state.current_player
        slot = _player_slot(player)
        piece = piece_of_action[action]  # int8, 0 for pass
        new_ppb = state.ppb + cover[action] * piece * player
        new_remaining = state.remaining.at[slot, piece].set(False)
        new_last = jnp.where(piece > 0, piece, state.last_piece[slot])
        return GameState(
            ppb=new_ppb,
            remaining=new_remaining,
            last_piece=state.last_piece.at[slot].set(new_last),
            current_player=(-player).astype(jnp.int8),
        )

    def _score(state: GameState, slot: jnp.ndarray) -> jnp.ndarray:
        """Mirrors ``BlokusDuoGame._calculate_score``."""
        remaining_row = state.remaining[slot]
        remaining_squares = jnp.matmul(
            remaining_row.astype(jnp.int32), piece_sizes, preferred_element_type=jnp.int32
        )
        all_placed = ~remaining_row[1:].any()
        bonus = 15 + jnp.where(state.last_piece[slot] == 1, 5, 0)
        return jnp.where(all_placed, bonus, -remaining_squares)

    @jax.jit
    def game_result(state: GameState, player: jnp.ndarray) -> jnp.ndarray:
        """Mirrors ``BlokusDuoGame.get_game_ended``."""
        ongoing = has_any_placement(state, jnp.int8(1)) | has_any_placement(state, jnp.int8(-1))
        white_score = _score(state, jnp.int32(0))
        black_score = _score(state, jnp.int32(1))
        draw = white_score == black_score
        white_win = white_score > black_score
        player_win = white_win == (player == 1)
        value = jnp.where(draw, DRAW_VALUE, jnp.where(player_win, 1.0, -1.0))
        return jnp.where(ongoing, 0.0, value).astype(jnp.float32)

    return JaxKernels(
        initial_state=initial_state,
        legal_mask=legal_mask,
        legal_mask_batch=jax.jit(jax.vmap(legal_mask)),
        has_any_placement=has_any_placement,
        step=step,
        step_batch=jax.jit(jax.vmap(step)),
        game_result=game_result,
        game_result_batch=jax.jit(jax.vmap(game_result, in_axes=(0, None))),
        score=jax.jit(_score),
        action_size=action_size,
        pass_index=pass_index,
    )
