/**
 * 44-channel board encoding — a port of `encode_planes_from_placement`
 * (games/blokusduo/board.py). Channel layout (44 x 14 x 14, row-major):
 *   0-20   current player's per-piece binary planes (piece id 1..21)
 *   21-41  opponent's per-piece planes
 *   42     aggregate current player, 43 aggregate opponent
 *
 * The board is canonicalised first (multiply by currentPlayer so the side to
 * move reads positive), exactly like `as_multi_channel(current_player)`.
 */

import type { GameState } from './types';

export const NUM_CHANNELS = 44;
export const NUM_CELLS = 196;

/** Encode the state's current player's canonical view. Values are exact 0.0/1.0 floats. */
export function encodePlanes(state: GameState): Float32Array {
  const planes = new Float32Array(NUM_CHANNELS * NUM_CELLS);
  const player = state.currentPlayer;
  for (let cell = 0; cell < NUM_CELLS; cell++) {
    const signed = state.ppb[cell]! * player; // canonical: own pieces positive
    if (signed === 0) continue;
    if (signed > 0) {
      planes[(signed - 1) * NUM_CELLS + cell] = 1;
      planes[42 * NUM_CELLS + cell] = 1;
    } else {
      planes[(21 + -signed - 1) * NUM_CELLS + cell] = 1;
      planes[43 * NUM_CELLS + cell] = 1;
    }
  }
  return planes;
}
