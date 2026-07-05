/**
 * Blokus Duo rules engine — a TypeScript port of the JAX kernel semantics
 * (games/blokusduo/jax/kernels.py), evaluated over the sparse cell lists in
 * the exported tables instead of dense int8 matmuls. Same conditions, same
 * state fields, same scoring; parity is pinned by the fixture battery.
 *
 * Rule conditions for player p (placement is legal iff ALL hold):
 *  - piece still in inventory
 *  - footprint cells all empty            (cover @ occ == 0)
 *  - no footprint-edge touch of own colour (edge @ own == 0)
 *  - some corner touch of own colour       (corner @ own > 0), OR on the
 *    player's first move: footprint covers the player's start cell
 *  - pass is legal iff no placement is
 */

import type { GameState, Player } from './types';
import { DRAW_VALUE } from './types';
import type { RulesTables } from './tables';

/** +1 -> slot 0 (White), -1 -> slot 1 (Black). */
export function playerSlot(player: Player): 0 | 1 {
  return player === 1 ? 0 : 1;
}

/** Fresh empty-board state; White to move. */
export function initialState(): GameState {
  const remaining = new Uint8Array(2 * 22).fill(1);
  remaining[0] = 0; // piece id 0 is padding in both slots
  remaining[22] = 0;
  return {
    ppb: new Int8Array(196),
    remaining,
    lastPiece: new Int8Array(2),
    currentPlayer: 1,
  };
}

export function cloneState(state: GameState): GameState {
  return {
    ppb: state.ppb.slice(),
    remaining: state.remaining.slice(),
    lastPiece: state.lastPiece.slice(),
    currentPlayer: state.currentPlayer,
  };
}

/** Compact serialisable key for MCTS transposition tables (mirrors state_key = ppb bytes). */
export function stateKey(state: GameState): string {
  // The signed placement board uniquely identifies the position (as in
  // BlokusDuoBoard.state_key); include the player so canonical/non-canonical
  // callers can't collide.
  let key = state.currentPlayer === 1 ? 'w:' : 'b:';
  for (let i = 0; i < state.ppb.length; i++) key += String.fromCharCode(state.ppb[i]! + 32);
  return key;
}

function isFirstMove(state: GameState, slot: number): boolean {
  for (let pieceId = 1; pieceId <= 21; pieceId++) {
    if (state.remaining[slot * 22 + pieceId] === 0) return false;
  }
  return true;
}

function isPlacementLegal(
  tables: RulesTables,
  state: GameState,
  player: Player,
  move: number,
  firstMove: boolean,
  slot: number,
): boolean {
  const { cells, adjCells, attachCells, cellsPerMove, adjPerMove, attachPerMove, nullCell } =
    tables;
  const ppb = state.ppb;

  if (state.remaining[slot * 22 + tables.piece[move]!] === 0) return false;

  // Overlap-free: every footprint cell empty. Cell lists are packed left and
  // NULL_CELL padded, so the first NULL ends the list.
  const cellBase = move * cellsPerMove;
  for (let k = 0; k < cellsPerMove; k++) {
    const cell = cells[cellBase + k]!;
    if (cell === nullCell) break;
    if (ppb[cell] !== 0) return false;
  }

  // No own-colour edge contact.
  const adjBase = move * adjPerMove;
  for (let k = 0; k < adjPerMove; k++) {
    const cell = adjCells[adjBase + k]!;
    if (cell === nullCell) break;
    if (ppb[cell]! * player > 0) return false;
  }

  if (firstMove) {
    // First move: footprint must cover the player's start cell (replaces the corner rule).
    const startCell = tables.startCells[slot]!;
    for (let k = 0; k < cellsPerMove; k++) {
      const cell = cells[cellBase + k]!;
      if (cell === nullCell) break;
      if (cell === startCell) return true;
    }
    return false;
  }

  // Some own-colour corner contact.
  const attachBase = move * attachPerMove;
  for (let k = 0; k < attachPerMove; k++) {
    const cell = attachCells[attachBase + k]!;
    if (cell === nullCell) break;
    if (ppb[cell]! * player > 0) return true;
  }
  return false;
}

/**
 * Legal action ids for `player` in `state`, ascending. Includes the pass id
 * (alone) exactly when no placement is legal — mirroring the kernel's
 * `mask.at[pass_index].set(~mask.any())`.
 */
export function legalMoves(tables: RulesTables, state: GameState, player: Player): number[] {
  const slot = playerSlot(player);
  const firstMove = isFirstMove(state, slot);
  const actions: number[] = [];
  for (let move = 0; move < tables.numMoves; move++) {
    if (isPlacementLegal(tables, state, player, move, firstMove, slot)) {
      actions.push(tables.actionId[move]!);
    }
  }
  if (actions.length === 0) return [tables.passIndex];
  actions.sort((a, b) => a - b);
  return actions;
}

/** Does `player` have any legal placement (pass excluded)? Early-exits. */
export function hasAnyPlacement(tables: RulesTables, state: GameState, player: Player): boolean {
  const slot = playerSlot(player);
  const firstMove = isFirstMove(state, slot);
  for (let move = 0; move < tables.numMoves; move++) {
    if (isPlacementLegal(tables, state, player, move, firstMove, slot)) return true;
  }
  return false;
}

/**
 * Apply `action` for the state's current player; returns a new state.
 * Pass leaves the board and inventory unchanged (as in the kernel, where the
 * pass row is all-zero). Throws on unknown action ids; legality of
 * placements is the caller's responsibility (UI/search only offer legal ids).
 */
export function step(tables: RulesTables, state: GameState, action: number): GameState {
  const next = cloneState(state);
  const player = state.currentPlayer;
  next.currentPlayer = (-player === 1 ? 1 : -1) as Player;

  if (action === tables.passIndex) return next;

  const move = tables.actionToMove[action]!;
  if (move < 0) throw new Error(`Action ${action} is not a placement or pass`);

  const slot = playerSlot(player);
  const pieceId = tables.piece[move]!;
  const cellBase = move * tables.cellsPerMove;
  for (let k = 0; k < tables.cellsPerMove; k++) {
    const cell = tables.cells[cellBase + k]!;
    if (cell === tables.nullCell) break;
    next.ppb[cell] = pieceId * player;
  }
  next.remaining[slot * 22 + pieceId] = 0;
  next.lastPiece[slot] = pieceId;
  return next;
}

/**
 * Final score for a player slot, mirroring `BlokusDuoGame._calculate_score`:
 * -(squares remaining), or +15 with all pieces placed (+5 more if the
 * monomino went last).
 */
export function score(tables: RulesTables, state: GameState, slot: 0 | 1): number {
  let remainingSquares = 0;
  let allPlaced = true;
  for (let pieceId = 1; pieceId <= 21; pieceId++) {
    if (state.remaining[slot * 22 + pieceId] !== 0) {
      allPlaced = false;
      remainingSquares += tables.pieceSizes[pieceId]!;
    }
  }
  if (!allPlaced) return -remainingSquares;
  return 15 + (state.lastPiece[slot] === 1 ? 5 : 0);
}

/**
 * Game result from `player`'s perspective: 0 ongoing, +1 win, -1 loss,
 * DRAW_VALUE draw. The game ends when NEITHER player has a legal placement.
 */
export function gameResult(tables: RulesTables, state: GameState, player: Player): number {
  if (hasAnyPlacement(tables, state, 1) || hasAnyPlacement(tables, state, -1)) return 0;
  const whiteScore = score(tables, state, 0);
  const blackScore = score(tables, state, 1);
  if (whiteScore === blackScore) return DRAW_VALUE;
  const whiteWins = whiteScore > blackScore;
  return whiteWins === (player === 1) ? 1 : -1;
}
