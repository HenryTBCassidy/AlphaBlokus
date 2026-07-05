/**
 * Piece-orientation helpers for the UI: grid transforms, rotate/flip cycling
 * between the 91 exported orientation ids, and the flat-action codec
 * (mirroring `ActionCodec`: index = y·(14·91) + x·91 + orientationId with
 * (x, y) board coords, bottom-left origin; the anchor is the top-left of the
 * piece's bounding box in array coords).
 */

import type { PiecesData } from '../engine/tables';

export type Grid = number[][];

export function rotateGrid(grid: Grid): Grid {
  // Counter-clockwise quarter turn, matching numpy's rot90 (pieces.py).
  const rows = grid.length;
  const cols = grid[0]!.length;
  const out: Grid = [];
  for (let i = 0; i < cols; i++) {
    const row: number[] = [];
    for (let j = 0; j < rows; j++) row.push(grid[j]![cols - 1 - i]!);
    out.push(row);
  }
  return out;
}

export function flipGrid(grid: Grid): Grid {
  // Horizontal mirror (numpy flip axis=1).
  return grid.map((row) => [...row].reverse());
}

function gridKey(grid: Grid): string {
  return grid.map((row) => row.join('')).join('|');
}

export interface OrientationMaps {
  /** orientationId -> orientationId of the quarter-rotated shape. */
  rotate: number[];
  /** orientationId -> orientationId of the mirrored shape. */
  flip: number[];
  /** pieceId -> its first (identity) orientationId. */
  firstOrientation: Map<number, number>;
  /** orientationId -> grid. */
  grids: Grid[];
  /** orientationId -> pieceId. */
  pieceOf: number[];
}

/**
 * Build rotate/flip cycling maps. Every rotation/mirror of a basis
 * orientation is itself a basis orientation of the same piece (the export
 * enumerates the full closure), so the lookups are total.
 */
export function buildOrientationMaps(pieces: PiecesData): OrientationMaps {
  const byPieceAndShape = new Map<string, number>();
  const grids: Grid[] = [];
  const pieceOf: number[] = [];
  const firstOrientation = new Map<number, number>();

  for (const entry of pieces.orientations) {
    grids[entry.orientationId] = entry.grid;
    pieceOf[entry.orientationId] = entry.pieceId;
    byPieceAndShape.set(`${entry.pieceId}:${gridKey(entry.grid)}`, entry.orientationId);
    if (!firstOrientation.has(entry.pieceId)) {
      firstOrientation.set(entry.pieceId, entry.orientationId);
    }
  }

  const lookup = (pieceId: number, grid: Grid): number => {
    const found = byPieceAndShape.get(`${pieceId}:${gridKey(grid)}`);
    if (found === undefined) {
      throw new Error(`No orientation of piece ${pieceId} matches the transformed grid`);
    }
    return found;
  };

  const rotate: number[] = [];
  const flip: number[] = [];
  for (const entry of pieces.orientations) {
    rotate[entry.orientationId] = lookup(entry.pieceId, rotateGrid(entry.grid));
    flip[entry.orientationId] = lookup(entry.pieceId, flipGrid(entry.grid));
  }
  return { rotate, flip, firstOrientation, grids, pieceOf };
}

export const NUM_ORIENTATIONS = 91;
export const BOARD_SIZE = 14;

/** Flat action id for anchoring `orientationId`'s bounding box at array (row, col). */
export function encodeAction(orientationId: number, row: number, col: number): number {
  const x = col;
  const y = BOARD_SIZE - 1 - row;
  return y * (BOARD_SIZE * NUM_ORIENTATIONS) + x * NUM_ORIENTATIONS + orientationId;
}

export interface DecodedAction {
  orientationId: number;
  row: number;
  col: number;
}

/** Inverse of `encodeAction` (placements only — never call with the pass id). */
export function decodeAction(action: number): DecodedAction {
  const orientationId = action % NUM_ORIENTATIONS;
  const remaining = (action - orientationId) / NUM_ORIENTATIONS;
  const x = remaining % BOARD_SIZE;
  const y = (remaining - x) / BOARD_SIZE;
  return { orientationId, row: BOARD_SIZE - 1 - y, col: x };
}
