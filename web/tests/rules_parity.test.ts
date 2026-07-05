/**
 * Rules/encoding parity: replay every fixture game through the TS engine and
 * assert it agrees with the Python reference engine exactly — same legal-move
 * sets, byte-identical 44-channel encodings, same game-ended values, same
 * board evolution. Fixtures come from scripts/generate_web_parity_fixtures.py.
 */

import { createHash } from 'node:crypto';

import { describe, expect, it } from 'vitest';

import { encodePlanes } from '../src/engine/encoding';
import { gameResult, initialState, legalMoves, step } from '../src/engine/rules';
import { DRAW_VALUE } from '../src/engine/types';
import type { RulesFixture } from './helpers';
import { loadFixture, loadManifest, loadTables, stateFromPly } from './helpers';

const manifest = loadManifest();
const tables = loadTables(manifest);
const fixture = loadFixture<RulesFixture>('rules_parity.json');

function sha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function encodingHash(state: Parameters<typeof encodePlanes>[0]): string {
  const planes = encodePlanes(state);
  return sha256(new Uint8Array(planes.buffer, planes.byteOffset, planes.byteLength));
}

describe('rules parity vs the Python reference engine', () => {
  it('loads the expected table shapes', () => {
    expect(tables.numMoves).toBe(13729);
    expect(tables.actionSize).toBe(17837);
    expect(tables.passIndex).toBe(17836);
  });

  fixture.games.forEach((plies, gameIndex) => {
    it(`game ${gameIndex}: every ply matches (${plies.length} plies)`, () => {
      // Replay from the initial position; the state must evolve exactly as
      // the reference recorded it, and every per-ply quantity must match.
      let state = initialState();
      plies.forEach((ply, plyIndex) => {
        const recorded = stateFromPly(ply);
        const at = `game ${gameIndex} ply ${plyIndex}`;

        expect(Array.from(state.ppb), `${at}: board drifted from reference`).toEqual(
          Array.from(recorded.ppb),
        );
        expect(Array.from(state.remaining), `${at}: inventory drifted`).toEqual(
          Array.from(recorded.remaining),
        );
        expect(Array.from(state.lastPiece), `${at}: lastPiece drifted`).toEqual(
          Array.from(recorded.lastPiece),
        );
        expect(state.currentPlayer, `${at}: player drifted`).toBe(ply.currentPlayer);

        const ended = gameResult(tables, state, ply.currentPlayer);
        if (ply.gameEnded === 0) {
          expect(ended, `${at}: gameEnded`).toBe(0);
        } else if (Math.abs(ply.gameEnded) === 1) {
          expect(ended, `${at}: gameEnded`).toBe(ply.gameEnded);
        } else {
          expect(ended, `${at}: draw value`).toBe(DRAW_VALUE);
        }

        if (ply.action === null) return; // terminal — reference stops here

        const legal = legalMoves(tables, state, ply.currentPlayer);
        expect(legal, `${at}: legal move set`).toEqual(ply.legal);
        expect(encodingHash(state), `${at}: 44-channel encoding`).toBe(ply.encodingSha256);

        state = step(tables, state, ply.action);
      });
    });
  });
});
