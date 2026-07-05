/** Node-side helpers for the parity suites: load exported assets + fixtures from disk. */

import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import type { Manifest, RulesTables } from '../src/engine/tables';
import { parseRulesTables } from '../src/engine/tables';
import type { GameState, Player } from '../src/engine/types';

const webRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
export const assetsDir = join(webRoot, 'public', 'assets');
export const fixturesDir = join(webRoot, 'tests', 'fixtures');

export function loadManifest(): Manifest {
  return JSON.parse(readFileSync(join(assetsDir, 'manifest.json'), 'utf-8')) as Manifest;
}

export function loadTables(manifest: Manifest): RulesTables {
  const blob = readFileSync(join(assetsDir, manifest.rules.path));
  return parseRulesTables(
    manifest,
    blob.buffer.slice(blob.byteOffset, blob.byteOffset + blob.byteLength) as ArrayBuffer,
  );
}

export function loadFixture<T>(name: string): T {
  return JSON.parse(readFileSync(join(fixturesDir, name), 'utf-8')) as T;
}

/** One recorded ply from scripts/generate_web_parity_fixtures.py. */
export interface FixturePly {
  ppbB64: string;
  currentPlayer: Player;
  remaining: [number[], number[]];
  lastPiece: [number, number];
  legal: number[];
  encodingSha256: string;
  gameEnded: number;
  action: number | null;
}

export interface RulesFixture {
  seed: number;
  games: FixturePly[][];
}

/** Rebuild a GameState from a recorded ply (absolute colours: +White / -Black). */
export function stateFromPly(ply: FixturePly): GameState {
  const ppb = new Int8Array(Buffer.from(ply.ppbB64, 'base64'));
  const remaining = new Uint8Array(2 * 22);
  for (const pieceId of ply.remaining[0]) remaining[pieceId] = 1;
  for (const pieceId of ply.remaining[1]) remaining[22 + pieceId] = 1;
  const lastPiece = new Int8Array([ply.lastPiece[0], ply.lastPiece[1]]);
  return { ppb, remaining, lastPiece, currentPlayer: ply.currentPlayer };
}
