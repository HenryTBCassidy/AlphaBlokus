/**
 * Search mechanics with a stub predictor (uniform priors, value 0): both
 * search policies must run their budget, stay inside the legal move set, and
 * play a full self-play game to a coherent terminal state.
 */

import { describe, expect, it } from 'vitest';

import { gumbelBestAction } from '../src/engine/gumbel';
import { Mcts } from '../src/engine/mcts';
import type { Prediction, Predictor } from '../src/engine/net';
import { gameResult, initialState, legalMoves, step } from '../src/engine/rules';
import { loadManifest, loadTables } from './helpers';

const manifest = loadManifest();
const tables = loadTables(manifest);

const uniformPredictor: Predictor = {
  predictBatch(planes: Float32Array[]): Promise<Prediction[]> {
    return Promise.resolve(
      planes.map(() => ({
        priors: new Float32Array(manifest.actionSize).fill(1 / manifest.actionSize),
        value: 0,
      })),
    );
  },
};

describe('play-time search', () => {
  it('PUCT plays a legal move and spends its budget', async () => {
    const root = initialState();
    const mcts = new Mcts(tables, uniformPredictor, 2.5);
    const result = await mcts.bestAction(root, 24);
    expect(result.simsRun).toBe(24);
    expect(legalMoves(tables, root, 1)).toContain(result.action);
  });

  it('Gumbel plays a legal move', async () => {
    const root = initialState();
    const mcts = new Mcts(tables, uniformPredictor, 2.5);
    const result = await gumbelBestAction(tables, mcts, root, 32, 8);
    expect(legalMoves(tables, root, 1)).toContain(result.action);
    expect(result.simsRun).toBeGreaterThan(0);
  });

  it('PUCT self-play reaches a terminal state with alternating players', async () => {
    let state = initialState();
    let plies = 0;
    while (gameResult(tables, state, state.currentPlayer) === 0) {
      const mcts = new Mcts(tables, uniformPredictor, 2.5);
      const { action } = await mcts.bestAction(state, 8);
      const legal = legalMoves(tables, state, state.currentPlayer);
      expect(legal, `ply ${plies}`).toContain(action);
      state = step(tables, state, action);
      plies++;
      expect(plies).toBeLessThan(200); // both players hold 21 pieces + passes
    }
    // Terminal: neither side may have a placement, only forced passes.
    expect(legalMoves(tables, state, 1)).toEqual([tables.passIndex]);
    expect(legalMoves(tables, state, -1)).toEqual([tables.passIndex]);
  });
});
