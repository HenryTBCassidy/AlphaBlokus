/**
 * Shared local-rules half of the `Engine` interface. Both engines answer
 * `legalMoves`/`applyMove`/`gameStatus` from the TS rules port (interaction
 * needs synchronous-speed legality); they differ only in where `bestMove`
 * comes from.
 */

import { gameResult, legalMoves, score, step } from './rules';
import type { LoadedAssets } from './tables';
import type { GameState, GameStatus } from './types';

export abstract class LocalRulesEngine {
  constructor(protected readonly assets: LoadedAssets) {}

  legalMoves(state: GameState): Promise<number[]> {
    return Promise.resolve(legalMoves(this.assets.tables, state, state.currentPlayer));
  }

  applyMove(state: GameState, action: number): GameState {
    return step(this.assets.tables, state, action);
  }

  gameStatus(state: GameState): GameStatus {
    const tables = this.assets.tables;
    const ended = gameResult(tables, state, 1);
    if (ended === 0) return { isOver: false };
    const scores: [number, number] = [score(tables, state, 0), score(tables, state, 1)];
    const winner = scores[0] === scores[1] ? 0 : scores[0] > scores[1] ? 1 : -1;
    return { isOver: true, scores, winner };
  }
}
