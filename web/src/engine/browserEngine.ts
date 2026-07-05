/**
 * The in-browser engine: TS rules + ONNX net + play-time search, all local.
 * Implements the `Engine` interface the UI depends on; the alternative
 * implementation is `ServerEngine` (HTTP to the full-strength Python stack).
 */

import { CPUCT, DIFFICULTY_LEVELS, GUMBEL_MAX_CONSIDERED } from './difficulty';
import { encodePlanes } from './encoding';
import { gumbelBestAction } from './gumbel';
import { LocalRulesEngine } from './localRules';
import { Mcts } from './mcts';
import type { Predictor } from './net';
import { OrtWebPredictor } from './net';
import { legalMoves } from './rules';
import type { LoadedAssets } from './tables';
import type { DifficultyLevel, Engine, EngineInfo, GameState, SearchResult } from './types';

export class BrowserEngine extends LocalRulesEngine implements Engine {
  private predictor: Predictor | null = null;
  private executionProvider = 'unknown';

  constructor(
    assets: LoadedAssets,
    private readonly assetsBaseUrl: string,
    private readonly netVariant: string = 'fp32',
  ) {
    super(assets);
  }

  /** Progress hook for the UI's thinking indicator. */
  onSearchProgress: ((done: number, total: number) => void) | null = null;

  async init(): Promise<EngineInfo> {
    const netFile = this.assets.manifest.net?.files[this.netVariant];
    if (!netFile) {
      throw new Error(
        `Manifest has no '${this.netVariant}' net variant — run scripts/export_web_assets.py with a checkpoint.`,
      );
    }
    const predictor = await OrtWebPredictor.create(
      `${this.assetsBaseUrl}/${netFile.path}`,
      this.assets.manifest.numChannels,
      this.assets.manifest.numCells,
      this.assets.manifest.actionSize,
    );
    this.predictor = predictor;
    this.executionProvider = predictor.executionProvider;
    return {
      name: `In-browser engine (${this.executionProvider})`,
      isFullStrength: false,
      difficulties: DIFFICULTY_LEVELS,
    };
  }

  async bestMove(state: GameState, difficulty: DifficultyLevel): Promise<SearchResult> {
    if (!this.predictor) throw new Error('Engine not initialised — call init() first.');
    const tables = this.assets.tables;
    const start = performance.now();
    const onProgress = this.onSearchProgress ?? undefined;

    if (difficulty.searchPolicy === 'policy') {
      const [prediction] = await this.predictor.predictBatch([encodePlanes(state)]);
      const legal = legalMoves(tables, state, state.currentPlayer);
      let best = legal[0]!;
      let bestPrior = -Infinity;
      for (const action of legal) {
        const prior = prediction!.priors[action]!;
        if (prior > bestPrior) {
          bestPrior = prior;
          best = action;
        }
      }
      return {
        action: best,
        value: prediction!.value,
        sims: 0,
        elapsedMs: performance.now() - start,
      };
    }

    // A fresh tree per move: no stale-subtree reuse concerns, and memory is
    // bounded by one move's search.
    const mcts = new Mcts(tables, this.predictor, CPUCT);
    if (difficulty.searchPolicy === 'gumbel') {
      const result = await gumbelBestAction(
        tables,
        mcts,
        state,
        difficulty.sims,
        GUMBEL_MAX_CONSIDERED,
        onProgress,
      );
      return {
        action: result.action,
        value: result.value,
        sims: result.simsRun,
        elapsedMs: performance.now() - start,
      };
    }

    const result = await mcts.bestAction(state, difficulty.sims, onProgress);
    return {
      action: result.action,
      value: result.value,
      sims: result.simsRun,
      elapsedMs: performance.now() - start,
    };
  }
}
