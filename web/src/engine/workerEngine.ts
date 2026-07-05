/**
 * Main-thread `Engine` that runs the expensive `bestMove` (net + MCTS) in a Web
 * Worker. Rules answer locally via `LocalRulesEngine` at synchronous speed
 * (interaction needs that), while search runs off-thread — so placing a move
 * repaints instantly, the "thinking" indicator shows with live progress, and
 * the page never freezes for the duration of a search. Public surface mirrors
 * `BrowserEngine`; selected in `main.ts` for static (in-browser) hosting.
 */

import { LocalRulesEngine } from './localRules';
import type { LoadedAssets } from './tables';
import type { DifficultyLevel, Engine, EngineInfo, GameState, SearchResult } from './types';

/** Main-thread → worker messages. */
export type WorkerRequest =
  | { type: 'init'; assetsBaseUrl: string; netVariant: string }
  | { type: 'bestMove'; id: number; state: GameState; difficulty: DifficultyLevel };

/** Worker → main-thread messages. */
export type WorkerResponse =
  | { type: 'ready'; info: EngineInfo }
  | { type: 'initError'; message: string }
  | { type: 'progress'; done: number; total: number }
  | { type: 'result'; id: number; result: SearchResult }
  | { type: 'error'; id: number; message: string };

export class WorkerEngine extends LocalRulesEngine implements Engine {
  /** Progress hook for the UI's thinking indicator (now genuinely live). */
  onSearchProgress: ((done: number, total: number) => void) | null = null;

  private readonly worker: Worker;
  private nextId = 0;
  private readonly pending = new Map<
    number,
    { resolve: (result: SearchResult) => void; reject: (error: Error) => void }
  >();
  private onReady: ((info: EngineInfo) => void) | null = null;
  private onReadyError: ((error: Error) => void) | null = null;

  constructor(
    assets: LoadedAssets,
    private readonly assetsBaseUrl: string,
    private readonly netVariant: string = 'fp32',
  ) {
    super(assets);
    this.worker = new Worker(new URL('./engineWorker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => this.handle(event.data);
  }

  init(): Promise<EngineInfo> {
    return new Promise<EngineInfo>((resolve, reject) => {
      this.onReady = resolve;
      this.onReadyError = reject;
      const message: WorkerRequest = {
        type: 'init',
        // Resolve to an absolute URL: inside the worker a relative base would
        // resolve against the worker's own location, not the page.
        assetsBaseUrl: new URL(this.assetsBaseUrl, location.href).href,
        netVariant: this.netVariant,
      };
      this.worker.postMessage(message);
    });
  }

  bestMove(state: GameState, difficulty: DifficultyLevel): Promise<SearchResult> {
    const id = this.nextId++;
    return new Promise<SearchResult>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      const message: WorkerRequest = { type: 'bestMove', id, state, difficulty };
      this.worker.postMessage(message);
    });
  }

  private handle(message: WorkerResponse): void {
    switch (message.type) {
      case 'ready':
        this.onReady?.(message.info);
        this.onReady = null;
        this.onReadyError = null;
        break;
      case 'initError':
        this.onReadyError?.(new Error(message.message));
        this.onReady = null;
        this.onReadyError = null;
        break;
      case 'progress':
        this.onSearchProgress?.(message.done, message.total);
        break;
      case 'result': {
        const entry = this.pending.get(message.id);
        this.pending.delete(message.id);
        entry?.resolve(message.result);
        break;
      }
      case 'error': {
        const entry = this.pending.get(message.id);
        this.pending.delete(message.id);
        entry?.reject(new Error(message.message));
        break;
      }
    }
  }
}
