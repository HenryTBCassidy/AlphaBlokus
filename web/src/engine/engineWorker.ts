/**
 * Web Worker host for the in-browser engine's search. It owns the ONNX net +
 * MCTS (via a `BrowserEngine`) so `bestMove` runs off the main thread; the page
 * stays responsive, the placed piece locks in immediately, and the thinking
 * indicator reports live progress instead of the tab freezing for the whole
 * search. Rules stay on the main thread (see `WorkerEngine`/`LocalRulesEngine`).
 */

import { BrowserEngine } from './browserEngine';
import { loadAssets } from './tables';
import type { DifficultyLevel, GameState } from './types';
import type { WorkerRequest, WorkerResponse } from './workerEngine';

// The worker global, typed minimally to avoid pulling in the WebWorker lib
// (which conflicts with the DOM lib the rest of the app compiles against).
const ctx = self as unknown as {
  postMessage(message: WorkerResponse): void;
  addEventListener(type: 'message', listener: (event: MessageEvent<WorkerRequest>) => void): void;
};

let engine: BrowserEngine | null = null;

ctx.addEventListener('message', (event) => {
  const message = event.data;
  if (message.type === 'init') {
    void handleInit(message.assetsBaseUrl, message.netVariant);
  } else {
    void handleBestMove(message.id, message.state, message.difficulty);
  }
});

async function handleInit(assetsBaseUrl: string, netVariant: string): Promise<void> {
  try {
    const assets = await loadAssets(assetsBaseUrl);
    const created = new BrowserEngine(assets, assetsBaseUrl, netVariant);
    const info = await created.init();
    created.onSearchProgress = (done, total) => ctx.postMessage({ type: 'progress', done, total });
    engine = created;
    ctx.postMessage({ type: 'ready', info });
  } catch (error) {
    ctx.postMessage({ type: 'initError', message: String(error) });
  }
}

async function handleBestMove(
  id: number,
  state: GameState,
  difficulty: DifficultyLevel,
): Promise<void> {
  try {
    if (!engine) throw new Error('Worker engine not initialised — send an init message first.');
    const result = await engine.bestMove(state, difficulty);
    ctx.postMessage({ type: 'result', id, result });
  } catch (error) {
    ctx.postMessage({ type: 'error', id, message: String(error) });
  }
}
