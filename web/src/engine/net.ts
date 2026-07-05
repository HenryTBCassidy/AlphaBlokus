/**
 * In-browser policy/value predictor over the exported ONNX net.
 *
 * Uses onnxruntime-web with the WebGPU execution provider when available,
 * falling back to WASM (SIMD). The WASM runtime files are bundled as hashed
 * assets via `?url` imports — no CDN calls at runtime. The model graph's
 * outputs are the torch forward's exactly: log-softmax policy + tanh value;
 * `postprocessOutputs` converts to priors, matching
 * `BaseNNetWrapper.predict_encoded` (softmaxed policy, scalar value).
 */

import * as ort from 'onnxruntime-web/webgpu';
import wasmFileUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url';
import wasmLoaderUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.mjs?url';

export interface Prediction {
  /** Softmaxed policy over the full action space. */
  priors: Float32Array;
  /** Value in [-1, 1] for the encoded position's side to move. */
  value: number;
}

/** Backend-agnostic prediction surface (the MCTS depends on this, not on ORT). */
export interface Predictor {
  predictBatch(planes: Float32Array[]): Promise<Prediction[]>;
}

/** exp(log_policy) + value unpacking, shared with the node-side parity tests. */
export function postprocessOutputs(
  logPolicy: Float32Array,
  values: Float32Array,
  batch: number,
  actionSize: number,
): Prediction[] {
  const predictions: Prediction[] = [];
  for (let row = 0; row < batch; row++) {
    const priors = new Float32Array(actionSize);
    const base = row * actionSize;
    for (let a = 0; a < actionSize; a++) priors[a] = Math.exp(logPolicy[base + a]!);
    predictions.push({ priors, value: values[row]! });
  }
  return predictions;
}

export class OrtWebPredictor implements Predictor {
  readonly executionProvider: string;
  private readonly session: ort.InferenceSession;
  private readonly numChannels: number;
  private readonly numCells: number;
  private readonly actionSize: number;

  private constructor(
    session: ort.InferenceSession,
    executionProvider: string,
    numChannels: number,
    numCells: number,
    actionSize: number,
  ) {
    this.session = session;
    this.executionProvider = executionProvider;
    this.numChannels = numChannels;
    this.numCells = numCells;
    this.actionSize = actionSize;
  }

  static async create(
    modelUrl: string,
    numChannels: number,
    numCells: number,
    actionSize: number,
  ): Promise<OrtWebPredictor> {
    ort.env.wasm.wasmPaths = { wasm: wasmFileUrl, mjs: wasmLoaderUrl };
    if (!globalThis.crossOriginIsolated) {
      // Static hosts rarely send COOP/COEP, so SharedArrayBuffer (multi-thread
      // WASM) is unavailable; pin one thread rather than warn-and-fallback.
      ort.env.wasm.numThreads = 1;
    }

    let lastError: unknown;
    for (const provider of ['webgpu', 'wasm'] as const) {
      try {
        const session = await ort.InferenceSession.create(modelUrl, {
          executionProviders: [provider],
        });
        return new OrtWebPredictor(session, provider, numChannels, numCells, actionSize);
      } catch (error) {
        lastError = error;
      }
    }
    throw new Error(`Failed to create ONNX session with webgpu or wasm: ${String(lastError)}`);
  }

  async predictBatch(planes: Float32Array[]): Promise<Prediction[]> {
    const batch = planes.length;
    const perBoard = this.numChannels * this.numCells;
    const input = new Float32Array(batch * perBoard);
    planes.forEach((board, index) => input.set(board, index * perBoard));

    const boardSide = Math.sqrt(this.numCells);
    const tensor = new ort.Tensor('float32', input, [
      batch,
      this.numChannels,
      boardSide,
      boardSide,
    ]);
    const outputs = await this.session.run({ board: tensor });
    const logPolicy = outputs['log_policy']!.data as Float32Array;
    const values = outputs['value']!.data as Float32Array;
    return postprocessOutputs(logPolicy, values, batch, this.actionSize);
  }
}
