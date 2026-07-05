/**
 * HTTP client to the local Python play server (`alphablokus-play`) — the
 * full-strength tier. Rules/legality stay local (TS port, parity-tested);
 * `bestMove` is answered by the real torch + MCTS stack. The server's reply
 * includes its own legal-move set for the position, which we cross-check
 * against the local rules at runtime: the Python side is authoritative, so a
 * mismatch is a loud bug, not a fallback.
 */

import { legalMoves } from './rules';
import { LocalRulesEngine } from './localRules';
import type { DifficultyLevel, Engine, EngineInfo, GameState, SearchResult } from './types';

interface ServerMeta {
  name: string;
  isFullStrength: boolean;
  actionSize: number;
  difficulties: DifficultyLevel[];
}

interface ServerBestMove {
  action: number;
  value: number;
  legal: number[];
  sims: number;
  elapsedMs: number;
}

export class ServerEngine extends LocalRulesEngine implements Engine {
  async init(): Promise<EngineInfo> {
    const meta = await fetchServerMeta();
    if (!meta) throw new Error('Play server not reachable at /api/meta');
    if (meta.actionSize !== this.assets.manifest.actionSize) {
      throw new Error(
        `Server action space (${meta.actionSize}) does not match assets (${this.assets.manifest.actionSize})`,
      );
    }
    return {
      name: meta.name,
      isFullStrength: meta.isFullStrength,
      difficulties: meta.difficulties,
    };
  }

  async bestMove(
    state: GameState,
    difficulty: DifficultyLevel,
    history: number[],
  ): Promise<SearchResult> {
    const start = performance.now();
    const response = await fetch('api/best-move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ history, difficulty: difficulty.id }),
    });
    if (!response.ok) {
      throw new Error(`Play server error ${response.status}: ${await response.text()}`);
    }
    const result = (await response.json()) as ServerBestMove;

    // Runtime cross-check: the authoritative Python legal set vs the TS port.
    const local = legalMoves(this.assets.tables, state, state.currentPlayer);
    if (!sameSet(local, result.legal)) {
      console.error(
        'Legal-move mismatch between the TS rules port and the Python engine — the Python side is authoritative.',
        { local, server: result.legal, history },
      );
    }

    return {
      action: result.action,
      value: result.value,
      sims: result.sims,
      elapsedMs: performance.now() - start,
    };
  }
}

/** Probe for the local play server; null on any failure (static hosting). */
export async function fetchServerMeta(timeoutMs = 1500): Promise<ServerMeta | null> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const response = await fetch('api/meta', { signal: controller.signal });
    clearTimeout(timer);
    if (!response.ok) return null;
    return (await response.json()) as ServerMeta;
  } catch {
    return null;
  }
}

function sameSet(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false;
  const sortedB = [...b].sort((x, y) => x - y);
  return a.every((value, index) => value === sortedB[index]);
}
