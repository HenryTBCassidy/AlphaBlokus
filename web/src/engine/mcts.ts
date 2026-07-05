/**
 * Play-time PUCT MCTS — a port of the search core in `search/mcts.py`.
 *
 * Same selection formula (`q + cpuct·prior·√N_total/(1+n)`, `√(N+EPS)` on
 * unvisited edges), same expansion (mask priors to legal moves, renormalise,
 * uniform fallback), same backprop sign-flip per ply. Self-play-only
 * machinery is deliberately dropped for interactive play: no Dirichlet root
 * noise, no virtual loss (sims run K=1 sequentially against the async net).
 *
 * Instead of canonical boards, nodes hold absolute states and everything is
 * evaluated for `state.currentPlayer` — the encoder canonicalises internally,
 * so the arithmetic seen by the net and the tree is identical to the Python
 * reference's canonical-form dance.
 */

import { encodePlanes } from './encoding';
import type { Predictor } from './net';
import { gameResult, legalMoves, stateKey, step } from './rules';
import type { RulesTables } from './tables';
import type { GameState } from './types';

const EPS = 1e-8;

interface Node {
  /** Ascending legal action ids. */
  acts: number[];
  /** Priors aligned to `acts`, masked + renormalised. */
  priors: Float64Array;
  /** Visit counts per edge. */
  n: Int32Array;
  /** Running-mean Q per edge (root-player-of-this-node perspective). */
  q: Float64Array;
  nTotal: number;
}

export interface MctsResult {
  action: number;
  /** Q of the chosen root edge — the mover's value estimate in [-1, 1]. */
  value: number;
  simsRun: number;
}

export class Mcts {
  private readonly nodes = new Map<string, Node>();
  private readonly endedCache = new Map<string, number>();

  constructor(
    private readonly tables: RulesTables,
    private readonly predictor: Predictor,
    private readonly cpuct: number,
  ) {}

  /** Run `sims` simulations from `root`, then pick the most-visited action. */
  async bestAction(
    root: GameState,
    sims: number,
    onProgress?: (done: number, total: number) => void,
  ): Promise<MctsResult> {
    for (let sim = 0; sim < sims; sim++) {
      await this.simulate(root);
      onProgress?.(sim + 1, sims);
    }

    const node = this.nodes.get(stateKey(root));
    if (!node || node.nTotal === 0) {
      // Degenerate budget (sims too small to visit an edge): fall back to the
      // root priors, mirroring the reference's legal-uniform guard.
      const expanded = node ?? (await this.expand(root));
      let best = 0;
      for (let i = 1; i < expanded.acts.length; i++) {
        if (expanded.priors[i]! > expanded.priors[best]!) best = i;
      }
      return { action: expanded.acts[best]!, value: 0, simsRun: sims };
    }

    let best = 0;
    for (let i = 1; i < node.acts.length; i++) {
      if (node.n[i]! > node.n[best]!) best = i;
    }
    return { action: node.acts[best]!, value: node.q[best]!, simsRun: sims };
  }

  /** Q-values + visit counts of the root's edges (used by the Gumbel driver). */
  async rootStats(root: GameState): Promise<{ node: Node }> {
    const key = stateKey(root);
    const node = this.nodes.get(key) ?? (await this.expand(root));
    return { node };
  }

  /**
   * One selection→expansion→evaluation→backprop simulation. Returns the value
   * seen above the root (the negated leaf value), as the reference `search` does.
   */
  async simulate(root: GameState): Promise<number> {
    let state = root;
    const path: { node: Node; edge: number }[] = [];

    for (;;) {
      const key = stateKey(state);

      let ended = this.endedCache.get(key);
      if (ended === undefined) {
        ended = gameResult(this.tables, state, state.currentPlayer);
        this.endedCache.set(key, ended);
      }
      if (ended !== 0) return this.backprop(path, ended);

      const node = this.nodes.get(key);
      if (!node) {
        const leaf = await this.expandWithValue(state, key);
        return this.backprop(path, leaf.value);
      }

      const edge = this.selectEdge(node);
      path.push({ node, edge });
      state = step(this.tables, state, node.acts[edge]!);
    }
  }

  private selectEdge(node: Node): number {
    const sqrtTotal = Math.sqrt(node.nTotal);
    const sqrtTotalEps = Math.sqrt(node.nTotal + EPS);
    let best = 0;
    let bestScore = -Infinity;
    for (let i = 0; i < node.acts.length; i++) {
      const visits = node.n[i]!;
      const score =
        visits > 0
          ? node.q[i]! + (this.cpuct * node.priors[i]! * sqrtTotal) / (1 + visits)
          : this.cpuct * node.priors[i]! * sqrtTotalEps;
      if (score > bestScore) {
        bestScore = score;
        best = i;
      }
    }
    return best;
  }

  private async expand(state: GameState): Promise<Node> {
    const { node } = await this.expandWithValue(state, stateKey(state));
    return node;
  }

  private async expandWithValue(
    state: GameState,
    key: string,
  ): Promise<{ node: Node; value: number }> {
    const existing = this.nodes.get(key);
    if (existing) return { node: existing, value: 0 };

    const [prediction] = await this.predictor.predictBatch([encodePlanes(state)]);
    const acts = legalMoves(this.tables, state, state.currentPlayer);
    const priors = new Float64Array(acts.length);
    let sum = 0;
    for (let i = 0; i < acts.length; i++) {
      priors[i] = prediction!.priors[acts[i]!]!;
      sum += priors[i]!;
    }
    if (sum > 0) {
      for (let i = 0; i < acts.length; i++) priors[i]! /= sum;
    } else {
      priors.fill(1 / acts.length);
    }

    const node: Node = {
      acts,
      priors,
      n: new Int32Array(acts.length),
      q: new Float64Array(acts.length),
      nTotal: 0,
    };
    this.nodes.set(key, node);
    return { node, value: prediction!.value };
  }

  private backprop(path: { node: Node; edge: number }[], leafValue: number): number {
    let value = -leafValue;
    for (let i = path.length - 1; i >= 0; i--) {
      const { node, edge } = path[i]!;
      if (node.n[edge]! !== 0) {
        node.q[edge] = (node.n[edge]! * node.q[edge]! + value) / (node.n[edge]! + 1);
        node.n[edge]!++;
      } else {
        node.q[edge] = value;
        node.n[edge] = 1;
      }
      node.nTotal++;
      value = -value;
    }
    return value;
  }
}
