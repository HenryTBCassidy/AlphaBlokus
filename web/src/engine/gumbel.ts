/**
 * Gumbel root search — an algorithmic reimplementation of the Gumbel
 * AlphaZero root procedure (Danihelka et al. 2022), as used by the jax
 * training pipeline via mctx (`search_policy: "gumbel"`). Not a line-level
 * port of mctx; the sequence is:
 *
 *  1. Evaluate the root; take log-priors over the legal actions.
 *  2. Sample one Gumbel per legal action; keep the top-m by `g + logits`
 *     (m = maxConsidered, 16 in the training config).
 *  3. Sequential Halving: spread the sim budget over phases, running each
 *     surviving candidate's sims through the shared PUCT tree (child-rooted
 *     descents), then halve, ranking by `g + logits + σ(q̂)` with mctx's
 *     σ(q̂) = (c_visit + max_visits)·c_scale·q̂.
 *  4. Play the last survivor.
 *
 * Fidelity caveats vs mctx (documented in the calibration note): interior
 * descents use PUCT rather than mctx's deterministic improved-policy
 * selection, and completed-Q mixing is replaced by the tree's running-mean Q.
 */

import type { Mcts } from './mcts';
import { step } from './rules';
import type { RulesTables } from './tables';
import type { GameState } from './types';

const C_VISIT = 50;
const C_SCALE = 1.0;

interface Candidate {
  action: number;
  gumbelLogit: number; // g_a + log π_a — fixed for the whole search
  simsRun: number;
  qSum: number;
}

function sampleGumbel(): number {
  // -log(-log(U)); clamp U away from 0/1 for numerical safety.
  const u = Math.min(Math.max(Math.random(), 1e-12), 1 - 1e-12);
  return -Math.log(-Math.log(u));
}

function candidateQ(candidate: Candidate): number {
  return candidate.simsRun > 0 ? candidate.qSum / candidate.simsRun : 0;
}

function rankScore(candidate: Candidate, maxSims: number): number {
  // mctx's σ transform of the Q estimate, added to the fixed Gumbel logits.
  const sigma = (C_VISIT + maxSims) * C_SCALE * candidateQ(candidate);
  return candidate.gumbelLogit + sigma;
}

export interface GumbelResult {
  action: number;
  value: number;
  simsRun: number;
}

/**
 * Pick a root action with Gumbel top-m + Sequential Halving, spending about
 * `sims` child descents in total through the shared `mcts` tree.
 */
export async function gumbelBestAction(
  tables: RulesTables,
  mcts: Mcts,
  root: GameState,
  sims: number,
  maxConsidered: number,
  onProgress?: (done: number, total: number) => void,
): Promise<GumbelResult> {
  const { node } = await mcts.rootStats(root);

  // Top-m by g + log π over the legal actions.
  const scored: Candidate[] = node.acts.map((action, index) => ({
    action,
    gumbelLogit: sampleGumbel() + Math.log(node.priors[index]! + 1e-38),
    simsRun: 0,
    qSum: 0,
  }));
  scored.sort((a, b) => b.gumbelLogit - a.gumbelLogit);
  let candidates = scored.slice(0, Math.max(1, Math.min(maxConsidered, scored.length)));

  if (candidates.length === 1) {
    return { action: candidates[0]!.action, value: 0, simsRun: 0 };
  }

  const phases = Math.ceil(Math.log2(candidates.length));
  let simsDone = 0;
  for (let phase = 0; phase < phases && candidates.length > 1; phase++) {
    const perCandidate = Math.max(1, Math.floor(sims / (phases * candidates.length)));
    for (const candidate of candidates) {
      const child = step(tables, root, candidate.action);
      for (let sim = 0; sim < perCandidate; sim++) {
        // simulate() returns the value seen above the child's root — i.e.
        // from the perspective of the player to move at `root`. That is
        // exactly Q(root, action) for this descent.
        const value = await mcts.simulate(child);
        candidate.qSum += value;
        candidate.simsRun++;
        simsDone++;
        onProgress?.(simsDone, sims);
      }
    }
    const maxSims = Math.max(...candidates.map((candidate) => candidate.simsRun));
    candidates.sort((a, b) => rankScore(b, maxSims) - rankScore(a, maxSims));
    candidates = candidates.slice(0, Math.ceil(candidates.length / 2));
  }

  const winner = candidates[0]!;
  return { action: winner.action, value: candidateQ(winner), simsRun: simsDone };
}
