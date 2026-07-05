/**
 * Difficulty levels: fixed {search policy, sim budget} pairs, so playing
 * strength is identical on every visitor's machine — hardware only changes
 * how long the engine thinks. Budgets bracket the training regimes (Gumbel
 * n=64 self-play; PUCT 800-sim arena); relative strength between levels is
 * measured by the sims-ladder round robin in
 * docs/research/web-play-calibration.md (plan step W13).
 */

import type { DifficultyLevel } from './types';

export const DIFFICULTY_LEVELS: DifficultyLevel[] = [
  {
    id: 'level-1',
    label: 'Level 1 — Instinct',
    searchPolicy: 'policy',
    sims: 0,
    description: 'Raw policy network, no search. Instant moves.',
  },
  {
    id: 'level-2',
    label: 'Level 2 — Quick',
    searchPolicy: 'puct',
    sims: 32,
    description: 'PUCT search, 32 simulations per move.',
  },
  {
    id: 'level-3',
    label: 'Level 3 — Club',
    searchPolicy: 'puct',
    sims: 128,
    description: 'PUCT search, 128 simulations per move.',
  },
  {
    id: 'level-4',
    label: 'Level 4 — Strong',
    searchPolicy: 'gumbel',
    sims: 256,
    description: 'Gumbel root + Sequential Halving, 256 simulations.',
  },
  {
    id: 'level-5',
    label: 'Level 5 — Max',
    searchPolicy: 'puct',
    sims: 800,
    description: 'PUCT search at the training-arena budget (800 simulations).',
  },
];

export const DEFAULT_DIFFICULTY = DIFFICULTY_LEVELS[2]!;

/** Root actions Sequential Halving considers (matches the training config). */
export const GUMBEL_MAX_CONSIDERED = 16;

/** Exploration constant — the production training value (blokus run configs). */
export const CPUCT = 2.5;
