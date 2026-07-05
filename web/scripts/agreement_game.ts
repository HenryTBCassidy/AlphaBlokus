/**
 * Agreement-game generator (plan step W11): the browser engine's stack
 * (TS rules + exported ONNX net, raw-policy argmax both sides) plays full
 * games and records every decision. `scripts/verify_web_agreement.py` then
 * replays the record through the reference Python engine + torch and asserts
 * agreement ply by ply.
 *
 * Run from web/:  npm run agreement   (writes tests/artifacts/agreement_games.json)
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { InferenceSession, Tensor } from 'onnxruntime-node';

import { encodePlanes, NUM_CHANNELS } from '../src/engine/encoding';
import { postprocessOutputs } from '../src/engine/net';
import { gameResult, initialState, legalMoves, score, step } from '../src/engine/rules';
import { loadManifest, loadTables, assetsDir } from '../tests/helpers';

const NUM_GAMES = 3;

const webRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const manifest = loadManifest();
const tables = loadTables(manifest);
const session = await InferenceSession.create(join(assetsDir, 'model.onnx'));

interface PlyRecord {
  action: number;
  legal: number[];
  value: number;
  chosenProb: number;
}

interface GameRecord {
  plies: PlyRecord[];
  finalScores: [number, number];
  result: number; // gameResult for player 1 at the terminal state
}

async function predict(state: ReturnType<typeof initialState>) {
  const planes = encodePlanes(state);
  const tensor = new Tensor('float32', planes, [1, NUM_CHANNELS, 14, 14]);
  const outputs = await session.run({ board: tensor });
  return postprocessOutputs(
    outputs['log_policy']!.data as Float32Array,
    outputs['value']!.data as Float32Array,
    1,
    manifest.actionSize,
  )[0]!;
}

async function playGame(): Promise<GameRecord> {
  let state = initialState();
  const plies: PlyRecord[] = [];
  while (gameResult(tables, state, state.currentPlayer) === 0) {
    const legal = legalMoves(tables, state, state.currentPlayer);
    const prediction = await predict(state);
    let action = legal[0]!;
    let bestProb = -Infinity;
    for (const candidate of legal) {
      const prob = prediction.priors[candidate]!;
      if (prob > bestProb) {
        bestProb = prob;
        action = candidate;
      }
    }
    plies.push({ action, legal, value: prediction.value, chosenProb: bestProb });
    state = step(tables, state, action);
  }
  return {
    plies,
    finalScores: [score(tables, state, 0), score(tables, state, 1)],
    result: gameResult(tables, state, 1),
  };
}

const games: GameRecord[] = [];
for (let index = 0; index < NUM_GAMES; index++) {
  games.push(await playGame());
  console.log(
    `game ${index}: ${games[index]!.plies.length} plies, scores ${games[index]!.finalScores}`,
  );
}

const outDir = join(webRoot, 'tests', 'artifacts');
mkdirSync(outDir, { recursive: true });
const outPath = join(outDir, 'agreement_games.json');
writeFileSync(outPath, JSON.stringify({ netVariant: 'fp32', games }));
console.log(`wrote ${outPath}`);
