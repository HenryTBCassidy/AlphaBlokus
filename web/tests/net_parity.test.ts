/**
 * Net-output parity: run the exported ONNX model (via onnxruntime-node — the
 * same graph the browser executes) on TS-encoded fixture positions and assert
 * the outputs match the recorded torch outputs within float tolerance.
 *
 * Skips (with a warning) when the model or fixture is absent — they require a
 * checkpoint export, which CI's rules-only job doesn't do.
 */

import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

import { InferenceSession, Tensor } from 'onnxruntime-node';
import { describe, expect, it } from 'vitest';

import { encodePlanes, NUM_CELLS, NUM_CHANNELS } from '../src/engine/encoding';
import { postprocessOutputs } from '../src/engine/net';
import type { GameState, Player } from '../src/engine/types';
import { assetsDir, fixturesDir, loadManifest } from './helpers';

interface NetFixturePosition {
  ppbB64: string;
  currentPlayer: Player;
  value: number;
  policySum: number;
  topActions: number[];
  topProbs: number[];
}

interface NetFixture {
  checkpoint: string;
  positions: NetFixturePosition[];
}

const modelPath = join(assetsDir, 'model.onnx');
const fixturePath = join(fixturesDir, 'net_parity.json');
const available = existsSync(modelPath) && existsSync(fixturePath);

// Tolerances: fp32 graph vs torch fp32 — conv reassociation differences only.
const POLICY_TOLERANCE = 1e-5;
const VALUE_TOLERANCE = 1e-5;

describe.skipIf(!available)('ONNX net output parity vs torch', () => {
  it('matches torch policy/value on every fixture position', async () => {
    const fixture = JSON.parse(readFileSync(fixturePath, 'utf-8')) as NetFixture;
    const manifest = loadManifest();
    const session = await InferenceSession.create(modelPath);

    let maxPolicyDiff = 0;
    let maxValueDiff = 0;
    for (const position of fixture.positions) {
      const state: GameState = {
        ppb: new Int8Array(Buffer.from(position.ppbB64, 'base64')),
        remaining: new Uint8Array(2 * 22), // encoding ignores inventories
        lastPiece: new Int8Array(2),
        currentPlayer: position.currentPlayer,
      };
      const planes = encodePlanes(state);
      const tensor = new Tensor('float32', planes, [1, NUM_CHANNELS, 14, 14]);
      const outputs = await session.run({ board: tensor });
      const [prediction] = postprocessOutputs(
        outputs['log_policy']!.data as Float32Array,
        outputs['value']!.data as Float32Array,
        1,
        manifest.actionSize,
      );

      maxValueDiff = Math.max(maxValueDiff, Math.abs(prediction!.value - position.value));
      position.topActions.forEach((action, rank) => {
        maxPolicyDiff = Math.max(
          maxPolicyDiff,
          Math.abs(prediction!.priors[action]! - position.topProbs[rank]!),
        );
      });
      // The torch argmax must be the ONNX argmax too — rank stability at the top.
      let argmax = 0;
      for (let a = 1; a < prediction!.priors.length; a++) {
        if (prediction!.priors[a]! > prediction!.priors[argmax]!) argmax = a;
      }
      expect(argmax, 'top-1 action drifted between torch and ONNX').toBe(position.topActions[0]);
    }

    expect(maxPolicyDiff).toBeLessThan(POLICY_TOLERANCE);
    expect(maxValueDiff).toBeLessThan(VALUE_TOLERANCE);
    // Reported for the calibration/fidelity note (W13).
    console.log(
      `net parity over ${fixture.positions.length} positions: ` +
        `max policy diff ${maxPolicyDiff.toExponential(2)}, max value diff ${maxValueDiff.toExponential(2)}`,
    );
  });
});

// NUM_CELLS is imported for the encoding invariant below; keep it exercised.
it('encoding constants agree with the manifest', () => {
  const manifest = loadManifest();
  expect(NUM_CHANNELS).toBe(manifest.numChannels);
  expect(NUM_CELLS).toBe(manifest.numCells);
});
