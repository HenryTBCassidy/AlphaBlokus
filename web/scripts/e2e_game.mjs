/**
 * End-to-end browser game (plan step W12): serve the BUILT app (`vite
 * preview`), drive a complete game in headless Chromium — random legal moves
 * for the human side via the page's test hook, real engine replies (ONNX +
 * search in the page) — and assert the game reaches a coherent final state.
 *
 * Run from web/:  npm run build && npm run e2e
 * Writes tests/artifacts/e2e_history.json (replayable by the Python engine).
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { chromium } from 'playwright';
import { preview } from 'vite';

const webRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const server = await preview({ root: webRoot, preview: { port: 5188 } });
const url = server.resolvedUrls.local[0];
console.log(`serving built app at ${url}`);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 950 } });
const pageErrors = [];
page.on('pageerror', (error) => pageErrors.push(String(error)));

await page.goto(url);
await page.waitForFunction(() => window.__alphablokus !== undefined, null, { timeout: 120000 });

// Level 2 (PUCT 32 sims) — real search on every engine move, tolerable runtime.
const summary = await page.evaluate(async () => {
  const { controller } = window.__alphablokus;
  const level = controller.info.difficulties.find((d) => d.id === 'level-2');
  await controller.newGame(1, level);

  let humanMoves = 0;
  let passes = 0;
  while (!controller.status.isOver) {
    if (controller.phase !== 'humanTurn') {
      throw new Error(`Unexpected phase ${controller.phase} with game not over`);
    }
    const legal = controller.legal;
    if (legal.length === 0) throw new Error('No legal moves while game not over');
    const action = legal[Math.floor(Math.random() * legal.length)];
    if (legal.length === 1) passes++;
    humanMoves++;
    await controller.humanMove(action);
    if (humanMoves > 120) throw new Error('Game did not terminate');
  }
  return {
    humanMoves,
    passes,
    history: controller.history,
    scores: controller.status.scores,
    winner: controller.status.winner,
    engineName: controller.info.name,
  };
});

const statusText = await page.locator('.status').textContent();
await page.screenshot({ path: join(webRoot, 'tests', 'artifacts', 'e2e_final.png') });
await browser.close();
await new Promise((resolve) => server.httpServer.close(resolve));

console.log(`engine: ${summary.engineName}`);
console.log(
  `game over after ${summary.history.length} plies (${summary.humanMoves} human moves, ${summary.passes} forced passes)`,
);
console.log(`final scores (white, black): ${summary.scores}, winner: ${summary.winner}`);
console.log(`status line: ${statusText}`);

if (pageErrors.length > 0) {
  console.error('PAGE ERRORS:', pageErrors);
  process.exit(1);
}
if (!summary.scores || summary.history.length < 4) {
  console.error('Game did not complete sanely');
  process.exit(1);
}
if (!/win|Draw|wins/i.test(statusText ?? '')) {
  console.error('Final status line does not announce a result');
  process.exit(1);
}

const outDir = join(webRoot, 'tests', 'artifacts');
mkdirSync(outDir, { recursive: true });
writeFileSync(
  join(outDir, 'e2e_history.json'),
  JSON.stringify({ history: summary.history, scores: summary.scores }),
);
console.log('e2e game OK — history written for Python replay');
