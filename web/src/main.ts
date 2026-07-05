// Entry point. The full UI lands in plan step W8; until then this boots the
// engine stack (assets + ONNX session) so the build pipeline exercises it.
import { OrtWebPredictor } from './engine/net';
import { loadAssets } from './engine/tables';

const app = document.getElementById('app');

async function boot(): Promise<void> {
  if (!app) return;
  app.textContent = 'Loading engine assets…';
  const assets = await loadAssets('./assets');
  const netFile = assets.manifest.net?.files['fp32'];
  if (!netFile) {
    app.textContent = 'Rules loaded; no net exported yet (run scripts/export_web_assets.py).';
    return;
  }
  const predictor = await OrtWebPredictor.create(
    `./assets/${netFile.path}`,
    assets.manifest.numChannels,
    assets.manifest.numCells,
    assets.manifest.actionSize,
  );
  app.textContent = `Engine ready (${predictor.executionProvider}). UI coming in W8.`;
}

boot().catch((error) => {
  if (app) app.textContent = `Engine failed to boot: ${String(error)}`;
});
