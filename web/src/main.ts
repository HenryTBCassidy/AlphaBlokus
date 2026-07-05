/**
 * Boot: load the exported assets, pick an engine backend, wire controller +
 * view. Engine selection (plan step W10): if the page is served by the local
 * Python server (`alphablokus-play`), its `/api/meta` endpoint responds and we
 * use the full-strength `ServerEngine`; otherwise (static hosting) the
 * in-browser engine runs everything locally.
 */

import './style.css';

import { BrowserEngine } from './engine/browserEngine';
import { loadAssets } from './engine/tables';
import type { Engine, Player } from './engine/types';
import { AppView, exposeTestHook } from './ui/app';
import { GameController } from './ui/controller';

const ASSETS_BASE = './assets';

async function boot(): Promise<void> {
  const root = document.getElementById('app');
  if (!root) return;
  root.textContent = 'Loading engine…';

  const assets = await loadAssets(ASSETS_BASE);
  const engine: Engine = new BrowserEngine(assets, ASSETS_BASE, netVariantFromQuery());
  const info = await engine.init();

  const view = new AppView(root, assets);
  const controller = new GameController(engine, info, view.render);
  if (engine instanceof BrowserEngine) {
    engine.onSearchProgress = (done, total) => controller.reportProgress(done, total);
  }
  view.attach(controller);
  exposeTestHook(controller, () => controller.state);

  const params = new URLSearchParams(location.search);
  const humanPlayer = (params.get('play') === 'black' ? -1 : 1) as Player;
  await controller.newGame(humanPlayer, controller.difficulty);
}

function netVariantFromQuery(): string {
  // ?net=fp16 / ?net=int8 opt into the smaller quantised downloads.
  return new URLSearchParams(location.search).get('net') ?? 'fp32';
}

boot().catch((error: unknown) => {
  const root = document.getElementById('app');
  if (root) {
    root.innerHTML = '';
    const message = document.createElement('div');
    message.className = 'boot-error';
    message.textContent = `Engine failed to boot: ${String(error)}`;
    root.append(message);
  }
});
