import { defineConfig } from 'vite';

// Everything must bundle — no CDN calls at runtime. The onnxruntime-web WASM
// binaries are pulled in as hashed assets via `?url` imports in src/engine/net.ts,
// so no copy plugin is needed here.
export default defineConfig({
  base: './',
  build: {
    target: 'es2022',
    // model.onnx (~10 MB) and rules.bin live in public/assets — copied verbatim.
    chunkSizeWarningLimit: 1500,
  },
  // vite pre-bundles CommonJS deps; onnxruntime-web ships ESM + workers that
  // must be left alone for the threaded WASM backend to boot.
  optimizeDeps: { exclude: ['onnxruntime-web'] },
});
