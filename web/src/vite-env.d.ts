/// <reference types="vite/client" />

// `?url` asset imports for package-export subpaths (the onnxruntime-web WASM
// runtime files). Vite resolves the export then emits the file as an asset.
declare module '*?url' {
  const url: string;
  export default url;
}
