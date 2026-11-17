import { copyFile, mkdir } from 'node:fs/promises';
const root = new URL('../', import.meta.url);
const destination = new URL('public/runtime/', root);
await mkdir(destination, { recursive: true });
for (const name of ['ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) {
  await copyFile(new URL(`node_modules/onnxruntime-web/dist/${name}`, root), new URL(name, destination));
}
await copyFile(new URL('third-party/ONNXRUNTIME-LICENSE.txt', root), new URL('LICENSE.txt', destination));
console.log('ONNX Runtime WASM assets prepared locally.');
