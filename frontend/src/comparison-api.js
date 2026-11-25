import { CLASSES } from './classes.js';
import { readImagePixels } from './image-preprocessing.js';
import { predictionResult } from './inference-utils.js';

export const MODEL_IDS = ['mobilenetv2', 'resnet50', 'vgg16'];
const asset = path => `${import.meta.env.BASE_URL}model/comparison/${path}`;
let catalogPromise;
let active;
let queue = Promise.resolve();

export async function comparisonCatalog() {
  if (!catalogPromise) catalogPromise = fetch(asset('manifest.json'), { cache: 'no-cache' }).then(async response => {
    if (!response.ok) throw new Error('Chưa tải được danh sách ba mô hình. Kiểm tra kết nối và thử lại.');
    const data = await response.json();
    if (JSON.stringify(data.classes) !== JSON.stringify(CLASSES.map(x => x.key)) ||
        MODEL_IDS.some(id => !data.models?.find(model => model.id === id))) throw new Error('Thông tin mô hình không khớp ứng dụng.');
    for (const model of data.models) {
      if (!MODEL_IDS.includes(model.id) || !/^[a-z0-9-]+\.onnx$/.test(model.file) ||
          !/^[a-f0-9]{64}$/.test(model.sha256) || !Number.isSafeInteger(model.size_bytes) || model.size_bytes < 1) {
        throw new Error('Thông tin file mô hình không hợp lệ.');
      }
    }
    return data;
  }).catch(error => { catalogPromise = undefined; throw error; });
  return catalogPromise;
}

async function sessionFor(model, onProgress) {
  if (active?.id === model.id && active.sha256 === model.sha256) return { session: active.session, loadedNow: false, loadMs: 0 };
  // Only one comparison session remains in RAM; model files can use the HTTP cache.
  if (active) { await active.session.release(); active = undefined; }
  const start = performance.now();
  const [ort, response] = await Promise.all([
    import('onnxruntime-web/wasm'),
    fetch(asset(`${model.file}?v=${model.sha256.slice(0, 12)}`), { cache: 'force-cache' })
  ]);
  if (!response.ok) throw new Error(`Không tải được ${model.name}. Hãy kiểm tra mạng và thử lại.`);
  let bytes;
  if (response.body?.getReader) {
    bytes = new Uint8Array(model.size_bytes);
    const reader = response.body.getReader();
    let offset = 0;
    let reportedPercent = -1;
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        if (offset + value.byteLength > bytes.byteLength) throw new Error('Kích thước file mô hình không đúng.');
        bytes.set(value, offset); offset += value.byteLength;
        const percent = Math.round(offset / bytes.byteLength * 100);
        if (percent !== reportedPercent) { onProgress?.(`Đang tải ${model.name}: ${percent}%`); reportedPercent = percent; }
      }
    } finally { reader.releaseLock(); }
    if (offset !== bytes.byteLength) throw new Error('File mô hình tải chưa đầy đủ. Hãy thử lại.');
  } else bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.byteLength !== model.size_bytes) throw new Error('File mô hình tải chưa đầy đủ.');
  if (!globalThis.crypto?.subtle) throw new Error('Hãy mở website bằng HTTPS hoặc localhost để xác minh mô hình.');
  const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('');
  if (hash !== model.sha256) throw new Error('Mô hình không đúng phiên bản. Vui lòng tải lại trang.');
  onProgress?.(`Đang khởi tạo ${model.name} trên thiết bị…`);
  ort.env.wasm.wasmPaths = new URL(`${import.meta.env.BASE_URL}runtime/`, window.location.origin).href;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = true;
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  active = { id: model.id, sha256: model.sha256, session };
  return { session, loadedNow: true, loadMs: performance.now() - start };
}

async function run(file, ids, { onProgress, onResult } = {}) {
  if (!file || !['image/jpeg', 'image/png', 'image/webp'].includes(file.type) || !file.size || file.size > 8 * 1024 * 1024) {
    throw new Error('Chọn ảnh JPG, PNG hoặc WEBP hợp lệ, dung lượng tối đa 8 MB.');
  }
  if (!ids.length || new Set(ids).size !== ids.length || ids.some(id => !MODEL_IDS.includes(id))) throw new Error('Chọn mô hình hợp lệ.');
  const catalog = await comparisonCatalog();
  onProgress?.('Đang chuẩn bị cùng một ảnh cho các mô hình…');
  const pixels = await readImagePixels(file);
  const digest = await crypto.subtle.digest('SHA-256', await file.arrayBuffer());
  const image_sha256 = Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
  const ort = await import('onnxruntime-web/wasm');
  const results = {};
  for (const id of ids) {
    const model = catalog.models.find(item => item.id === id);
    let input;
    let output;
    try {
      const { session, loadedNow, loadMs } = await sessionFor(model, onProgress);
      onProgress?.(`Đang phân tích bằng ${model.name}…`);
      // A worker can transfer/detach its input buffer; preserve the shared pixels for later models.
      input = new ort.Tensor('float32', new Float32Array(pixels), [1, 224, 224, 3]);
      const started = performance.now();
      output = await session.run({ [model.input_name]: input });
      const scores = Array.from(output[model.output_name].data);
      const inference_ms = performance.now() - started;
      const result = { ...predictionResult(scores), model_id: id, model_name: model.name,
        model_version: model.model_version, model_sha256: model.sha256, inference_ms,
        load_ms: loadMs, first_run_after_load: loadedNow, model_bytes: model.size_bytes };
      results[id] = result;
      onResult?.(id, result);
    } catch (error) {
      console.error(`TeaLeafAI ${model.name}:`, error);
      const message = /memory|allocat|WebAssembly/i.test(error.message)
        ? 'Thiết bị chưa đủ bộ nhớ để chạy mô hình này. Hãy đóng bớt tab hoặc thử trên máy tính.'
        : error instanceof TypeError ? 'Chưa tải được mô hình. Hãy kiểm tra mạng và thử lại.'
          : error.message || 'Thiết bị chưa chạy được mô hình này.';
      results[id] = { model_id: id, error: message };
      if (active?.id === id) { const previous = active; active = undefined; try { await previous.session.release(); } catch { /* allow a fresh retry */ } }
      onResult?.(id, results[id]);
    } finally {
      input?.dispose?.();
      if (output) Object.values(output).forEach(tensor => tensor.dispose?.());
    }
  }
  onProgress?.('Đã hoàn tất các mô hình đã chọn.');
  return { image_sha256, protocol_id: catalog.protocol_id, results };
}

export function compareImage(file, ids = MODEL_IDS, options) {
  // Serializes model creation/release even if an external caller submits overlapping runs.
  const task = queue.then(() => run(file, ids, options));
  queue = task.catch(() => {});
  return task;
}
