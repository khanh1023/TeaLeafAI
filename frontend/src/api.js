import { CLASSES } from './classes.js';
import { predictionResult } from './inference-utils.js';
import { readImagePixels } from './image-preprocessing.js';

export const ENGINE = import.meta.env.VITE_INFERENCE_MODE === 'api' ? 'api' : 'browser';
export const BASE = ENGINE === 'api'
  ? (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '')
  : 'Trình duyệt · xử lý ngay trên thiết bị';
const asset = path => `${import.meta.env.BASE_URL}${path}`;
let sessionPromise;
let loadedSession;
let metadataPromise;

async function metadata() {
  if (!metadataPromise) metadataPromise = fetch(asset('model/manifest.json'), {cache:'no-cache'})
    .then(async response => {
      if (!response.ok) throw new Error('Không tải được thông tin mô hình. Kiểm tra kết nối Internet.');
      const data = await response.json();
      if (JSON.stringify(data.classes) !== JSON.stringify(CLASSES.map(x => x.key))) throw new Error('Danh sách lớp của mô hình không khớp ứng dụng.');
      return data;
    }).catch(error => { metadataPromise = undefined; throw error; });
  return metadataPromise;
}

export async function health() {
  if (ENGINE === 'api') {
    const response = await fetch(`${BASE}/api/health`, {signal:AbortSignal.timeout(20000)});
    if (!response.ok) throw new Error('Không kết nối được máy chủ AI.');
    return {...await response.json(), engine:'api'};
  }
  const manifest = await metadata();
  return {
    status:'ok', engine:'browser', model_loaded:Boolean(loadedSession),
    model_file_exists:true, classes:manifest.classes, model_bytes:manifest.size_bytes
  };
}

async function loadSession(onProgress) {
  if (loadedSession) return loadedSession;
  if (!sessionPromise) sessionPromise = (async () => {
    const manifest = await metadata();
    onProgress?.({message:'Đang tải mô hình AI về thiết bị (khoảng 9 MB)…',percent:10});
    const [ort, response] = await Promise.all([
      import('onnxruntime-web/wasm'),
      fetch(asset(`model/${manifest.file}?v=${manifest.sha256.slice(0,12)}`))
    ]);
    if (!response.ok) throw new Error('Không tải được file mô hình. Kiểm tra Internet và thử lại.');
    const buffer = await response.arrayBuffer();
    if (buffer.byteLength !== manifest.size_bytes) throw new Error('File mô hình tải chưa đầy đủ. Vui lòng thử lại.');
    if (globalThis.crypto?.subtle) {
      const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',buffer)), b => b.toString(16).padStart(2,'0')).join('');
      if (hash !== manifest.sha256) throw new Error('File mô hình không đúng phiên bản. Hãy tải lại trang.');
    }
    onProgress?.({message:'Đang khởi tạo bộ phân tích trên thiết bị…',percent:45});
    ort.env.wasm.wasmPaths = new URL(asset('runtime/'), window.location.origin).href;
    ort.env.wasm.numThreads = 1; // Works on free hosts without cross-origin isolation.
    ort.env.wasm.proxy = true; // Keep inference off the UI thread.
    loadedSession = await ort.InferenceSession.create(buffer, {executionProviders:['wasm'],graphOptimizationLevel:'all'});
    return loadedSession;
  })().catch(error => { sessionPromise = undefined; loadedSession = undefined; throw error; });
  return sessionPromise;
}

export async function diagnose(file, {onProgress} = {}) {
  if (!file || !['image/jpeg','image/png','image/webp'].includes(file.type)) throw new Error('Chỉ hỗ trợ ảnh JPG, PNG hoặc WEBP.');
  if (file.size > 8 * 1024 * 1024) throw new Error('Ảnh không được vượt quá 8 MB.');
  if (ENGINE === 'api') {
    onProgress?.({message:'Đang gửi ảnh đến máy chủ AI đã cấu hình…',percent:35});
    const form = new FormData();
    form.append('file',file);
    const response = await fetch(`${BASE}/api/diagnosis`, {method:'POST',body:form,signal:AbortSignal.timeout(120000)});
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'Máy chủ chưa phân tích được ảnh.');
    onProgress?.({message:'Hoàn tất',percent:100});
    return {...result, engine:'api'};
  }
  try {
    onProgress?.({message:'Đang đọc và chuẩn bị ảnh…',percent:5});
    const data = await readImagePixels(file);
    const [session, ort] = await Promise.all([loadSession(onProgress),import('onnxruntime-web/wasm')]);
    onProgress?.({message:'Đang phân tích ảnh trên thiết bị…',percent:75});
    const output = await session.run({images:new ort.Tensor('float32',data,[1,224,224,3])});
    const result = predictionResult(output.probabilities.data);
    onProgress?.({message:'Hoàn tất',percent:100});
    return result;
  } catch (error) {
    console.error('TeaLeafAI inference:', error);
    if (error instanceof TypeError || /wasm|backend|memory|WebAssembly/i.test(error.message)) {
      throw new Error('Không khởi tạo được AI trên thiết bị này. Hãy thử Chrome/Edge/Firefox mới nhất, kiểm tra mạng và đóng bớt tab.');
    }
    throw error;
  }
}
