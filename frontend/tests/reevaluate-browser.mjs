/** Evaluate the real browser diagnose() pipeline on the unchanged held-out test files.
 * Run a Vite server first. Requires Playwright + Chromium (not part of lightweight CI).
 * TEALEAF_BASE_URL, PLAYWRIGHT_MODULE, CHROMIUM_PATH supported.
 */
import assert from 'node:assert/strict';
import {readFile, mkdir, writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';

const root = fileURLToPath(new URL('../../', import.meta.url));
const output = path.join(root, 'training/reports/reevaluation');
const base = process.env.TEALEAF_BASE_URL || 'http://127.0.0.1:5179';
const moduleName = process.env.PLAYWRIGHT_MODULE || 'playwright';
const {chromium} = await import(path.isAbsolute(moduleName) ? pathToFileURL(moduleName).href : moduleName);
const manifestBytes = await readFile(path.join(root, 'training/reports/expanded-dataset-manifest.json'));
const manifest = JSON.parse(manifestBytes);
const model = JSON.parse(await readFile(path.join(root, 'frontend/public/model/manifest.json'), 'utf8'));
assert.equal(createHash('sha256').update(manifestBytes).digest('hex'), model.dataset_manifest_sha256);
const entries = manifest.filter(r => r.split === 'test');
const browser = await chromium.launch({headless:true, ...(process.env.CHROMIUM_PATH ? {executablePath:process.env.CHROMIUM_PATH} : {})});
const predictions = [], errors = [], requests = [];
const started = Date.now();
try {
  const page = await browser.newPage({viewport:{width:1280,height:900}});
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => requests.push({url:request.url(),method:request.method()}));
  await page.goto(base+'/chan-doan', {waitUntil:'networkidle'});
  const engine = await page.evaluate(async () => (await import('/src/api.js')).ENGINE);
  assert.equal(engine, 'browser');
  for (const [index, record] of entries.entries()) {
    const bytes = await readFile(path.join(root, 'training/dataset', record.path));
    assert.equal(createHash('sha256').update(bytes).digest('hex'), record.sha256);
    const result = await page.evaluate(async ({base64,name}) => {
      const {diagnose} = await import('/src/api.js');
      const bytes = Uint8Array.from(atob(base64), char => char.charCodeAt(0));
      return diagnose(new File([bytes], name, {type:'image/jpeg'}));
    }, {base64:bytes.toString('base64'),name:path.basename(record.path)});
    assert.equal(result.model_version, model.model_version);
    assert.equal(result.engine, 'browser');
    assert.equal(Object.keys(result.probabilities).length, model.classes.length);
    predictions.push({id:record.id, path:record.path, source:record.source, actual:record.label,
                      predicted:result.predicted_class, probabilities:result.probabilities,
                      correct:record.label===result.predicted_class});
    if ((index+1)%25===0 || index+1===entries.length) {
      console.log(`Browser ${index+1}/${entries.length}; correct so far: ${predictions.filter(r=>r.correct).length}`, new Date().toISOString());
    }
  }
  assert.deepEqual(errors, []);
  assert.equal(requests.filter(r => r.method==='POST').length, 0);
  const external = requests.filter(r => !r.url.startsWith(base+'/') && !r.url.startsWith('blob:') && !r.url.startsWith('data:'));
  assert.deepEqual(external, []);
  const modelDownloads = requests.filter(r => /\.onnx(?:\?|$)/.test(r.url)).length;
  assert.equal(modelDownloads, 1);
  const result = {evaluated_at:new Date().toISOString(), engine:'browser',
    model_version:model.model_version, onnx_sha256:model.sha256,
    dataset_manifest_sha256:model.dataset_manifest_sha256,
    preprocessing:'Actual frontend readImagePixels(): createImageBitmap -> Canvas drawImage 224x224 -> raw RGB 0–255 -> ONNX Runtime Web WASM.',
    user_agent:await page.evaluate(()=>navigator.userAgent),
    browser_version:browser.version(), model_downloads:modelDownloads, image_uploads:0,
    elapsed_seconds:(Date.now()-started)/1000, predictions};
  await mkdir(output, {recursive:true});
  await writeFile(path.join(output, 'browser-predictions.json'), JSON.stringify(result,null,2));
  const correct = predictions.filter(r=>r.correct).length;
  console.log(JSON.stringify({test_images:entries.length, correct, accuracy:correct/entries.length,elapsed_seconds:result.elapsed_seconds},null,2));
} finally {
  await browser.close();
}
