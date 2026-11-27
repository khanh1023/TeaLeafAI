/** Real UI inference QA for a versioned default model and three-model catalog.
 * Uses the actual built or live app; never mocks predictions or claims accuracy
 * from its one-image smoke sample. Compatible with Vite dev and production.
 * TEALEAF_BASE_URL, TEALEAF_DATASET_DIR, TEALEAF_QA_OUTPUT, PLAYWRIGHT_MODULE,
 * CHROMIUM_PATH, TEALEAF_QA_TRUSTED_SPKI and TEALEAF_QA_COMPARISON=0 supported.
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = fileURLToPath(new URL('../../', import.meta.url));
const data = path.resolve(process.env.TEALEAF_DATASET_DIR || path.join(root, 'training/dataset-v2'));
const output = path.resolve(process.env.TEALEAF_QA_OUTPUT || `/workspace/work/v2-training-20261126/qa-${Date.now()}`);
const base = (process.env.TEALEAF_BASE_URL || 'http://127.0.0.1:5184').replace(/\/$/, '');
const runComparison = process.env.TEALEAF_QA_COMPARISON !== '0';
const pw = process.env.PLAYWRIGHT_MODULE || '/opt/codex/runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const { chromium } = await import(path.isAbsolute(pw) ? pathToFileURL(pw).href : pw);
const manifestBytes = await readFile(path.join(data, 'manifest.json'));
const datasetSha = createHash('sha256').update(manifestBytes).digest('hex');
const records = JSON.parse(manifestBytes);
const expectedModel = JSON.parse(await readFile(path.join(root, 'frontend/public/model/manifest.json'), 'utf8'));
const expectedCatalog = JSON.parse(await readFile(path.join(root, 'frontend/public/model/comparison/manifest.json'), 'utf8'));
assert.equal(expectedModel.dataset_manifest_sha256, datasetSha, 'Default model must use the requested dataset.');
if (runComparison) assert.equal(expectedCatalog.dataset_manifest_sha256, datasetSha, 'Comparison models must use the requested dataset.');
const sample = records.find(record => record.split === 'test' && record.label === 'healthy');
assert.ok(sample, 'A frozen healthy test sample exists.');
const photo = path.join(data, sample.path);
assert.equal(createHash('sha256').update(await readFile(photo)).digest('hex'), sample.sha256);
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true,
  executablePath: process.env.CHROMIUM_PATH || '/usr/bin/chromium',
  args: ['--no-sandbox', '--disable-dev-shm-usage', '--disk-cache-size=1', ...(process.env.TEALEAF_QA_TRUSTED_SPKI
    ? ['--ignore-certificate-errors-spki-list=' + process.env.TEALEAF_QA_TRUSTED_SPKI] : [])] });
const errors = [], requests = [], failedResponses = [];
const report = { base_url: base, dataset_manifest_sha256: datasetSha,
  sample: { id: sample.id, label: sample.label, path: sample.path },
  scope: 'One-image functional UI smoke test; not a test-set accuracy measurement.' };

function validProbabilities(result) {
  assert.equal(Object.keys(result.probabilities).length, expectedModel.classes.length);
  const values = Object.values(result.probabilities);
  assert.ok(values.every(value => Number.isFinite(value) && value >= 0 && value <= 1));
  assert.ok(Math.abs(values.reduce((a, b) => a + b, 0) - 1) < .001);
  assert.ok(expectedModel.classes.includes(result.predicted_class));
}

try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 }, acceptDownloads: true });
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => requests.push({ url: request.url(), method: request.method() }));
  page.on('response', response => { if (response.status() >= 400) failedResponses.push({ url: response.url(), status: response.status() }); });
  await page.goto(base + '/', { waitUntil: 'networkidle' });
  await page.locator('main h1').waitFor();
  const actualModel = await page.evaluate(async () => (await fetch('/model/manifest.json', { cache: 'no-cache' })).json());
  assert.equal(actualModel.sha256, expectedModel.sha256, 'Deployment serves the new default model manifest.');
  assert.equal(actualModel.dataset_manifest_sha256, datasetSha);
  await page.goto(base + '/thu-vien', { waitUntil: 'networkidle' });
  assert.equal(await page.locator('.library-card-link').count(), 8);
  await page.locator('.library-card-link[href="/thu-vien/algal_leaf"]').click();
  await page.locator('.disease-page[data-disease-key="algal_leaf"]').waitFor();
  await page.goto(base + '/chan-doan', { waitUntil: 'networkidle' });
  await page.getByRole('heading', { name: 'Phân tích ảnh lá chè', exact: true }).waitFor();
  await page.getByRole('button', { name: 'Chụp bằng camera', exact: true }).waitFor();
  await page.locator('input[aria-label="Chọn ảnh lá chè"]').setInputFiles(photo);
  await page.getByRole('button', { name: 'Phân tích ảnh', exact: true }).click();
  await page.locator('.result-main').waitFor({ timeout: 180000 });
  const history = await page.evaluate(() => JSON.parse(localStorage.getItem('teaLeafHistory')));
  assert.equal(history.length, 1);
  assert.equal(history[0].model_version, expectedModel.model_version);
  validProbabilities(history[0]);
  const advice = page.locator('[data-testid="result-advice"]');
  assert.equal(await advice.getAttribute('data-disease-key'), history[0].predicted_class);
  assert.equal(await page.locator('.result-panel .prob').count(), 8);
  const jsonPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Tải kết quả JSON', exact: true }).click();
  const jsonFile = path.join(output, 'default-result.json');
  await (await jsonPromise).saveAs(jsonFile);
  assert.equal(JSON.parse(await readFile(jsonFile, 'utf8')).model_version, expectedModel.model_version);
  await page.screenshot({ path: path.join(output, 'default-desktop.png'), fullPage: true });
  report.default = history[0];

  // Releasing the default page context keeps memory bounded before large networks.
  await page.close();
  const comparison = await browser.newPage({ viewport: { width: 1440, height: 1100 }, acceptDownloads: true });
  comparison.on('pageerror', error => errors.push(error.message));
  comparison.on('request', request => requests.push({ url: request.url(), method: request.method() }));
  await comparison.goto(base + '/so-sanh', { waitUntil: 'networkidle' });
  await comparison.getByRole('heading', { name: 'Thử ba mô hình phân loại', exact: true }).waitFor();
  if (runComparison) {
    const actualCatalog = await comparison.evaluate(async () => (await fetch('/model/comparison/manifest.json', { cache: 'no-cache' })).json());
    assert.equal(actualCatalog.protocol_id, expectedCatalog.protocol_id);
    assert.equal(actualCatalog.dataset_manifest_sha256, datasetSha);
    await comparison.locator('input[aria-label="Chọn ảnh đối chiếu ba mô hình"]').setInputFiles(photo);
    await comparison.locator('#actual-label').selectOption(sample.label);
    await comparison.getByRole('button', { name: 'Chạy cả ba mô hình', exact: true }).click();
    await comparison.waitForFunction(() => document.querySelectorAll('.comparison-prediction').length === 3, undefined, { timeout: 300000 });
    await comparison.getByText('Đã lưu kết quả trên thiết bị.', { exact: false }).waitFor();
    const trials = await comparison.evaluate(() => JSON.parse(localStorage.getItem('teaLeafComparisonTrialsV1')));
    assert.equal(trials.length, 1);
    assert.equal(trials[0].protocol_id, expectedCatalog.protocol_id);
    assert.equal(trials[0].actual_label, sample.label);
    assert.equal(Object.keys(trials[0].results).length, 3);
    for (const model of expectedCatalog.models) {
      const result = trials[0].results[model.id];
      assert.equal(result.model_sha256, model.sha256);
      assert.equal(result.model_version, model.model_version);
      assert.ok(result.inference_ms > 0);
      validProbabilities(result);
    }
    const csvPromise = comparison.waitForEvent('download');
    await comparison.getByRole('button', { name: 'Tải CSV', exact: true }).click();
    const csvPath = path.join(output, 'comparison.csv');
    await (await csvPromise).saveAs(csvPath);
    const csv = await readFile(csvPath, 'utf8');
    for (const model of expectedCatalog.models) assert.ok(csv.includes(model.model_version));
    const wordPromise = comparison.waitForEvent('download');
    await comparison.getByRole('button', { name: 'Tải Word', exact: true }).click();
    const wordPath = path.join(output, 'comparison.docx');
    await (await wordPromise).saveAs(wordPath);
    const wordBytes = await readFile(wordPath);
    assert.equal(wordBytes.subarray(0, 2).toString(), 'PK');
    assert.ok(wordBytes.length > 2000);
    await comparison.screenshot({ path: path.join(output, 'comparison-desktop.png'), fullPage: true });
    report.comparison = trials[0];
  }
  for (const width of [390, 320]) {
    await comparison.setViewportSize({ width, height: 844 });
    assert.ok(await comparison.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `No horizontal overflow at ${width}px.`);
  }
  await comparison.screenshot({ path: path.join(output, 'comparison-mobile.png'), fullPage: true });
  await comparison.close();
  assert.deepEqual(errors, []);
  assert.equal(requests.filter(request => request.method === 'POST').length, 0, 'No image uploads.');
  assert.deepEqual(failedResponses, []);
  report.browser = browser.version();
  report.tested_at = new Date().toISOString();
  report.uncaught_errors = errors;
  report.image_uploads = 0;
  report.passed = true;
  await writeFile(path.join(output, 'qa.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ output, passed: true, default_model: expectedModel.model_version,
    comparison_protocol: runComparison ? expectedCatalog.protocol_id : 'not evaluated', dataset_manifest_sha256: datasetSha }, null, 2));
} finally { await browser.close(); }
