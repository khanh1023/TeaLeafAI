/** Integration QA of three real browser models, shared-cohort metrics and DOCX export. */
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { readFile, mkdir, writeFile } from 'node:fs/promises';

const root = fileURLToPath(new URL('../../', import.meta.url));
const output = process.env.TEALEAF_COMPARISON_QA || '/workspace/work/model-comparison-20261125/browser-qa';
const base = process.env.TEALEAF_BASE_URL || 'http://127.0.0.1:5184';
const module = process.env.PLAYWRIGHT_MODULE || '/opt/codex/runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const { chromium } = await import(pathToFileURL(module).href);
await mkdir(output, { recursive: true });
const manifest = JSON.parse(await readFile(path.join(root, 'training/reports/expanded-dataset-manifest.json')));
const record = manifest.find(item => item.split === 'test' && item.label === 'healthy');
const photo = path.join(root, 'training/dataset', record.path);
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH || '/usr/bin/chromium',
  // Optional, test-only pin for environments whose HTTPS proxy CA is absent from Chromium.
  args: ['--no-sandbox', '--disable-dev-shm-usage', ...(process.env.TEALEAF_QA_TRUSTED_SPKI
    ? ['--ignore-certificate-errors-spki-list=' + process.env.TEALEAF_QA_TRUSTED_SPKI] : [])] });
const errors = [], requests = [];
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 }, acceptDownloads: true });
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => requests.push({ url: request.url(), method: request.method() }));
  await page.goto(base + '/so-sanh', { waitUntil: 'networkidle' });
  await page.getByRole('heading', { name: 'Thử ba mô hình phân loại' }).waitFor();
  await page.locator('input[aria-label="Chọn ảnh đối chiếu ba mô hình"]').setInputFiles(photo);
  await page.locator('#actual-label').selectOption('healthy');
  await page.getByRole('button', { name: 'Chạy cả ba mô hình', exact: true }).click();
  await page.waitForFunction(() => document.querySelectorAll('.comparison-prediction').length === 3, undefined, { timeout: 240000 });
  await page.getByText('Đã lưu kết quả trên thiết bị.', { exact: false }).waitFor();
  const rows = await page.evaluate(() => JSON.parse(localStorage.getItem('teaLeafComparisonTrialsV1')));
  assert.equal(rows.length, 1);
  assert.equal(rows[0].actual_label, 'healthy');
  assert.equal(Object.keys(rows[0].results).length, 3);
  for (const [id, result] of Object.entries(rows[0].results)) {
    assert.equal(result.model_id, id);
    assert.ok(result.inference_ms > 0);
    assert.equal(Object.keys(result.probabilities).length, 8);
    assert.ok(Math.abs(Object.values(result.probabilities).reduce((a, b) => a + b, 0) - 1) < .001);
  }
  // Ground-truth corrections update stored cohort metrics without another inference.
  await page.locator('#actual-label').selectOption('white_spot');
  assert.equal(await page.evaluate(() => JSON.parse(localStorage.getItem('teaLeafComparisonTrialsV1'))[0].actual_label), 'white_spot');
  await page.locator('#actual-label').selectOption('healthy');

  // Each individual model remains usable; exact-image history stays deduplicated.
  for (const name of ['MobileNetV2', 'ResNet50', 'VGG16']) {
    await page.getByRole('button', { name: 'Chạy ' + name, exact: true }).click();
    await page.getByRole('button', { name: 'Chạy cả ba mô hình', exact: true }).waitFor({ state: 'visible' });
    await page.waitForFunction(() => !document.querySelector('.comparison-controls .primary').disabled, undefined, { timeout: 180000 });
    assert.equal(await page.evaluate(() => JSON.parse(localStorage.getItem('teaLeafComparisonTrialsV1')).length), 1);
  }
  const wordPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Tải Word', exact: true }).click();
  const word = await wordPromise;
  assert.ok(word.suggestedFilename().endsWith('.docx'));
  await word.saveAs(path.join(output, 'website-export.docx'));
  const csvPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Tải CSV', exact: true }).click();
  const csv = await csvPromise;
  await csv.saveAs(path.join(output, 'website-export.csv'));
  await page.evaluate(() => { document.activeElement?.blur(); window.scrollTo(0, 0); });
  await page.screenshot({ path: path.join(output, 'three-models-desktop.png'), fullPage: true });
  const saved = await page.evaluate(() => JSON.parse(localStorage.getItem('teaLeafComparisonTrialsV1')));
  const catalog = await page.evaluate(async url => (await fetch(url, { cache: 'no-cache' })).json(), base + '/model/comparison/manifest.json');
  const data = { sample_id: record.id, actual: record.label, filename: path.basename(photo), browser: browser.version(),
    viewport: { width: 1440, height: 1100 }, catalog, trials: saved, uncaught_errors: errors,
    image_uploads: requests.filter(request => request.method === 'POST').length };
  assert.deepEqual(errors, []);
  assert.equal(data.image_uploads, 0);
  await writeFile(path.join(output, 'browser-qa.json'), JSON.stringify(data, null, 2));
  await page.close();

  const mobile = await browser.newPage({ viewport: { width: 390, height: 844 } });
  await mobile.goto(base + '/so-sanh', { waitUntil: 'networkidle' });
  await mobile.locator('input[aria-label="Chọn ảnh đối chiếu ba mô hình"]').setInputFiles(photo);
  for (const width of [390, 320]) {
    await mobile.setViewportSize({ width, height: 844 });
    assert.ok(await mobile.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'No page overflow at ' + width);
    for (const name of ['MobileNetV2', 'ResNet50', 'VGG16']) await mobile.getByRole('button', { name: 'Chạy ' + name, exact: true }).waitFor();
  }
  await mobile.setViewportSize({ width: 390, height: 844 });
  await mobile.screenshot({ path: path.join(output, 'three-models-mobile.png'), fullPage: true });
  await mobile.close();
  console.log(JSON.stringify({ output, tested: 'Three real models, individual reruns, deduplication, corrected labels, Word/CSV,390/320px layout,zero image uploads',
    predictions: Object.fromEntries(Object.entries(saved[0].results).map(([id, r]) => [id, { class: r.predicted_class, score: r.confidence }])) }, null, 2));
} finally { await browser.close(); }
