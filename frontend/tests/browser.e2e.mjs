/**
 * Browser smoke test against a running TeaLeafAI server.
 * Setup: npm install --no-save playwright && npx playwright install chromium
 * Run: node tests/browser.e2e.mjs
 * Optional: TEALEAF_BASE_URL, TEALEAF_SAMPLE, PLAYWRIGHT_MODULE, CHROMIUM_PATH.
 */
import assert from 'node:assert/strict';
import { readdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../../', import.meta.url));
const base = process.env.TEALEAF_BASE_URL || 'http://127.0.0.1:5173';
const moduleName = process.env.PLAYWRIGHT_MODULE;
const { chromium } = await import(moduleName && path.isAbsolute(moduleName) ? pathToFileURL(moduleName).href : moduleName || 'playwright');
async function findImage(folder) {
  for (const item of await readdir(folder, { withFileTypes:true })) {
    const target = path.join(folder,item.name);
    if (item.isDirectory()) { const result = await findImage(target); if (result) return result; }
    else if (/\.(jpe?g|png|webp)$/i.test(item.name)) return target;
  }
}
const sample = process.env.TEALEAF_SAMPLE || (existsSync(path.join(root,'DuLieuGoc')) ? await findImage(path.join(root,'DuLieuGoc')) : path.join(root,'examples/sample-leaf.jpg'));
assert.ok(sample, 'Set TEALEAF_SAMPLE to an image when the dataset is absent.');
const browser = await chromium.launch({ headless:true, ...(process.env.CHROMIUM_PATH ? { executablePath:process.env.CHROMIUM_PATH } : {}) });
const page = await browser.newPage({ viewport:{ width:1365,height:1000 } });
const errors=[];
const requests=[];
page.on('pageerror', error => errors.push(error.message));
page.on('request', request => requests.push({ url:request.url(), method:request.method(), resource:request.resourceType() }));
page.on('console', entry => { if(entry.type()==='error') console.error('Browser console:',entry.text()); });
try {
  await page.goto(`${base.replace(/\/$/,'')}/chan-doan`, { waitUntil:'networkidle' });
  await page.getByLabel('Chọn ảnh lá chè', {exact:true}).setInputFiles(sample);
  const run = page.getByRole('button', { name:/Phân tích ảnh/ });
  await run.click();
  await page.getByText('So sánh 8 nhóm phân loại', {exact:true}).waitFor({timeout:120000});
  assert.equal(await page.locator('.prob').count(),8,'Show eight probabilities.');
  const firstResult = await page.locator('.result-main').innerText();
  const modelRequests = () => requests.filter(x=>/\.onnx(?:\?|$)/.test(x.url));
  assert.equal(modelRequests().length,1,'Load model once.');
  await run.click();
  await page.getByText('So sánh 8 nhóm phân loại', {exact:true}).waitFor({timeout:120000});
  assert.equal(await page.locator('.result-main').innerText(),firstResult,'Repeated inference is deterministic.');
  assert.equal(modelRequests().length,1,'Reuse model for second analysis.');
  assert.equal(await page.locator('.prob').count(),8);
  assert.deepEqual(errors,[],'No uncaught browser errors.');
  const nonLocal=requests.filter(x=>!x.url.startsWith(base)&&!x.url.startsWith('blob:')&&!x.url.startsWith('data:'));
  assert.deepEqual(nonLocal,[],'No external image upload or CDN dependency.');
  assert.equal(requests.filter(x=>x.method==='POST').length,0,'No image POST request in browser mode.');
  const downloadPromise=page.waitForEvent('download');
  await page.getByRole('button',{name:'Tải kết quả JSON'}).click();
  const download=await downloadPromise;
  assert.equal(download.suggestedFilename(),'tea-leaf-ai-ket-qua.json');
  const chunks=[];
  for await (const chunk of await download.createReadStream()) chunks.push(chunk);
  const exported=JSON.parse(Buffer.concat(chunks).toString('utf8'));
  assert.equal(Object.keys(exported.probabilities).length,8,'JSON contains eight model outputs.');
  assert.ok(exported.confidence>=0 && exported.confidence<=1);
  assert.equal(exported.engine,'browser');
  assert.equal(await page.evaluate(() => JSON.parse(localStorage.getItem('teaLeafHistory')).length),2,'Both runs are saved locally.');
  if(process.env.TEALEAF_SCREENSHOT) await page.screenshot({path:process.env.TEALEAF_SCREENSHOT,fullPage:true});
  for(const width of [1280,390,375]) {
    await page.setViewportSize({width,height:844});
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth<=window.innerWidth),`Diagnosis at ${width}px has no horizontal overflow.`);
  }
  await page.getByRole('button',{name:'Mở menu',exact:true}).click();
  await page.getByRole('link',{name:'Lịch sử',exact:true}).click();
  await page.getByRole('heading',{name:'Lịch sử phân tích',exact:true}).waitFor();
  assert.equal(await page.locator('tbody tr').count(),2,'Mobile history shows both results.');
  for(const width of [375,390,1280]) {
    await page.setViewportSize({width,height:844});
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth<=window.innerWidth),`History at ${width}px has no horizontal overflow.`);
  }
  const csvPromise=page.waitForEvent('download');
  await page.getByRole('button',{name:'Xuất CSV',exact:true}).click();
  const csvDownload=await csvPromise;
  assert.equal(csvDownload.suggestedFilename(),'tea-leaf-ai-lich-su.csv');
  const csvChunks=[];
  for await (const chunk of await csvDownload.createReadStream()) csvChunks.push(chunk);
  const csv=Buffer.concat(csvChunks).toString('utf8');
  assert.ok(csv.includes(path.basename(sample)),'CSV contains the saved sample filename.');
  await page.evaluate(() => localStorage.setItem('teaLeafHistory','{invalid-json'));
  await page.reload();
  await page.getByRole('heading',{name:'Chưa có lần phân tích nào',exact:true}).waitFor();
  assert.deepEqual(errors,[],'Corrupt history does not crash the page.');
  console.log(JSON.stringify({status:'PASS',sample,result:firstResult,modelDownloads:modelRequests().length,imageUploads:0,checks:['browser ONNX/WASM inference','eight probabilities','repeat inference/session reuse','JSON and CSV export','no external requests','no uncaught browser errors','mobile navigation/layout','history corruption recovery']},null,2));
} finally {
  await browser.close();
}
