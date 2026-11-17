/** Regression for a valid portrait JPEG when ImageBitmap fails.
 * Requires Playwright + Chromium and a running website.
 * Optional TEALEAF_BASE_URL, PLAYWRIGHT_MODULE, CHROMIUM_PATH.
 */
import assert from 'node:assert/strict';
import path from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
const moduleName=process.env.PLAYWRIGHT_MODULE;
const {chromium}=await import(moduleName && path.isAbsolute(moduleName) ? pathToFileURL(moduleName).href : moduleName || 'playwright');
const base=process.env.TEALEAF_BASE_URL || 'http://127.0.0.1:5173';
const sample=fileURLToPath(new URL('../../examples/healthy-leaf.jpg',import.meta.url));
const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH ? {executablePath:process.env.CHROMIUM_PATH} : {})});
try {
 for(const mode of ['bitmap-rejects','bitmap-unavailable','canvas-blocked']){
  const context=await browser.newContext({viewport:{width:390,height:844}});
  const page=await context.newPage();const posts=[];const modelRequests=[];const errors=[];
  page.on('request',r=>{if(r.method()==='POST')posts.push(r.url());if(/\.onnx(?:\?|$)/.test(r.url()))modelRequests.push(r.url());});
  page.on('pageerror',e=>errors.push(e.message));
  await page.addInitScript(mode=>{
   if(mode==='bitmap-rejects')window.createImageBitmap=async()=>{throw new DOMException('Decoder unavailable','InvalidStateError');};
   if(mode==='bitmap-unavailable')window.createImageBitmap=undefined;
   if(mode==='canvas-blocked')HTMLCanvasElement.prototype.getContext=()=>null;
  },mode);
  await page.goto(base+'/chan-doan',{waitUntil:'networkidle'});
  await page.getByLabel('Chọn ảnh lá chè',{exact:true}).setInputFiles(sample);
  await page.waitForFunction(()=>document.querySelector('.preview img')?.naturalWidth>0);
  assert.equal(await page.locator('.preview img').evaluate(e=>e.naturalWidth),768);
  await page.getByRole('button',{name:'Phân tích ảnh',exact:true}).click();
  if(mode==='canvas-blocked'){
   const text=await page.getByRole('alert').innerText();assert.match(text,/vùng xử lý ảnh/);assert.doesNotMatch(text,/ảnh.*hợp lệ/);
   assert.equal(modelRequests.length,0,'Do not load the model if preprocessing is blocked.');
  }else{
   await page.getByText('So sánh 8 nhóm phân loại',{exact:true}).waitFor({timeout:120000});
   assert.equal(await page.locator('.prob').count(),8);
   assert.equal(await page.getByRole('alert').count(),0);
   const geometry=await page.locator('.preview img').evaluate(img=>({image:img.getBoundingClientRect().height,frame:img.parentElement.getBoundingClientRect().height,page:document.documentElement.scrollWidth,viewport:document.documentElement.clientWidth}));
   assert.ok(geometry.image<=geometry.frame+1,'Portrait preview stays inside frame.');
   assert.ok(geometry.page<=geometry.viewport+1,'Long portrait upload stays inside mobile layout.');
   assert.equal(modelRequests.length,1);
  }
  assert.deepEqual(posts,[]);assert.deepEqual(errors,[]);
  console.log(`PASS: ${mode} — valid portrait JPEG, honest error reporting, no external uploads.`);
  await context.close();
 }
}finally{await browser.close();}
