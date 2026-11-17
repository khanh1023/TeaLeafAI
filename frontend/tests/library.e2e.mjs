/** Library navigation + real inference care mapping. Optional Playwright dependency.
 * TEALEAF_BASE_URL, PLAYWRIGHT_MODULE, CHROMIUM_PATH supported.
 */
import assert from 'node:assert/strict';
import path from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {DISEASES} from '../src/diseases.js';
const moduleName=process.env.PLAYWRIGHT_MODULE;
const {chromium}=await import(moduleName && path.isAbsolute(moduleName) ? pathToFileURL(moduleName).href : moduleName || 'playwright');
const base=process.env.TEALEAF_BASE_URL || 'http://127.0.0.1:5173';
const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH ? {executablePath:process.env.CHROMIUM_PATH} : {})});
try {
 const page=await browser.newPage({viewport:{width:1280,height:900}});const errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto(base+'/thu-vien',{waitUntil:'networkidle'});
 assert.equal(await page.locator('.library-card-link').count(),8);
 assert.equal(await page.getByRole('heading',{name:'Hướng xử lý ban đầu',exact:true}).count(),0,'Summary cards do not show full detail.');
 for(const disease of DISEASES){
  await page.getByRole('link',{name:`Xem chi tiết ${disease.label}`,exact:true}).click();
  await page.getByRole('heading',{name:disease.label,exact:true}).waitFor();
  assert.equal(new URL(page.url()).pathname,`/thu-vien/${disease.key}`);
  assert.equal(await page.locator('.care-steps li').count(),disease.actions.length);
  assert.equal(await page.locator('.disease-sources li').count(),disease.sources.length);
  if(disease.key==='healthy')await page.getByText('Không cần dùng thuốc chỉ dựa trên kết quả “lá khỏe mạnh” của mô hình.',{exact:true}).waitFor();
  if(disease.identityNote)await page.getByText(disease.identityNote,{exact:true}).waitFor();
  await page.getByRole('link',{name:'Về thư viện 8 nhóm',exact:true}).click();
 }
 await page.getByLabel('Tìm nhóm phân loại').fill('healthy');assert.equal(await page.locator('.library-card-link').count(),1);
 await page.getByRole('link',{name:'Xem chi tiết Lá khỏe mạnh',exact:true}).focus();await page.keyboard.press('Enter');
 await page.getByRole('heading',{name:'Lá khỏe mạnh',exact:true}).waitFor();
 await page.reload();await page.getByRole('heading',{name:'Lá khỏe mạnh',exact:true}).waitFor();
 await page.goto(base+'/thu-vien/constructor');await page.getByRole('heading',{name:'Chưa có thông tin cho nhóm này',exact:true}).waitFor();
 for(const width of [320,390,768,1280]){
  await page.setViewportSize({width,height:900});
  for(const disease of DISEASES){
   await page.goto(`${base}/thu-vien/${disease.key}`,{waitUntil:'networkidle'});
   assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=document.documentElement.clientWidth+1),`No detail overflow: ${width}/${disease.key}`);
  }
 }
 await page.setViewportSize({width:390,height:844});
 await page.goto(base+'/chan-doan',{waitUntil:'networkidle'});
 for(const filename of ['healthy-leaf.jpg','sample-leaf.jpg']){
  await page.getByLabel('Chọn ảnh lá chè',{exact:true}).setInputFiles(fileURLToPath(new URL(`../../examples/${filename}`,import.meta.url)));
  assert.equal(await page.getByTestId('result-advice').count(),0,'Old advice clears when a new photo is selected.');
  await page.getByRole('button',{name:'Phân tích ảnh',exact:true}).click();
  await page.getByText('So sánh 8 nhóm phân loại',{exact:true}).waitFor({timeout:120000});
  const predicted=await page.evaluate(()=>JSON.parse(localStorage.getItem('teaLeafHistory'))[0].predicted_class);
  const advice=page.getByTestId('result-advice');await advice.waitFor();
  assert.equal(await advice.getAttribute('data-disease-key'),predicted);
  assert.ok(await advice.locator('.care-steps li').count()>0);
  if(predicted==='healthy')await advice.getByRole('heading',{name:'Chăm sóc và theo dõi lá khỏe',exact:true}).waitFor();
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=document.documentElement.clientWidth+1),'No advice overflow on mobile.');
 }
 const destination=await page.getByTestId('result-advice').getAttribute('data-disease-key');
 await page.getByTestId('result-advice').getByRole('link',{name:'Xem chi tiết, phòng ngừa và tài liệu'}).click();
 assert.equal(new URL(page.url()).pathname,`/thu-vien/${destination}`);
 assert.deepEqual(errors,[]);
 console.log('PASS: 8 concise cards, all detailed routes/care/source links, keyboard and refresh, unknown route, 320–1280px layouts, actual healthy/disease inference advice and clearing stale guidance.');
} finally {await browser.close();}
