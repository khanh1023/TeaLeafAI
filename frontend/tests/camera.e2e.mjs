/** Camera QA uses Chromium's fake camera, not a physical camera.
 * Setup: npm install --no-save playwright && npx playwright install chromium
 * Run against a running website: node tests/camera.e2e.mjs
 * Optional TEALEAF_BASE_URL, PLAYWRIGHT_MODULE and CHROMIUM_PATH.
 */
import assert from 'node:assert/strict';
import path from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
const moduleName=process.env.PLAYWRIGHT_MODULE;
const {chromium}=await import(moduleName && path.isAbsolute(moduleName) ? pathToFileURL(moduleName).href : moduleName || 'playwright');
const base=process.env.TEALEAF_BASE_URL || 'http://127.0.0.1:5173';
const browser=await chromium.launch({headless:true,args:['--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream'],...(process.env.CHROMIUM_PATH ? {executablePath:process.env.CHROMIUM_PATH} : {})});
try {
 const page=await browser.newPage({viewport:{width:390,height:844}});
 const errors=[];const posts=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(r.method()==='POST')posts.push(r.url());});
 await page.addInitScript(()=>{
  window.__cameraCalls=[];window.__cameraStreams=[];
  const original=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia=async constraints=>{
   window.__cameraCalls.push(constraints);
   if(window.__denyCamera)throw new DOMException('Permission denied','NotAllowedError');
   const stream=await original(constraints);window.__cameraStreams.push(stream);return stream;
  };
 });
 await page.goto(base+'/chan-doan',{waitUntil:'networkidle'});
 assert.equal(await page.evaluate(()=>window.__cameraCalls.length),0,'Camera stays off on page load.');
 assert.equal(await page.getByLabel('Chụp ảnh bằng camera điện thoại',{exact:true}).getAttribute('capture'),'environment');
 await page.getByRole('button',{name:'Chụp bằng camera',exact:true}).click();
 await page.waitForFunction(()=>document.querySelector('video')?.videoWidth>0);
 await page.getByRole('button',{name:'Chụp ảnh',exact:true}).waitFor();
 await page.waitForFunction(()=>!Array.from(document.querySelectorAll('button')).find(b=>b.textContent==='Chụp ảnh')?.disabled);
 assert.equal(await page.evaluate(()=>window.__cameraCalls[0].audio),false,'No microphone requested.');
 assert.equal(await page.evaluate(()=>window.__cameraCalls[0].video.facingMode.ideal),'environment','Rear camera preferred.');
 await page.getByRole('button',{name:'Đổi camera',exact:true}).click();
 await page.waitForFunction(()=>window.__cameraStreams.length>=2 && document.querySelector('video')?.videoWidth>0);
 assert.equal(await page.evaluate(()=>window.__cameraStreams[0].getTracks().every(t=>t.readyState==='ended')),true,'Old camera stops on switch.');
 await page.getByRole('button',{name:'Chụp ảnh',exact:true}).click();
 await page.getByRole('dialog').waitFor({state:'detached'});
 const caption=await page.locator('.file-caption').innerText();assert.match(caption,/la-che-.*\.jpg/);
 assert.equal(await page.evaluate(()=>window.__cameraStreams.every(s=>s.getTracks().every(t=>t.readyState==='ended'))),true,'Camera stops after capture.');
 assert.ok(await page.getByRole('button',{name:'Phân tích ảnh',exact:true}).isEnabled());
 await page.getByRole('button',{name:'Phân tích ảnh',exact:true}).click();
 await page.getByText('So sánh 8 nhóm phân loại',{exact:true}).waitFor({timeout:120000});
 assert.equal(await page.locator('.prob').count(),8,'Captured JPEG flows through actual browser inference.');
 await page.getByRole('button',{name:'Chụp bằng camera',exact:true}).click();
 await page.waitForFunction(()=>document.querySelector('video')?.videoWidth>0);
 await page.keyboard.press('Escape');
 await page.getByRole('dialog').waitFor({state:'detached'});
 assert.equal(await page.evaluate(()=>window.__cameraStreams.every(s=>s.getTracks().every(t=>t.readyState==='ended'))),true,'Escape closes and stops camera.');
 await page.evaluate(()=>window.__denyCamera=true);
 await page.getByRole('button',{name:'Chụp bằng camera',exact:true}).click();
 await page.getByRole('alert').filter({hasText:'Bạn chưa cho phép camera'}).waitFor();
 assert.ok(await page.getByRole('button',{name:'Dùng camera của điện thoại',exact:true}).isEnabled(),'Native fallback remains available.');
 const nativeChooser=page.waitForEvent('filechooser');
 await page.getByRole('button',{name:'Dùng camera của điện thoại',exact:true}).click();
 const chooser=await nativeChooser;
 await chooser.setFiles(fileURLToPath(new URL('../../examples/sample-leaf.jpg',import.meta.url)));
 await page.getByRole('dialog').waitFor({state:'detached'});
 assert.match(await page.locator('.file-caption').innerText(),/sample-leaf.jpg/,'Native fallback feeds the selected image into preview.');
 assert.deepEqual(errors,[]);assert.deepEqual(posts,[]);
 for(const width of [320,390,768,1280]){
  await page.setViewportSize({width,height:900});
  for(const route of ['/','/chan-doan','/thu-vien','/lich-su','/huong-dan','/quan-tri']){
   await page.goto(base+route,{waitUntil:'networkidle'});
   const metrics=await page.evaluate(()=>({width:document.documentElement.clientWidth,scroll:document.documentElement.scrollWidth,fonts:Array.from(document.querySelectorAll('h1,h2,h3')).map(e=>({family:getComputedStyle(e).fontFamily,style:getComputedStyle(e).fontStyle}))}));
   assert.ok(metrics.scroll<=metrics.width+1,JSON.stringify({route,...metrics}));
   assert.ok(metrics.fonts.every(f=>f.family.startsWith('Arial')&&f.style==='normal'),JSON.stringify({route,...metrics}));
  }
 }
 console.log('PASS: camera off until opened, rear preference/no audio, switch and cleanup, JPEG capture + real 8-class inference, Escape, permission denial fallback, all-page Vietnamese fonts and 320–1280px layout.');
} finally {await browser.close();}
