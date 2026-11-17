import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';

test('portable server serves assets, WASM MIME and React routes after extracting a ZIP', {timeout:10000}, async t => {
  const root = await mkdtemp(path.join(tmpdir(),'tealeaf-server-'));
  await mkdir(path.join(root,'frontend/scripts'),{recursive:true});
  await mkdir(path.join(root,'frontend/dist/runtime'),{recursive:true});
  await writeFile(path.join(root,'frontend/scripts/serve-dist.mjs'),await readFile(new URL('../scripts/serve-dist.mjs',import.meta.url)));
  await writeFile(path.join(root,'frontend/dist/index.html'),'<h1>TeaLeafAI</h1>');
  await writeFile(path.join(root,'frontend/dist/runtime/sample.wasm'),'test-wasm');
  const child=spawn(process.execPath,[path.join(root,'frontend/scripts/serve-dist.mjs')],{env:{...process.env,PORT:'0'},stdio:['ignore','pipe','pipe']});
  t.after(async()=> { child.kill(); await rm(root,{recursive:true,force:true}); });
  const base=await new Promise((resolve,reject)=>{
    child.on('error',reject); child.on('exit',code=>reject(new Error(`Server exited: ${code}`)));
    child.stdout.on('data',bytes=>{const match=bytes.toString().match(/http:\/\/127\.0\.0\.1:\d+/); if(match)resolve(match[0]);});
  });
  for(const route of ['/','/chan-doan','/lich-su']){
    const response=await fetch(base+route);
    assert.equal(response.status,200);
    assert.match(await response.text(),/TeaLeafAI/);
  }
  const wasm=await fetch(base+'/runtime/sample.wasm');
  assert.equal(wasm.status,200);assert.equal(wasm.headers.get('content-type'),'application/wasm');
  assert.equal(await (await fetch(base+'/model/missing.onnx')).status,404);
});
