/** Evaluate the unchanged held-out test files with the actual browser diagnose().
 * Run Vite first. Requires Playwright + Chromium (not part of lightweight CI).
 * TEALEAF_BASE_URL, TEALEAF_EVAL_OUTPUT, PLAYWRIGHT_MODULE, CHROMIUM_PATH supported.
 * No historical report is overwritten: the default output uses a unique run directory.
 * Timing is local headless-browser timing, not a benchmark of users' devices/networks.
 */
import assert from 'node:assert/strict';
import {readFile, mkdir, writeFile, access} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {performance as nodePerformance} from 'node:perf_hooks';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';

const root = fileURLToPath(new URL('../../', import.meta.url));
const runId = new Date().toISOString().replaceAll(':','-');
const output = path.resolve(process.env.TEALEAF_EVAL_OUTPUT || path.join(root, 'training/reports', `browser-evaluation-${runId}`));
const base = (process.env.TEALEAF_BASE_URL || 'http://127.0.0.1:5179').replace(/\/$/,'');
const moduleName = process.env.PLAYWRIGHT_MODULE || 'playwright';
const {chromium} = await import(path.isAbsolute(moduleName) ? pathToFileURL(moduleName).href : moduleName);
const manifestBytes = await readFile(path.join(root, 'training/reports/expanded-dataset-manifest.json'));
const manifest = JSON.parse(manifestBytes);
const model = JSON.parse(await readFile(path.join(root, 'frontend/public/model/manifest.json'), 'utf8'));
const lock = JSON.parse(await readFile(path.join(root, 'frontend/package-lock.json'), 'utf8'));
const manifestSha = createHash('sha256').update(manifestBytes).digest('hex');
const modelSha = createHash('sha256').update(await readFile(path.join(root,'frontend/public/model',model.file))).digest('hex');
assert.equal(manifestSha, model.dataset_manifest_sha256);
assert.equal(modelSha, model.sha256);
if (process.env.TEALEAF_EXPECTED_MODEL_SHA256) assert.equal(modelSha, process.env.TEALEAF_EXPECTED_MODEL_SHA256);
const entries = manifest.filter(r => r.split === 'test');
assert.equal(entries.length,Number(process.env.TEALEAF_EXPECTED_TEST_COUNT || 329));
await mkdir(output,{recursive:true});
const outputFile = path.join(output,'browser-predictions.json');
try { await access(outputFile); throw new Error(`Output already exists; select a fresh TEALEAF_EVAL_OUTPUT: ${outputFile}`); }
catch(error) { if (error.code !== 'ENOENT') throw error; }

function stats(values) {
  const xs=values.filter(Number.isFinite).sort((a,b)=>a-b);
  return {
    count:xs.length, mean_ms:xs.reduce((a,b)=>a+b,0)/xs.length,
    median_ms:xs.length%2 ? xs[(xs.length-1)/2] : (xs[xs.length/2-1]+xs[xs.length/2])/2,
    p95_ms:xs[Math.ceil(xs.length*.95)-1],min_ms:xs[0],max_ms:xs.at(-1),
    percentile_definition:'p95 uses nearest-rank ceil(0.95*n); median averages the two middle values for even n.'
  };
}
const browser = await chromium.launch({headless:true, ...(process.env.CHROMIUM_PATH ? {executablePath:process.env.CHROMIUM_PATH} : {})});
const predictions = [], errors = [], requests = [], repeats = [];
const started = Date.now();
let firstResources=[];
try {
  const page = await browser.newPage({viewport:{width:1280,height:900}});
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => requests.push({url:request.url(),method:request.method()}));
  await page.goto(base+'/chan-doan', {waitUntil:'networkidle'});
  const engine = await page.evaluate(async () => (await import('/src/api.js')).ENGINE);
  assert.equal(engine, 'browser');

  async function measure(record) {
    const readStart=nodePerformance.now();
    const bytes=await readFile(path.join(root,'training/dataset',record.path));
    const fileReadMs=nodePerformance.now()-readStart;
    assert.equal(createHash('sha256').update(bytes).digest('hex'),record.sha256);
    const encodeStart=nodePerformance.now();
    const base64=bytes.toString('base64');
    const encodeMs=nodePerformance.now()-encodeStart;
    const outerStart=nodePerformance.now();
    const measured=await page.evaluate(async ({base64,name})=>{
      const {diagnose}=await import('/src/api.js');
      const bytes=Uint8Array.from(atob(base64),char=>char.charCodeAt(0));
      const file=new File([bytes],name,{type:'image/jpeg'});
      const phases=[];
      const started=performance.now();
      const result=await diagnose(file,{onProgress:value=>phases.push({percent:value.percent,message:value.message,elapsed_ms:performance.now()-started})});
      const diagnoseMs=performance.now()-started;
      const runStart=phases.find(value=>value.percent===75)?.elapsed_ms;
      const runEnd=phases.find(value=>value.percent===100)?.elapsed_ms;
      return {result,timing:{browser_diagnose_ms:diagnoseMs,browser_model_execution_stage_ms:runEnd-runStart,progress_phases:phases}};
    },{base64,name:path.basename(record.path)});
    const evaluateRoundtripMs=nodePerformance.now()-outerStart;
    assert.equal(measured.result.model_version,model.model_version);
    assert.equal(measured.result.engine,'browser');
    assert.equal(Object.keys(measured.result.probabilities).length,model.classes.length);
    return {result:measured.result,timing:{...measured.timing,node_file_read_ms:fileReadMs,node_base64_encode_ms:encodeMs,node_evaluate_roundtrip_ms:evaluateRoundtripMs,
      automation_and_serialization_overhead_ms:Math.max(0,evaluateRoundtripMs-measured.timing.browser_diagnose_ms)}};
  }

  for (const [index,record] of entries.entries()) {
    const {result,timing}=await measure(record);
    predictions.push({id:record.id,path:record.path,source:record.source,actual:record.label,
      predicted:result.predicted_class,probabilities:result.probabilities,
      correct:record.label===result.predicted_class,cold_start:index===0,timing});
    if(index===0) firstResources=await page.evaluate(()=>performance.getEntriesByType('resource').filter(r=>/\.onnx(?:\?|$)|\.wasm(?:\?|$)/.test(r.name)).map(r=>({
      url:r.name,initiator_type:r.initiatorType,start_time_ms:r.startTime,response_end_ms:r.responseEnd,duration_ms:r.duration,
      transfer_size_bytes:r.transferSize,encoded_body_size_bytes:r.encodedBodySize,decoded_body_size_bytes:r.decodedBodySize
    })));
    if((index+1)%25===0 || index+1===entries.length) console.log(`Browser ${index+1}/${entries.length}; correct so far: ${predictions.filter(r=>r.correct).length}`,new Date().toISOString());
  }

  // Ten deterministic, evenly spaced test records, one extra warm pass each.
  const repeatIndices=Array.from({length:10},(_,i)=>Math.floor(i*(entries.length-1)/9));
  for(const index of repeatIndices) {
    const record=entries[index];
    const measured=await measure(record);
    const original=predictions[index];
    assert.equal(measured.result.predicted_class,original.predicted);
    const maxDifference=Math.max(...model.classes.map(key=>Math.abs(measured.result.probabilities[key]-original.probabilities[key])));
    assert.ok(maxDifference<1e-6);
    repeats.push({index,id:record.id,path:record.path,predicted:measured.result.predicted_class,max_probability_difference:maxDifference,timing:measured.timing});
  }
  assert.deepEqual(errors,[]);
  assert.equal(requests.filter(r=>r.method==='POST').length,0);
  const external=requests.filter(r=>!r.url.startsWith(base+'/') && !r.url.startsWith('blob:') && !r.url.startsWith('data:'));
  assert.deepEqual(external,[]);
  const modelDownloads=requests.filter(r=>/\.onnx(?:\?|$)/.test(r.url)).length;
  assert.equal(modelDownloads,1);

  const runtimeModuleUrl=requests.find(r=>/onnxruntime-web[_-]wasm\.js/.test(r.url))?.url;
  assert.ok(runtimeModuleUrl,'Observed loaded Vite ORT module URL.');
  const device=await page.evaluate(async(runtimeModuleUrl)=>{
    const ort=await import(runtimeModuleUrl);
    return {user_agent:navigator.userAgent,platform:navigator.platform,hardware_concurrency:navigator.hardwareConcurrency,
      device_memory_gib:navigator.deviceMemory??null,language:navigator.language,device_pixel_ratio:devicePixelRatio,
      viewport:{width:innerWidth,height:innerHeight},cross_origin_isolated:crossOriginIsolated,secure_context:isSecureContext,
      runtime:{wasm_threads:ort.env.wasm.numThreads,wasm_proxy:ort.env.wasm.proxy,wasm_simd:ort.env.wasm.simd??null}};
  },runtimeModuleUrl);
  const first=predictions[0];
  const firstPhases=first.timing.progress_phases;
  const phase=percent=>firstPhases.find(p=>p.percent===percent)?.elapsed_ms;
  const timingDefinitions={
    browser_diagnose_ms:'Browser performance.now() immediately before await diagnose(file) to after resolve; includes image decode/Canvas resize, model execution, tensor/output handling, and cached metadata. File construction, disk I/O, base64 encoding, and Playwright transport are excluded. First call additionally includes model download/hash/WASM import/session initialization.',
    browser_model_execution_stage_ms:'Approximate execution stage: onProgress 75% to 100%; includes tensor construction, session.run, output copying and predictionResult; excludes image decoding and session initialization. This is not a direct isolated kernel benchmark.',
    warm_summary:'First cold call excluded; 328 remaining held-out samples in manifest order. Ten extra deterministic samples are summarized separately, not counted as additional test observations.',
    model_download_ms:'PerformanceResourceTiming.duration for initial .onnx resource from request start to response end on localhost. Not internet download performance. WASM resource timing also recorded.',
    automation_and_serialization_overhead_ms:'Node page.evaluate round-trip minus browser diagnose wall time; includes IPC, payload/file preparation and result serialization, not model latency.',
    environment_limit:'Single headless Chromium process on this Cloud Linux host, with concurrent work potentially sharing CPU. No inference or deployment-performance claim for other devices.'
  };
  const warm=predictions.slice(1);
  const warmSummaries={
    diagnose:stats(warm.map(r=>r.timing.browser_diagnose_ms)),
    model_execution_stage:stats(warm.map(r=>r.timing.browser_model_execution_stage_ms)),
    evaluate_roundtrip:stats(warm.map(r=>r.timing.node_evaluate_roundtrip_ms)),
    automation_overhead:stats(warm.map(r=>r.timing.automation_and_serialization_overhead_ms)),
    disk_read:stats(warm.map(r=>r.timing.node_file_read_ms)),
    base64_encode:stats(warm.map(r=>r.timing.node_base64_encode_ms))
  };
  const correct=predictions.filter(r=>r.correct).length;
  const initialTiming={
    first_record:first.id,cold_diagnose_ms:first.timing.browser_diagnose_ms,
    model_download_resources:firstResources,
    model_download_ms:firstResources.find(r=>/\.onnx(?:\?|$)/.test(r.url))?.duration_ms??null,
    download_module_and_hash_stage_ms:phase(45)-phase(10),
    session_initialization_stage_ms:phase(75)-phase(45),progress_phases:firstPhases
  };
  const metadata={
    evaluated_at:new Date().toISOString(),run_id:runId,engine:'browser',model_version:model.model_version,onnx_sha256:modelSha,
    dataset_manifest_sha256:manifestSha,test_images:entries.length,correct,accuracy:correct/entries.length,
    preprocessing:'Actual frontend readImagePixels(): createImageBitmap -> Canvas drawImage 224x224 -> raw RGB 0–255 -> ONNX Runtime Web WASM.',
    user_agent:device.user_agent,browser_version:browser.version(),device,
    host:{platform:os.platform(),release:os.release(),architecture:os.arch(),cpu_model:os.cpus()[0]?.model,logical_cpus:os.cpus().length,total_memory_bytes:os.totalmem(),node:process.version},
    runtime:{package:'onnxruntime-web',version:lock.packages?.['node_modules/onnxruntime-web']?.version,execution_provider:'wasm',...device.runtime},
    model_downloads:modelDownloads,image_uploads:0,external_requests:external.length,uncaught_browser_errors:errors,
    elapsed_seconds:(Date.now()-started)/1000,timing_definitions:timingDefinitions,initial_timing:initialTiming,
    warm_timing: warmSummaries,repeat_warm_timing:{selection:'Ten indices floor(i*(n-1)/9), i=0..9, in frozen test manifest order; one additional warm call per selected sample.',indices:repeatIndices,
      diagnose:stats(repeats.map(r=>r.timing.browser_diagnose_ms)),model_execution_stage:stats(repeats.map(r=>r.timing.browser_model_execution_stage_ms))}
  };
  await writeFile(outputFile,JSON.stringify({...metadata,predictions,warm_repeats:repeats},null,2),{flag:'wx'});
  await writeFile(path.join(output,'browser-summary.json'),JSON.stringify(metadata,null,2),{flag:'wx'});
  console.log(JSON.stringify({output,test_images:entries.length,correct,accuracy:metadata.accuracy,cold_diagnose_ms:initialTiming.cold_diagnose_ms,model_download_ms:initialTiming.model_download_ms,warm_diagnose:warmSummaries.diagnose,repeat_warm_diagnose:metadata.repeat_warm_timing.diagnose,elapsed_seconds:metadata.elapsed_seconds},null,2));
} finally {
  await browser.close();
}
