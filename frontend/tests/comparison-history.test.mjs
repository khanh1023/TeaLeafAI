import test from 'node:test';
import assert from 'node:assert/strict';
import { comparisonSummary, comparisonCSV, readComparisonTrials, saveComparisonTrial, clearComparisonTrials } from '../src/comparison-history.js';

const keys=['mobilenetv2','resnet50','vgg16'];
const result=(id,label='healthy')=>({model_id:id,model_name:id,model_version:'v1',model_sha256:'a'.repeat(64),
  predicted_class:label,confidence:.9,inference_ms:20,load_ms:100,first_run_after_load:true});
const trial=(image,actual='healthy',ids=keys)=>({image_sha256:image,protocol_id:'p1',filename:image+'.jpg',
  actual_label:actual,results:Object.fromEntries(ids.map(id=>[id,result(id)]))});

test('Accuracy uses the same labelled, fully evaluated images for every model',()=>{
  const common=trial('a');common.results.resnet50=result('resnet50','white_spot');
  const rows=comparisonSummary([common,trial('b','healthy',['mobilenetv2']),trial('c','')]);
  assert.equal(rows[0].runs,3);assert.equal(rows[0].shared_runs,2);
  assert.deepEqual(rows.map(row=>row.labelled),[1,1,1]);
  assert.deepEqual(rows.map(row=>row.accuracy),[1,0,1]);
  assert.ok(comparisonSummary([trial('a','')]).every(row=>row.accuracy===null));
});

test('Same image/protocol merges model runs without counting duplicates, and versions remain separate',()=>{
  let saved='[]';globalThis.localStorage={getItem:()=>saved,setItem:(_,value)=>{saved=value;},removeItem:()=>{saved='[]';}};
  try {
    const a=trial('same','healthy',['mobilenetv2']);assert.ok(saveComparisonTrial(a,a.filename,a.actual_label));
    const b=trial('same','healthy',['resnet50','vgg16']);assert.ok(saveComparisonTrial(b,b.filename,b.actual_label));
    assert.equal(readComparisonTrials('p1').length,1);assert.equal(comparisonSummary(readComparisonTrials('p1'))[0].labelled,1);
    assert.ok(saveComparisonTrial({...a,protocol_id:'p2'},a.filename,''));
    assert.equal(readComparisonTrials('p2').length,1);assert.equal(readComparisonTrials('p1').length,1);
    assert.ok(clearComparisonTrials());assert.equal(readComparisonTrials().length,0);
  } finally {delete globalThis.localStorage;}
});

test('CSV keeps output scores distinct from correctness and escapes spreadsheet formulas',()=>{
  const a={...trial('x',''),filename:'=malicious.jpg'};
  const csv=comparisonCSV([a]);
  assert.ok(csv.includes("'=malicious.jpg"));assert.ok(csv.includes('Điểm đầu ra'));
  const rows=csv.split('\r\n').slice(1);
  assert.equal(rows.length,3);
  assert.ok(rows.every(row=>row.split(',')[9]==='""')); // Unknown labels have no correctness value.
  assert.ok(rows.every(row=>row.split(',')[12]==='"true"')); // Cold-start indicator is independent.
});
