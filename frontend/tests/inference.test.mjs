import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {rgbaToRGB,predictionResult} from '../src/inference-utils.js';
import {CLASSES} from '../src/classes.js';

test('RGB input preserves raw pixel range and NHWC order (no double normalization)',()=>{
  assert.deepEqual(Array.from(rgbaToRGB(new Uint8ClampedArray([255,128,0,255,1,2,3,255]))),[255,128,0,1,2,3]);
});
test('all 8 output positions map to the expected labels',()=>{
  for(let i=0;i<8;i++){
    const values = new Float32Array(8); values[i]=1;
    const result=predictionResult(values);
    assert.equal(result.predicted_class,CLASSES[i].key);
    assert.equal(result.confidence,1);
    assert.equal(Object.keys(result.probabilities).length,8);
  }
});
test('corrupted/nonfinite model output is rejected',()=>{
  for(const values of [[1],[NaN,0,0,0,0,0,0,1],[Infinity,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[-1,2,0,0,0,0,0,0]]) assert.throws(()=>predictionResult(values));
});
test('browser model integrity and class order match Python model',()=>{
  const manifest=JSON.parse(readFileSync(new URL('../public/model/manifest.json',import.meta.url)));
  const bytes=readFileSync(new URL('../public/model/tea-leaf.onnx',import.meta.url));
  assert.equal(bytes.length,manifest.size_bytes);
  assert.equal(createHash('sha256').update(bytes).digest('hex'),manifest.sha256);
  assert.deepEqual(manifest.classes,CLASSES.map(x=>x.key));
  const python=JSON.parse(readFileSync(new URL('../../backend/models/classes.json',import.meta.url)));
  assert.deepEqual(manifest.classes,python);
});
