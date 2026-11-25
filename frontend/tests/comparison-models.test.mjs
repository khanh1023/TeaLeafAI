import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { CLASSES } from '../src/classes.js';

test('Three distinct trained browser models match the catalog, class order and file hashes',async()=>{
  const base=new URL('../public/model/comparison/',import.meta.url);
  const catalog=JSON.parse(await readFile(new URL('manifest.json',base),'utf8'));
  assert.deepEqual(catalog.classes,CLASSES.map(c=>c.key));
  assert.deepEqual(catalog.models.map(m=>m.id),['mobilenetv2','resnet50','vgg16']);
  assert.equal(new Set(catalog.models.map(m=>m.sha256)).size,3);
  assert.equal(new Set(catalog.models.map(m=>m.keras_sha256)).size,3);
  assert.match(catalog.protocol_id,/comparison-frozen-20261125-seed42-[a-f0-9]{16}/);
  for(const model of catalog.models){
    const bytes=await readFile(new URL(model.file,base));
    assert.equal(bytes.length,model.size_bytes);
    assert.equal(createHash('sha256').update(bytes).digest('hex'),model.sha256);
    assert.equal(model.parity_top1_disagreements,0);
    assert.ok(model.parity_samples>=8);
    assert.ok(model.parity_max_absolute_error<1e-3);
    assert.equal(model.native_test.test_images,329);
    assert.ok(model.input_name&&model.output_name);
  }
});
