import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { CLASSES } from '../src/classes.js';

test('Three distinct trained browser models match the catalog, class order and file hashes',async()=>{
  const base=new URL('../public/model/comparison/',import.meta.url);
  const catalog=JSON.parse(await readFile(new URL('manifest.json',base),'utf8'));
  assert.deepEqual(catalog.classes,CLASSES.map(c=>c.key));
  assert.deepEqual(catalog.models.map(m=>m.id),['mobilenetv2','resnet50','vgg16']);
  assert.equal(new Set(catalog.models.map(m=>m.sha256)).size,3);
  assert.equal(new Set(catalog.models.map(m=>m.keras_sha256)).size,3);
  assert.match(catalog.protocol_id,/^comparison-[a-z0-9-]+-[a-f0-9]{16}$/);
  assert.match(catalog.dataset_manifest_sha256,/^[a-f0-9]{64}$/);
  const reports = new URL('../../training/reports/', import.meta.url);
  const candidates = [new URL('expanded-dataset-manifest.json', reports)];
  for (const entry of await readdir(reports, { withFileTypes: true })) {
    if (entry.isDirectory() && entry.name.startsWith('dataset-')) candidates.push(new URL(entry.name + '/manifest.json', reports));
  }
  let dataset;
  for (const candidate of candidates) {
    let bytes;
    try { bytes = await readFile(candidate); } catch (error) { if (error.code === 'ENOENT') continue; throw error; }
    if (createHash('sha256').update(bytes).digest('hex') === catalog.dataset_manifest_sha256) { dataset = JSON.parse(bytes); break; }
  }
  assert.ok(dataset, 'Catalog must point to an available, versioned dataset manifest.');
  const expectedTestCount = dataset.filter(record => record.split === 'test').length;
  assert.ok(expectedTestCount > 0);
  for(const model of catalog.models){
    const bytes=await readFile(new URL(model.file,base));
    assert.equal(bytes.length,model.size_bytes);
    assert.equal(createHash('sha256').update(bytes).digest('hex'),model.sha256);
    assert.equal(model.parity_top1_disagreements,0);
    assert.ok(model.parity_samples>=8);
    assert.ok(model.parity_max_absolute_error<1e-3);
    assert.equal(model.native_test.test_images,expectedTestCount);
    assert.ok(Number.isInteger(model.native_test.correct) && model.native_test.correct >= 0 && model.native_test.correct <= expectedTestCount);
    assert.ok(Math.abs(model.native_test.accuracy - model.native_test.correct / expectedTestCount) < 1e-12);
    assert.ok(model.input_name&&model.output_name);
  }
});
