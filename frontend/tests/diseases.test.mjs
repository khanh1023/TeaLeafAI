import test from 'node:test';
import assert from 'node:assert/strict';
import {CLASSES} from '../src/classes.js';
import {DISEASES,DISEASE_BY_KEY,SOURCES} from '../src/diseases.js';
test('all eight model labels have complete referenced care information',()=>{
 assert.deepEqual(DISEASES.map(x=>x.key),CLASSES.map(x=>x.key));
 for(const item of DISEASES){
  assert.ok(item.summary.length>20);
  for(const field of ['signs','conditions','actions','prevention'])assert.ok(item[field].length>0 && item[field].every(x=>typeof x==='string'&&x.length>10),`${item.key}: ${field}`);
  assert.ok(item.expert.length>20);
  assert.ok(item.sources.length>0 && item.sources.every(key=>SOURCES[key]?.url.startsWith('https://')));
 }
});
test('unknown route keys cannot resolve to object prototype properties',()=>{
 for(const key of ['missing','constructor','toString','__proto__'])assert.equal(DISEASE_BY_KEY[key],undefined);
});
test('healthy care and uncertain dataset labels retain their specific guidance',()=>{
 assert.match(DISEASE_BY_KEY.healthy.prevention.join(' '),/Không cần dùng thuốc/);
 for(const key of ['gray_light','red_leaf_spot','white_spot'])assert.ok(DISEASE_BY_KEY[key].identityNote);
});
