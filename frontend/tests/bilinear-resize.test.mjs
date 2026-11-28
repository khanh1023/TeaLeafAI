import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {resizeRGBBilinear} from '../src/bilinear-resize.js';

test('Antialiased RGB resize matches independent Pillow reference pixels for downscale, upscale, boundaries and unchanged axes', async () => {
  const {cases} = JSON.parse(await readFile(new URL('./fixtures/pillow-bilinear.json', import.meta.url), 'utf8'));
  for (const fixture of cases) {
    const actual = resizeRGBBilinear(fixture.input,fixture.width,fixture.height,fixture.outWidth,fixture.outHeight);
    assert.deepEqual([...actual],fixture.expected);
  }
});

test('Downsampling removes checkerboard aliasing and preserves constant RGB colours', () => {
  const checker = new Uint8Array(8*8*4);
  for(let i=0;i<64;i++) checker.set([(Math.floor(i/8)+i%8)%2*255,80,240,255],i*4);
  const result=resizeRGBBilinear(checker,8,8,1,1);
  assert.deepEqual([...result],[128,80,240]);
  assert.throws(()=>resizeRGBBilinear(checker,0,8));
  assert.throws(()=>resizeRGBBilinear(checker,9,8));
});
