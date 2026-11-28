/** Separable, antialiased bilinear RGB resize with 22-bit integer coefficients.
 * Matches Pillow's 8-bit BILINEAR convention (pixel centres, normalized boundary
 * weights, rounded horizontal intermediate). See public/licenses/Pillow-LICENSE.txt
 * and https://github.com/python-pillow/Pillow/blob/12.1.0/src/libImaging/Resample.c.
 * Input RGBA must already be opaque/composited by the image decoder.
 */
const PRECISION = 2 ** 22;

function coefficients(input, output) {
  const scale = input / output;
  const filterScale = Math.max(1, scale);
  return Array.from({length:output}, (_, i) => {
    const center = (i + .5) * scale;
    const start = Math.max(0, Math.trunc(center - filterScale + .5));
    const end = Math.min(input, Math.trunc(center + filterScale + .5));
    const values = Array.from({length:end-start}, (_, j) => Math.max(0, 1 - Math.abs((j + start - center + .5) / filterScale)));
    const sum = values.reduce((a, b) => a + b, 0);
    return {start, weights:values.map(v => Math.floor(v / sum * PRECISION + .5))};
  });
}

export function resizeRGBBilinear(rgba, width, height, outWidth=224, outHeight=224) {
  if (![width,height,outWidth,outHeight].every(v => Number.isSafeInteger(v) && v > 0)
      || rgba.length !== width * height * 4) throw new Error('Kích thước điểm ảnh không hợp lệ.');
  const horizontal = new Uint8Array(outWidth * height * 3);
  const xCoefficients = coefficients(width, outWidth);
  const yCoefficients = coefficients(height, outHeight);
  const round = value => Math.max(0, Math.min(255, Math.floor(value / PRECISION)));
  for (let y=0;y<height;y++) for (let x=0;x<outWidth;x++) {
    const {start,weights} = xCoefficients[x];
    let red=PRECISION/2, green=PRECISION/2, blue=PRECISION/2;
    for (let k=0;k<weights.length;k++) {
      const index=(y*width+start+k)*4, weight=weights[k];
      red+=rgba[index]*weight; green+=rgba[index+1]*weight; blue+=rgba[index+2]*weight;
    }
    const index=(y*outWidth+x)*3;
    horizontal[index]=round(red); horizontal[index+1]=round(green); horizontal[index+2]=round(blue);
  }
  const rgb = new Float32Array(outWidth * outHeight * 3);
  for (let y=0;y<outHeight;y++) for (let x=0;x<outWidth;x++) {
    const {start,weights} = yCoefficients[y];
    let red=PRECISION/2, green=PRECISION/2, blue=PRECISION/2;
    for (let k=0;k<weights.length;k++) {
      const index=((start+k)*outWidth+x)*3, weight=weights[k];
      red+=horizontal[index]*weight; green+=horizontal[index+1]*weight; blue+=horizontal[index+2]*weight;
    }
    const index=(y*outWidth+x)*3;
    rgb[index]=round(red); rgb[index+1]=round(green); rgb[index+2]=round(blue);
  }
  return rgb;
}
