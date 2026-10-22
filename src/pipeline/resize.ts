// Exact ports of the two image resizes the reference pre-processing uses:
//  - PIL's Image.resize(..., BICUBIC) (torchvision's Resize on a PIL image), 8-bit fixed point;
//  - OpenCV's cv2.resize(..., INTER_AREA), including its linear-style path when enlarging.
// Images are interleaved uint8 with `c` channels.

// ---------- PIL (libImaging/Resample.c) ----------

const PRECISION_BITS = 32 - 8 - 2;

function bicubic(x: number): number {
  const a = -0.5;
  if (x < 0) x = -x;
  if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
  return 0;
}

/** precompute_coeffs + normalize_coeffs_8bpc: integer weights per output pixel. */
function pilCoeffs(inSize: number, outSize: number) {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1);
  const support = 2 * filterscale;
  const ss = 1 / filterscale;
  const bounds: [number, number][] = [];
  const weights: number[][] = [];
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    const k: number[] = [];
    let ww = 0;
    for (let x = 0; x < xmax; x++) {
      const w = bicubic((x + xmin - center + 0.5) * ss);
      k.push(w);
      ww += w;
    }
    weights.push(
      k.map((w) => {
        const v = (ww !== 0 ? w / ww : w) * (1 << PRECISION_BITS);
        return v < 0 ? Math.trunc(-0.5 + v) : Math.trunc(0.5 + v);
      }),
    );
    bounds.push([xmin, xmax]);
  }
  return { bounds, weights };
}

const clip8 = (ss: number) => {
  const v = Math.floor(ss / (1 << PRECISION_BITS));
  return v < 0 ? 0 : v > 255 ? 255 : v;
};

/** PIL Image.resize((nw, nh), Image.BICUBIC) for an interleaved uint8 image. */
export function pilResizeBicubic(src: Uint8Array, w: number, h: number, c: number, nw: number, nh: number): Uint8Array {
  let img = src;
  let iw = w;
  let ih = h;
  const needH = nw !== w;
  const needV = nh !== h;
  let vc: ReturnType<typeof pilCoeffs> | null = null;
  let yFirst = 0;
  if (needV) {
    vc = pilCoeffs(h, nh);
    // Only the rows the vertical pass reads are resampled horizontally (as PIL does).
    yFirst = vc.bounds[0][0];
    const last = vc.bounds[nh - 1][0] + vc.bounds[nh - 1][1];
    if (needH) {
      img = src.subarray(yFirst * w * c, last * w * c);
      ih = last - yFirst;
    }
  }
  if (needH) {
    const { bounds, weights } = pilCoeffs(w, nw);
    const out = new Uint8Array(nw * ih * c);
    for (let y = 0; y < ih; y++)
      for (let x = 0; x < nw; x++) {
        const [xmin, n] = bounds[x];
        const k = weights[x];
        for (let ch = 0; ch < c; ch++) {
          let ss = 1 << (PRECISION_BITS - 1);
          for (let t = 0; t < n; t++) ss += img[(y * w + xmin + t) * c + ch] * k[t];
          out[(y * nw + x) * c + ch] = clip8(ss);
        }
      }
    img = out;
    iw = nw;
  }
  if (needV && vc) {
    const out = new Uint8Array(iw * nh * c);
    const off = needH ? yFirst : 0;
    for (let y = 0; y < nh; y++) {
      const [ymin, n] = vc.bounds[y];
      const k = vc.weights[y];
      for (let x = 0; x < iw; x++)
        for (let ch = 0; ch < c; ch++) {
          let ss = 1 << (PRECISION_BITS - 1);
          for (let t = 0; t < n; t++) ss += img[((ymin - off + t) * iw + x) * c + ch] * k[t];
          out[(y * iw + x) * c + ch] = clip8(ss);
        }
    }
    img = out;
  }
  return img === src ? src.slice() : img;
}

// ---------- OpenCV (imgproc/src/resize.cpp) ----------

/** cvRound on a float32 value: round half to even (lrint). */
function cvRound(v: number): number {
  const f = Math.floor(v);
  const d = v - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}
const sat8 = (v: number) => (v < 0 ? 0 : v > 255 ? 255 : v);

/** computeResizeAreaTab: (src index, float32 weight) lists per destination index. */
function areaTab(ssize: number, dsize: number, scale: number): [number, number][][] {
  const tab: [number, number][][] = [];
  for (let dx = 0; dx < dsize; dx++) {
    const fsx1 = dx * scale;
    const fsx2 = fsx1 + scale;
    const cellWidth = Math.min(scale, ssize - fsx1);
    let sx1 = Math.ceil(fsx1);
    let sx2 = Math.floor(fsx2);
    sx2 = Math.min(sx2, ssize - 1);
    sx1 = Math.min(sx1, sx2);
    const t: [number, number][] = [];
    if (sx1 - fsx1 > 1e-3) t.push([sx1 - 1, Math.fround((sx1 - fsx1) / cellWidth)]);
    for (let sx = sx1; sx < sx2; sx++) t.push([sx, Math.fround(1 / cellWidth)]);
    if (fsx2 - sx2 > 1e-3) t.push([sx2, Math.fround(Math.min(Math.min(fsx2 - sx2, 1), cellWidth) / cellWidth)]);
    tab.push(t);
  }
  return tab;
}

/** 2-tap tables for the enlarging INTER_AREA path, in OpenCV's 11-bit fixed point. */
function linearAreaTab(ssize: number, dsize: number) {
  const inv = dsize / ssize;
  const scale = 1 / inv;
  const sx: number[] = [];
  const a: [number, number][] = [];
  for (let dx = 0; dx < dsize; dx++) {
    let s = Math.floor(dx * scale);
    let fx = Math.fround((dx + 1) - (s + 1) * inv);
    fx = fx <= 0 ? 0 : Math.fround(fx - Math.floor(fx));
    if (s < 0) {
      fx = 0;
      s = 0;
    }
    if (s >= ssize - 1) {
      fx = 0;
      s = ssize - 1;
    }
    sx.push(s);
    a.push([cvRound(Math.fround(Math.fround(1 - fx) * 2048)), cvRound(Math.fround(fx * 2048))]);
  }
  return { sx, a };
}

/** cv2.resize(src, (nw, nh), interpolation=INTER_AREA) for an interleaved uint8 image. */
export function cvResizeArea(src: Uint8Array, w: number, h: number, c: number, nw: number, nh: number): Uint8Array {
  if (nw === w && nh === h) return src.slice();
  const scaleX = w / nw;
  const scaleY = h / nh;
  const out = new Uint8Array(nw * nh * c);
  if (scaleX >= 1 && scaleY >= 1) {
    const ix = Math.round(scaleX);
    const iy = Math.round(scaleY);
    if (Math.abs(scaleX - ix) < 1e-9 && Math.abs(scaleY - iy) < 1e-9 && ix * nw === w && iy * nh === h) {
      // resizeAreaFast: integer factors, average with float scale then cvRound.
      const s = Math.fround(1 / (ix * iy));
      for (let y = 0; y < nh; y++)
        for (let x = 0; x < nw; x++)
          for (let ch = 0; ch < c; ch++) {
            let sum = 0;
            for (let dy = 0; dy < iy; dy++) for (let dx = 0; dx < ix; dx++) sum += src[((y * iy + dy) * w + x * ix + dx) * c + ch];
            out[(y * nw + x) * c + ch] = sat8(cvRound(Math.fround(sum * s)));
          }
      return out;
    }
    // resizeArea_: separable float32 accumulation.
    const xt = areaTab(w, nw, scaleX);
    const yt = areaTab(h, nh, scaleY);
    const row = new Float32Array(nw * c);
    const acc = new Float32Array(nw * c);
    for (let y = 0; y < nh; y++) {
      acc.fill(0);
      for (const [sy, beta] of yt[y]) {
        row.fill(0);
        for (let x = 0; x < nw; x++)
          for (const [sx, alpha] of xt[x])
            for (let ch = 0; ch < c; ch++) row[x * c + ch] = Math.fround(row[x * c + ch] + Math.fround(alpha * src[(sy * w + sx) * c + ch]));
        for (let k = 0; k < nw * c; k++) acc[k] = Math.fround(acc[k] + Math.fround(beta * row[k]));
      }
      for (let k = 0; k < nw * c; k++) out[y * nw * c + k] = sat8(cvRound(acc[k]));
    }
    return out;
  }
  // Enlarging (in at least one direction): bilinear-style taps, fixed point.
  const X = linearAreaTab(w, nw);
  const Y = linearAreaTab(h, nh);
  const hrow = (sy: number) => {
    const r = new Int32Array(nw * c);
    for (let x = 0; x < nw; x++) {
      const s = X.sx[x];
      const [a0, a1] = X.a[x];
      const s1 = Math.min(s + 1, w - 1);
      for (let ch = 0; ch < c; ch++) r[x * c + ch] = src[(sy * w + s) * c + ch] * a0 + src[(sy * w + s1) * c + ch] * a1;
    }
    return r;
  };
  for (let y = 0; y < nh; y++) {
    const s = Y.sx[y];
    const [b0, b1] = Y.a[y];
    const r0 = hrow(s);
    const r1 = hrow(Math.min(s + 1, h - 1));
    // VResizeLinearVec_32s8u: rows to 16 bits (>> 4), high-half multiply (>> 16), then (+2) >> 2.
    for (let k = 0; k < nw * c; k++) out[y * nw * c + k] = sat8(((((r0[k] >> 4) * b0) >> 16) + (((r1[k] >> 4) * b1) >> 16) + 2) >> 2);
  }
  return out;
}
