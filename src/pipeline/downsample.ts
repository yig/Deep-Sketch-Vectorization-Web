// Dual contouring downsampling (utils/ndc_tools.py: downsample_all with return_lines=True).
// Coarsens the 2x dual-contouring grid by `ratio`, keeping the fine geometry inside
// under-sampled blocks (predicted USM, multiple crossings of a coarse edge, endpoints).

import { connectedComponents, type Mask } from "./imgproc";
import { cutBranch, type EdgeMap, type LineMap } from "./dualContouring";

/** out[I,J] = a[I,J] | a[I+1,J] (mode 'x'), | a[I,J+1] ('y'), or both ('xy'); 0 past the end. */
function expand(a: Mask, h: number, w: number, mode: "x" | "y" | "xy"): Mask {
  const out = new Uint8Array(h * w);
  for (let i = 0; i < h; i++)
    for (let j = 0; j < w; j++) {
      const p = i * w + j;
      let v = a[p];
      if (mode !== "y" && i + 1 < h) v |= a[p + w];
      if (mode !== "x" && j + 1 < w) v |= a[p + 1];
      out[p] = v ? 1 : 0;
    }
  return out;
}

function upsample(a: Mask, hd: number, wd: number, r: number): Mask {
  const h = hd * r;
  const w = wd * r;
  const out = new Uint8Array(h * w);
  for (let i = 0; i < h; i++) for (let j = 0; j < w; j++) out[i * w + j] = a[((i / r) | 0) * wd + ((j / r) | 0)];
  return out;
}

/**
 * downsample_all. Returns flat segments: the fine segments kept inside under-sampled areas
 * (x line map then y line map, raster order), then the coarse segments.
 */
export function downsampleAll(
  edges: EdgeMap,
  pt: Float32Array,
  usm: Mask,
  valence: Int32Array,
  ptType: Int32Array,
  lx: LineMap,
  ly: LineMap,
  r: number,
): number[] {
  const { h, w } = edges;
  if (h % r !== 0 || w % r !== 0) throw new Error(`grid ${h}x${w} not divisible by down rate ${r}`);
  const hd = h / r;
  const wd = w / r;
  const nd = hd * wd;
  const ex = edges.x;
  const ey = edges.y;

  // Block statistics.
  const usmPreDown = new Uint8Array(nd);
  const xDown = new Int32Array(nd); // crossings of each coarse cell's top edge
  const yDown = new Int32Array(nd); // crossings of each coarse cell's left edge
  const hasEnd = new Uint8Array(nd);
  const allSharp = new Uint8Array(nd);
  for (let I = 0; I < hd; I++)
    for (let J = 0; J < wd; J++) {
      const q = I * wd + J;
      let u = 0;
      let e1 = 0;
      let s2 = 1;
      for (let a = 0; a < r; a++)
        for (let b = 0; b < r; b++) {
          const p = (I * r + a) * w + J * r + b;
          u += usm[p];
          if (ptType[p] === 1) e1 = 1;
          if (ptType[p] !== 2) s2 = 0;
        }
      usmPreDown[q] = u > 0 ? 1 : 0;
      hasEnd[q] = e1;
      allSharp[q] = s2;
      let xs = 0;
      let ys = 0;
      for (let c = 0; c < r; c++) {
        xs += ex[I * r * w + J * r + c];
        ys += ey[(I * r + c) * w + J * r];
      }
      xDown[q] = xs;
      yDown[q] = ys;
    }
  const at = (a: Int32Array, I: number, J: number) => (I < hd && J < wd ? a[I * wd + J] : 0);

  // Under-sampling introduced by the downsampling itself.
  const usmTop = new Uint8Array(nd);
  const usmLeft = new Uint8Array(nd);
  const usmBottom = new Uint8Array(nd);
  const usmRight = new Uint8Array(nd);
  const usm4 = new Uint8Array(nd);
  const usmXY = new Uint8Array(nd);
  for (let I = 0; I < hd; I++)
    for (let J = 0; J < wd; J++) {
      const q = I * wd + J;
      const top = xDown[q];
      const bottom = at(xDown, I + 1, J);
      const left = yDown[q];
      const right = at(yDown, I, J + 1);
      const kp = hasEnd[q] && !allSharp[q] ? 1 : 0;
      usmLeft[q] = left > 1 || kp ? 1 : 0;
      usmTop[q] = top > 1 || kp ? 1 : 0;
      usmBottom[q] = bottom > 1 || kp ? 1 : 0;
      usmRight[q] = right > 1 || kp ? 1 : 0;
      usm4[q] = (top >= 1 ? 1 : 0) + (bottom >= 1 ? 1 : 0) + (left >= 1 ? 1 : 0) + (right >= 1 ? 1 : 0) === 4 ? 1 : 0;
      usmXY[q] = top + left > 1 ? 1 : 0;
    }
  const top2 = expand(usmTop, hd, wd, "x");
  const left2 = expand(usmLeft, hd, wd, "y");
  const four2 = expand(usm4, hd, wd, "xy");
  const usmDownsample = new Uint8Array(nd);
  for (let q = 0; q < nd; q++) usmDownsample[q] = left2[q] || usmRight[q] || top2[q] || usmBottom[q] || four2[q] || usmXY[q] ? 1 : 0;

  // Drop USM regions introduced by the downsampling that contain no keypoint. (The reference
  // indexes with a uint8 mask here, which NumPy treats as row indices; fixed.) Regions are found on
  // the fine grid; each coarse block lies in a single region.
  {
    const fine = upsample(usmDownsample, hd, wd, r);
    const { labels, count } = connectedComponents(fine, h, w, 4);
    const hasKey = new Uint8Array(count + 1);
    for (let p = 0; p < h * w; p++) if (labels[p] && ptType[p] > 0) hasKey[labels[p]] = 1;
    for (let I = 0; I < hd; I++)
      for (let J = 0; J < wd; J++) {
        const l = labels[I * r * w + J * r];
        if (l && !hasKey[l]) usmDownsample[I * wd + J] = 0;
      }
  }

  let usmDown = new Uint8Array(nd);
  for (let q = 0; q < nd; q++) usmDown[q] = usmPreDown[q] || usmDownsample[q] ? 1 : 0;

  // Coarse edge flags outside the under-sampled blocks.
  const exd = new Uint8Array(nd);
  const eyd = new Uint8Array(nd);
  for (let q = 0; q < nd; q++) {
    exd[q] = xDown[q] >= 1 && !usmDown[q] ? 1 : 0;
    eyd[q] = yDown[q] >= 1 && !usmDown[q] ? 1 : 0;
  }

  // Which fine segments to keep.
  const lineMask = upsample(usmDown, hd, wd, r);
  const lineMaskXBase = upsample(expand(usmDown, hd, wd, "x"), hd, wd, r);
  const lineMaskYBase = upsample(expand(usmDown, hd, wd, "y"), hd, wd, r);
  const n = h * w;
  const lm = new Uint8Array(n);
  for (let i = 0; i < h; i++)
    for (let j = 0; j < w; j++) {
      const p = i * w + j;
      const mx = i % r === 0 ? 0 : lineMaskXBase[p];
      const my = j % r === 0 ? 0 : lineMaskYBase[p];
      lm[p] = mx || my || lineMask[p] ? 1 : 0;
    }
  const reserved: number[] = [];
  for (const [m, base] of [
    [lx, lineMaskYBase],
    [ly, lineMaskXBase],
  ] as const) {
    for (let p = 0; p < n; p++) {
      if (!(lm[p] || base[p]) || !m.mask[p]) continue;
      reserved.push(m.seg[4 * p], m.seg[4 * p + 1], m.seg[4 * p + 2], m.seg[4 * p + 3]);
    }
  }

  // Coarse vertices: average of the fine vertices in each block (endpoints win if the block has
  // no junction), and per-side averages of the fine vertices whose links cross each block side.
  const center = new Float64Array(nd * 2);
  const topPt = new Float64Array(nd * 2);
  const leftPt = new Float64Array(nd * 2);
  const bottomPt = new Float64Array(nd * 2);
  const rightPt = new Float64Array(nd * 2);
  for (let I = 0; I < hd; I++)
    for (let J = 0; J < wd; J++) {
      const q = I * wd + J;
      let cx = 0, cy = 0, cn = 0, vx = 0, vy = 0, vn = 0, v3 = 0;
      for (let a = 0; a < r; a++)
        for (let b = 0; b < r; b++) {
          const p = (I * r + a) * w + J * r + b;
          const v = valence[p];
          if (v > 0) {
            cx = Math.fround(cx + pt[2 * p]);
            cy = Math.fround(cy + pt[2 * p + 1]);
            cn++;
          }
          if (v === 1) {
            vx = Math.fround(vx + pt[2 * p]);
            vy = Math.fround(vy + pt[2 * p + 1]);
            vn++;
          }
          if (v > 2) v3 = 1;
        }
      const useEnd = vn > 0 && !v3;
      center[2 * q] = useEnd ? vx / vn : cx / Math.max(cn, 1);
      center[2 * q + 1] = useEnd ? vy / vn : cy / Math.max(cn, 1);
      // Top: row I*r, links up (edge x). Bottom: row I*r+r-1, links down. Left/right analogous.
      const side = (out: Float64Array, cells: (c: number) => number, flag: (p: number) => number) => {
        let sx = 0, sy = 0, k = 0;
        for (let c = 0; c < r; c++) {
          const p = cells(c);
          if (flag(p)) {
            sx = Math.fround(sx + pt[2 * p]);
            sy = Math.fround(sy + pt[2 * p + 1]);
            k++;
          }
        }
        out[2 * q] = sx / Math.max(k, 1);
        out[2 * q + 1] = sy / Math.max(k, 1);
      };
      side(topPt, (c) => I * r * w + J * r + c, (p) => ex[p]);
      side(bottomPt, (c) => (I * r + r - 1) * w + J * r + c, (p) => (p + w < n ? ex[p + w] : 0));
      side(leftPt, (c) => (I * r + c) * w + J * r, (p) => ey[p]);
      side(rightPt, (c) => (I * r + c) * w + J * r + r - 1, (p) => ((p % w) + 1 < w ? ey[p + 1] : 0));
    }

  const coarse = cutBranch({ h: hd, w: wd, x: exd, y: eyd });
  usmDown = expand(usmDown, hd, wd, "xy");
  const down: number[] = [];
  const differ = (a: Float64Array, qa: number, b: Float64Array, qb: number) =>
    a[2 * qa] !== b[2 * qb] || a[2 * qa + 1] !== b[2 * qb + 1];
  const add = (a: Float64Array, qa: number, b: Float64Array, qb: number) =>
    down.push(a[2 * qa], a[2 * qa + 1], b[2 * qb], b[2 * qb + 1]);
  for (let I = 0; I < hd; I++)
    for (let J = 0; J < wd; J++) {
      const q = I * wd + J;
      if (coarse.x[q] && I !== 0) {
        const qp = q - wd;
        const up = usmDown[qp];
        const cur = usmDown[q];
        if (!up && cur) {
          if (differ(center, qp, topPt, q)) add(center, qp, topPt, q);
        } else if (up && !cur) {
          if (differ(bottomPt, qp, center, q)) add(bottomPt, qp, center, q);
        } else if (!up && !cur) {
          if (differ(center, qp, center, q)) add(center, qp, center, q);
        } else if (differ(bottomPt, qp, topPt, q)) add(bottomPt, qp, topPt, q);
      }
      if (coarse.y[q] && J !== 0) {
        const qp = q - 1;
        const lf = usmDown[qp];
        const cur = usmDown[q];
        if (!lf && cur) {
          if (differ(center, qp, leftPt, q)) add(center, qp, leftPt, q);
        } else if (lf && !cur) {
          if (differ(rightPt, qp, center, q)) add(rightPt, qp, center, q);
        } else if (!lf && !cur) {
          if (differ(center, qp, center, q)) add(center, qp, center, q);
        } else if (differ(rightPt, qp, leftPt, q)) add(rightPt, qp, leftPt, q);
      }
    }
  return reserved.concat(down);
}
