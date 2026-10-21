// Keypoint and under-sampling-map extraction from the distance-field network's auxiliary outputs
// (predict_s1.predict_UDF, utils/keypt_tools.py).

import { connectedComponents, peakLocalMax, type Mask } from "./imgproc";

/** Keypoints as flat [x0, y0, x1, y1, ...] arrays. */
export interface Keypoints {
  end: number[];
  sharp: number[];
  junc: number[];
}

/** Aux channel order of the exported UDF network. */
export const AUX = { all: 0, end: 1, sharp: 2, junc: 3, usm: 4 } as const;

/** USM from the network: usm < thr or udf_all < thr / 3 (both clipped to [0, 1]). */
export function underSamplingMap(aux: Float32Array, h: number, w: number, thr = 0.5): Mask {
  const n = h * w;
  const out = new Uint8Array(n);
  const all = AUX.all * n;
  const usm = AUX.usm * n;
  for (let p = 0; p < n; p++) {
    const u = Math.min(1, Math.max(0, aux[usm + p]));
    const a = Math.min(1, Math.max(0, aux[all + p]));
    out[p] = u < thr || a < Math.fround(thr / 3) ? 1 : 0;
  }
  return out;
}

/** udf_filter: clip to [0,1]; set 4-connected regions of (udf < thr) whose minimum exceeds 0.1 to 1. */
export function udfFilter(src: Float32Array, h: number, w: number, thr = 0.99): Float32Array {
  const n = h * w;
  const udf = new Float32Array(n);
  const m = new Uint8Array(n);
  for (let p = 0; p < n; p++) {
    udf[p] = Math.min(1, Math.max(0, src[p]));
    m[p] = udf[p] < Math.fround(thr) ? 1 : 0;
  }
  const { labels, count } = connectedComponents(m, h, w, 4);
  const mins = new Float64Array(count + 1).fill(Infinity);
  for (let p = 0; p < n; p++) if (labels[p] && udf[p] < mins[labels[p]]) mins[labels[p]] = udf[p];
  for (let p = 0; p < n; p++) if (labels[p] && mins[labels[p]] > 0.1) udf[p] = 1;
  return udf;
}

/** to_img_coord / get_coords_from_heatmap: peaks of 255 - 255 * udf, returned as (x, y) in grid units. */
export function udfPeaks(udf: Float32Array, h: number, w: number, thrRel: number, ptNum = 1000): number[] {
  const n = h * w;
  const hm = new Float32Array(n);
  for (let p = 0; p < n; p++) {
    const v = Math.fround(255 - Math.fround(udf[p] * 255));
    hm[p] = Math.min(255, Math.max(0, v));
  }
  let coords = peakLocalMax(hm, h, w, 2, thrRel);
  if (coords.length > ptNum) {
    const order = coords.map((c, k) => [hm[c[0] * w + c[1]], k] as const);
    order.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    coords = order.slice(-ptNum).map(([, k]) => coords[k]);
  }
  const out: number[] = [];
  for (const [i, j] of coords) out.push(j, i);
  return out;
}

/**
 * Keypoints in input-pixel units (the reference halves the 2x-grid peak coordinates).
 * Thresholds follow extract_keypts: end 0.8, sharp 0.8, junc 0.5.
 */
export function extractKeypoints(aux: Float32Array, h: number, w: number): Keypoints {
  const n = h * w;
  const channel = (c: number) => udfFilter(aux.subarray(c * n, (c + 1) * n), h, w);
  const half = (a: number[]) => a.map((v) => v / 2);
  return {
    end: half(udfPeaks(channel(AUX.end), h, w, 0.8)),
    sharp: half(udfPeaks(channel(AUX.sharp), h, w, 0.8)),
    junc: half(udfPeaks(channel(AUX.junc), h, w, 0.5)),
  };
}
