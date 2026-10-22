// Raster primitives replicating the OpenCV / scikit-image / SciPy calls used by the reference
// implementation. All images are row-major with explicit (h, w); masks are Uint8Array of 0/1.

export type Mask = Uint8Array<ArrayBuffer>;

/**
 * cv2.connectedComponents: labels 1..n assigned in raster order of each component's first pixel.
 * With `sameValue`, neighbors only connect if their (nonzero) values are equal.
 */
export function connectedComponents(
  mask: ArrayLike<number>,
  h: number,
  w: number,
  connectivity: 4 | 8,
  sameValue = false,
): { labels: Int32Array; count: number } {
  const labels = new Int32Array(h * w);
  const stack = new Int32Array(h * w);
  let next = 0;
  for (let start = 0; start < h * w; start++) {
    if (!mask[start] || labels[start]) continue;
    next++;
    labels[start] = next;
    let top = 0;
    stack[top++] = start;
    while (top > 0) {
      const p = stack[--top];
      const i = (p / w) | 0;
      const j = p - i * w;
      for (let di = -1; di <= 1; di++) {
        const ii = i + di;
        if (ii < 0 || ii >= h) continue;
        for (let dj = -1; dj <= 1; dj++) {
          if (di === 0 && dj === 0) continue;
          if (connectivity === 4 && di !== 0 && dj !== 0) continue;
          const jj = j + dj;
          if (jj < 0 || jj >= w) continue;
          const q = ii * w + jj;
          if (mask[q] && !labels[q] && (!sameValue || mask[q] === mask[p])) {
            labels[q] = next;
            stack[top++] = q;
          }
        }
      }
    }
  }
  return { labels, count: next };
}

/** Binary dilation (cv2.dilate, default border: pixels outside the image never contribute). */
export function dilate(mask: Mask, h: number, w: number, kernel: number[][]): Mask {
  return morph(mask, h, w, kernel, true);
}

/** Binary erosion (cv2.erode, default border: pixels outside the image never remove). */
export function erode(mask: Mask, h: number, w: number, kernel: number[][]): Mask {
  return morph(mask, h, w, kernel, false);
}

function morph(mask: Mask, h: number, w: number, kernel: number[][], isDilate: boolean): Mask {
  const kh = kernel.length;
  const kw = kernel[0].length;
  const ci = kh >> 1;
  const cj = kw >> 1;
  const out = new Uint8Array(h * w);
  for (let i = 0; i < h; i++) {
    for (let j = 0; j < w; j++) {
      let v = isDilate ? 0 : 1;
      for (let a = 0; a < kh && v === (isDilate ? 0 : 1); a++) {
        const ii = i + a - ci;
        if (ii < 0 || ii >= h) continue;
        for (let b = 0; b < kw; b++) {
          if (!kernel[a][b]) continue;
          const jj = j + b - cj;
          if (jj < 0 || jj >= w) continue;
          const m = mask[ii * w + jj];
          if (isDilate && m) {
            v = 1;
            break;
          }
          if (!isDilate && !m) {
            v = 0;
            break;
          }
        }
      }
      out[i * w + j] = v;
    }
  }
  return out;
}

/**
 * scipy.ndimage.correlate(mode='constant', cval=0) for odd square kernels, on numeric input.
 * scipy.signal.convolve2d(mode='same', fill 0) equals this with the kernel flipped (see convolve2dSame).
 */
export function correlate(img: ArrayLike<number>, h: number, w: number, kernel: number[][]): Float64Array {
  const kh = kernel.length;
  const kw = kernel[0].length;
  const ci = kh >> 1;
  const cj = kw >> 1;
  const out = new Float64Array(h * w);
  for (let i = 0; i < h; i++) {
    for (let j = 0; j < w; j++) {
      let s = 0;
      for (let a = 0; a < kh; a++) {
        const ii = i + a - ci;
        if (ii < 0 || ii >= h) continue;
        for (let b = 0; b < kw; b++) {
          const k = kernel[a][b];
          if (k === 0) continue;
          const jj = j + b - cj;
          if (jj < 0 || jj >= w) continue;
          s += k * img[ii * w + jj];
        }
      }
      out[i * w + j] = s;
    }
  }
  return out;
}

export function convolve2dSame(img: ArrayLike<number>, h: number, w: number, kernel: number[][]): Float64Array {
  const flipped = kernel.map((row) => row.slice().reverse()).reverse();
  return correlate(img, h, w, flipped);
}

// 8-neighborhood directions in OpenCV's order: 0 = +x, then counter-clockwise on screen (y down).
const DIR_DI = [0, -1, -1, -1, 0, 1, 1, 1];
const DIR_DJ = [1, 1, 0, -1, -1, -1, 0, 1];

/**
 * Outer border of the (single) foreground region of `mask`, traced like cv2.findContours with
 * CHAIN_APPROX_NONE (Suzuki–Abe border following, 8-connected foreground), plus the number of
 * contours RETR_TREE would return (1 outer + number of holes). Points are returned as [i, j].
 * Assumes the foreground is one 8-connected component (callers pass one 4-connected region).
 */
export function regionContour(mask: Mask, h: number, w: number): { contour: [number, number][]; count: number } {
  let start = -1;
  for (let p = 0; p < h * w; p++) {
    if (mask[p]) {
      start = p;
      break;
    }
  }
  if (start < 0) return { contour: [], count: 0 };
  const at = (i: number, j: number) => i >= 0 && i < h && j >= 0 && j < w && mask[i * w + j] !== 0;
  const i0 = (start / w) | 0;
  const j0 = start - i0 * w;

  const contour: [number, number][] = [];
  // Find the first foreground neighbor clockwise from the left neighbor.
  let s = 4;
  let found = false;
  for (let k = 0; k < 7; k++) {
    s = (s - 1) & 7;
    if (at(i0 + DIR_DI[s], j0 + DIR_DJ[s])) {
      found = true;
      break;
    }
  }
  if (!found) {
    contour.push([i0, j0]);
  } else {
    const i1 = i0 + DIR_DI[s];
    const j1 = j0 + DIR_DJ[s];
    let i3 = i0;
    let j3 = j0;
    for (;;) {
      let i4 = 0;
      let j4 = 0;
      for (let k = 0; k < 8; k++) {
        s++;
        i4 = i3 + DIR_DI[s & 7];
        j4 = j3 + DIR_DJ[s & 7];
        if (at(i4, j4)) break;
      }
      s &= 7;
      contour.push([i3, j3]);
      if (i4 === i0 && j4 === j0 && i3 === i1 && j3 === j1) break;
      i3 = i4;
      j3 = j4;
      s = (s + 4) & 7;
    }
  }

  // Holes: 4-connected background components that do not touch the image border.
  const bg = new Uint8Array(h * w);
  for (let p = 0; p < h * w; p++) bg[p] = mask[p] ? 0 : 1;
  const { labels, count } = connectedComponents(bg, h, w, 4);
  const touches = new Uint8Array(count + 1);
  for (let j = 0; j < w; j++) {
    touches[labels[j]] = 1;
    touches[labels[(h - 1) * w + j]] = 1;
  }
  for (let i = 0; i < h; i++) {
    touches[labels[i * w]] = 1;
    touches[labels[i * w + w - 1]] = 1;
  }
  let holes = 0;
  for (let l = 1; l <= count; l++) if (!touches[l]) holes++;
  return { contour, count: 1 + holes };
}

// scikit-image 0.26 _fast_skeletonize lookup table (Zhang–Suen variant).
// prettier-ignore
const SKEL_LUT = new Uint8Array([
  0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0,
  3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  2, 0, 0, 0, 3, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0,
  3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0, 0, 0, 3, 1, 0, 0, 1, 3, 0, 0, 0, 0,
  0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1, 3,
  0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0,
  2, 0, 0, 0,
]);

/** skimage.morphology.skeletonize (2D, method='zhang'). */
export function skeletonize(mask: Mask, h: number, w: number): Mask {
  const H = h + 2;
  const W = w + 2;
  const skel = new Uint8Array(H * W);
  for (let i = 0; i < h; i++) for (let j = 0; j < w; j++) skel[(i + 1) * W + j + 1] = mask[i * w + j] ? 1 : 0;
  const cleaned = skel.slice();
  let removed = true;
  while (removed) {
    removed = false;
    for (let pass = 0; pass < 2; pass++) {
      const first = pass === 0;
      for (let r = 1; r < H - 1; r++) {
        for (let c = 1; c < W - 1; c++) {
          const p = r * W + c;
          if (!skel[p]) continue;
          const n = SKEL_LUT[
            skel[p - W - 1] + 2 * skel[p - W] + 4 * skel[p - W + 1] + 8 * skel[p + 1] +
            16 * skel[p + W + 1] + 32 * skel[p + W] + 64 * skel[p + W - 1] + 128 * skel[p - 1]
          ];
          if (n === 0) continue;
          if (n === 3 || (n === 1 && first) || (n === 2 && !first)) {
            cleaned[p] = 0;
            removed = true;
          }
        }
      }
      skel.set(cleaned);
    }
  }
  const out = new Uint8Array(h * w);
  for (let i = 0; i < h; i++) for (let j = 0; j < w; j++) out[i * w + j] = skel[(i + 1) * W + j + 1];
  return out;
}

/**
 * skimage.feature.peak_local_max(image, min_distance, threshold_rel, exclude_border=False),
 * returning [row, col] pairs, highest intensity first (stable on ties, raster order).
 */
export function peakLocalMax(
  img: ArrayLike<number>,
  h: number,
  w: number,
  minDistance: number,
  thresholdRel: number,
): [number, number][] {
  let mn = Infinity;
  let mx = -Infinity;
  for (let p = 0; p < h * w; p++) {
    if (img[p] < mn) mn = img[p];
    if (img[p] > mx) mx = img[p];
  }
  const threshold = Math.max(mn, thresholdRel * mx);
  // maximum_filter with a (2d+1)^2 footprint, mode='nearest' (== max over the clipped window).
  const rowMax = new Float64Array(h * w);
  for (let i = 0; i < h; i++) {
    for (let j = 0; j < w; j++) {
      let m = -Infinity;
      for (let jj = Math.max(0, j - minDistance); jj <= Math.min(w - 1, j + minDistance); jj++) {
        const v = img[i * w + jj];
        if (v > m) m = v;
      }
      rowMax[i * w + j] = m;
    }
  }
  const cand: number[] = [];
  let allPeaks = true;
  const isPeak = new Uint8Array(h * w);
  for (let i = 0; i < h; i++) {
    for (let j = 0; j < w; j++) {
      let m = -Infinity;
      for (let ii = Math.max(0, i - minDistance); ii <= Math.min(h - 1, i + minDistance); ii++) {
        const v = rowMax[ii * w + j];
        if (v > m) m = v;
      }
      const p = i * w + j;
      if (img[p] === m) isPeak[p] = 1;
      else allPeaks = false;
    }
  }
  if (allPeaks) return [];
  for (let p = 0; p < h * w; p++) if (isPeak[p] && img[p] > threshold) cand.push(p);
  // Stable sort by descending intensity.
  const order = cand.map((p, k) => [p, k]);
  order.sort((a, b) => img[b[0]] - img[a[0]] || a[1] - b[1]);
  const sorted = order.map((o) => o[0]);
  if (minDistance <= 1) return sorted.map((p) => [(p / w) | 0, p % w]);
  // ensure_spacing: greedy; reject later peaks at Chebyshev distance < minDistance from a kept one.
  const kept: [number, number][] = [];
  const rejected = new Uint8Array(h * w);
  for (const p of sorted) {
    if (rejected[p]) continue;
    const i = (p / w) | 0;
    const j = p % w;
    kept.push([i, j]);
    const r = Math.ceil(minDistance) - 1;
    for (let ii = Math.max(0, i - r); ii <= Math.min(h - 1, i + r); ii++)
      for (let jj = Math.max(0, j - r); jj <= Math.min(w - 1, j + r); jj++) rejected[ii * w + jj] = 1;
  }
  return kept;
}
