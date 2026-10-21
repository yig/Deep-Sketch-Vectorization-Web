// Dual contouring on the 2x grid: edge flags from the NDC network, edge-flag cleanup
// (utils/ndc_tools.py: cut_branch, remove_stray, remove_parallel, fill_edge_hole) and line maps.
//
// Conventions (same as the reference):
//  - The grid has h x w cells (h = 2H, w = 2W for an H x W input image), indexed (i, j) row-major.
//  - edgeX[i,j] ("edge flag along x") connects cell (i-1, j) to cell (i, j): a vertical polyline step.
//  - edgeY[i,j] connects cell (i, j-1) to cell (i, j).
//  - ptMap[i,j] = (x, y) vertex of cell (i, j), in input-image pixel units.

import { connectedComponents, convolve2dSame, correlate, dilate, skeletonize, type Mask } from "./imgproc";

export interface Grid {
  h: number;
  w: number;
}

export interface EdgeMap extends Grid {
  x: Mask;
  y: Mask;
}

/** Per-cell stored segment (start, end), like the reference's object-array line maps. */
export interface LineMap extends Grid {
  mask: Uint8Array;
  /** [sx, sy, ex, ey] per cell. */
  seg: Float64Array;
}

const CROSS = [
  [0, 1, 0],
  [1, 1, 1],
  [0, 1, 0],
];

/** Shift so out[i,j] = a[i-di, j-dj] (0 outside): roll_edge(a, di, 'x') / roll_edge(a, dj, 'y'). */
export function shift(a: ArrayLike<number>, h: number, w: number, di: number, dj: number): Mask {
  const out = new Uint8Array(h * w);
  for (let i = 0; i < h; i++) {
    const si = i - di;
    if (si < 0 || si >= h) continue;
    for (let j = 0; j < w; j++) {
      const sj = j - dj;
      if (sj < 0 || sj >= w) continue;
      out[i * w + j] = a[si * w + sj] ? 1 : 0;
    }
  }
  return out;
}

/** Edge flags from NDC logits [4, h, w] (none / x / y / both) by argmax. */
export function edgeMapFromLogits(logits: Float32Array, h: number, w: number): EdgeMap {
  const n = h * w;
  const x = new Uint8Array(n);
  const y = new Uint8Array(n);
  for (let p = 0; p < n; p++) {
    let best = 0;
    let bv = logits[p];
    for (let c = 1; c < 4; c++) {
      const v = logits[c * n + p];
      if (v > bv) {
        bv = v;
        best = c;
      }
    }
    x[p] = best === 1 || best === 3 ? 1 : 0;
    y[p] = best === 2 || best === 3 ? 1 : 0;
  }
  return { h, w, x, y };
}

/** Vertex map from NDC offsets [2, h, w]: offset + cell base coordinate (grid size 0.5 px). */
export function vertexMapFromOffsets(offset: Float32Array, h: number, w: number): Float32Array {
  const n = h * w;
  const pt = new Float32Array(n * 2);
  for (let i = 0; i < h; i++) {
    for (let j = 0; j < w; j++) {
      const p = i * w + j;
      pt[2 * p] = Math.fround(offset[p] + (0.25 + 0.5 * j));
      pt[2 * p + 1] = Math.fround(offset[n + p] + (0.25 + 0.5 * i));
    }
  }
  return pt;
}

/** get_valence_np: per-cell valence (number of incident edges) and vertex mask. */
export function valenceMap(e: EdgeMap): { valence: Int32Array; ptMask: Mask } {
  const { h, w } = e;
  const valence = new Int32Array(h * w);
  const ptMask = new Uint8Array(h * w);
  for (let i = 0; i < h; i++) {
    for (let j = 0; j < w; j++) {
      const p = i * w + j;
      const gx = (i + 1 < h ? e.x[p + w] : 0) + e.x[p];
      const gy = (j + 1 < w ? e.y[p + 1] : 0) + e.y[p];
      valence[p] = gx + gy;
      ptMask[p] = gx || gy ? 1 : 0;
    }
  }
  return { valence, ptMask };
}

/** cut_branch: remove 1-cell spurs hanging off junctions. */
export function cutBranch(e: EdgeMap): EdgeMap {
  const { h, w } = e;
  const n = h * w;
  const { valence } = valenceMap(e);
  const end = new Uint8Array(n);
  for (let p = 0; p < n; p++) end[p] = valence[p] === 1 ? 1 : 0;
  const near = convolve2dSame(end, h, w, CROSS);
  let branch = new Uint8Array(n);
  for (let p = 0; p < n; p++) branch[p] = near[p] > 0 && valence[p] > 2 ? 1 : 0;
  branch = dilate(branch, h, w, CROSS);
  for (let p = 0; p < n; p++) branch[p] = branch[p] && end[p] ? 1 : 0;
  const branchFree = new Uint8Array(n);
  for (let p = 0; p < n; p++) branchFree[p] = branch[p] && !(e.x[p] || e.y[p]) ? 1 : 0;
  const bx = shift(branchFree, h, w, 1, 0);
  const by = shift(branchFree, h, w, 0, 1);
  const x = new Uint8Array(n);
  const y = new Uint8Array(n);
  for (let p = 0; p < n; p++) {
    x[p] = e.x[p] && !branch[p] && !bx[p] ? 1 : 0;
    y[p] = e.y[p] && !branch[p] && !by[p] ? 1 : 0;
  }
  return { h, w, x, y };
}

/** remove_stray: drop 8-connected components smaller than thr cells. */
export function removeStray(skel: Mask, h: number, w: number, thr = 6): Mask {
  const { labels, count } = connectedComponents(skel, h, w, 8);
  const size = new Int32Array(count + 1);
  for (let p = 0; p < h * w; p++) size[labels[p]]++;
  const out = new Uint8Array(h * w);
  for (let p = 0; p < h * w; p++) out[p] = skel[p] && size[labels[p]] >= thr ? 1 : 0;
  return out;
}

/** remove_parallel: skeletonize the occupancy map to drop redundant parallel edge flags. */
export function removeParallel(occupancy: Mask, e: EdgeMap, strayThr = 6): Mask {
  const { h, w } = e;
  const n = h * w;
  let skel = skeletonize(occupancy, h, w);
  skel = removeStray(skel, h, w, strayThr);
  // Fill 8-way -45 degree diagonals to 4-way connections.
  const below = shift(skel, h, w, -1, 0);
  const left = shift(skel, h, w, 0, 1);
  for (let p = 0; p < n; p++) if (below[p] && !skel[p] && left[p]) skel[p] = 1;
  const s2 = new Uint8Array(n);
  for (let p = 0; p < n; p++) s2[p] = (e.x[p] || e.y[p]) && skel[p] ? 1 : 0;
  const below2 = shift(s2, h, w, -1, 0);
  const a = new Uint8Array(n);
  for (let p = 0; p < n; p++) a[p] = below2[p] && !s2[p] ? 1 : 0;
  const aShift = shift(shift(a, h, w, 0, -1), h, w, 1, 0);
  const up = shift(s2, h, w, 1, 0);
  const out = new Uint8Array(n);
  for (let p = 0; p < n; p++) out[p] = s2[p] || (aShift[p] && up[p]) ? 1 : 0;
  return out;
}

const eq = (arr: Float64Array, v: number) => {
  const m = new Uint8Array(arr.length);
  for (let p = 0; p < arr.length; p++) m[p] = arr[p] === v ? 1 : 0;
  return m;
};

/** fill_edge_hole: restore broken lines with 3x3 kernels and remove links between adjacent T-junctions. */
export function fillEdgeHole(e: EdgeMap): EdgeMap {
  const { h, w } = e;
  const n = h * w;
  const kxHole = [
    [-1, 1, -1],
    [-1, -1, -1],
    [-1, 1, -1],
  ];
  const kyHole = [
    [-1, -1, -1],
    [1, -1, 1],
    [-1, -1, -1],
  ];
  const holeX = eq(convolve2dSame(e.x, h, w, kxHole), 2);
  const holeY = eq(convolve2dSame(e.y, h, w, kyHole), 2);
  const ex = new Uint8Array(n);
  const ey = new Uint8Array(n);
  for (let p = 0; p < n; p++) {
    ex[p] = e.x[p] || holeX[p] ? 1 : 0;
    ey[p] = e.y[p] || holeY[p] ? 1 : 0;
  }
  const kxy1Hole = [[1, -1, 0], [1, -1, 0], [-1, 1, 0]];
  const kxy2Hole = [[-1, 1, 0], [-1, 1, 0], [1, -1, 0]];
  const kyx1Hole = [[1, 1, -1], [-1, -1, 1], [0, 0, 0]];
  const kyx2Hole = [[-1, -1, 1], [1, 1, -1], [0, 0, 0]];
  const kyx1Over = [[1, 1, -1], [-1, 1, 1], [0, 0, 0]];
  const kyx2Over = [[-1, -1, 1], [1, 1, 1], [0, 0, 0]];
  const kxy1Over = [[1, -1, 0], [1, 1, 0], [-1, 1, 0]];
  const kxy2Over = [[-1, 1, 0], [-1, 1, 0], [1, 1, 0]];
  const hy1 = eq(correlate(ex, h, w, kxy1Hole), 3);
  const hy2 = eq(correlate(ex, h, w, kxy2Hole), 3);
  const hy3 = eq(correlate(ex, h, w, kxy1Over), 4);
  const hy4 = eq(correlate(ex, h, w, kxy2Over), 4);
  const hx1 = eq(correlate(ey, h, w, kyx1Hole), 3);
  const hx2 = eq(correlate(ey, h, w, kyx2Hole), 3);
  const hx3 = eq(correlate(ey, h, w, kyx1Over), 4);
  const hx4 = eq(correlate(ey, h, w, kyx2Over), 4);
  const hy4s = shift(hy4, h, w, 1, 0);
  const hx4s = shift(hx4, h, w, 0, 1);
  // Valence of the *input* edge map (as in the reference).
  const { valence } = valenceMap(e);
  const v3 = new Float64Array(n);
  for (let p = 0; p < n; p++) v3[p] = valence[p] >= 3 ? 1 : 0;
  const bx = eq(correlate(v3, h, w, [[0, 1, 0], [0, 1, 0], [0, 0, 0]]), 2);
  const by = eq(correlate(v3, h, w, [[0, 0, 0], [1, 1, 0], [0, 0, 0]]), 2);
  const x = new Uint8Array(n);
  const y = new Uint8Array(n);
  for (let p = 0; p < n; p++) {
    x[p] = (ex[p] || hx1[p] || hx2[p] || hx3[p] || hx4[p]) && !(hy4s[p] || hy3[p]) && !bx[p] ? 1 : 0;
    y[p] = (ey[p] || hy1[p] || hy2[p] || hy3[p] || hy4[p]) && !(hx4s[p] || hx3[p]) && !by[p] ? 1 : 0;
  }
  return { h, w, x, y };
}

function maskEdges(e: EdgeMap, m: Mask): EdgeMap {
  const n = e.h * e.w;
  const x = new Uint8Array(n);
  const y = new Uint8Array(n);
  for (let p = 0; p < n; p++) {
    x[p] = e.x[p] && m[p] ? 1 : 0;
    y[p] = e.y[p] && m[p] ? 1 : 0;
  }
  return { h: e.h, w: e.w, x, y };
}

/** The `refine=True` edge cleanup in map_to_lines. */
export function refineEdgeMap(e: EdgeMap): EdgeMap {
  const cut = cutBranch(e);
  const occ = new Uint8Array(e.h * e.w);
  for (let p = 0; p < occ.length; p++) occ[p] = cut.x[p] || cut.y[p] ? 1 : 0;
  const skel = removeParallel(occ, cut);
  const filled = fillEdgeHole(maskEdges(cut, skel));
  return cutBranch(maskEdges(filled, skel));
}

export function emptyLineMap(h: number, w: number): LineMap {
  return { h, w, mask: new Uint8Array(h * w), seg: new Float64Array(h * w * 4) };
}

export function cloneLineMap(m: LineMap): LineMap {
  return { h: m.h, w: m.w, mask: m.mask.slice(), seg: m.seg.slice() };
}

/** map_to_lines (without refine): per-cell line maps from edge flags and vertices. */
export function lineMaps(e: EdgeMap, pt: Float32Array): { x: LineMap; y: LineMap } {
  const { h, w } = e;
  const lx = emptyLineMap(h, w);
  const ly = emptyLineMap(h, w);
  const set = (m: LineMap, p: number, a: number, b: number) => {
    if (pt[2 * a] === pt[2 * b] && pt[2 * a + 1] === pt[2 * b + 1]) return;
    m.mask[p] = 1;
    m.seg[4 * p] = pt[2 * a];
    m.seg[4 * p + 1] = pt[2 * a + 1];
    m.seg[4 * p + 2] = pt[2 * b];
    m.seg[4 * p + 3] = pt[2 * b + 1];
  };
  for (let i = 1; i < h; i++) for (let j = 0; j < w; j++) if (e.x[i * w + j]) set(lx, i * w + j, (i - 1) * w + j, i * w + j);
  for (let i = 0; i < h; i++) for (let j = 1; j < w; j++) if (e.y[i * w + j]) set(ly, i * w + j, i * w + j - 1, i * w + j);
  return { x: lx, y: ly };
}

/** linemap_to_lines: all stored segments, x map first then y map, raster order. Flat [sx,sy,ex,ey,...]. */
export function lineMapsToSegments(lx: LineMap, ly: LineMap): number[] {
  const out: number[] = [];
  for (const m of [lx, ly]) {
    for (let p = 0; p < m.h * m.w; p++) {
      if (!m.mask[p]) continue;
      out.push(m.seg[4 * p], m.seg[4 * p + 1], m.seg[4 * p + 2], m.seg[4 * p + 3]);
    }
  }
  return out;
}
