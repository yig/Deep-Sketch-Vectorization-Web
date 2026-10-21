// Topology refinement ("USM surgery") from utils/ndc_tools.py: refine_topology, align_keypt,
// find_connection_points, update_line_maps and helpers.
//
// The reference runs surgery incrementally (and differently) for user edits. Here one pure function,
// `surgery`, maps (marks, USM) to lines, and both the automatic pass and every edited state go through
// it: the automatic pass is surgery(autoMarks, autoUsm). See DESIGN.md §8.

import { connectedComponents, dilate, erode, regionContour, type Mask } from "./imgproc";
import { cloneLineMap, lineMapsToSegments, valenceMap, type EdgeMap, type LineMap } from "./dualContouring";
import { downsampleAll } from "./downsample";
import type { Keypoints } from "./keypoints";

type Pt = [number, number];

interface ConnectionPoint {
  pt: Pt;
  tan: Pt;
  i: number;
  j: number;
  /** 0 top, 1 left, 2 bottom, 3 right (direction from the boundary cell to this neighbor). */
  edge: number;
}

/**
 * align_keypt: snap each keypoint to its nearest dual-contouring vertex of the given valence
 * (if closer than thr); marks ptType there. Returns the snapped positions (flat), or the input
 * unchanged if there is no vertex of that valence.
 */
export function alignKeypoints(
  pt: Float32Array,
  keypts: number[],
  valence: Int32Array,
  val: number,
  ptType: Int32Array,
  thr: number,
): number[] {
  const cells: number[] = [];
  for (let p = 0; p < valence.length; p++) if (valence[p] === val) cells.push(p);
  if (cells.length === 0) return keypts;
  const out: number[] = [];
  for (let k = 0; k < keypts.length / 2; k++) {
    const kx = keypts[2 * k];
    const ky = keypts[2 * k + 1];
    let best = Infinity;
    let bestCell = -1;
    for (const c of cells) {
      const dx = pt[2 * c] - kx;
      const dy = pt[2 * c + 1] - ky;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d < best) {
        best = d;
        bestCell = c;
      }
    }
    if (best < thr) {
      ptType[bestCell] = val;
      out.push(pt[2 * bestCell], pt[2 * bestCell + 1]);
    }
  }
  return out;
}

function unitTangent(sx: number, sy: number, ex: number, ey: number): Pt {
  const dx = ex - sx;
  const dy = ey - sy;
  const l = Math.hypot(dx, dy);
  // The reference raises ZeroDivisionError here; a zero tangent keeps going instead.
  return l === 0 ? [0, 0] : [dx / l, dy / l];
}

/**
 * find_connection_points: where polylines enter the region from outside the USM.
 * null = the region's contour is not a single simple border (it has holes).
 * The contour is traced on the region's bounding box (padded by 1), which gives the same result.
 */
function findConnectionPoints(
  hasVertex: Mask,
  pt: Float32Array,
  cells: number[],
  usm: Mask,
  h: number,
  w: number,
): ConnectionPoint[] | null {
  let i0 = h, i1 = 0, j0 = w, j1 = 0;
  for (const p of cells) {
    const i = (p / w) | 0;
    const j = p - i * w;
    if (i < i0) i0 = i;
    if (i > i1) i1 = i;
    if (j < j0) j0 = j;
    if (j > j1) j1 = j;
  }
  const bh = i1 - i0 + 3;
  const bw = j1 - j0 + 3;
  const local = new Uint8Array(bh * bw);
  for (const p of cells) {
    const i = (p / w) | 0;
    local[(i - i0 + 1) * bw + (p - i * w - j0 + 1)] = 1;
  }
  const { contour, count } = regionContour(local, bh, bw);
  if (count !== 1) return null;
  const res: ConnectionPoint[] = [];
  const DI = [-1, 0, 1, 0];
  const DJ = [0, -1, 0, 1];
  for (const [li, lj] of contour) {
    const i = li + i0 - 1;
    const j = lj + j0 - 1;
    if (i === 0 || j === 0) continue;
    if (i > h - 5 || j > w - 5) continue;
    const p = i * w + j;
    if (!hasVertex[p]) continue;
    for (let d = 0; d < 4; d++) {
      const q = (i + DI[d]) * w + (j + DJ[d]);
      if (!hasVertex[q] || usm[q]) continue;
      const sx = pt[2 * q];
      const sy = pt[2 * q + 1];
      res.push({ pt: [sx, sy], tan: unitTangent(sx, sy, pt[2 * p], pt[2 * p + 1]), i: i + DI[d], j: j + DJ[d], edge: d });
    }
  }
  return res;
}

/** Matching keypoint vertices of one type inside a region: [] (none), one point, or several. */
type Match = Pt[];

function queryKeypointsByRegion(marks: MarkGrid, cells: number[], pt: Float32Array, ptMask: Mask) {
  const find = (val: number): Match => {
    const res: Match = [];
    for (const p of cells) if (marks.type[p] === val) res.push([marks.pos[2 * p], marks.pos[2 * p + 1]]);
    return res;
  };
  let sx = 0;
  let sy = 0;
  let n = 0;
  for (const p of cells) {
    if (ptMask[p]) {
      // NumPy float32 mean along axis 0: sequential float32 sum, float32 divide.
      sx = Math.fround(sx + pt[2 * p]);
      sy = Math.fround(sy + pt[2 * p + 1]);
      n++;
    }
  }
  const avg: Pt | null = n > 0 ? [Math.fround(sx / n), Math.fround(sy / n)] : null;
  return { end: find(1), sharp: find(2), junc: find(3), avg };
}

/** svgpathtools Line.intersect for two line segments (bounding-box rejection, then solve). */
function segmentsIntersect(p0: Pt, p1: Pt, q0: Pt, q1: Pt): boolean {
  if (Math.min(q0[0], q1[0]) > Math.max(p0[0], p1[0]) || Math.max(q0[0], q1[0]) < Math.min(p0[0], p1[0])) return false;
  if (Math.min(q0[1], q1[1]) > Math.max(p0[1], p1[1]) || Math.max(q0[1], q1[1]) < Math.min(p0[1], p1[1])) return false;
  const a = [p0[0], p1[0]];
  const b = [p0[1], p1[1]];
  const c = [q0[0], q1[0]];
  const d = [q0[1], q1[1]];
  const denom = (a[1] - a[0]) * (d[0] - d[1]) - (b[1] - b[0]) * (c[0] - c[1]);
  if (Math.abs(denom) <= 1e-8) return false;
  const t1 = (c[0] * (b[0] - d[1]) - c[1] * (b[0] - d[0]) - a[0] * (d[0] - d[1])) / denom;
  const t2 = -(a[1] * (b[0] - d[0]) - a[0] * (b[1] - d[0]) - c[0] * (b[0] - b[1])) / denom;
  return t1 >= 0 && t1 <= 1 && t2 >= 0 && t2 <= 1;
}

const same = (a: Pt, b: Pt) => a[0] === b[0] && a[1] === b[1];
const absDot = (a: Pt, b: Pt) => Math.abs(a[0] * b[0] + a[1] * b[1]);

class XJunctionError extends Error {}

/** The valence-4 "parallel lines" case of update_line_maps (inside the reference's try block). */
function xJunction(cps: ConnectionPoint[], added: Pt[]): void {
  const P = cps.map((c) => c.pt);
  const pairs: [number, number][] = [
    [0, 1], [2, 3],
    [1, 2], [0, 3],
    [0, 2], [1, 3],
  ];
  const tansEdge = pairs.map(([s, e]) => {
    // Line.unit_tangent divides by zero for a degenerate segment.
    if (same(P[s], P[e])) throw new XJunctionError();
    return unitTangent(P[s][0], P[s][1], P[e][0], P[e][1]);
  });
  const dir = tansEdge.map((t) => cps.reduce((s, c) => s + absDot(t, c.tan), 0) / 4);
  let best = 0;
  for (let k = 1; k < 6; k++) if (dir[k] >= dir[best]) best = k; // argsort()[-1]: last of the maxima
  let groupA: [number, number];
  let groupB: [number, number];
  if ((best === 0 && dir[1] > 0.9) || (best === 1 && dir[0] > 0.9)) [groupA, groupB] = [[0, 1], [2, 3]];
  else if ((best === 2 && dir[3] > 0.9) || (best === 3 && dir[2] > 0.9)) [groupA, groupB] = [[1, 2], [0, 3]];
  else if ((best === 4 && dir[5] > 0.9) || (best === 5 && dir[4] > 0.9)) [groupA, groupB] = [[0, 2], [1, 3]];
  else throw new XJunctionError(); // UnboundLocalError in the reference
  // update_x_junction
  const [p0, p1, p2, p3] = [P[groupA[0]], P[groupA[1]], P[groupB[0]], P[groupB[1]]];
  let done = false;
  if (!same(p0, p1) && !same(p2, p3)) {
    if (!segmentsIntersect(p0, p1, p2, p3)) {
      added.push(p0, p1, p2, p3);
      done = true;
    }
  }
  if (!same(p0, p3) && !same(p2, p1) && !done) {
    if (!segmentsIntersect(p0, p3, p2, p1)) added.push(p0, p3, p2, p1);
  }
}

/**
 * update_line_maps (manual = false, full_auto_mode = true). Mutates the line maps.
 * Returns null (valence 0), or whether the region was rebuilt, the added segments' endpoints, and
 * the cells the updated USM covers (the region, plus the connection cells if rebuilt).
 */
function updateLineMaps(
  lx: LineMap,
  ly: LineMap,
  pt: Float32Array,
  ptMask: Mask,
  marks: MarkGrid,
  cps: ConnectionPoint[],
  cells: number[],
): { applied: boolean; added: Pt[]; usmCells: number[] } | null {
  const valence = cps.length;
  const added: Pt[] = [];
  const { end, sharp, junc, avg } = queryKeypointsByRegion(marks, cells, pt, ptMask);
  const connect = (p: Pt) => {
    for (const c of cps) added.push(c.pt, p);
  };
  const connectAvg = () => {
    if (avg) connect(avg);
  };
  let update = false;
  if (valence === 0) return null;
  if (valence === 1) {
    if (end.length === 1) {
      connect(end[0]);
      update = true;
    }
  } else if (valence === 2) {
    update = true;
    if (sharp.length === 1) connect(sharp[0]);
    else if (junc.length === 1) connect(junc[0]);
    else if (end.length >= 1 && sharp.length >= 1 && junc.length >= 1) connect(sharp[0]);
    else if (sharp.length > 1) update = false;
    else added.push(cps[0].pt, cps[1].pt);
  } else if (valence === 3) {
    update = true;
    if (junc.length === 1) connect(junc[0]);
    else if (sharp.length === 1) connect(sharp[0]);
    else if (end.length >= 1 && sharp.length >= 1 && junc.length >= 1) connect(junc[0]);
    else if (junc.length > 1) update = false;
    else {
      // Join the two most parallel connections; attach the third to their midpoint.
      let bi = 0;
      let bj = 1;
      let bv = -Infinity;
      for (let a = 0; a < 3; a++)
        for (let b = 0; b < 3; b++) {
          const v = b > a ? absDot(cps[a].tan, cps[b].tan) : 0;
          if (v > bv) {
            bv = v;
            bi = a;
            bj = b;
          }
        }
      const k = [0, 1, 2].find((x) => x !== bi && x !== bj)!;
      const p0 = cps[bi].pt;
      const p1 = cps[bj].pt;
      added.push(p0, p1, cps[k].pt, [Math.fround(Math.fround(p0[0] + p1[0]) / 2), Math.fround(Math.fround(p0[1] + p1[1]) / 2)]);
    }
  } else if (valence === 4) {
    update = true;
    if (junc.length === 1) connect(junc[0]);
    else if (sharp.length === 1) connect(sharp[0]);
    else if (end.length >= 1 && sharp.length >= 1 && junc.length >= 1) connect(junc[0]);
    else {
      try {
        xJunction(cps, added);
      } catch (e) {
        if (!(e instanceof XJunctionError)) throw e;
        connectAvg();
      }
    }
  } else {
    update = true;
    if (junc.length === 1) connect(junc[0]);
    else if (end.length >= 1 && sharp.length >= 1 && junc.length >= 1) connect(junc[0]);
    else connectAvg();
  }
  // The reference asserts here (and would crash the run); treat it as "not refined" instead.
  if (update && added.length === 0) update = false;
  const usmCells = cells.slice();
  if (update) {
    for (const p of cells) {
      lx.mask[p] = 0;
      ly.mask[p] = 0;
    }
    for (const c of cps) {
      const p = c.i * lx.w + c.j;
      if (c.edge === 2) lx.mask[p] = 0;
      if (c.edge === 3) ly.mask[p] = 0;
      usmCells.push(p);
    }
  }
  return { applied: update, added, usmCells };
}

const BOX3 = [
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
];

/** Dual contouring result before topology refinement. Never mutated. */
export interface DCBase {
  edges: EdgeMap;
  pt: Float32Array;
  valence: Int32Array;
  ptMask: Mask;
  /** Line maps before surgery. */
  lx: LineMap;
  ly: LineMap;
}

export function dcBase(edges: EdgeMap, pt: Float32Array, lx: LineMap, ly: LineMap): DCBase {
  const { valence, ptMask } = valenceMap(edges);
  return { edges, pt, valence, ptMask, lx, ly };
}

/** A keypoint as the surgery uses it: a type, the grid cell it belongs to, and its position. */
export interface Mark {
  /** 1 end point, 2 sharp turn, 3 junction. */
  type: 1 | 2 | 3;
  cell: number;
  x: number;
  y: number;
}

/** Marks rasterized on the grid: type per cell (0 none; the last mark in a cell wins) and position. */
export interface MarkGrid {
  type: Int32Array;
  pos: Float64Array;
}

export function markGrid(marks: Mark[], n: number): MarkGrid {
  const type = new Int32Array(n);
  const pos = new Float64Array(2 * n);
  for (const m of marks) {
    type[m.cell] = m.type;
    pos[2 * m.cell] = m.x;
    pos[2 * m.cell + 1] = m.y;
  }
  return { type, pos };
}

/** Marks from a type grid whose keypoints sit on dual-contouring vertices (raster order). */
function marksFromTypes(ptType: Int32Array, pt: Float32Array): Mark[] {
  const marks: Mark[] = [];
  for (let p = 0; p < ptType.length; p++)
    if (ptType[p] > 0) marks.push({ type: ptType[p] as 1 | 2 | 3, cell: p, x: pt[2 * p], y: pt[2 * p + 1] });
  return marks;
}

/**
 * The reference's keypoint alignment in refine_topology: snap detected keypoints to vertices of the
 * matching valence. Returns the resulting marks and the aligned keypoint lists the reference returns.
 */
export function autoMarks(base: DCBase, detected: Keypoints): { marks: Mark[]; aligned: Keypoints } {
  const { pt, valence } = base;
  const ptType = new Int32Array(valence.length);
  const aligned: Keypoints = {
    end: alignKeypoints(pt, detected.end, valence, 1, ptType, 3.5),
    sharp: alignKeypoints(pt, detected.sharp, valence, 2, ptType, 3.5),
    junc: detected.junc,
  };
  // T/X junctions are sometimes valence-2 vertices: mark them too (result discarded, as in the reference).
  alignKeypoints(pt, detected.junc, valence, 2, ptType, 1.5);
  aligned.junc = alignKeypoints(pt, detected.junc, valence, 3, ptType, 3.5);
  return { marks: marksFromTypes(ptType, pt), aligned };
}

/**
 * Snap a user-placed keypoint to the nearest dual-contouring vertex within `radius` (input pixels).
 * Without one, the keypoint keeps its position and belongs to the cell that contains it.
 */
export function snapMark(base: DCBase, type: 1 | 2 | 3, x: number, y: number, radius = 1.5): Mark {
  const { h, w } = base.edges;
  const { pt, valence } = base;
  const ci = Math.floor(2 * y);
  const cj = Math.floor(2 * x);
  const r = Math.ceil(2 * radius) + 2;
  let best = radius;
  let cell = -1;
  for (let i = Math.max(0, ci - r); i <= Math.min(h - 1, ci + r); i++)
    for (let j = Math.max(0, cj - r); j <= Math.min(w - 1, cj + r); j++) {
      const p = i * w + j;
      if (!valence[p]) continue;
      const d = Math.hypot(pt[2 * p] - x, pt[2 * p + 1] - y);
      if (d <= best) {
        best = d;
        cell = p;
      }
    }
  if (cell >= 0) return { type, cell, x: pt[2 * cell], y: pt[2 * cell + 1] };
  const i = Math.min(h - 1, Math.max(0, ci));
  const j = Math.min(w - 1, Math.max(0, cj));
  return { type, cell: i * w + j, x, y };
}

/**
 * The reference's USM regions for the automatic pass: close the predicted USM (3x3 dilate, then
 * erode) and label its 4-connected components. Returns 0 (none), 1 (surgery is attempted) or
 * 2 (kept out of surgery: larger than long side / 50 cells, or closing merged several predicted
 * regions). Every nonzero cell keeps its fine geometry during downsampling.
 */
export function autoUsm(base: DCBase, usmPred: Mask): Uint8Array {
  const { h, w } = base.edges;
  const n = h * w;
  const longSide = Math.max(h, w);
  const regionsRaw = connectedComponents(usmPred, h, w, 4).labels;
  const usm = erode(dilate(usmPred, h, w, BOX3), h, w, BOX3);
  const { labels, count } = connectedComponents(usm, h, w, 4);
  const size = new Int32Array(count + 1);
  const raw: Set<number>[] = Array.from({ length: count + 1 }, () => new Set());
  for (let p = 0; p < n; p++) {
    if (!labels[p]) continue;
    size[labels[p]]++;
    if (regionsRaw[p]) raw[labels[p]].add(regionsRaw[p]);
  }
  const out = new Uint8Array(n);
  for (let p = 0; p < n; p++) {
    const l = labels[p];
    if (l) out[p] = size[l] > longSide / 50 || raw[l].size > 1 ? 2 : 1;
  }
  return out;
}

/** Per-region outcome of the surgery. */
export const RegionStatus = {
  /** No cell of the region is active (value 1): geometry is only kept fine during downsampling. */
  Inactive: 0,
  /** The region's border is not a single contour (it has holes), or no polyline enters it. */
  Invalid: 1,
  /** No rule applies with the keypoints inside (e.g. a junction region without a junction keypoint). */
  NotApplied: 2,
  Applied: 3,
} as const;

export interface SurgeryResult {
  /** Flat segments [sx, sy, ex, ey, ...] after downsampling, followed by the surgery's segments. */
  lines: number[];
  /** Added segments per refined region label. */
  refined: Map<number, number[]>;
  /** 4-connected components of the nonzero USM, labelled in raster order. */
  labels: Int32Array;
  /** Status per label (index 0 unused). */
  status: Uint8Array;
  /** As refine_topology returns them: region labels, extended by the connection cells when applied. */
  usmApplied: Int32Array;
  usmUncertain: Int32Array;
  /** Inputs handed to downsampleAll, for tests. */
  debug: { usmRegions: Mask; valence: Int32Array; ptType: Int32Array; lineMapX: LineMap; lineMapY: LineMap } | null;
}

/**
 * USM surgery (update_line_maps for each region) followed by dual contouring downsampling.
 * `usm` holds 0 / 1 / 2 per cell (see autoUsm); a region gets surgery if any of its cells is 1.
 * Pure: the base is not modified. With autoUsm and autoMarks this is refine_topology(...,
 * downsample=True, manual=False, full_auto_mode=True); with user-edited marks and USM it is the
 * interactive refinement.
 */
export function surgery(base: DCBase, marks: Mark[], usm: Uint8Array, downRate: number, keepDebug = false): SurgeryResult {
  const { h, w } = base.edges;
  const n = h * w;
  const { pt, valence, ptMask } = base;
  const lx = cloneLineMap(base.lx);
  const ly = cloneLineMap(base.ly);
  const grid = markGrid(marks, n);
  const { labels, count } = connectedComponents(usm, h, w, 4);
  const status = new Uint8Array(count + 1);
  const usmApplied = new Int32Array(n);
  const usmUncertain = new Int32Array(n);
  const refined = new Map<number, number[]>();
  const usmRegions = new Uint8Array(n);
  for (let p = 0; p < n; p++) usmRegions[p] = usm[p] ? 1 : 0;

  const cells: number[][] = Array.from({ length: count + 1 }, () => []);
  const active = new Uint8Array(count + 1);
  for (let p = 0; p < n; p++) {
    if (!labels[p]) continue;
    cells[labels[p]].push(p);
    if (usm[p] === 1) active[labels[p]] = 1;
  }
  const hasVertex = new Uint8Array(n);
  for (let p = 0; p < n; p++) hasVertex[p] = valence[p] !== 0 ? 1 : 0;

  const refinedLines: number[] = [];
  for (let idx = 1; idx <= count; idx++) {
    if (!active[idx]) continue;
    const rc = cells[idx];
    status[idx] = RegionStatus.Invalid;
    const cps = findConnectionPoints(hasVertex, pt, rc, usmRegions, h, w);
    if (cps === null) continue;
    const r = updateLineMaps(lx, ly, pt, ptMask, grid, cps, rc);
    if (r === null) continue;
    status[idx] = r.applied ? RegionStatus.Applied : RegionStatus.NotApplied;
    const target = r.applied ? usmApplied : usmUncertain;
    for (const p of r.usmCells) target[p] = idx;
    if (r.applied) {
      const flat = r.added.flat();
      refined.set(idx, flat);
      refinedLines.push(...flat);
    }
  }

  const debug = keepDebug ? { usmRegions, valence, ptType: grid.type.slice(), lineMapX: cloneLineMap(lx), lineMapY: cloneLineMap(ly) } : null;
  const lines = count === 0 ? lineMapsToSegments(lx, ly) : downsampleAll(base.edges, pt, usmRegions, valence, grid.type, lx, ly, downRate);
  lines.push(...refinedLines);
  return { lines, refined, labels, status, usmApplied, usmUncertain, debug };
}

/** Keypoint lists (flat x, y) from marks, in mark order. */
export function marksToKeypoints(marks: Mark[]): Keypoints {
  const k: Keypoints = { end: [], sharp: [], junc: [] };
  for (const m of marks) (m.type === 1 ? k.end : m.type === 2 ? k.sharp : k.junc).push(m.x, m.y);
  return k;
}
