// Stroke grouping: utils/svg_tools.py simplify_graph(mode='hybird').
// Chains the dual-contouring segments into long strokes, walking through junctions along the
// straightest continuation.
//
// Deliberate deviation: the reference picks start points with random.sample and iterates Python
// sets in hash order. Here every choice is deterministic: the smallest point index is taken first.
// Point indices come from sorting the unique points lexicographically (as np.unique does).

import type { Keypoints } from "./keypoints";

export type Polyline = number[]; // flat [x0, y0, x1, y1, ...]

class Graph {
  pts: Float64Array;
  nbrs: number[][];
  constructor(segments: ArrayLike<number>) {
    const m = segments.length / 4;
    const idx = Array.from({ length: 2 * m }, (_, k) => k);
    const P = (k: number, c: number) => segments[2 * k + c];
    idx.sort((a, b) => P(a, 0) - P(b, 0) || P(a, 1) - P(b, 1));
    const inv = new Int32Array(2 * m);
    const pts: number[] = [];
    let u = -1;
    for (let r = 0; r < idx.length; r++) {
      const k = idx[r];
      if (r === 0 || P(k, 0) !== P(idx[r - 1], 0) || P(k, 1) !== P(idx[r - 1], 1)) {
        u++;
        pts.push(P(k, 0), P(k, 1));
      }
      inv[k] = u;
    }
    this.pts = Float64Array.from(pts);
    const sets: Set<number>[] = Array.from({ length: u + 1 }, () => new Set());
    for (let s = 0; s < m; s++) {
      const a = inv[2 * s];
      const b = inv[2 * s + 1];
      sets[a].add(b);
      sets[b].add(a);
    }
    this.nbrs = sets.map((s) => [...s].sort((a, b) => a - b));
  }
  get size() {
    return this.nbrs.length;
  }
}

const sortedAsc = (s: Iterable<number>) => [...s].sort((a, b) => a - b);
const popMin = (s: Set<number>) => {
  const v = Math.min(...s);
  s.delete(v);
  return v;
};

/** find_alignment: mark vertices (of the given valences) nearest to each keypoint within thr. */
function alignTo(g: Graph, valence: Int32Array, keep: (v: number) => boolean, keypts: number[], val: number, thr: number, ptType: Int32Array) {
  const cand: number[] = [];
  for (let p = 0; p < g.size; p++) if (keep(valence[p])) cand.push(p);
  if (cand.length === 0 || keypts.length === 0) return;
  for (let k = 0; k < keypts.length / 2; k++) {
    let best = Infinity;
    let bi = -1;
    for (const c of cand) {
      const d = Math.hypot(g.pts[2 * c] - keypts[2 * k], g.pts[2 * c + 1] - keypts[2 * k + 1]);
      if (d < best) {
        best = d;
        bi = c;
      }
    }
    if (best <= thr) ptType[bi] = val;
  }
}

/**
 * Group segments (flat [sx, sy, ex, ey, ...], zero-length segments already removed) into polylines.
 * `keypoints` must be in the same units as the segments.
 */
export function groupStrokes(segments: ArrayLike<number>, keypoints: Keypoints | null, skipLen = 4): Polyline[] {
  const g = new Graph(segments);
  const n = g.size;
  const valence = new Int32Array(n);
  for (let p = 0; p < n; p++) valence[p] = g.nbrs[p].length;
  const ptType = new Int32Array(n).fill(-1);
  if (keypoints) {
    const v0 = valence.slice();
    alignTo(g, v0, (v) => v === 1, keypoints.end, 1, 1.5, ptType);
    alignTo(g, v0, (v) => v === 2, keypoints.sharp, 2, 1.5, ptType);
    alignTo(g, v0, (v) => v >= 3, keypoints.junc, 3, 5, ptType);
  }
  const unvisited = new Set<number>();
  for (let p = 0; p < n; p++) unvisited.add(p);
  const pt = (k: number) => [g.pts[2 * k], g.pts[2 * k + 1]] as const;

  // A stroke is a sequence of vertex indices; segments are consecutive pairs.
  let stroke: number[] = [];

  const visit = (idx: number): boolean => {
    if (valence[idx] === 0) return false;
    valence[idx]--;
    if (valence[idx] === 0) unvisited.delete(idx);
    return true;
  };
  const removal = (idxs: Iterable<number>, val: Int32Array): Set<number> => {
    const out = new Set<number>();
    for (const i of idxs) {
      if (val[i] > 0) {
        val[i]--;
        if (val[i] === 0) out.add(i);
      }
    }
    return out;
  };
  const avail = (k: number, exclude: Set<number>) => {
    const s = new Set<number>();
    for (const q of g.nbrs[k]) if (unvisited.has(q) && !exclude.has(q)) s.add(q);
    return s;
  };

  type Hop = [number, number, boolean, boolean];

  const dfsHop = (cur: number, pre: number, preDfs: boolean): Hop => {
    const a = pt(stroke[stroke.length - 2]);
    const b = pt(stroke[stroke.length - 1]);
    const pl = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const preTan = [(b[0] - a[0]) / pl, (b[1] - a[1]) / pl];
    const next = avail(cur, new Set([pre]));
    if (next.size === 0) return [cur, pre, false, false];
    if (next.size === 1) {
      const npt = popMin(next);
      stroke.push(npt);
      visit(npt);
      return preDfs ? [npt, cur, false, false] : [npt, cur, true, true];
    }
    let maxDot = -2;
    let best = -1;
    const c = pt(cur);
    for (const i of sortedAsc(next)) {
      const q = pt(i);
      const l = Math.hypot(q[0] - c[0], q[1] - c[1]);
      const d = ((q[0] - c[0]) / l) * preTan[0] + ((q[1] - c[1]) / l) * preTan[1];
      if (maxDot < d) {
        maxDot = d;
        best = i;
      }
    }
    stroke.push(best);
    visit(best);
    return [best, cur, true, false];
  };

  const nextHop = (cur: number, pre: number, dfs: boolean, preDfs: boolean): Hop => {
    if (dfs) return visit(cur) ? dfsHop(cur, pre, preDfs) : [cur, pre, false, false];
    const val = valence.slice();
    const nodes: Set<number>[] = [new Set(), new Set(), new Set()];
    let wbrpt = new Set<number>();
    const lv1 = avail(cur, new Set([pre]));
    nodes[0] = lv1;
    if (lv1.size === 0) return [cur, pre, false, false];
    wbrpt = removal(sortedAsc(lv1), val);
    for (let k = 0; k < lv1.size; k++) for (const q of removal([cur], val)) wbrpt.add(q);
    const commit = () => {
      for (const q of wbrpt) unvisited.delete(q);
      valence.set(val);
    };
    const free = (q: number) => unvisited.has(q) && !wbrpt.has(q);
    // Level 2.
    const rmLv1: number[] = [];
    for (const p of sortedAsc(nodes[0])) {
      const nb = new Set<number>();
      for (const q of g.nbrs[p]) if (free(q) && q !== cur && q !== pre) nb.add(q);
      for (const q of removal(sortedAsc(nb), val)) wbrpt.add(q);
      if (nb.size === 0) {
        rmLv1.push(p);
        continue;
      }
      for (const q of sortedAsc(nb)) {
        for (const r of removal([p], val)) wbrpt.add(r);
        nodes[1].add(q);
      }
    }
    if (rmLv1.length > 0 && nodes[1].size > 0) for (const p of rmLv1) nodes[0].delete(p);
    const loop2 = [...nodes[0]].filter((q) => nodes[1].has(q));
    if (loop2.length === 1) {
      nodes[0] = new Set(loop2);
      nodes[1].delete(loop2[0]);
    }
    if (nodes[0].size === 1 && nodes[1].size === 1) {
      const n0 = popMin(nodes[0]);
      const n1 = popMin(nodes[1]);
      stroke.push(n0, n1);
      commit();
      return [n1, n0, false, false];
    }
    if (nodes[0].size === 1 && nodes[1].size === 0) {
      const n0 = popMin(nodes[0]);
      stroke.push(n0);
      commit();
      return [n0, cur, false, false];
    }
    if (nodes[0].size > 1 && nodes[1].size > 1) return visit(cur) ? dfsHop(cur, pre, preDfs) : [cur, pre, false, false];
    // Level 3.
    const rmLv2: number[] = [];
    for (const p of sortedAsc(nodes[1])) {
      const nb = new Set<number>();
      for (const q of g.nbrs[p]) if (free(q) && !nodes[0].has(q) && q !== cur && q !== pre) nb.add(q);
      for (const q of removal(sortedAsc(nb), val)) wbrpt.add(q);
      if (nb.size === 0) {
        rmLv2.push(p);
        continue;
      }
      for (const q of sortedAsc(nb)) {
        for (const r of removal([p], val)) wbrpt.add(r);
        nodes[2].add(q);
      }
    }
    if (rmLv2.length > 0 && nodes[2].size > 0) for (const p of rmLv2) nodes[1].delete(p);
    if (nodes[0].size >= 1 && nodes[1].size === 1 && nodes[2].size === 1) {
      const n0 = popMin(nodes[0]);
      const n1 = popMin(nodes[1]);
      const n2 = popMin(nodes[2]);
      stroke.push(n0, n1, n2);
      commit();
      return [n2, n1, false, false];
    }
    if (nodes[0].size > 1) return visit(cur) ? dfsHop(cur, pre, preDfs) : [cur, pre, false, false];
    if (nodes[0].size === 1 && (nodes[1].size > 1 || nodes[2].size > 1)) {
      const n0 = popMin(nodes[0]);
      stroke.push(n0);
      visit(n0);
      visit(cur);
      return [n0, cur, false, false];
    }
    return [cur, pre, false, false];
  };

  const walk = (cur: number, pre: number) => {
    let dfs = false;
    let preDfs = false;
    while (unvisited.size > 0) {
      const [c, p, d, pd] = nextHop(cur, pre, dfs, preDfs);
      dfs = d;
      preDfs = pd;
      if (c === cur && p === pre) break;
      cur = c;
      pre = p;
    }
  };

  const strokes: Polyline[] = [];
  while (unvisited.size > 0) {
    let sp = -1;
    for (const q of unvisited) if (ptType[q] === 1 && (sp < 0 || q < sp)) sp = q;
    if (sp < 0) sp = Math.min(...unvisited);
    const eps = avail(sp, new Set());
    visit(sp);
    const skipSp = eps.size === 1;
    if (eps.size === 0) {
      if (valence.every((v) => v === 0)) break;
      continue;
    }
    const ep = popMin(eps);
    visit(ep);
    stroke = [ep, sp];
    if (!skipSp) {
      walk(sp, ep);
      stroke.reverse();
    } else {
      stroke = [sp, ep];
    }
    walk(ep, sp);
    if (stroke.length - 1 > skipLen) {
      const line: Polyline = [];
      for (const k of stroke) line.push(g.pts[2 * k], g.pts[2 * k + 1]);
      strokes.push(line);
    }
  }
  return strokes;
}
