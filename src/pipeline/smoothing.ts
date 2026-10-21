// Optional stroke smoothing: Ramer–Douglas–Peucker (the `rdp` package, algo="iter") and
// Schneider's Bézier fitting (utils/fitCurves.py, "Graphics Gems", 1990).

import type { Polyline } from "./grouping";

/** flatten_paths_to_pts_array(unique=True) for a polyline: drop consecutive duplicate points. */
function dedupe(line: Polyline): number[][] {
  const pts: number[][] = [];
  for (let k = 0; k < line.length / 2; k++) {
    const p = [line[2 * k], line[2 * k + 1]];
    const q = pts[pts.length - 1];
    if (q && q[0] === p[0] && q[1] === p[1]) continue;
    pts.push(p);
  }
  return pts;
}

function pldist(p: number[], s: number[], e: number[]): number {
  if (s[0] === e[0] && s[1] === e[1]) return Math.hypot(p[0] - s[0], p[1] - s[1]);
  const cross = (e[0] - s[0]) * (s[1] - p[1]) - (e[1] - s[1]) * (s[0] - p[0]);
  return Math.abs(cross) / Math.hypot(e[0] - s[0], e[1] - s[1]);
}

export function rdp(line: Polyline, epsilon: number): Polyline {
  const M = dedupe(line);
  const keep = new Uint8Array(M.length).fill(1);
  const stack: [number, number][] = [[0, M.length - 1]];
  while (stack.length) {
    const [s, e] = stack.pop()!;
    let dmax = 0;
    let index = s;
    for (let i = s + 1; i < e; i++) {
      if (!keep[i]) continue;
      const d = pldist(M[i], M[s], M[e]);
      if (d > dmax) {
        index = i;
        dmax = d;
      }
    }
    if (dmax > epsilon) {
      stack.push([s, index], [index, e]);
    } else {
      for (let i = s + 1; i < e; i++) keep[i] = 0;
    }
  }
  const out: Polyline = [];
  M.forEach((p, i) => keep[i] && out.push(p[0], p[1]));
  return out;
}

type V = [number, number];
/** A cubic Bézier: [p0, c1, c2, p3]. */
export type Cubic = [V, V, V, V];

const sub = (a: V, b: V): V => [a[0] - b[0], a[1] - b[1]];
const add = (a: V, b: V): V => [a[0] + b[0], a[1] + b[1]];
const mul = (a: V, s: number): V => [a[0] * s, a[1] * s];
const dot = (a: V, b: V) => a[0] * b[0] + a[1] * b[1];
// Same operation order as NumPy (linalg.norm = sqrt(dot); vector * scalar * scalar left to right).
const norm = (a: V) => Math.sqrt(a[0] * a[0] + a[1] * a[1]);
const normalize = (a: V): V => {
  const l = norm(a);
  return [a[0] / l, a[1] / l];
};

function q(c: Cubic, t: number): V {
  const s = 1 - t;
  return add(add(add(mul(c[0], s ** 3), mul(c[1], 3 * s ** 2 * t)), mul(c[2], 3 * s * t ** 2)), mul(c[3], t ** 3));
}
function qprime(c: Cubic, t: number): V {
  const s = 1 - t;
  return add(add(mul(sub(c[1], c[0]), 3 * s ** 2), mul(sub(c[2], c[1]), 6 * s * t)), mul(sub(c[3], c[2]), 3 * t ** 2));
}
function qprimeprime(c: Cubic, t: number): V {
  return add(mul(add(sub(c[2], mul(c[1], 2)), c[0]), 6 * (1 - t)), mul(add(sub(c[3], mul(c[2], 2)), c[1]), 6 * t));
}

function generateBezier(points: V[], u: number[], lt: V, rt: V): Cubic {
  const C = [
    [0, 0],
    [0, 0],
  ];
  const X = [0, 0];
  const first = points[0];
  const last = points[points.length - 1];
  for (let i = 0; i < points.length; i++) {
    const a0 = mul(mul(mul(lt, 3), (1 - u[i]) ** 2), u[i]);
    const a1 = mul(mul(mul(rt, 3), 1 - u[i]), u[i] ** 2);
    C[0][0] += dot(a0, a0);
    C[0][1] += dot(a0, a1);
    C[1][0] += dot(a0, a1);
    C[1][1] += dot(a1, a1);
    const tmp = sub(points[i], q([first, first, last, last], u[i]));
    X[0] += dot(a0, tmp);
    X[1] += dot(a1, tmp);
  }
  const det = C[0][0] * C[1][1] - C[1][0] * C[0][1];
  const detC0X = C[0][0] * X[1] - C[1][0] * X[0];
  const detXC1 = X[0] * C[1][1] - X[1] * C[0][1];
  const al = det === 0 ? 0 : detXC1 / det;
  const ar = det === 0 ? 0 : detC0X / det;
  const segLength = norm(sub(first, last));
  const eps = 1e-6 * segLength;
  if (al < eps || ar < eps) return [first, add(first, mul(lt, segLength / 3)), add(last, mul(rt, segLength / 3)), last];
  return [first, add(first, mul(lt, al)), add(last, mul(rt, ar)), last];
}

function chordLength(points: V[]): number[] {
  const u = [0];
  for (let i = 1; i < points.length; i++) u.push(u[i - 1] + norm(sub(points[i], points[i - 1])));
  const total = u[u.length - 1];
  return u.map((v) => v / total);
}

function maxError(points: V[], bez: Cubic, u: number[]): [number, number] {
  let maxDist = 0;
  let split = points.length / 2;
  for (let i = 0; i < points.length; i++) {
    const d = norm(sub(q(bez, u[i]), points[i])) ** 2;
    if (d > maxDist) {
      maxDist = d;
      split = i;
    }
  }
  return [maxDist, split];
}

function newtonRaphson(bez: Cubic, p: V, u: number): number {
  const d = sub(q(bez, u), p);
  const qp = qprime(bez, u);
  const qpp = qprimeprime(bez, u);
  const num = d[0] * qp[0] + d[1] * qp[1];
  const den = qp[0] ** 2 + qp[1] ** 2 + d[0] * qpp[0] + d[1] * qpp[1];
  return den === 0 ? u : u - num / den;
}

function fitCubic(points: V[], lt: V, rt: V, error: number): Cubic[] {
  if (points.length === 2) {
    const dist = norm(sub(points[0], points[1])) / 3;
    return [[points[0], add(points[0], mul(lt, dist)), add(points[1], mul(rt, dist)), points[1]]];
  }
  let u = chordLength(points);
  let bez = generateBezier(points, u, lt, rt);
  let [err, split] = maxError(points, bez, u);
  if (err < error) return [bez];
  if (err < error ** 2) {
    for (let k = 0; k < 20; k++) {
      const uPrime = points.map((p, i) => newtonRaphson(bez, p, u[i]));
      bez = generateBezier(points, uPrime, lt, rt);
      [err, split] = maxError(points, bez, uPrime);
      if (err < error) return [bez];
      u = uPrime;
    }
  }
  const center = normalize(sub(points[split - 1], points[split + 1]));
  return [
    ...fitCubic(points.slice(0, split + 1), lt, center, error),
    ...fitCubic(points.slice(split), mul(center, -1), rt, error),
  ];
}

/** fitCurve(points, maxError) on a polyline (as fitBezier would, were it not broken in the reference). */
export function fitBezier(line: Polyline, error = 0.1): Cubic[] {
  const pts: V[] = [];
  for (let k = 0; k < line.length / 2; k++) pts.push([line[2 * k], line[2 * k + 1]]);
  const lt = normalize(sub(pts[1], pts[0]));
  const rt = normalize(sub(pts[pts.length - 2], pts[pts.length - 1]));
  return fitCubic(pts, lt, rt, error);
}
