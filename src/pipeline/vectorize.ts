// End-to-end post-processing (predict_s1.py's automatic path) from the two networks' outputs.

import { edgeMapFromLogits, lineMaps, refineEdgeMap, vertexMapFromOffsets } from "./dualContouring";
import { extractKeypoints, underSamplingMap, type Keypoints } from "./keypoints";
import { autoMarks, autoUsm, dcBase, marksToKeypoints, surgery, type DCBase, type Mark, type RegionInfo } from "./topology";
import { groupStrokes, type Polyline } from "./grouping";
import { fitBezier, rdp, type Cubic } from "./smoothing";

export interface VectorizeOptions {
  /** Dual contouring downsampling rate on the 2x grid (must divide 2H and 2W). */
  downRate: number;
  /** Drop strokes with at most this many segments. */
  skipLen: number;
  smoothing: "none" | "rdp" | "bezier";
  rdpEpsilon: number;
}

export const DEFAULT_OPTIONS: VectorizeOptions = { downRate: 8, skipLen: 4, smoothing: "none", rdpEpsilon: 0.4 };

export interface NetworkOutputs {
  /** Input image size (pixels); the grids are 2H x 2W. */
  H: number;
  W: number;
  aux: Float32Array;
  edge: Float32Array;
  offset: Float32Array;
}

export interface VectorizeResult {
  /** Strokes in input-pixel units (the reference's 2x SVG coordinates halved). */
  strokes: Polyline[];
  /** Bézier fits per stroke when smoothing = "bezier". */
  beziers: Cubic[][] | null;
  /** Keypoints used (input-pixel units). */
  marks: Mark[];
  /** USM used: 0 none, 1 surgery attempted, 2 detail kept only (see autoUsm). */
  usm: Uint8Array;
  /** Region label per cell, and the surgery's outcome and explanation per label. */
  regionLabels: Int32Array;
  regionStatus: Uint8Array;
  regionInfo: RegionInfo[];
  /** Segments before stroke grouping, input-pixel units. */
  segments: number[];
  timings: Record<string, number>;
}

/** Everything that depends only on the network outputs: computed once per image. */
export interface Prepared {
  h: number;
  w: number;
  base: DCBase;
  /** Keypoints detected by the network (input-pixel units). */
  detected: Keypoints;
  /** The automatic pass's marks and USM: the starting point for editing. */
  autoMarks: Mark[];
  autoUsm: Uint8Array;
  timings: Record<string, number>;
}

function stopwatch(timings: Record<string, number>) {
  let t = performance.now();
  return (name: string) => {
    const now = performance.now();
    timings[name] = now - t;
    t = now;
  };
}

export function prepare(net: NetworkOutputs): Prepared {
  const h = 2 * net.H;
  const w = 2 * net.W;
  const timings: Record<string, number> = {};
  const lap = stopwatch(timings);
  const usmPred = underSamplingMap(net.aux, h, w);
  const detected = extractKeypoints(net.aux, h, w);
  lap("keypoints");
  const edges = refineEdgeMap(edgeMapFromLogits(net.edge, h, w));
  const pt = vertexMapFromOffsets(net.offset, h, w);
  const { x, y } = lineMaps(edges, pt);
  const base = dcBase(edges, pt, x, y);
  lap("dual contouring");
  return { h, w, base, detected, autoMarks: autoMarks(base, detected).marks, autoUsm: autoUsm(base, usmPred), timings };
}

/** Topology refinement, downsampling, grouping and smoothing for given marks and USM (fast). */
export function finish(prep: Prepared, marks: Mark[], usm: Uint8Array, options: Partial<VectorizeOptions> = {}): VectorizeResult {
  const opt = { ...DEFAULT_OPTIONS, ...options };
  const timings: Record<string, number> = {};
  const lap = stopwatch(timings);
  const res = surgery(prep.base, marks, usm, opt.downRate);
  lap("topology + downsampling");

  // The reference writes the segments to an SVG at 2x scale and groups them there; zero-length
  // segments are dropped on the way.
  const segs2x: number[] = [];
  for (let k = 0; k < res.lines.length; k += 4) {
    const [a, b, c, d] = [res.lines[k] * 2, res.lines[k + 1] * 2, res.lines[k + 2] * 2, res.lines[k + 3] * 2];
    if (a !== c || b !== d) segs2x.push(a, b, c, d);
  }
  // Keypoints in the same 2x units as the segments. (The reference passes 1x keypoints here; fixed.)
  const kp = marksToKeypoints(marks);
  const x2 = (a: number[]) => a.map((v) => v * 2);
  let strokes = groupStrokes(segs2x, { end: x2(kp.end), sharp: x2(kp.sharp), junc: x2(kp.junc) }, opt.skipLen);
  lap("grouping");
  let beziers: Cubic[][] | null = null;
  if (opt.smoothing === "rdp") strokes = strokes.map((s) => rdp(s, opt.rdpEpsilon));
  if (opt.smoothing === "bezier") beziers = strokes.map((s) => fitBezier(s).map((c) => c.map(([px, py]) => [px / 2, py / 2]) as Cubic));
  lap("smoothing");
  return {
    strokes: strokes.map((s) => s.map((v) => v / 2)),
    beziers,
    marks,
    usm,
    regionLabels: res.labels,
    regionStatus: res.status,
    regionInfo: res.info,
    segments: segs2x.map((v) => v / 2),
    timings,
  };
}

/** The automatic pipeline (predict_s1.py) from the network outputs. */
export function vectorizeFromNetworks(net: NetworkOutputs, options: Partial<VectorizeOptions> = {}): VectorizeResult {
  const prep = prepare(net);
  const res = finish(prep, prep.autoMarks, prep.autoUsm, options);
  res.timings = { ...prep.timings, ...res.timings };
  return res;
}

const fmt = (v: number) => (Math.round(v * 1000) / 1000).toString();

/** SVG document. `scale` maps input-pixel units to output units (e.g. back to the original image size). */
export function toSvg(result: VectorizeResult, width: number, height: number, scale = 1, strokeWidth = 1): string {
  const paths: string[] = [];
  if (result.beziers) {
    for (const cubics of result.beziers) {
      if (cubics.length === 0) continue;
      let d = `M${fmt(cubics[0][0][0] * scale)},${fmt(cubics[0][0][1] * scale)}`;
      for (const [, c1, c2, p] of cubics) d += `C${[c1, c2, p].map(([x, y]) => `${fmt(x * scale)},${fmt(y * scale)}`).join(" ")}`;
      paths.push(d);
    }
  } else {
    for (const s of result.strokes) {
      let d = "";
      for (let k = 0; k < s.length; k += 2) d += `${k === 0 ? "M" : "L"}${fmt(s[k] * scale)},${fmt(s[k + 1] * scale)}`;
      paths.push(d);
    }
  }
  const body = paths.map((d) => `  <path d="${d}"/>`).join("\n");
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${fmt(width)}" height="${fmt(height)}" viewBox="0 0 ${fmt(width)} ${fmt(height)}">
<g fill="none" stroke="#000" stroke-width="${strokeWidth}" stroke-linecap="round" stroke-linejoin="round">
${body}
</g>
</svg>
`;
}
