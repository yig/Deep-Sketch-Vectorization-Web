// End-to-end post-processing (predict_s1.py's automatic path) from the two networks' outputs.

import { edgeMapFromLogits, lineMaps, refineEdgeMap, vertexMapFromOffsets } from "./dualContouring";
import { extractKeypoints, underSamplingMap, type Keypoints } from "./keypoints";
import { refineTopology } from "./topology";
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
  /** Detected keypoints (input-pixel units), before alignment. */
  keypoints: Keypoints;
  /** Under-sampling map on the 2x grid as predicted, and the regions where surgery was applied. */
  usm: Uint8Array;
  usmApplied: Int32Array;
  /** Segments before stroke grouping, input-pixel units. */
  segments: number[];
  timings: Record<string, number>;
}

export function vectorizeFromNetworks(net: NetworkOutputs, options: Partial<VectorizeOptions> = {}): VectorizeResult {
  const opt = { ...DEFAULT_OPTIONS, ...options };
  const h = 2 * net.H;
  const w = 2 * net.W;
  const timings: Record<string, number> = {};
  let t = performance.now();
  const lap = (name: string) => {
    const now = performance.now();
    timings[name] = now - t;
    t = now;
  };

  const usm = underSamplingMap(net.aux, h, w);
  const keypoints = extractKeypoints(net.aux, h, w);
  lap("keypoints");
  const edges = refineEdgeMap(edgeMapFromLogits(net.edge, h, w));
  const pt = vertexMapFromOffsets(net.offset, h, w);
  const { x, y } = lineMaps(edges, pt);
  lap("dual contouring");
  const res = refineTopology(edges, pt, usm, x, y, keypoints, opt.downRate);
  lap("topology + downsampling");

  // The reference writes the segments to an SVG at 2x scale and groups them there; zero-length
  // segments are dropped on the way.
  const segs2x: number[] = [];
  for (let k = 0; k < res.lines.length; k += 4) {
    const [a, b, c, d] = [res.lines[k] * 2, res.lines[k + 1] * 2, res.lines[k + 2] * 2, res.lines[k + 3] * 2];
    if (a !== c || b !== d) segs2x.push(a, b, c, d);
  }
  // Keypoints in the same 2x units as the segments. (The reference passes 1x keypoints here; fixed.)
  const x2 = (a: number[]) => a.map((v) => v * 2);
  let strokes = groupStrokes(segs2x, { end: x2(res.keypoints.end), sharp: x2(res.keypoints.sharp), junc: x2(res.keypoints.junc) }, opt.skipLen);
  lap("grouping");
  let beziers: Cubic[][] | null = null;
  if (opt.smoothing === "rdp") strokes = strokes.map((s) => rdp(s, opt.rdpEpsilon));
  if (opt.smoothing === "bezier") beziers = strokes.map((s) => fitBezier(s).map((c) => c.map(([px, py]) => [px / 2, py / 2]) as Cubic));
  lap("smoothing");
  return {
    strokes: strokes.map((s) => s.map((v) => v / 2)),
    beziers,
    keypoints,
    usm,
    usmApplied: res.usmApplied,
    segments: segs2x.map((v) => v / 2),
    timings,
  };
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
