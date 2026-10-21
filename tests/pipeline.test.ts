// Golden tests: each stage of the TypeScript post-processing against the reference Python
// pipeline's intermediates (generate with `cd tools && uv run python make_fixtures.py IMAGE...`).
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { readNpy } from "./npy";
import {
  cutBranch,
  fillEdgeHole,
  lineMaps,
  lineMapsToSegments,
  refineEdgeMap,
  removeParallel,
  edgeMapFromLogits,
  type EdgeMap,
  type LineMap,
} from "../src/pipeline/dualContouring";
import { extractKeypoints, udfFilter, underSamplingMap, type Keypoints } from "../src/pipeline/keypoints";
import { refineTopology } from "../src/pipeline/topology";
import { downsampleAll } from "../src/pipeline/downsample";
import { groupStrokes } from "../src/pipeline/grouping";
import { fitBezier, rdp } from "../src/pipeline/smoothing";

const ROOT = join(__dirname, "fixtures");
const fixtures = existsSync(ROOT) ? readdirSync(ROOT).filter((d) => existsSync(join(ROOT, d, "meta.json"))) : [];

type Num = ArrayLike<number> | BigInt64Array;
const mismatches = (a: Num, b: Num) => {
  let n = 0;
  for (let i = 0; i < a.length; i++) if (Number(a[i]) !== Number(b[i])) n++;
  return n;
};
const maxAbs = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};

describe.skipIf(fixtures.length === 0)("golden fixtures", () => {
  describe.each(fixtures)("%s", (name) => {
    const dir = join(ROOT, name);
    const npy = (f: string) => readNpy(join(dir, f + ".npy"));
    const json = (f: string) => JSON.parse(readFileSync(join(dir, f + ".json"), "utf8"));
    const meta = json("meta");
    const H = 2 * meta.h;
    const W = 2 * meta.w;
    const edges = (f: string): EdgeMap => {
      const d = npy(f).data as Uint8Array;
      const x = new Uint8Array(H * W);
      const y = new Uint8Array(H * W);
      for (let p = 0; p < H * W; p++) {
        x[p] = d[2 * p];
        y[p] = d[2 * p + 1];
      }
      return { h: H, w: W, x, y };
    };
    const lineMap = (f: string): LineMap => ({
      h: H,
      w: W,
      mask: npy(f + "_mask").data as Uint8Array<ArrayBuffer>,
      seg: Float64Array.from(npy(f).data as Float32Array),
    });
    const kp = (o: Record<string, number[][]>, scale = 1): Keypoints => ({
      end: o.end_point.flat().map((v) => v * scale),
      sharp: o.sharp_turn.flat().map((v) => v * scale),
      junc: o.junc.flat().map((v) => v * scale),
    });
    const ptMap = () => npy("pt_map").data as Float32Array;

    test("USM and keypoints", () => {
      const aux = npy("aux").data as Float32Array;
      expect(mismatches(underSamplingMap(aux, H, W), npy("usm_pre").data)).toBe(0);
      const filtered = npy("udf_filtered").data as Float32Array;
      for (let c = 0; c < 4; c++) {
        const got = udfFilter(aux.subarray(c * H * W, (c + 1) * H * W), H, W);
        expect(mismatches(got, filtered.subarray(c * H * W, (c + 1) * H * W))).toBe(0);
      }
      const k = extractKeypoints(aux, H, W);
      const ref = kp(json("keypts_2x"), 0.5);
      expect(k.end).toEqual(ref.end);
      expect(k.sharp).toEqual(ref.sharp);
      expect(k.junc).toEqual(ref.junc);
    });

    test("edge flags and cleanup", () => {
      const logits = npy("edge_logits").data as Float32Array;
      const raw = edgeMapFromLogits(logits, H, W);
      const refRaw = edges("edge_raw");
      expect(mismatches(raw.x, refRaw.x) + mismatches(raw.y, refRaw.y)).toBe(0);
      const cut = cutBranch(refRaw);
      const refCut = edges("edge_cut1");
      expect(mismatches(cut.x, refCut.x) + mismatches(cut.y, refCut.y)).toBe(0);
      const occ = refCut.x.map((v, p) => v | refCut.y[p]);
      expect(mismatches(removeParallel(occ, refCut), npy("skel_parallel").data)).toBe(0);
      const skel = npy("skel_parallel").data as Uint8Array;
      const masked: EdgeMap = { h: H, w: W, x: refCut.x.map((v, p) => v & skel[p]), y: refCut.y.map((v, p) => v & skel[p]) };
      const filled = fillEdgeHole(masked);
      const refFilled = edges("edge_filled");
      expect(mismatches(filled.x, refFilled.x) + mismatches(filled.y, refFilled.y)).toBe(0);
      const fin = refineEdgeMap(refRaw);
      const refFin = edges("edge_final");
      expect(mismatches(fin.x, refFin.x) + mismatches(fin.y, refFin.y)).toBe(0);
    });

    test("line maps", () => {
      const { x, y } = lineMaps(edges("edge_final"), ptMap());
      const rx = lineMap("linemap_x");
      const ry = lineMap("linemap_y");
      expect(mismatches(x.mask, rx.mask) + mismatches(y.mask, ry.mask)).toBe(0);
      expect(maxAbs(x.seg, rx.seg) + maxAbs(y.seg, ry.seg)).toBe(0);
      const segs = lineMapsToSegments(x, y);
      const ref = npy("lines_raw").data as Float64Array;
      expect(segs.length).toBe(ref.length);
      expect(maxAbs(segs, ref)).toBe(0);
    });

    test("topology refinement (USM surgery)", () => {
      const lx = lineMap("linemap_x");
      const ly = lineMap("linemap_y");
      const res = refineTopology(edges("edge_final"), ptMap(), npy("usm_pre").data as Uint8Array<ArrayBuffer>, lx, ly, kp(json("keypts_2x"), 0.5), meta.down_rate, true);
      const ka = kp(json("keypts_aligned"));
      expect(res.keypoints.end).toEqual(ka.end.map(Math.fround));
      expect(res.keypoints.sharp).toEqual(ka.sharp.map(Math.fround));
      expect(res.keypoints.junc).toEqual(ka.junc.map(Math.fround));
      const dbg = res.debug!;
      expect(mismatches(dbg.usmRegions, npy("down_usm_in").data)).toBe(0);
      expect(mismatches(dbg.valence, npy("valence_map").data)).toBe(0);
      expect(mismatches(dbg.ptType, npy("pt_type").data)).toBe(0);
      expect(mismatches(dbg.lineMapX.mask, npy("linemap_x_surgery_mask").data)).toBe(0);
      expect(mismatches(dbg.lineMapY.mask, npy("linemap_y_surgery_mask").data)).toBe(0);
      expect(mismatches(res.usmApplied, npy("usm_applied").data)).toBe(0);
      expect(mismatches(res.usmUncertain, npy("usm_uncertain").data)).toBe(0);
      const regions = json("refined_regions") as Record<string, number[][]>;
      expect([...res.refined.keys()].map(String)).toEqual(Object.keys(regions));
      for (const [k, v] of res.refined) expect(maxAbs(v, regions[String(k)].flat())).toBeLessThan(1e-9);
      const ref = npy("lines_refine").data as Float64Array;
      expect(res.lines.length).toBe(ref.length);
      expect(maxAbs(res.lines, ref)).toBeLessThan(1e-4);
    });

    test("dual contouring downsampling", () => {
      const lines = downsampleAll(
        edges("edge_final"),
        ptMap(),
        npy("down_usm_in").data as Uint8Array<ArrayBuffer>,
        npy("valence_map").data as Int32Array,
        npy("pt_type").data as Int32Array,
        lineMap("linemap_x_surgery"),
        lineMap("linemap_y_surgery"),
        meta.down_rate,
      );
      const ref = npy("lines_down").data as Float64Array;
      expect(lines.length).toBe(ref.length);
      expect(maxAbs(lines, ref)).toBeLessThan(1e-4);
    });

    test("stroke grouping (structural)", () => {
      const segs = Array.from(npy("grouping_in").data as Float64Array);
      const strokes = groupStrokes(segs, kp(json("keypts_aligned")), meta.skip_len);
      const ref = json("strokes") as number[][][][];
      const key = (ax: number, ay: number, bx: number, by: number) =>
        ax < bx || (ax === bx && ay < by) ? `${ax},${ay},${bx},${by}` : `${bx},${by},${ax},${ay}`;
      const input = new Set<string>();
      for (let s = 0; s < segs.length; s += 4) input.add(key(segs[s], segs[s + 1], segs[s + 2], segs[s + 3]));
      let covered = 0;
      for (const st of strokes) {
        for (let k = 0; k + 3 < st.length; k += 2) {
          expect(input.has(key(st[k], st[k + 1], st[k + 2], st[k + 3]))).toBe(true);
          covered++;
        }
      }
      const refCovered = ref.reduce((s, st) => s + st.length, 0);
      // Grouping order differs from the reference's random / hash-order choices; totals stay close.
      expect(Math.abs(covered - refCovered) / refCovered).toBeLessThan(0.05);
      expect(Math.abs(strokes.length - ref.length) / ref.length).toBeLessThan(0.25);
    });

    test("RDP and Bezier smoothing", () => {
      const ref = json("strokes") as number[][][][];
      const polys = ref.map((st) => [...st[0][0], ...st.flatMap((seg) => seg[1])]);
      const refRdp = json("strokes_rdp") as number[][][][];
      polys.forEach((p, i) => {
        const got = rdp(p, meta.eps);
        const want = [...refRdp[i][0][0], ...refRdp[i].flatMap((seg) => seg[1])];
        expect(got).toEqual(want);
      });
      const refBez = json("strokes_bezier") as number[][][][];
      polys.forEach((p, i) => {
        const got = fitBezier(p).flat(2);
        expect(got.length).toBe(refBez[i].flat(2).length);
        // Relative: near-degenerate fits put control points thousands of px away.
        const want = refBez[i].flat(2);
        got.forEach((v, k) => expect(Math.abs(v - want[k])).toBeLessThan(1e-6 * Math.max(1, Math.abs(want[k]))));
      });
    });
  });
});
