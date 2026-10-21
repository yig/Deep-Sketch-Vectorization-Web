// End to end: exported ONNX models (onnxruntime-web, WASM backend) with tiling, then the TypeScript
// post-processing, against the PyTorch reference fixtures. Needs public/models (tools/export_onnx.py)
// and tests/fixtures (tools/make_fixtures.py).
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import * as ort from "onnxruntime-web";
import { readNpy } from "./npy";
import { runNdcTiled, runUdfTiled, UDF_HALO, NDC_HALO, type RunFn } from "../src/inference/tiling";
import { vectorizeFromNetworks } from "../src/pipeline/vectorize";
import { edgeMapFromLogits, vertexMapFromOffsets } from "../src/pipeline/dualContouring";

const ROOT = join(__dirname, "fixtures");
const MODELS = join(__dirname, "..", "public", "models");
const fixtures = existsSync(ROOT) ? readdirSync(ROOT).filter((d) => existsSync(join(ROOT, d, "meta.json"))) : [];
const ready = fixtures.length > 0 && existsSync(join(MODELS, "udf_light.onnx"));

async function session(name: string): Promise<RunFn> {
  const s = await ort.InferenceSession.create(readFileSync(join(MODELS, name)), { executionProviders: ["wasm"] });
  return async (input, h, w) => {
    const out = await s.run({ [s.inputNames[0]]: new ort.Tensor("float32", input, [1, 1, h, w]) });
    return s.outputNames.map((n) => out[n].data as Float32Array);
  };
}

const maxAbs = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};

describe.skipIf(!ready)("ONNX + tiling + post-processing", () => {
  test.each(fixtures)("%s", { timeout: 600_000 }, async (name) => {
    const dir = join(ROOT, name);
    const meta = JSON.parse(readFileSync(join(dir, "meta.json"), "utf8"));
    const { h: H, w: W, variant } = meta;
    const img = readNpy(join(dir, "image.npy")).data as Float32Array;

    const udfRun = await session(`udf_${variant}.onnx`);
    const { udf, aux } = await runUdfTiled(udfRun, img, H, W, { tile: 128, halo: UDF_HALO });
    expect(maxAbs(udf, readNpy(join(dir, "udf.npy")).data as Float32Array)).toBeLessThan(1e-3);
    expect(maxAbs(aux, readNpy(join(dir, "aux.npy")).data as Float32Array)).toBeLessThan(1e-3);

    const ndcRun = await session(`ndc_${variant}.onnx`);
    const { edge, offset } = await runNdcTiled(ndcRun, udf, 2 * H, 2 * W, { tile: 256, halo: NDC_HALO });
    const refEdge = readNpy(join(dir, "edge_raw.npy")).data as Uint8Array;
    const e = edgeMapFromLogits(edge, 2 * H, 2 * W);
    let flips = 0;
    for (let p = 0; p < 4 * H * W; p++) flips += Number(e.x[p] !== refEdge[2 * p]) + Number(e.y[p] !== refEdge[2 * p + 1]);
    // ONNX vs PyTorch float differences may flip a handful of near-tied edge flags.
    expect(flips).toBeLessThan(10);
    const pt = vertexMapFromOffsets(offset, 2 * H, 2 * W);
    expect(maxAbs(pt, readNpy(join(dir, "pt_map.npy")).data as Float32Array)).toBeLessThan(1e-3);

    const res = vectorizeFromNetworks({ H, W, aux, edge, offset }, { downRate: meta.down_rate, skipLen: meta.skip_len });
    const ref = readNpy(join(dir, "grouping_in.npy")).data as Float64Array;
    if (flips === 0) {
      expect(res.segments.length).toBe(ref.length);
      expect(maxAbs(res.segments, Array.from(ref, (v) => v / 2))).toBeLessThan(1e-3);
    } else {
      expect(Math.abs(res.segments.length - ref.length) / ref.length).toBeLessThan(0.01);
    }
    expect(res.strokes.length).toBeGreaterThan(0);
  });

  test("vertex map from offsets", () => {
    for (const name of fixtures) {
      const dir = join(ROOT, name);
      const { h, w } = JSON.parse(readFileSync(join(dir, "meta.json"), "utf8"));
      const pt = vertexMapFromOffsets(readNpy(join(dir, "offset.npy")).data as Float32Array, 2 * h, 2 * w);
      expect(maxAbs(pt, readNpy(join(dir, "pt_map.npy")).data as Float32Array)).toBe(0);
    }
  });
});
