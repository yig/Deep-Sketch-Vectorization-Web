// Pre-processing (line thinning, Anime2Sketch) through onnxruntime-web (WASM, Node) against the
// PyTorch reference (tools/make_preprocess_fixtures.py). Run with `npm run test:e2e`.
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import * as ort from "onnxruntime-web";
import { readNpy } from "./npy";
import { runSameSizeTiled, THIN_HALO } from "../src/inference/tiling";
import { anime2sketch, thin, type Net } from "../src/pipeline/preprocess";
import { prepareImage } from "../src/pipeline/image";

const ROOT = join(__dirname, "fixtures", "preprocess");
const MODELS = join(__dirname, "..", "public", "models");
const images = existsSync(ROOT) ? readdirSync(ROOT).filter((d) => existsSync(join(ROOT, d, "rgb.npy"))) : [];
const ready = images.length > 0 && existsSync(join(MODELS, "thin.onnx")) && existsSync(join(MODELS, "anime2sketch.onnx"));

async function net(name: string, channels: number): Promise<Net> {
  const s = await ort.InferenceSession.create(readFileSync(join(MODELS, name)), { executionProviders: ["wasm"] });
  return async (x, h, w) => (await s.run({ [s.inputNames[0]]: new ort.Tensor("float32", x, [1, channels, h, w]) }))[s.outputNames[0]].data as Float32Array;
}

const diffs = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let n = 0;
  let max = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d) n++;
    max = Math.max(max, d);
  }
  return { n, max, frac: n / a.length };
};

describe.skipIf(!ready)("pre-processing networks", () => {
  test.each(images)("%s", { timeout: 600_000 }, async (d) => {
    const dir = join(ROOT, d);
    const rgbN = readNpy(join(dir, "rgb.npy"));
    const [h, w] = rgbN.shape;
    const rgb = rgbN.data as Uint8Array;
    const gray = readNpy(join(dir, "gray.npy")).data as Uint8Array;
    const thinNet = await net("thin.onnx", 1);
    // Small tiles to exercise the tiling.
    const tiled = (x: Float32Array, hh: number, ww: number) =>
      runSameSizeTiled(async (t, th, tw) => [await thinNet(t, th, tw)], x, hh, ww, { tile: 96, halo: THIN_HALO });

    // Tiling is exact: tiled == whole-image output.
    const x = Float32Array.from(gray, (v) => v / 255);
    const whole = await thinNet(x, h, w);
    expect(diffs(await tiled(x, h, w), whole).max).toBeLessThan(1e-5);

    // Thinning: the near-white background comes out of the sigmoid at ~0.99999, so x * 255 then
    // truncation sits on the 254 / 255 boundary and ONNX vs PyTorch float noise flips it.
    const t = await thin(gray, w, h, tiled);
    const dt = diffs(t, readNpy(join(dir, "thin.npy")).data as Uint8Array);
    expect(dt.max).toBeLessThanOrEqual(1);
    expect(dt.frac).toBeLessThan(0.05);

    // Anime2Sketch with fp32 weights reproduces the reference; fp16-stored weights stay close.
    const refA = readNpy(join(dir, "a2s.npy")).data as Uint8Array;
    const a32 = await anime2sketch(rgb, w, h, await net("anime2sketch.onnx", 3));
    const d32 = diffs(a32, refA);
    expect(d32.max).toBeLessThanOrEqual(1);
    expect(d32.frac).toBeLessThan(0.01);
    const a16 = await anime2sketch(rgb, w, h, await net("anime2sketch.w16.onnx", 3));
    const d16 = diffs(a16, refA);
    expect(d16.max).toBeLessThanOrEqual(3);
    expect(d16.frac).toBeLessThan(0.05);

    // Both, on the reference's Anime2Sketch output (thinning amplifies its one-level differences
    // locally), and the final network input load_img produces for each mode.
    const both = await thin(refA, w, h, tiled);
    expect(diffs(both, readNpy(join(dir, "both.npy")).data as Uint8Array).max).toBeLessThanOrEqual(1);
    for (const [mode, img] of [["none", gray], ["thin", t], ["a2s", refA], ["both", both]] as const) {
      const ref = readNpy(join(dir, `load_${mode}.npy`)).data as Float32Array;
      const got = prepareImage(img, w, h, Math.max(w, h)).data;
      expect(got.length).toBe(ref.length);
      expect(diffs(got, ref).max).toBeLessThanOrEqual(1 / 255 + 1e-6);
    }
  });
});
