// The PIL and OpenCV resize ports against the reference libraries' outputs.
import { existsSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { readNpy } from "./npy";
import { cvResizeArea, pilResizeBicubic } from "../src/pipeline/resize";

const ROOT = join(__dirname, "fixtures", "preprocess");
const have = existsSync(join(ROOT, "resize", "src.npy"));

const diffs = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let n = 0;
  let max = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d) n++;
    max = Math.max(max, d);
  }
  return { n, max };
};

describe.skipIf(!have)("resize ports", () => {
  // Guarded: vitest runs a skipped describe's body while collecting tests.
  const src = have ? readNpy(join(ROOT, "resize", "src.npy")) : null!;
  const sizes = have ? readdirSync(join(ROOT, "resize")).filter((f) => /^\d+x\d+\.npy$/.test(f)) : [];

  test.each(sizes)("cv2 INTER_AREA 512x512 -> %s", (f) => {
    const [w, h] = f.replace(".npy", "").split("x").map(Number);
    const ref = readNpy(join(ROOT, "resize", f)).data as Uint8Array;
    const got = cvResizeArea(src.data as Uint8Array, 512, 512, 3, w, h);
    expect(got.length).toBe(ref.length);
    expect(diffs(got, ref)).toEqual({ n: 0, max: 0 });
  });

  const images = have ? readdirSync(ROOT).filter((d) => existsSync(join(ROOT, d, "rgb.npy"))) : [];
  test.each(images)("PIL bicubic to 512x512 (%s)", (d) => {
    const rgb = readNpy(join(ROOT, d, "rgb.npy"));
    const [h, w] = rgb.shape;
    const ref = readNpy(join(ROOT, d, "a2s_in.npy")).data as Float32Array; // [3,512,512] normalized
    const got = pilResizeBicubic(rgb.data as Uint8Array, w, h, 3, 512, 512);
    const back = new Uint8Array(512 * 512 * 3);
    for (let c = 0; c < 3; c++)
      for (let p = 0; p < 512 * 512; p++) back[p * 3 + c] = Math.round((ref[c * 512 * 512 + p] * 0.5 + 0.5) * 255);
    expect(diffs(got, back)).toEqual({ n: 0, max: 0 });
  });
});
