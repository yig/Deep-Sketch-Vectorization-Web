// Optional pre-processing before vectorization, as in predict_s1.load_img:
//  - line thinning (Edgar Simo-Serra's line normalizer, network/thin.py), on the grayscale image;
//  - Anime2Sketch (Xiaoyu Xiang, network/anime2sketch.py): line extraction from a color image at a
//    fixed 512 x 512, resized back to the image size;
//  - both: line extraction, then thinning.
// Both run at the source resolution, before the image is resized to the working size.

import { cvResizeArea, pilResizeBicubic } from "./resize";

export interface PreprocessOptions {
  lineExtraction: boolean;
  thinning: boolean;
}

export const NO_PREPROCESS: PreprocessOptions = { lineExtraction: false, thinning: false };

export const preprocessKey = (p: PreprocessOptions) => (p.lineExtraction ? "a2s" : "") + (p.thinning ? "+thin" : "");

/** Runs a network on an [1, c, h, w] float32 input; returns its first output. */
export type Net = (input: Float32Array, h: number, w: number) => Promise<Float32Array>;

/** Thinning: input gray / 255, output sigmoid, then (x * 255).astype(int) as the reference does. */
export async function thin(gray: Uint8Array, w: number, h: number, runTiled: (img: Float32Array, h: number, w: number) => Promise<Float32Array>) {
  const x = new Float32Array(w * h);
  for (let p = 0; p < x.length; p++) x[p] = gray[p] / 255;
  const y = await runTiled(x, h, w);
  const out = new Uint8Array(w * h);
  for (let p = 0; p < out.length; p++) out[p] = Math.min(255, Math.max(0, Math.trunc(Math.fround(y[p] * 255))));
  return out;
}

/**
 * Anime2Sketch on an interleaved RGB image: PIL bicubic resize to 512 x 512, normalize to [-1, 1],
 * run, map back to 0..255 (truncating, as tensor_to_img does), cv2 INTER_AREA back to w x h.
 * Returns one channel (the reference's three channels are identical).
 */
export async function anime2sketch(rgb: Uint8Array, w: number, h: number, net: Net) {
  const S = 512;
  const small = pilResizeBicubic(rgb, w, h, 3, S, S);
  const x = new Float32Array(3 * S * S);
  for (let c = 0; c < 3; c++)
    for (let p = 0; p < S * S; p++) x[c * S * S + p] = Math.fround((Math.fround(small[3 * p + c] / 255) - 0.5) / 0.5);
  const y = await net(x, S, S);
  const sk = new Uint8Array(S * S);
  for (let p = 0; p < S * S; p++) {
    const v = Math.fround(Math.fround(Math.fround(y[p] + 1) / 2) * 255);
    sk[p] = Math.min(255, Math.max(0, Math.trunc(v)));
  }
  return cvResizeArea(sk, S, S, 1, w, h);
}

/** RGBA (canvas) to RGB, compositing transparent pixels over white like rgbaToGray does. */
export function rgbaToRgb(rgba: Uint8ClampedArray | Uint8Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h * 3);
  for (let p = 0; p < w * h; p++) {
    const a = rgba[4 * p + 3] / 255;
    for (let c = 0; c < 3; c++) out[3 * p + c] = Math.round(rgba[4 * p + c] * a + 255 * (1 - a));
  }
  return out;
}
