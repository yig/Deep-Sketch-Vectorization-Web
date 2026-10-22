// Input preparation (predict_s1.load_img / open_img): grayscale, optional downscale so the long
// side is at most `maxSide` (cv2 INTER_AREA), crop to multiples of 8, scale to [0, 1].

import { cvResizeArea } from "./resize";

export interface PreparedImage {
  /** [H x W] float32 in [0, 1], white background = 1. */
  data: Float32Array;
  H: number;
  W: number;
  /** Size after resizing, before cropping to multiples of 8. */
  resizedW: number;
  resizedH: number;
  /** Original image size. */
  sourceW: number;
  sourceH: number;
}

/**
 * RGBA (as from canvas getImageData) to 8-bit luma with PIL's convert("L") weights.
 * Deviation: transparent pixels are composited over white (PIL ignores alpha, which turns
 * transparent PNG backgrounds black).
 */
export function rgbaToGray(rgba: Uint8ClampedArray | Uint8Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let p = 0; p < w * h; p++) {
    const a = rgba[4 * p + 3] / 255;
    const r = rgba[4 * p] * a + 255 * (1 - a);
    const g = rgba[4 * p + 1] * a + 255 * (1 - a);
    const b = rgba[4 * p + 2] * a + 255 * (1 - a);
    out[p] = (Math.round(r) * 19595 + Math.round(g) * 38470 + Math.round(b) * 7471 + 0x8000) >> 16;
  }
  return out;
}

export function prepareImage(gray: Uint8Array, w: number, h: number, maxSide: number): PreparedImage {
  let img = gray;
  let rw = w;
  let rh = h;
  const long = Math.max(w, h);
  if (long > maxSide) {
    const ratio = maxSide / long;
    rh = Math.trunc(h * ratio);
    rw = Math.trunc(w * ratio);
    img = cvResizeArea(gray, w, h, 1, rw, rh);
  }
  const H = rh - (rh % 8);
  const W = rw - (rw % 8);
  if (H === 0 || W === 0) throw new Error("image is too small (needs at least 8 x 8 pixels)");
  const data = new Float32Array(H * W);
  for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) data[i * W + j] = img[i * rw + j] / 255;
  return { data, H, W, resizedW: rw, resizedH: rh, sourceW: w, sourceH: h };
}
