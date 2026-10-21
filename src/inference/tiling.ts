// Tiled inference for the fully convolutional networks. Overlapping tiles reproduce whole-image
// inference exactly as long as the halo covers the receptive field (verified offline: 64 input
// pixels for the UDF network, 32 grid cells for the NDC network) and tile origins stay on the
// UDF network's 8-pixel downsampling lattice.

/** Runs one network on a [1, 1, h, w] input; returns its outputs as flat NCHW arrays. */
export type RunFn = (input: Float32Array, h: number, w: number) => Promise<Float32Array[]>;

export interface TileSpec {
  tile: number;
  halo: number;
}

export const UDF_HALO = 64;
export const NDC_HALO = 32;

function* tiles(h: number, w: number, { tile, halo }: TileSpec) {
  for (let y0 = 0; y0 < h; y0 += tile)
    for (let x0 = 0; x0 < w; x0 += tile) {
      const y1 = Math.min(y0 + tile, h);
      const x1 = Math.min(x0 + tile, w);
      yield { y0, x0, y1, x1, ty0: Math.max(0, y0 - halo), tx0: Math.max(0, x0 - halo), ty1: Math.min(h, y1 + halo), tx1: Math.min(w, x1 + halo) };
    }
}

function crop(src: Float32Array, sw: number, y0: number, x0: number, h: number, w: number): Float32Array {
  const out = new Float32Array(h * w);
  for (let i = 0; i < h; i++) out.set(src.subarray((y0 + i) * sw + x0, (y0 + i) * sw + x0 + w), i * w);
  return out;
}

/** Copy rows/cols [r0, r1) x [c0, c1) of each channel of a tile output into the full output. */
function paste(
  dst: Float32Array, dh: number, dw: number,
  src: Float32Array, sh: number, sw: number,
  channels: number, r0: number, r1: number, c0: number, c1: number, dr: number, dc: number,
) {
  for (let c = 0; c < channels; c++)
    for (let r = r0; r < r1; r++) {
      const s = c * sh * sw + r * sw;
      dst.set(src.subarray(s + c0, s + c1), c * dh * dw + (r + dr) * dw + (c0 + dc));
    }
}

/**
 * UDF network on an H x W image (multiples of 8). Returns udf [(2H+1) x (2W+1)] and aux [5 x 2H x 2W].
 * `tile` must be a multiple of 8.
 */
export async function runUdfTiled(run: RunFn, img: Float32Array, H: number, W: number, spec: TileSpec, onTile?: (k: number, n: number) => void) {
  if (H % 8 || W % 8 || spec.tile % 8 || spec.halo % 8) throw new Error("UDF tiling needs multiples of 8");
  const uh = 2 * H + 1;
  const uw = 2 * W + 1;
  const udf = new Float32Array(uh * uw);
  const aux = new Float32Array(5 * 4 * H * W);
  const all = [...tiles(H, W, spec)];
  for (let k = 0; k < all.length; k++) {
    const t = all[k];
    onTile?.(k, all.length);
    const th = t.ty1 - t.ty0;
    const tw = t.tx1 - t.tx0;
    const [u, a] = await run(crop(img, W, t.ty0, t.tx0, th, tw), th, tw);
    // aux: pixel-centered 2x samples; this tile owns rows [2*y0, 2*y1).
    paste(aux, 2 * H, 2 * W, a, 2 * th, 2 * tw, 5, 2 * (t.y0 - t.ty0), 2 * (t.y1 - t.ty0), 2 * (t.x0 - t.tx0), 2 * (t.x1 - t.tx0), 2 * t.ty0, 2 * t.tx0);
    // udf: corner samples; this tile owns [2*y0, 2*y1), plus the last row/column at the image edge.
    const er = 2 * (t.y1 - t.ty0) + (t.y1 === H ? 1 : 0);
    const ec = 2 * (t.x1 - t.tx0) + (t.x1 === W ? 1 : 0);
    paste(udf, uh, uw, u, 2 * th + 1, 2 * tw + 1, 1, 2 * (t.y0 - t.ty0), er, 2 * (t.x0 - t.tx0), ec, 2 * t.ty0, 2 * t.tx0);
  }
  return { udf, aux };
}

/** NDC network on a (h+1) x (w+1) UDF. Returns edge logits [4 x h x w] and offsets [2 x h x w]. */
export async function runNdcTiled(run: RunFn, udf: Float32Array, h: number, w: number, spec: TileSpec, onTile?: (k: number, n: number) => void) {
  const edge = new Float32Array(4 * h * w);
  const offset = new Float32Array(2 * h * w);
  const all = [...tiles(h, w, spec)];
  for (let k = 0; k < all.length; k++) {
    const t = all[k];
    onTile?.(k, all.length);
    const th = t.ty1 - t.ty0;
    const tw = t.tx1 - t.tx0;
    const [e, o] = await run(crop(udf, w + 1, t.ty0, t.tx0, th + 1, tw + 1), th + 1, tw + 1);
    const r0 = t.y0 - t.ty0;
    const c0 = t.x0 - t.tx0;
    paste(edge, h, w, e, th, tw, 4, r0, r0 + t.y1 - t.y0, c0, c0 + t.x1 - t.x0, t.ty0, t.tx0);
    paste(offset, h, w, o, th, tw, 2, r0, r0 + t.y1 - t.y0, c0, c0 + t.x1 - t.x0, t.ty0, t.tx0);
  }
  return { edge, offset };
}

/**
 * Largest tile (multiple of 8) whose biggest intermediate tensor fits in `maxBytes`.
 * `bytesPerCell` is channels x bytes-per-element of the widest tensor at the network's
 * output resolution; `scale` is output cells per input pixel (2 for UDF, 1 for NDC).
 */
export function chooseTile(maxBytes: number, bytesPerCell: number, scale: number, halo: number, limit: number): number {
  let tile = 8;
  for (let t = 8; t <= limit; t += 8) {
    const side = (t + 2 * halo) * scale;
    if (side * side * bytesPerCell <= maxBytes) tile = t;
  }
  return tile;
}
