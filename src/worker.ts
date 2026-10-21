// Web Worker: runs the networks and the post-processing off the main thread.
// Network outputs are cached per (image, model, size) so changing post-processing options is cheap.

import { detectDevice, loadNetworks, setAppBase, type DeviceOverrides, type ModelVariant } from "./inference/session";
import { runNdcTiled, runUdfTiled, NDC_HALO, UDF_HALO } from "./inference/tiling";
import { prepareImage, type PreparedImage } from "./pipeline/image";
import { toSvg, vectorizeFromNetworks, type NetworkOutputs, type VectorizeOptions } from "./pipeline/vectorize";
import type { Keypoints } from "./pipeline/keypoints";

export interface RunRequest {
  type: "run";
  id: number;
  /** Grayscale source image (8-bit). */
  gray: Uint8Array;
  width: number;
  height: number;
  /** Identifies the image content, for caching network outputs. */
  imageKey: string;
  model: ModelVariant;
  maxSide: number;
  options: VectorizeOptions;
}

export type WorkerRequest = { type: "init"; appBase: string; overrides: DeviceOverrides } | RunRequest;

export interface RunResult {
  type: "result";
  id: number;
  svg: string;
  /** Strokes in source-image pixels. */
  strokes: number[][];
  keypoints: Keypoints;
  /** USM on the 2x grid of the prepared image: 0 none, 1 predicted, 2 refined by surgery. */
  usm: Uint8Array;
  gridW: number;
  gridH: number;
  /** Source pixels per prepared-image pixel. */
  scale: number;
  prepared: { W: number; H: number; resizedW: number; resizedH: number };
  timings: Record<string, number>;
  device: string;
  tiles: { udf: number; ndc: number };
}

export type WorkerResponse =
  | { type: "ready"; device: string }
  | { type: "progress"; id: number; message: string; frac: number }
  | RunResult
  | { type: "error"; id: number; message: string };

const post = (msg: WorkerResponse, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(msg, transfer);

let cached: { key: string; prepared: PreparedImage; net: NetworkOutputs; timings: Record<string, number> } | null = null;

async function run(req: RunRequest) {
  const progress = (message: string, frac: number) => post({ type: "progress", id: req.id, message, frac });
  const key = `${req.imageKey}|${req.model}|${req.maxSide}`;
  const nets = await loadNetworks(req.model, progress);
  if (cached?.key !== key) {
    cached = null;
    const t0 = performance.now();
    const prepared = prepareImage(req.gray, req.width, req.height, req.maxSide);
    const { H, W } = prepared;
    const { udf, aux } = await runUdfTiled(nets.udf, prepared.data, H, W, { tile: nets.udfTile, halo: UDF_HALO }, (k, n) =>
      progress(`Distance field network: tile ${k + 1} / ${n}`, k / n),
    );
    const t1 = performance.now();
    const { edge, offset } = await runNdcTiled(nets.ndc, udf, 2 * H, 2 * W, { tile: nets.ndcTile, halo: NDC_HALO }, (k, n) =>
      progress(`Line reconstruction network: tile ${k + 1} / ${n}`, k / n),
    );
    const t2 = performance.now();
    cached = { key, prepared, net: { H, W, aux, edge, offset }, timings: { "distance field network": t1 - t0, "line network": t2 - t1 } };
  }
  progress("Post-processing", 1);
  const { prepared, net } = cached;
  const res = vectorizeFromNetworks(net, req.options);
  const scale = prepared.sourceW / prepared.resizedW;
  const svg = toSvg(res, prepared.sourceW, prepared.sourceH, scale, Math.max(1, scale));
  const usm = new Uint8Array(res.usm.length);
  for (let p = 0; p < usm.length; p++) usm[p] = res.usmApplied[p] ? 2 : res.usm[p] ? 1 : 0;
  const strokes = res.strokes.map((s) => s.map((v) => v * scale));
  const sc = (a: number[]) => a.map((v) => v * scale);
  post(
    {
      type: "result",
      id: req.id,
      svg,
      strokes,
      keypoints: { end: sc(res.keypoints.end), sharp: sc(res.keypoints.sharp), junc: sc(res.keypoints.junc) },
      usm,
      gridW: 2 * net.W,
      gridH: 2 * net.H,
      scale,
      prepared: { W: prepared.W, H: prepared.H, resizedW: prepared.resizedW, resizedH: prepared.resizedH },
      timings: { ...cached.timings, ...res.timings },
      device: nets.device.description,
      tiles: { udf: nets.udfTile, ndc: nets.ndcTile },
    },
    [usm.buffer],
  );
}

self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  const req = e.data;
  if (req.type === "init") {
    setAppBase(req.appBase);
    const d = await detectDevice(req.overrides);
    post({ type: "ready", device: d.description });
    return;
  }
  try {
    await run(req);
  } catch (err) {
    post({ type: "error", id: req.id, message: err instanceof Error ? err.message : String(err) });
  }
};
