// Web Worker: runs the networks and the post-processing off the main thread.
// Network outputs and everything derived from them alone are cached per (image, model, size), so
// option changes and edits only re-run the fast finishing stage.

import { detectDevice, loadNetworks, loadPreprocessor, setAppBase } from "./inference/session";
import { runNdcTiled, runSameSizeTiled, runUdfTiled, NDC_HALO, THIN_HALO, UDF_HALO } from "./inference/tiling";
import { prepareImage, rgbaToGray, type PreparedImage } from "./pipeline/image";
import { anime2sketch, preprocessKey, rgbaToRgb, thin, type PreprocessOptions } from "./pipeline/preprocess";
import { finish, prepare, toSvg, type Prepared } from "./pipeline/vectorize";
import { RegionStatus, snapMark, type Mark } from "./pipeline/topology";
import { CellStatus, prepKeyOf, type RunRequest, type RunResult, type WorkerRequest, type WorkerResponse } from "./protocol";

const post = (msg: WorkerResponse, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(msg, transfer);

let cached: { key: string; image: PreparedImage; prep: Prepared; timings: Record<string, number> } | null = null;
/** The last pre-processed image (source size), per image and pre-processing options. */
let preCached: { key: string; gray: Uint8Array; timings: Record<string, number> } | null = null;
/** Key of the pre-processed image the main thread last received. */
let preSent = "";

type Progress = (message: string, frac: number) => void;

/** The grayscale image to vectorize: the source, or its pre-processed version. */
async function sourceGray(req: RunRequest, progress: Progress) {
  const { width: w, height: h, preprocess: pre } = req;
  const key = `${req.imageKey}|${preprocessKey(pre)}`;
  if (preCached?.key === key) return preCached;
  const timings: Record<string, number> = {};
  let t = 0;
  let gray: Uint8Array;
  if (pre.lineExtraction) {
    const a2s = await loadPreprocessor("anime2sketch", progress);
    progress("Line extraction (Anime2Sketch)", 0.5);
    t = performance.now();
    gray = await anime2sketch(rgbaToRgb(req.rgba, w, h), w, h, async (x, hh, ww) => (await a2s.run(x, hh, ww))[0]);
    timings["line extraction"] = performance.now() - t;
  } else gray = rgbaToGray(req.rgba, w, h);
  if (pre.thinning) {
    const net = await loadPreprocessor("thin", progress);
    t = performance.now();
    gray = await thin(gray, w, h, (x, hh, ww) =>
      runSameSizeTiled(net.run, x, hh, ww, { tile: net.tile, halo: THIN_HALO }, (k, n) => progress(`Line thinning: tile ${k + 1} / ${n}`, k / n)),
    );
    timings["line thinning"] = performance.now() - t;
  }
  preCached = { key, gray, timings };
  return preCached;
}

const anyPreprocess = (p: PreprocessOptions) => p.lineExtraction || p.thinning;

async function run(req: RunRequest) {
  const progress = (message: string, frac: number) => post({ type: "progress", id: req.id, message, frac });
  const key = prepKeyOf(req.imageKey, req.model, req.maxSide, req.preprocess);
  const nets = await loadNetworks(req.model, progress);
  if (cached?.key !== key) {
    cached = null;
    const src = await sourceGray(req, progress);
    const t0 = performance.now();
    const image = prepareImage(src.gray, req.width, req.height, req.maxSide);
    const { H, W } = image;
    const { udf, aux } = await runUdfTiled(nets.udf, image.data, H, W, { tile: nets.udfTile, halo: UDF_HALO }, (k, n) =>
      progress(`Distance field network: tile ${k + 1} / ${n}`, k / n),
    );
    const t1 = performance.now();
    const { edge, offset } = await runNdcTiled(nets.ndc, udf, 2 * H, 2 * W, { tile: nets.ndcTile, halo: NDC_HALO }, (k, n) =>
      progress(`Line network: tile ${k + 1} / ${n}`, k / n),
    );
    const t2 = performance.now();
    progress("Post-processing", 1);
    const prep = prepare({ H, W, aux, edge, offset });
    cached = { key, image, prep, timings: { ...src.timings, "distance field network": t1 - t0, "line network": t2 - t1, ...prep.timings } };
  }
  progress("Post-processing", 1);
  const { image, prep } = cached;
  const scale = image.sourceW / image.resizedW;
  const n = prep.h * prep.w;

  const edited = req.edit !== null && req.edit.prepKey === key && req.edit.state.usm.length === n;
  let marks: Mark[] = prep.autoMarks;
  let usm = prep.autoUsm;
  if (edited) {
    const state = req.edit!.state;
    usm = state.usm;
    marks = state.marks.map((m) =>
      m.cell !== null && m.cell >= 0 && m.cell < n
        ? { type: m.type, cell: m.cell, x: m.x / scale, y: m.y / scale }
        : snapMark(prep.base, m.type, m.x / scale, m.y / scale),
    );
  }
  const res = finish(prep, marks, usm, req.options);
  const svg = toSvg(res, image.sourceW, image.sourceH, scale, Math.max(1, scale));

  const cellStatus = new Uint8Array(n);
  const regions = { applied: 0, notApplied: 0, invalid: 0 };
  for (let l = 1; l < res.regionStatus.length; l++) {
    const s = res.regionStatus[l];
    if (s === RegionStatus.Applied) regions.applied++;
    else if (s === RegionStatus.NotApplied) regions.notApplied++;
    else if (s === RegionStatus.Invalid) regions.invalid++;
  }
  const code = [CellStatus.Detail, CellStatus.Invalid, CellStatus.NotApplied, CellStatus.Applied];
  for (let p = 0; p < n; p++) if (res.regionLabels[p]) cellStatus[p] = code[res.regionStatus[res.regionLabels[p]]];
  const usmOut = usm.slice();
  // Send the pre-processed image for display when it changed.
  let preprocessed: RunResult["preprocessed"] = null;
  if (anyPreprocess(req.preprocess) && preCached && preSent !== preCached.key) {
    preprocessed = { key: preCached.key, gray: preCached.gray.slice() };
    preSent = preCached.key;
  }

  post(
    {
      type: "result",
      id: req.id,
      prepKey: key,
      edited,
      preprocessed,
      preKey: anyPreprocess(req.preprocess) && preCached ? preCached.key : "",
      svg,
      strokes: res.strokes.map((s) => s.map((v) => v * scale)),
      marks: marks.map((m) => ({ type: m.type, cell: m.cell, x: m.x * scale, y: m.y * scale })),
      usm: usmOut,
      cellStatus,
      regionLabels: res.regionLabels,
      regionInfo: res.regionInfo,
      regions,
      gridW: prep.w,
      gridH: prep.h,
      scale,
      prepared: { W: image.W, H: image.H, resizedW: image.resizedW, resizedH: image.resizedH },
      timings: { ...cached.timings, ...res.timings },
      device: nets.device.description,
      tiles: { udf: nets.udfTile, ndc: nets.ndcTile },
    },
    [usmOut.buffer, cellStatus.buffer, res.regionLabels.buffer, ...(preprocessed ? [preprocessed.gray.buffer] : [])],
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
