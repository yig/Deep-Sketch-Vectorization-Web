// Messages between the main thread and the worker (src/worker.ts). Kept separate so the main
// thread does not import the worker module itself.

import type { DeviceOverrides, ModelVariant } from "./inference/session";
import type { VectorizeOptions } from "./pipeline/vectorize";

/** A keypoint as the UI sees it, in source-image pixels. `cell` is set for keypoints the worker placed. */
export interface UiMark {
  type: 1 | 2 | 3;
  x: number;
  y: number;
  cell: number | null;
}

/** User edits: keypoints and the USM on the 2x grid (0 none, 1 refine, 2 keep detail only). */
export interface EditState {
  marks: UiMark[];
  usm: Uint8Array;
}

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
  /** Edits made against the result with this `prepKey`; ignored if the key no longer matches. */
  edit: { prepKey: string; state: EditState } | null;
}

export type WorkerRequest = { type: "init"; appBase: string; overrides: DeviceOverrides } | RunRequest;

/** Per-cell display code for the USM overlay. */
export const CellStatus = {
  None: 0,
  /** Kept at full resolution during downsampling, no surgery (USM value 2, region not activated). */
  Detail: 1,
  /** Surgery not possible: the region has holes, or no polyline enters it. */
  Invalid: 2,
  /** Surgery attempted, but no rule applies with the keypoints inside. */
  NotApplied: 3,
  Applied: 4,
} as const;

export interface RunResult {
  type: "result";
  id: number;
  /** Changes whenever the network outputs change; edits are only valid for the same key. */
  prepKey: string;
  /** Whether `edit` was applied (false: automatic result). */
  edited: boolean;
  svg: string;
  /** Strokes in source-image pixels. */
  strokes: number[][];
  /** Keypoints used, source-image pixels. */
  marks: UiMark[];
  /** USM used (0 / 1 / 2) and per-cell status (CellStatus), on the 2x grid of the prepared image. */
  usm: Uint8Array;
  cellStatus: Uint8Array;
  regions: { applied: number; notApplied: number; invalid: number };
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

