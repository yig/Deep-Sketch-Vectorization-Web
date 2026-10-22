// Saving and loading the edit state (keypoints + USM) as JSON, to resume an editing session later.

import type { ModelVariant } from "./inference/session";
import type { EditState, UiMark } from "./protocol";
import type { PreprocessOptions } from "./pipeline/preprocess";

const FORMAT = "deepsketch-edits";
/** 2: adds `preprocess` (version 1 files had none). */
const VERSION = 2;

export interface EditFile {
  format: typeof FORMAT;
  version: number;
  /** The source image: content hash (as used for caching), name and size. */
  image: { key: string; name: string; width: number; height: number };
  model: ModelVariant;
  maxSide: number;
  preprocess: PreprocessOptions;
  /** The 2x dual-contouring grid the USM lives on. */
  grid: { w: number; h: number };
  marks: UiMark[];
  /** Run-length encoded USM: [value, count, value, count, ...]. */
  usm: number[];
}

function rle(a: Uint8Array): number[] {
  const out: number[] = [];
  for (let p = 0; p < a.length; ) {
    let q = p;
    while (q < a.length && a[q] === a[p]) q++;
    out.push(a[p], q - p);
    p = q;
  }
  return out;
}

function unrle(runs: number[], n: number): Uint8Array {
  const a = new Uint8Array(n);
  let p = 0;
  for (let k = 0; k + 1 < runs.length; k += 2) {
    const [v, c] = [runs[k], runs[k + 1]];
    if (!Number.isInteger(c) || c < 0 || p + c > n || ![0, 1, 2].includes(v)) throw new Error("corrupt USM data");
    a.fill(v, p, p + c);
    p += c;
  }
  if (p !== n) throw new Error("USM size does not match the grid");
  return a;
}

export function encodeEdits(
  state: EditState,
  image: EditFile["image"],
  model: ModelVariant,
  maxSide: number,
  preprocess: PreprocessOptions,
  grid: EditFile["grid"],
): string {
  const file: EditFile = { format: FORMAT, version: VERSION, image, model, maxSide, preprocess, grid, marks: state.marks, usm: rle(state.usm) };
  return JSON.stringify(file);
}

export function decodeEdits(text: string): { file: EditFile; state: EditState } {
  const f = JSON.parse(text) as EditFile;
  if (f.format !== FORMAT) throw new Error("not a Deep Sketch Vectorization edits file");
  if (f.version > VERSION) throw new Error(`edits file version ${f.version} is newer than this app supports`);
  if (f.model !== "light" && f.model !== "full") throw new Error("unknown model in edits file");
  f.preprocess = { lineExtraction: !!f.preprocess?.lineExtraction, thinning: !!f.preprocess?.thinning };
  const n = f.grid.w * f.grid.h;
  const marks: UiMark[] = f.marks.map((m) => {
    if (![1, 2, 3].includes(m.type) || !Number.isFinite(m.x) || !Number.isFinite(m.y)) throw new Error("corrupt keypoint data");
    return { type: m.type, x: m.x, y: m.y, cell: Number.isInteger(m.cell) && m.cell! >= 0 && m.cell! < n ? m.cell : null };
  });
  return { file: f, state: { marks, usm: unrle(f.usm, n) } };
}
