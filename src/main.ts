// Main-thread UI: image input, options, worker orchestration, and result display.

import type { RunResult, WorkerRequest, WorkerResponse } from "./protocol";
import { rgbaToGray } from "./pipeline/image";
import type { VectorizeOptions } from "./pipeline/vectorize";
import { Editor, type MarkType, type Tool } from "./editor";
import { decodeEdits, encodeEdits } from "./editfile";
import { prepKeyOf } from "./protocol";
import type { PreprocessOptions } from "./pipeline/preprocess";
import { countPageview, countVectorized } from "./analytics";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
const send = (msg: WorkerRequest) => worker.postMessage(msg, msg.type === "run" ? [msg.rgba.buffer] : []);
countPageview();

interface Source {
  name: string;
  url: string;
  rgba: Uint8Array;
  width: number;
  height: number;
  key: string;
}

let source: Source | null = null;
/** The current source has been counted as vectorized (re-runs and edits do not count again). */
let counted = false;
let result: RunResult | null = null;
let runId = 0;
let pending: number | null = null;
/** A run was requested while another was in flight: run again when it returns. */
let dirty = false;
/** Editor version each request was made at. */
const requestVersion = new Map<number, number>();

const ui = {
  file: $<HTMLInputElement>("file"),
  maxSide: $<HTMLSelectElement>("maxSide"),
  downRate: $<HTMLSelectElement>("downRate"),
  smoothing: $<HTMLSelectElement>("smoothing"),
  eps: $<HTMLInputElement>("eps"),
  epsField: $("epsField"),
  skipLen: $<HTMLInputElement>("skipLen"),
  opacity: $<HTMLInputElement>("opacity"),
  showVector: $<HTMLInputElement>("showVector"),
  colorStrokes: $<HTMLInputElement>("colorStrokes"),
  showKeypoints: $<HTMLInputElement>("showKeypoints"),
  showUsm: $<HTMLInputElement>("showUsm"),
  legend: $("legend"),
  download: $<HTMLButtonElement>("download"),
  progress: $("progress"),
  status: $("status"),
  device: $("device"),
  stats: $("stats"),
  stage: $("stage"),
  empty: $("empty"),
  canvas: $("canvas"),
  input: $<HTMLImageElement>("input"),
  usm: $<HTMLCanvasElement>("usm"),
  vector: $("vector"),
  keypoints: $("keypoints") as unknown as SVGSVGElement,
  toolbar: $("toolbar"),
  markTypes: $("markTypes"),
  brushField: $("brushField"),
  brushSize: $<HTMLInputElement>("brushSize"),
  undo: $<HTMLButtonElement>("undo"),
  redo: $<HTMLButtonElement>("redo"),
  resetEdits: $<HTMLButtonElement>("resetEdits"),
  editHint: $("editHint"),
  saveEdits: $<HTMLButtonElement>("saveEdits"),
  lineExtraction: $<HTMLInputElement>("lineExtraction"),
  thinning: $<HTMLInputElement>("thinning"),
  loadEdits: $<HTMLInputElement>("loadEdits"),
};

const editor = new Editor({
  stage: ui.stage,
  canvas: ui.canvas,
  usm: ui.usm,
  keypoints: ui.keypoints,
  lasso: $("lasso") as unknown as SVGSVGElement,
});
editor.onChange = () => {
  updateToolbar();
  run();
};
editor.onStatus = (t) => setStatus(t);

const STATUS_WORDS: Record<number, string> = { 1: "Detail only", 2: "Can't rebuild", 3: "Not rebuilt", 4: "Rebuilt" };
let hoverText: string | null = null;
editor.onHover = (h) => {
  if (!h) hoverText = null;
  else {
    const { info, status } = h;
    const [e, s, j] = info.keypoints;
    const kp = [e && `${e} end point${e > 1 ? "s" : ""}`, s && `${s} sharp turn${s > 1 ? "s" : ""}`, j && `${j} junction${j > 1 ? "s" : ""}`].filter(Boolean).join(", ");
    const parts = [`${info.cells} cells`];
    if (info.entering >= 0) parts.push(`${info.entering} line${info.entering === 1 ? "" : "s"} entering`);
    parts.push(kp ? `keypoints inside: ${kp}` : "no keypoints inside");
    hoverText = `${STATUS_WORDS[status] ?? "Region"}: ${info.explanation}. (${parts.join(" · ")})`;
  }
  updateToolbar();
};

const preprocessOptions = (): PreprocessOptions => ({ lineExtraction: ui.lineExtraction.checked, thinning: ui.thinning.checked });

/** The last pre-processed image received from the worker, shown instead of the source when in use. */
let preImage: { key: string; url: string } | null = null;
function setPreprocessedImage(p: { key: string; gray: Uint8Array } | null) {
  if (preImage) URL.revokeObjectURL(preImage.url);
  preImage = null;
  if (!p || !source) return;
  const c = document.createElement("canvas");
  c.width = source.width;
  c.height = source.height;
  const ctx = c.getContext("2d")!;
  const img = ctx.createImageData(source.width, source.height);
  for (let i = 0; i < p.gray.length; i++) img.data.set([p.gray[i], p.gray[i], p.gray[i], 255], 4 * i);
  ctx.putImageData(img, 0, 0);
  const key = p.key;
  c.toBlob((b) => {
    if (!b) return;
    preImage = { key, url: URL.createObjectURL(b) };
    if (result?.preKey === key) showInput(key);
  });
}
/** Show the source, or the pre-processed image the current result used. */
function showInput(preKey: string) {
  if (!source) return;
  const url = preKey && preImage?.key === preKey ? preImage.url : source.url;
  if (ui.input.src !== url) ui.input.src = url;
}

const model = () => (document.querySelector<HTMLInputElement>('input[name="model"]:checked')!.value as "light" | "full");
const options = (): VectorizeOptions => ({
  downRate: Number(ui.downRate.value),
  skipLen: Number(ui.skipLen.value),
  smoothing: ui.smoothing.value as VectorizeOptions["smoothing"],
  rdpEpsilon: Number(ui.eps.value),
});

/**
 * Shows a status message. Progress and results (`oneLine`) are truncated to one line, so the status
 * keeps its height while a slider re-runs the pipeline; other messages, like errors, wrap.
 */
function setStatus(text: string, frac: number | null = null, oneLine = frac !== null) {
  ui.status.textContent = text;
  ui.status.title = text;
  ui.status.classList.toggle("one-line", oneLine);
  // Hide the bar without collapsing its space, so the panel below doesn't jump.
  ui.progress.style.visibility = frac === null ? "hidden" : "visible";
  if (frac !== null) (ui.progress.firstElementChild as HTMLElement).style.width = `${Math.round(frac * 100)}%`;
}

async function hashBytes(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-1", bytes as Uint8Array<ArrayBuffer>);
  return Array.from(new Uint8Array(digest).slice(0, 8), (b) => b.toString(16).padStart(2, "0")).join("");
}

async function loadImage(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  try {
    const bitmap = await createImageBitmap(blob);
    const c = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = c.getContext("2d")!;
    ctx.drawImage(bitmap, 0, 0);
    const { data } = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
    const gray = rgbaToGray(data, bitmap.width, bitmap.height);
    if (source) URL.revokeObjectURL(source.url);
    setPreprocessedImage(null);
    source = { name, url, rgba: new Uint8Array(data.buffer), width: bitmap.width, height: bitmap.height, key: await hashBytes(gray) };
    result = null;
    counted = false;
    ui.input.src = url;
    ui.empty.hidden = true;
    ui.canvas.hidden = false;
    ui.toolbar.hidden = false;
    ui.vector.innerHTML = "";
    ui.download.disabled = true;
    updateToolbar();
    editor.setSource(bitmap.width, bitmap.height);
    runFresh();
  } catch (e) {
    URL.revokeObjectURL(url);
    setStatus(`Could not read image: ${e instanceof Error ? e.message : e}`);
  }
}

function run() {
  if (!source) return;
  if (pending !== null) {
    dirty = true;
    return;
  }
  const id = ++runId;
  pending = id;
  requestVersion.set(id, editor.version);
  setStatus("Working…", 0);
  send({
    type: "run",
    id,
    rgba: source.rgba.slice(),
    width: source.width,
    height: source.height,
    imageKey: source.key,
    preprocess: preprocessOptions(),
    model: model(),
    maxSide: Number(ui.maxSide.value),
    options: options(),
    edit: editor.editRequest(),
  });
}

/** A new image: drop any queued run for the previous one. */
function runFresh() {
  pending = null;
  dirty = false;
  run();
}

// Debounce re-runs triggered by sliders.
let rerunTimer = 0;
const rerunSoon = () => {
  clearTimeout(rerunTimer);
  rerunTimer = window.setTimeout(run, 200);
};

worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
  const msg = e.data;
  if (msg.type === "ready") {
    ui.device.textContent = `Backend: ${msg.device}`;
    setStatus(source ? "Ready" : "Open a sketch to begin.");
    return;
  }
  if (msg.id !== pending) return; // superseded (a new image was loaded)
  if (msg.type === "progress") {
    if (!dirty) setStatus(msg.message, msg.frac);
    return;
  }
  pending = null;
  if (msg.type === "error") setStatus(`Error: ${msg.message}`);
  else {
    result = msg;
    if (!counted) {
      counted = true;
      countVectorized();
    }
    ui.device.textContent = `Backend: ${msg.device}`;
    editor.applyResult(msg, requestVersion.get(msg.id) ?? -1);
    if (msg.preprocessed) setPreprocessedImage(msg.preprocessed);
    showInput(msg.preKey);
    const { applied, notApplied, invalid } = msg.regions;
    const regions = applied + notApplied + invalid;
    setStatus(
      `${msg.strokes.length} strokes${msg.edited ? " (edited)" : ""}` +
        (regions ? ` · ${applied} of ${regions} USM region${regions > 1 ? "s" : ""} refined` : ""),
      null,
      true,
    );
    renderResult();
    renderStats();
    updateToolbar();
    ui.download.disabled = false;
  }
  requestVersion.delete(msg.id);
  if (dirty) {
    dirty = false;
    run();
  }
};
worker.onerror = (e) => setStatus(`Worker error: ${e.message}`);
const params = new URLSearchParams(location.search);
send({
  type: "init",
  appBase: new URL(import.meta.env.BASE_URL, location.href).href,
  overrides: {
    backend: params.get("backend") === "wasm" ? "wasm" : undefined,
    budgetMB: Number(params.get("budget")) || undefined,
  },
});

// ---------- rendering ----------

const STROKE_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#db2777", "#65a30d", "#4f46e5", "#ca8a04"];

function renderResult() {
  if (!result || !source) return;
  // Display copy of the SVG: optional per-stroke colors and a constant on-screen line width.
  const doc = new DOMParser().parseFromString(result.svg, "image/svg+xml").documentElement as unknown as SVGSVGElement;
  doc.removeAttribute("width");
  doc.removeAttribute("height");
  doc.setAttribute("preserveAspectRatio", "none");
  const g = doc.querySelector("g")!;
  g.setAttribute("vector-effect", "non-scaling-stroke");
  g.setAttribute("stroke-width", "1.5");
  doc.querySelectorAll("path").forEach((p, i) => {
    p.setAttribute("vector-effect", "non-scaling-stroke");
    if (ui.colorStrokes.checked) p.setAttribute("stroke", STROKE_COLORS[i % STROKE_COLORS.length]);
  });
  ui.vector.replaceChildren(doc);
  applyView();
}

function applyView() {
  ui.input.style.opacity = String(Number(ui.opacity.value) / 100);
  $("opacityOut").textContent = `${ui.opacity.value}%`;
  ui.vector.style.display = ui.showVector.checked ? "" : "none";
  editor.showKeypoints = ui.showKeypoints.checked;
  editor.showUsm = ui.showUsm.checked;
  editor.render();
  ui.legend.hidden = !(ui.showKeypoints.checked || ui.showUsm.checked || editor.tool !== "pan");
}

// ---------- editing toolbar ----------

const HINTS: Record<Tool, string> = {
  pan: "Drag to pan, scroll to zoom. Pick a tool to refine the result.",
  keypoint: "Click to add a keypoint · drag to move · double-click to change its type · Alt-click or Delete to remove",
  brush: "Paint under-sampled regions (USM). Each region is rebuilt by connecting the lines entering it to the keypoint inside.",
  eraser: "Erase under-sampled regions: their original dual-contouring lines come back.",
  lasso: "Draw a loop to fill it with USM. Hold Alt while releasing to erase inside the loop instead.",
};

function setTool(t: Tool) {
  editor.tool = t;
  ui.stage.dataset.tool = t;
  updateToolbar();
  applyView();
  editor.layout(); // the toolbar's height may have changed
}

function updateToolbar() {
  for (const b of ui.toolbar.querySelectorAll<HTMLButtonElement>("[data-tool]")) b.setAttribute("aria-pressed", String(b.dataset.tool === editor.tool));
  for (const b of ui.markTypes.querySelectorAll<HTMLButtonElement>("[data-type]"))
    b.setAttribute("aria-pressed", String(Number(b.dataset.type) === editor.markType));
  ui.markTypes.hidden = editor.tool !== "keypoint";
  ui.brushField.hidden = editor.tool !== "brush" && editor.tool !== "eraser";
  ui.saveEdits.disabled = !editor.isEdited || !result;
  ui.loadEdits.disabled = !source;
  $("loadEditsLabel").classList.toggle("disabled", !source);
  ui.undo.disabled = !editor.canUndo;
  ui.redo.disabled = !editor.canRedo;
  ui.resetEdits.disabled = !editor.isEdited;
  ui.editHint.hidden = !source;
  ui.editHint.textContent = hoverText ?? HINTS[editor.tool];
  ui.editHint.classList.toggle("info", hoverText !== null);
}

for (const b of ui.toolbar.querySelectorAll<HTMLButtonElement>("[data-tool]")) b.addEventListener("click", () => setTool(b.dataset.tool as Tool));
for (const b of ui.markTypes.querySelectorAll<HTMLButtonElement>("[data-type]"))
  b.addEventListener("click", () => {
    editor.setMarkType(Number(b.dataset.type) as MarkType);
    updateToolbar();
  });
ui.brushSize.addEventListener("input", () => (editor.brushRadius = Number(ui.brushSize.value)));
ui.undo.addEventListener("click", () => editor.undo());
ui.redo.addEventListener("click", () => editor.redo());
ui.resetEdits.addEventListener("click", () => editor.resetToAuto());
$("zoomIn").addEventListener("click", () => editor.zoomBy(1.5));
$("zoomOut").addEventListener("click", () => editor.zoomBy(1 / 1.5));
$("zoomFit").addEventListener("click", () => editor.fit());

/** Keyboard shortcuts stay active unless the user is typing into a field. */
function isTextEntry(t: EventTarget | null): boolean {
  const el = t as HTMLElement | null;
  if (!el) return false;
  if (el.isContentEditable || el.tagName === "TEXTAREA" || el.tagName === "SELECT") return true;
  return el.tagName === "INPUT" && ["text", "number", "search", "email", "url", "password"].includes((el as HTMLInputElement).type);
}

window.addEventListener("keydown", (e) => {
  if (isTextEntry(e.target) || !source) return;
  const mod = e.metaKey || e.ctrlKey;
  const k = e.key.toLowerCase();
  if (mod && k === "z") {
    e.preventDefault();
    if (e.shiftKey) editor.redo();
    else editor.undo();
  } else if (mod && k === "y") {
    e.preventDefault();
    editor.redo();
  } else if (mod) return;
  else if (k === "v") setTool("pan");
  else if (k === "k") setTool("keypoint");
  else if (k === "b") setTool("brush");
  else if (k === "e") setTool("eraser");
  else if (k === "l") setTool("lasso");
  else if (k === "1" || k === "2" || k === "3") {
    editor.setMarkType(Number(k) as MarkType);
    updateToolbar();
  } else if (k === "delete" || k === "backspace") editor.deleteSelected();
  else if (k === "0") editor.fit();
  else if (k === "+" || k === "=") editor.zoomBy(1.5);
  else if (k === "-") editor.zoomBy(1 / 1.5);
});
setTool("pan");

function renderStats() {
  if (!result) return;
  const rows: [string, string][] = [
    ["working size", `${result.prepared.W} × ${result.prepared.H}`],
    ["tiles (udf / ndc)", `${result.tiles.udf} / ${result.tiles.ndc}`],
    ...Object.entries(result.timings).map(([k, v]) => [k, `${v < 10 ? v.toFixed(1) : Math.round(v)} ms`] as [string, string]),
  ];
  ui.stats.innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}

// ---------- events ----------

ui.file.addEventListener("change", () => {
  const f = ui.file.files?.[0];
  if (f) loadImage(f, f.name);
  ui.file.value = "";
  ui.file.blur();
});

for (const el of document.querySelectorAll<HTMLInputElement>('input[name="model"]')) el.addEventListener("change", run);
ui.maxSide.addEventListener("change", run);
ui.lineExtraction.addEventListener("change", run);
ui.thinning.addEventListener("change", run);
ui.downRate.addEventListener("change", run);
ui.smoothing.addEventListener("change", () => {
  ui.epsField.hidden = ui.smoothing.value !== "rdp";
  run();
});
ui.eps.addEventListener("input", () => {
  $("epsOut").textContent = ui.eps.value;
  rerunSoon();
});
ui.skipLen.addEventListener("input", () => {
  $("skipOut").textContent = ui.skipLen.value;
  rerunSoon();
});
for (const el of [ui.opacity, ui.showVector, ui.showKeypoints, ui.showUsm]) el.addEventListener("input", applyView);
ui.colorStrokes.addEventListener("input", renderResult);

ui.download.addEventListener("click", () => {
  if (!result || !source) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([result.svg], { type: "image/svg+xml" }));
  a.download = `${baseName()}.svg`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

const baseName = () => (source?.name ?? "sketch").replace(/\.[^.]+$/, "");

ui.saveEdits.addEventListener("click", () => {
  const state = editor.current;
  const grid = editor.gridSize;
  if (!source || !state || !grid) return;
  const text = encodeEdits(
    state,
    { key: source.key, name: source.name, width: source.width, height: source.height },
    model(),
    Number(ui.maxSide.value),
    preprocessOptions(),
    grid,
  );
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "application/json" }));
  a.download = `${baseName()}.edits.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

ui.loadEdits.addEventListener("change", () => {
  const f = ui.loadEdits.files?.[0];
  ui.loadEdits.value = "";
  ui.loadEdits.blur(); // keep keyboard shortcuts working
  if (f) loadEditsFile(f);
});

async function loadEditsFile(f: File) {
  if (!source) {
    setStatus("Open the image the edits belong to first.");
    return;
  }
  try {
    const { file, state } = decodeEdits(await f.text());
    if (file.image.key !== source.key) {
      setStatus(`These edits are for a different image (${file.image.name}). Open that image first.`);
      return;
    }
    // Switch to the model, working size and pre-processing the edits were made with.
    const radio = document.querySelector<HTMLInputElement>(`input[name="model"][value="${file.model}"]`)!;
    const pre = file.preprocess;
    const changed =
      !radio.checked ||
      Number(ui.maxSide.value) !== file.maxSide ||
      ui.lineExtraction.checked !== pre.lineExtraction ||
      ui.thinning.checked !== pre.thinning;
    ui.lineExtraction.checked = pre.lineExtraction;
    ui.thinning.checked = pre.thinning;
    radio.checked = true;
    if (![...ui.maxSide.options].some((o) => Number(o.value) === file.maxSide)) ui.maxSide.add(new Option(`${file.maxSide} px`, String(file.maxSide)));
    ui.maxSide.value = String(file.maxSide);
    editor.importState(prepKeyOf(source.key, file.model, file.maxSide, pre), state);
    if (changed) run();
    setStatus("Loaded edits.");
  } catch (e) {
    setStatus(`Could not load edits: ${e instanceof Error ? e.message : e}`);
  }
}

// Drag & drop and paste.
const stage = document.body;
stage.addEventListener("dragover", (e) => {
  e.preventDefault();
  ui.stage.classList.add("dragging");
});
stage.addEventListener("dragleave", (e) => {
  if (e.target === ui.stage || !ui.stage.contains(e.relatedTarget as Node)) ui.stage.classList.remove("dragging");
});
stage.addEventListener("drop", (e) => {
  e.preventDefault();
  ui.stage.classList.remove("dragging");
  const files = [...(e.dataTransfer?.files ?? [])];
  const img = files.find((f) => f.type.startsWith("image/"));
  const edits = files.find((f) => f.name.endsWith(".json"));
  if (img) loadImage(img, img.name);
  else if (edits) loadEditsFile(edits);
});
window.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items ?? [])].find((i) => i.type.startsWith("image/"));
  const f = item?.getAsFile();
  if (f) loadImage(f, "pasted.png");
});

// A procedurally drawn sample doodle (no third-party artwork needed).
$("sample").addEventListener("click", async () => {
  const c = new OffscreenCanvas(512, 400);
  const ctx = c.getContext("2d")!;
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, 512, 400);
  ctx.strokeStyle = "#222";
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  let seed = 7;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5) * 2;
  const stroke = (pts: [number, number][], width: number) => {
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(pts[0][0] + rand(), pts[0][1] + rand());
    for (let k = 1; k < pts.length; k++) ctx.lineTo(pts[k][0] + rand(), pts[k][1] + rand());
    ctx.stroke();
  };
  const arc = (cx: number, cy: number, rx: number, ry: number, a0: number, a1: number, n = 40) =>
    Array.from({ length: n + 1 }, (_, k) => {
      const a = a0 + ((a1 - a0) * k) / n;
      return [cx + rx * Math.cos(a), cy + ry * Math.sin(a)] as [number, number];
    });
  // A house with a tree and the sun.
  stroke([[90, 330], [430, 330]], 3);
  stroke([[140, 330], [140, 200], [250, 110], [360, 200], [360, 330]], 3.5);
  stroke([[140, 200], [360, 200]], 2.5);
  stroke([[225, 330], [225, 255], [275, 255], [275, 330]], 3);
  stroke([[165, 225], [205, 225], [205, 260], [165, 260], [165, 225]], 2.5);
  stroke([[185, 225], [185, 260]], 2);
  stroke([[165, 242], [205, 242]], 2);
  stroke([[300, 90], [300, 150]], 3);
  stroke([[285, 90], [315, 90]], 3);
  stroke([[440, 330], [440, 240]], 4);
  stroke(arc(440, 205, 38, 42, Math.PI * 0.62, Math.PI * 2.38), 3);
  stroke(arc(85, 75, 28, 28, 0, Math.PI * 2), 3);
  for (let k = 0; k < 8; k++) {
    const a = (k / 8) * Math.PI * 2;
    stroke([[85 + 38 * Math.cos(a), 75 + 38 * Math.sin(a)], [85 + 52 * Math.cos(a), 75 + 52 * Math.sin(a)]], 2.5);
  }
  loadImage(await c.convertToBlob({ type: "image/png" }), "sample.png");
});
