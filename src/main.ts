// Main-thread UI: image input, options, worker orchestration, and result display.

import type { RunResult, WorkerRequest, WorkerResponse } from "./worker";
import { rgbaToGray } from "./pipeline/image";
import type { VectorizeOptions } from "./pipeline/vectorize";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
const send = (msg: WorkerRequest) => worker.postMessage(msg, msg.type === "run" ? [msg.gray.buffer] : []);

interface Source {
  name: string;
  url: string;
  gray: Uint8Array;
  width: number;
  height: number;
  key: string;
}

let source: Source | null = null;
let result: RunResult | null = null;
let runId = 0;
let pending: number | null = null;

const ui = {
  file: $<HTMLInputElement>("file"),
  maxSide: $<HTMLSelectElement>("maxSide"),
  downRate: $<HTMLSelectElement>("downRate"),
  smoothing: $<HTMLSelectElement>("smoothing"),
  eps: $<HTMLInputElement>("eps"),
  epsField: $("epsField"),
  skipLen: $<HTMLInputElement>("skipLen"),
  opacity: $<HTMLInputElement>("opacity"),
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
};

const model = () => (document.querySelector<HTMLInputElement>('input[name="model"]:checked')!.value as "light" | "full");
const options = (): VectorizeOptions => ({
  downRate: Number(ui.downRate.value),
  skipLen: Number(ui.skipLen.value),
  smoothing: ui.smoothing.value as VectorizeOptions["smoothing"],
  rdpEpsilon: Number(ui.eps.value),
});

function setStatus(text: string, frac: number | null = null) {
  ui.status.textContent = text;
  ui.progress.hidden = frac === null;
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
    source = { name, url, gray, width: bitmap.width, height: bitmap.height, key: await hashBytes(gray) };
    result = null;
    ui.input.src = url;
    ui.empty.hidden = true;
    ui.canvas.hidden = false;
    fitCanvas();
    ui.vector.innerHTML = "";
    ui.keypoints.innerHTML = "";
    ui.usm.getContext("2d")?.clearRect(0, 0, ui.usm.width, ui.usm.height);
    ui.download.disabled = true;
    run();
  } catch (e) {
    URL.revokeObjectURL(url);
    setStatus(`Could not read image: ${e instanceof Error ? e.message : e}`);
  }
}

function run() {
  if (!source) return;
  const id = ++runId;
  pending = id;
  setStatus("Starting…", 0);
  send({
    type: "run",
    id,
    gray: source.gray.slice(),
    width: source.width,
    height: source.height,
    imageKey: source.key,
    model: model(),
    maxSide: Number(ui.maxSide.value),
    options: options(),
  });
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
  if (msg.id !== pending) return; // superseded
  if (msg.type === "progress") setStatus(msg.message, msg.frac);
  else if (msg.type === "error") {
    pending = null;
    setStatus(`Error: ${msg.message}`);
  } else {
    pending = null;
    result = msg;
    ui.device.textContent = `Backend: ${msg.device}`;
    setStatus(`${msg.strokes.length} strokes`);
    renderResult();
    renderStats();
    ui.download.disabled = false;
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
  const { width: w, height: h } = source;
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

  // Keypoints.
  ui.keypoints.setAttribute("viewBox", `0 0 ${w} ${h}`);
  ui.keypoints.setAttribute("preserveAspectRatio", "none");
  const r = Math.max(w, h) / 160;
  const dots: string[] = [];
  const add = (pts: number[], color: string) => {
    for (let k = 0; k < pts.length; k += 2) dots.push(`<circle cx="${pts[k]}" cy="${pts[k + 1]}" r="${r}" fill="${color}" fill-opacity="0.85"/>`);
  };
  add(result.keypoints.end, "var(--end)");
  add(result.keypoints.sharp, "var(--sharp)");
  add(result.keypoints.junc, "var(--junc)");
  ui.keypoints.innerHTML = dots.join("");

  // Under-sampling map, drawn on the 2x grid and stretched over the prepared (cropped) area.
  const { gridW, gridH, usm, scale } = result;
  ui.usm.width = gridW;
  ui.usm.height = gridH;
  const ctx = ui.usm.getContext("2d")!;
  const img = ctx.createImageData(gridW, gridH);
  for (let p = 0; p < usm.length; p++) {
    if (!usm[p]) continue;
    const [cr, cg, cb] = usm[p] === 2 ? [234, 88, 12] : [147, 51, 234];
    img.data.set([cr, cg, cb, usm[p] === 2 ? 130 : 100], 4 * p);
  }
  ctx.putImageData(img, 0, 0);
  ui.usm.style.width = `${((result.prepared.W * scale) / w) * 100}%`;
  ui.usm.style.height = `${((result.prepared.H * scale) / h) * 100}%`;
  applyView();
}

/** Scale the image (and its overlays) to fit the stage. */
function fitCanvas() {
  if (!source) return;
  const pad = 48;
  const aw = Math.max(64, ui.stage.clientWidth - pad);
  const ah = Math.max(64, (window.innerWidth > 760 ? ui.stage.clientHeight : window.innerHeight * 0.7) - pad);
  const s = Math.min(aw / source.width, ah / source.height, 4);
  ui.canvas.style.width = `${Math.round(source.width * s)}px`;
  ui.canvas.style.height = `${Math.round(source.height * s)}px`;
}
window.addEventListener("resize", fitCanvas);

function applyView() {
  ui.input.style.opacity = String(Number(ui.opacity.value) / 100);
  $("opacityOut").textContent = `${ui.opacity.value}%`;
  ui.keypoints.style.display = ui.showKeypoints.checked ? "" : "none";
  ui.usm.style.display = ui.showUsm.checked ? "" : "none";
  ui.legend.hidden = !(ui.showKeypoints.checked || ui.showUsm.checked);
}

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
});

for (const el of document.querySelectorAll<HTMLInputElement>('input[name="model"]')) el.addEventListener("change", run);
ui.maxSide.addEventListener("change", run);
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
for (const el of [ui.opacity, ui.showKeypoints, ui.showUsm]) el.addEventListener("input", applyView);
ui.colorStrokes.addEventListener("input", renderResult);

ui.download.addEventListener("click", () => {
  if (!result || !source) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([result.svg], { type: "image/svg+xml" }));
  a.download = source.name.replace(/\.[^.]+$/, "") + ".svg";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

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
  const f = [...(e.dataTransfer?.files ?? [])].find((f) => f.type.startsWith("image/"));
  if (f) loadImage(f, f.name);
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
