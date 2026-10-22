// onnxruntime-web sessions for the two networks: backend selection (WebGPU, else WASM), model
// download with Cache Storage, and tile sizes derived from the device's buffer limits.

import * as ort from "onnxruntime-web/webgpu";
import { chooseTile, NDC_HALO, THIN_HALO, UDF_HALO, type RunFn } from "./tiling";

export type ModelVariant = "light" | "full";
export type Backend = "webgpu" | "wasm";

const CACHE = "deepsketch-models-v1";

// Resolved against the page's base URL, which the main thread sends (the worker's own location
// is inside assets/).
let modelBase = "";

/** Set where models (VITE_MODEL_BASE, default <app>/models/) and the ORT runtime (<app>/ort/) live. */
export function setAppBase(appBase: string) {
  modelBase = new URL(import.meta.env.VITE_MODEL_BASE ?? "models/", appBase).href;
  // onnxruntime-web's WebAssembly runtime, copied to public/ort/ by scripts/copy-ort.mjs.
  ort.env.wasm.wasmPaths = new URL("ort/", appBase).href;
}

/** Widest intermediate tensor per output cell (channels x 4 bytes); see DESIGN.md §3.2. */
const UDF_BYTES_PER_CELL: Record<ModelVariant, number> = { light: 96 * 4, full: 256 * 4 };
const NDC_BYTES_PER_CELL = 3 * 64 * 4;

async function fetchModel(name: string, onProgress?: (loaded: number, total: number) => void): Promise<Uint8Array> {
  const url = new URL(name, modelBase).href;
  let cache: Cache | null = null;
  try {
    cache = await caches.open(CACHE);
    const hit = await cache.match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch {
    cache = null; // Cache Storage unavailable (e.g. insecure context); just download.
  }
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`could not download ${name} (${res.status})`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress?.(loaded, total);
  }
  const bytes = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    bytes.set(c, off);
    off += c.length;
  }
  try {
    await cache?.put(url, new Response(bytes, { headers: { "content-type": "application/octet-stream" } }));
  } catch {
    // Quota exceeded etc.: not fatal.
  }
  return bytes;
}

export interface DeviceInfo {
  backend: Backend;
  /** Bytes allowed for the largest single intermediate tensor. */
  tensorBudget: number;
  description: string;
}

let deviceInfo: Promise<DeviceInfo> | null = null;
const DEFAULT_GPU_BUDGET = 256 << 20;

/** Optional overrides, e.g. from URL parameters: ?backend=wasm&budget=256 (MB). */
export interface DeviceOverrides {
  backend?: Backend;
  budgetMB?: number;
}

export function detectDevice(overrides: DeviceOverrides = {}): Promise<DeviceInfo> {
  deviceInfo ??= (async (): Promise<DeviceInfo> => {
    const d = await detect(overrides.backend);
    if (overrides.budgetMB) d.tensorBudget = overrides.budgetMB << 20;
    return d;
  })();
  return deviceInfo;
}

async function detect(force?: Backend): Promise<DeviceInfo> {
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  if (gpu && force !== "wasm") {
    try {
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      if (adapter) {
        const limit = Math.min(adapter.limits.maxStorageBufferBindingSize, adapter.limits.maxBufferSize);
        const info = adapter.info;
        return {
          backend: "webgpu",
          tensorBudget: Math.min(limit, DEFAULT_GPU_BUDGET),
          description: `WebGPU${info?.vendor ? ` (${[info.vendor, info.architecture].filter(Boolean).join(" ")})` : ""}`,
        };
      }
    } catch {
      // fall through to WASM
    }
  }
  const threads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;
  ort.env.wasm.numThreads = threads;
  return { backend: "wasm", tensorBudget: 256 << 20, description: `WebAssembly (CPU, ${threads} thread${threads > 1 ? "s" : ""})` };
}

export interface Networks {
  variant: ModelVariant;
  udf: RunFn;
  ndc: RunFn;
  udfTile: number;
  ndcTile: number;
  device: DeviceInfo;
}

const loaded = new Map<ModelVariant, Promise<Networks>>();

// The WebGPU execution provider cannot create two sessions concurrently.
let sessionQueue: Promise<unknown> = Promise.resolve();

async function createSession(bytes: Uint8Array, backend: Backend, channels = 1): Promise<RunFn> {
  const created = sessionQueue.then(() =>
    ort.InferenceSession.create(bytes, {
      executionProviders: backend === "webgpu" ? ["webgpu", "wasm"] : ["wasm"],
      graphOptimizationLevel: "all",
    }),
  );
  sessionQueue = created.catch(() => {});
  const session = await created;
  return async (input, h, w) => {
    const feeds = { [session.inputNames[0]]: new ort.Tensor("float32", input, [1, channels, h, w]) };
    const out = await session.run(feeds);
    const res = session.outputNames.map((n) => out[n].data as Float32Array);
    for (const n of session.outputNames) out[n].dispose();
    return res;
  };
}

export function loadNetworks(variant: ModelVariant, onProgress?: (message: string, frac: number) => void): Promise<Networks> {
  let p = loaded.get(variant);
  if (!p) {
    p = (async () => {
      const device = await detectDevice();
      const names = [`udf_${variant}.onnx`, `ndc_${variant}.onnx`];
      const bytes: Uint8Array[] = [];
      for (const name of names) {
        bytes.push(
          await fetchModel(name, (l, t) => onProgress?.(`Downloading ${name} (${(l / 1e6).toFixed(1)}${t ? ` / ${(t / 1e6).toFixed(1)}` : ""} MB)`, t ? l / t : 0)),
        );
      }
      onProgress?.("Initializing networks", 1);
      const udf = await createSession(bytes[0], device.backend);
      const ndc = await createSession(bytes[1], device.backend);
      return {
        variant,
        udf,
        ndc,
        device,
        udfTile: chooseTile(device.tensorBudget, UDF_BYTES_PER_CELL[variant], 2, UDF_HALO, 1024),
        ndcTile: chooseTile(device.tensorBudget, NDC_BYTES_PER_CELL, 1, NDC_HALO, 2048),
      };
    })();
    p.catch(() => loaded.delete(variant));
    loaded.set(variant, p);
  }
  return p;
}

export type Preprocessor = "thin" | "anime2sketch";

const PREPROCESSOR_FILES: Record<Preprocessor, string> = { thin: "thin.onnx", anime2sketch: "anime2sketch.w16.onnx" };
const preprocessors = new Map<Preprocessor, Promise<{ run: RunFn; tile: number }>>();

/** The optional pre-processing networks, downloaded on first use. */
export function loadPreprocessor(name: Preprocessor, onProgress?: (message: string, frac: number) => void) {
  let p = preprocessors.get(name);
  if (!p) {
    p = (async () => {
      const device = await detectDevice();
      const file = PREPROCESSOR_FILES[name];
      const bytes = await fetchModel(file, (l, t) =>
        onProgress?.(`Downloading ${file} (${(l / 1e6).toFixed(1)}${t ? ` / ${(t / 1e6).toFixed(1)}` : ""} MB)`, t ? l / t : 0),
      );
      onProgress?.(`Initializing ${name === "thin" ? "line thinning" : "Anime2Sketch"}`, 1);
      const run = await createSession(bytes, device.backend, name === "thin" ? 1 : 3);
      // Line thinning's widest tensor: 64 channels at full resolution.
      return { run, tile: name === "thin" ? chooseTile(device.tensorBudget, 64 * 4, 1, THIN_HALO, 4096) : 512 };
    })();
    p.catch(() => preprocessors.delete(name));
    preprocessors.set(name, p);
  }
  return p;
}
