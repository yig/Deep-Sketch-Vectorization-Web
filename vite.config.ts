import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

// Cross-origin isolation enables multi-threaded WebAssembly (onnxruntime-web's CPU fallback).
const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  base: "./",
  server: { headers: isolation },
  preview: { headers: isolation },
  resolve: {
    // Use ORT's build that loads its WebAssembly runtime from env.wasm.wasmPaths (public/ort/),
    // rather than the bundle that inlines it (whose WASM threads cannot start inside a Vite chunk).
    alias: { "onnxruntime-web/webgpu": fileURLToPath(new URL("node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs", import.meta.url)) },
  },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  worker: { format: "es" },
  build: { target: "es2022" },
  test: {
    include: ["tests/**/*.test.ts"],
    exclude: process.env.E2E ? [] : ["tests/e2e.test.ts"],
    testTimeout: 120_000,
  },
});
