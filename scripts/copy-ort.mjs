// Copy onnxruntime-web's WebAssembly runtime into public/ort/ so it is served as static files.
// (The bundled ORT build cannot spawn its WASM threads from inside a Vite chunk.)
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(root, "node_modules", "onnxruntime-web", "dist");
const dst = join(root, "public", "ort");
mkdirSync(dst, { recursive: true });
for (const f of ["ort-wasm-simd-threaded.asyncify.mjs", "ort-wasm-simd-threaded.asyncify.wasm"]) copyFileSync(join(src, f), join(dst, f));
