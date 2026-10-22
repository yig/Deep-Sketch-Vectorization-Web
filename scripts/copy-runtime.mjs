// Copy runtime files that must be served as-is (not bundled) into public/:
// - onnxruntime-web's WebAssembly runtime -> public/ort/ (the bundled ORT build cannot spawn its
//   WASM threads from inside a Vite chunk);
// - coi-serviceworker -> public/coi-serviceworker.js (it registers itself by its own URL, so it
//   must be a separate file at the app root).
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const modules = join(root, "node_modules");
const pub = join(root, "public");
mkdirSync(join(pub, "ort"), { recursive: true });
for (const f of ["ort-wasm-simd-threaded.asyncify.mjs", "ort-wasm-simd-threaded.asyncify.wasm"])
  copyFileSync(join(modules, "onnxruntime-web", "dist", f), join(pub, "ort", f));
copyFileSync(join(modules, "coi-serviceworker", "coi-serviceworker.min.js"), join(pub, "coi-serviceworker.js"));
