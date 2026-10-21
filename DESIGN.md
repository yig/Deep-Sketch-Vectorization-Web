# Deep Sketch Vectorization in the Browser: Design & Architecture

This is a browser port of `../Deep-Sketch-Vectorization`, the reference implementation of *Deep Sketch
Vectorization via Implicit Surface Extraction* (Yan et al., SIGGRAPH 2024; paper source in
`../paper/main-latexpand.tex`; AGENTS.md says `../main-latexpand.tex`, but the file is in `../paper/`).

**Status (2026-10-22):** the automatic pipeline, equivalent to `predict_s1.py`, runs in the browser
on WebGPU or WebAssembly. Every post-processing stage matches the Python reference on golden
fixtures. Interactive refinement (editing keypoints and the USM) is the next milestone.

## 1. Quick start

```sh
npm install
npm run models     # once: exports ../pretrained/*.pth to public/models/*.onnx (needs uv; see tools/)
npm run dev        # http://localhost:5173
npm test           # golden tests (needs tests/fixtures, see §6)
npm run test:e2e   # ONNX + tiling + post-processing vs PyTorch, in Node (slow: ~4 min)
npm run build      # static site in dist/ (models are copied from public/models)
```

URL parameters for debugging: `?backend=wasm` forces the CPU backend; `?budget=512` sets the
per-tensor GPU memory budget in MB (default 256), which controls the tile size (§4.2).

## 2. Decisions

These were agreed with the user on 2026-10-22, after the strategy discussion that AGENTS.md asks for:

| topic | decision | alternatives considered |
|---|---|---|
| NN inference | ONNX Runtime Web: WebGPU execution provider, WASM fallback; tiled | TensorFlow.js (fragile conversion chain); hand-written WebGPU kernels (only if ever needed for speed) |
| post-processing | hand-port to TypeScript, verified stage by stage against the Python reference | Pyodide running the original Python (adds 30–60 MB and a slow startup); Pyodide first, then migrate |
| UI | new UI around in-browser state | reuse the eel-based `web/` UI |
| first milestone | automatic pipeline (`predict_s1.py` CLI path) | full interactive parity first |
| models | ship light (default) and full; user-selectable | full only / light only |

## 3. Pipeline

```
image ──► prepareImage ──► UDF net (tiled) ──► udf (2H+1)²         ──► NDC net (tiled) ──► edge logits, offsets (2H×2W)
          gray, ≤512 px,                      aux (5 × 2H × 2W):                            │
          crop to ×8                           all/end/sharp/junc UDFs, USM                 │
                                                  │                                          ▼
                                                  ├─► underSamplingMap ──────┐   edgeMapFromLogits → refineEdgeMap
                                                  └─► extractKeypoints ──────┤   vertexMapFromOffsets → lineMaps
                                                                             ▼
                                         refineTopology (align keypoints, USM surgery) → downsampleAll
                                                                             │
                                         segments ×2 → groupStrokes → [rdp | fitBezier] → toSvg
```

All grids follow the reference's conventions (`src/pipeline/dualContouring.ts` header). The
grid is 2H × 2W cells for an H × W input. `edge.x[i,j]` links cell (i−1, j) to (i, j), and
`edge.y[i,j]` links (i, j−1) to (i, j). Vertex `pt[i,j]` is (x, y) in input-pixel units, equal
to the offset plus (0.25 + 0.5j, 0.25 + 0.5i).

### Module map

| file | reference | contents |
|---|---|---|
| `src/pipeline/image.ts` | `predict_s1.load_img` | RGBA→gray (PIL weights), INTER_AREA downscale, crop to multiples of 8 |
| `src/pipeline/imgproc.ts` | cv2 / skimage / scipy | connected components (raster-order labels like cv2), dilate/erode, correlate/convolve2d, Suzuki border following (cv2.findContours, APPROX_NONE) + hole count, skimage 0.26 Zhang–Suen skeletonize (its exact LUT), peak_local_max, INTER_AREA |
| `src/pipeline/keypoints.ts` | `predict_UDF`, `keypt_tools.py` | USM threshold, `udf_filter`, keypoint peaks |
| `src/pipeline/dualContouring.ts` | `ndc_tools.py` | edge flags, vertex map, `cut_branch`, `remove_stray`, `remove_parallel`, `fill_edge_hole`, line maps |
| `src/pipeline/topology.ts` | `refine_topology` & co. | keypoint alignment, USM regions, connection points, per-valence surgery, X-junction pairing |
| `src/pipeline/downsample.ts` | `downsample_all` | dual-contouring downsampling preserving under-sampled blocks |
| `src/pipeline/grouping.ts` | `svg_tools.simplify_graph` (`'hybird'`) | stroke chaining |
| `src/pipeline/smoothing.ts` | `rdp` package, `fitCurves.py` | RDP (iterative), Schneider Bézier fitting |
| `src/pipeline/vectorize.ts` | `predict_s1.__main__` | orchestration, SVG output |
| `src/inference/tiling.ts` | — | overlapping-tile execution for both networks, tile sizing |
| `src/inference/session.ts` | — | ORT sessions, backend detection, model download + Cache Storage |
| `src/worker.ts` | — | Web Worker: runs everything; caches network outputs per (image, model, size) |
| `src/main.ts`, `index.html`, `src/style.css` | — | UI |
| `tools/export_onnx.py` | — | checkpoint → ONNX export + check against PyTorch |
| `tools/make_fixtures.py` | — | runs the reference pipeline and dumps every intermediate |
| `scripts/copy-ort.mjs` | — | copies ORT's WASM runtime into `public/ort/` |

The automatic path corresponds to `refine_topology(..., downsample=True, manual=False,
full_auto_mode=True)` with `down_rate=8`, `map_to_lines(refine=True)`, `skip_len=4`, and input
resized to a long side of ≤ 512 (the web UI's `open_img` behavior).

## 4. Inference

### 4.1 Export (`tools/export_onnx.py`)

- Opset 17, eval mode with BatchNorm folded, dynamic H/W. Outputs: `udf [1,1,2H+1,2W+1]` and `aux [1,5,2H,2W]`
  (UDF net); `edge [1,4,2H,2W]` logits and `offset [1,2,2H,2W]` (NDC net; the clip to [0,1] is in the graph,
  and the unused fifth logit channel is dropped).
- `padding='same'` is rewritten as explicit padding: ORT rejects `auto_pad=SAME_*` with dilation.
- **Concat-free ResNeXt.** `plusconv(cat(branch_1..8))` is exported as `Σ_k conv1x1_k(branch_k)`.
  It's the same function, but the 8 × C-channel concat is never materialized. For the full model at 2×
  resolution on a 512² image, that tensor alone was 2048 ch × 1024² × 4 B = 8 GiB, which is the
  reference README's "8 GB VRAM".
- The export is checked against the *unmodified* PyTorch modules: max abs error ≤ 1e-5 (UDF), ≤ 3e-4 (NDC logits).

| model | params | ONNX fp32 | fp16 (exported, unused so far) |
|---|---|---|---|
| udf_light | 2.3 M | 9 MB | 4.5 MB |
| udf_full | 14.7 M | 56 MB | 28 MB |
| ndc_light / ndc_full (different weights, same architecture) | 1.1 M | 4.3 MB | 2.2 MB |

### 4.2 Tiling (`src/inference/tiling.ts`)

Both networks are fully convolutional, so overlapping tiles reproduce whole-image inference
**exactly** (measured offline with ORT: max error 0.0000) under two conditions:
- UDF net: a halo of **64 input pixels** (32 px gave errors up to 6e-3), and tile origins and sizes
  that are multiples of 8, the net's downsampling lattice. Each tile owns the 2× samples of its interior; the
  `2H+1`-th corner row and column come from the tiles on the image edge.
- NDC net: a halo of **32 grid cells**.

The tile size is the largest multiple of 8 whose widest tensor fits the per-tensor budget:
`min(adapter maxStorageBufferBindingSize, maxBufferSize, 256 MB)` on WebGPU, and 256 MB on WASM.
The widest tensors are C × 4 bytes per cell, with C = 96 (light) or 256 (full) for the UDF net and
3 × 64 for the NDC net. With the defaults on a 512² image, tiles are 288 px (UDF light), 128 px (UDF full),
and 520 cells (NDC).

### 4.3 Runtime

- The worker imports `onnxruntime-web/webgpu`, aliased in `vite.config.ts` to ORT's non-bundled
  build. ORT's WASM runtime (`ort-wasm-simd-threaded.asyncify.{mjs,wasm}`, 26 MB) is served from `public/ort/`.
  The bundled ORT build cannot start its WASM threads inside a Vite chunk: it fetches itself by filename and gets a 404.
- The page sends its base URL to the worker. Models and the ORT runtime resolve against it, not against
  the worker's own `assets/` URL.
- The WebGPU EP cannot create two sessions concurrently, so they are created one after the other.
- Multi-threaded WASM needs cross-origin isolation. The dev and preview servers send COOP/COEP headers;
  without them (e.g. GitHub Pages) the WASM backend falls back to 1 thread. WebGPU is unaffected.
- Models are fetched from `VITE_MODEL_BASE` (default `<app>/models/`) and kept in Cache Storage.

### 4.4 Measured (Apple M5 Max, headless Chrome)

| run | UDF net | NDC net | post-processing |
|---|---|---|---|
| light, WebGPU, 512 × 400 sample (before the topology speedup) | 0.97 s | 0.41 s | 0.87 s |
| full, WebGPU, 512² benchmark image, 256 MB budget (128 px tiles) | 3.3 s | 0.53 s | 0.63 s |
| same, `?budget=1024` (384 px tiles) | 1.75 s | 0.66 s | — |
| light, WASM 8 threads, 512 × 400 sample | ≈ 5 s | ≈ 6 s | 13 s wall in total |

Post-processing is dominated by the edge cleanup (≈ 0.45 s). Changing post-processing options re-uses
the cached network outputs.

## 5. Data model

The reference's npz files and NumPy object arrays become typed arrays: masks are `Uint8Array`,
the vertex map is `Float32Array` (h·w·2), and line maps are `{mask: Uint8Array, seg: Float64Array(h·w·4)}`.
Keypoints are flat `[x0, y0, …]` arrays per type. Segments are flat `[sx, sy, ex, ey, …]`. Strokes are flat polylines.

Float behavior follows NumPy where it changes outcomes. The vertex map stays float32. Region
averages and midpoints use sequential float32 sums with a float32 divide, as `np.mean` does on
float32 arrays. Bézier fitting uses NumPy's operation order. With these, the topology stage matches
the reference to the last bit.

## 6. Verification

`tools/make_fixtures.py IMAGE… [--variant light|full]` runs the reference pipeline (PyTorch, CPU) and
writes `tests/fixtures/<image>-<variant>/` (100–160 MB each, gitignored). Each directory holds `image`,
`udf`, `aux`, `usm_pre`, `udf_filtered`, `keypts_2x.json`, `edge_logits`, `offset`, `pt_map`, `edge_raw`,
`edge_cut1`, `skel_parallel`, `edge_filled`, `edge_final`, `lines_raw`, `linemap_{x,y}[_mask]`, `lines_refine`,
`usm_applied`, `usm_uncertain`, `keypts_aligned.json`, `refined_regions.json`, the downsampling inputs
(`down_usm_in`, `valence_map`, `pt_type`, `linemap_{x,y}_surgery`), `lines_down`, `grouping_in`, and
`strokes{,_rdp,_bezier}.json`.
Current fixtures: three light-model and one full-model run on `../SketchVG/png_hard_texture` images.

`tests/pipeline.test.ts` (28 tests, ~9 s) runs each TypeScript stage on the fixture's inputs:

- **exact:** USM, filtered UDFs, keypoints, edge flags, every edge-cleanup step, line maps, aligned keypoints,
  USM labels (applied and uncertain), surgery line maps, refined segments per region, RDP.
- **≤ 1e-4:** refined and downsampled segment coordinates (float64 vs. NumPy float32 sums).
- **relative 1e-6:** Bézier control points. Near-degenerate fits put control points thousands of px away.
- **structural:** stroke grouping (§7, deviation 1). Every stroke segment must be an input segment;
  covered segments within 5% of the reference, and stroke count within 25%.

`tests/e2e.test.ts` (`npm run test:e2e`) runs the ONNX models in onnxruntime-web (WASM, Node) with
tiling, then the TypeScript pipeline. It passes on all fixtures: UDF and aux within 1e-3 of PyTorch,
no edge-flag flips, and segments identical to the reference's grouping input.

The deployed app was also driven in headless Chrome (Playwright, WebGPU and WASM). On the procedural
sample, the reference produces 54 strokes and the port 55. The stroke fragmentation seen with "Color
strokes" is present in the reference too.

## 7. Known issues

### Deliberate deviations from the reference

1. **Deterministic stroke grouping.** The reference picks start points with `random.sample` and
   iterates Python sets in hash order. The port always takes the smallest point index (points are
   ordered as `np.unique` orders them). Results differ in chaining, not in coverage.
2. **Alpha.** Transparent pixels are composited over white. PIL's `convert("L")` drops alpha, which
   turns transparent backgrounds black.
3. **No crashes.** Where the reference would raise, the port continues:
   - a zero-length tangent (ZeroDivisionError) gets a zero tangent;
   - a valence-4 region whose two pairings both intersect (AssertionError) is left unrefined.
4. **Bézier smoothing works.** The reference's `fitBezier` raises `AttributeError` (it calls
   `.length()` on a list), and `simplify_SVG` swallows it, so `--bezier` is a silent no-op there.
   The port runs the underlying Schneider fit (`fitCurve(points, 0.1)`).

### Reference bugs reproduced for parity (candidates to fix, pending discussion)

1. `downsample_all`: `usm_downsample[r_mask] = False` indexes with a **uint8** array. NumPy treats
   that as integer row indices, so instead of dropping keypoint-free USM regions it clears coarse
   rows 0 and 1 (whenever any region lacks a keypoint).
2. Stroke grouping receives keypoints in **1× pixel units** but segments in **2× SVG units**, so its
   keypoint alignment (end/sharp/junction start points) matches the wrong vertices.
3. `refine_topology` only aligns junction keypoints to valence-3 vertices (plus a pass at valence 2),
   never valence 4.
4. Schneider fitting puts control points far outside the curve for tiny segments: overshooting
   control points that, rendered, would show as loops. It's visible only with Bézier smoothing.
5. Grouping quality: straight edges often split into several strokes, and parallel duplicate segments
   survive along some lines (also in the reference output).

## 8. Next steps

- Interactive refinement: edit keypoints and USM regions, then re-run `refine_topology` in manual
  mode on the changed regions only, with undo. That needs the manual-mode branches of
  `update_line_maps` / `query_keypt_by_region_no_alginment` and `update_usm`.
- Decide which reference bugs (§7) to fix, behind a "reference parity" switch.
- fp16 models for the full variant (half the download). This needs an accuracy check against fp32,
  since a flipped edge flag changes topology.
- Hosting: the models could be served from Hugging Face (`VITE_MODEL_BASE`) instead of the site.
- Optional pre-processing networks: the line thinner (1 MB) and Anime2Sketch (208 MB, lazy).
