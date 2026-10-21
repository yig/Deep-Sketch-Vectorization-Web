# Deep Sketch Vectorization in the Browser: Design & Architecture

This is a browser port of `../Deep-Sketch-Vectorization`, the reference implementation of *Deep Sketch
Vectorization via Implicit Surface Extraction* (Yan et al., SIGGRAPH 2024; paper source in
`../paper/main-latexpand.tex`).

**Status (2026-10-22):** the automatic pipeline, equivalent to `predict_s1.py`, runs in the browser
on WebGPU or WebAssembly. Every post-processing stage matches the Python reference on golden
fixtures (reference with two bug fixes applied, §7). A first version of interactive refinement
(editing keypoints and the USM, §8) works. Commit `14b4fec` is the bug-for-bug parity version. Commit `14b4fec` is the bug-for-bug parity version.

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
| `src/pipeline/topology.ts` | `refine_topology` & co. | keypoint alignment (`autoMarks`), USM regions (`autoUsm`), pure `surgery` (connection points, per-valence rules, X-junction pairing) |
| `src/pipeline/downsample.ts` | `downsample_all` | dual-contouring downsampling preserving under-sampled blocks |
| `src/pipeline/grouping.ts` | `svg_tools.simplify_graph` (`'hybird'`) | stroke chaining |
| `src/pipeline/smoothing.ts` | `rdp` package, `fitCurves.py` | RDP (iterative), Schneider Bézier fitting |
| `src/pipeline/vectorize.ts` | `predict_s1.__main__` | `prepare` (per image) and `finish` (per edit / option change), SVG output |
| `src/inference/tiling.ts` | — | overlapping-tile execution for both networks, tile sizing |
| `src/inference/session.ts` | — | ORT sessions, backend detection, model download + Cache Storage |
| `src/worker.ts` | `server.py` | Web Worker: runs everything; caches network outputs and `Prepared` per (image, model, size) |
| `src/protocol.ts` | — | main thread ↔ worker messages, edit state, per-cell status codes |
| `src/editor.ts` | `web/js/canvas.js` | interactive refinement: zoom / pan, keypoint and USM tools, undo / redo |
| `src/main.ts`, `index.html`, `src/style.css` | — | UI |
| `tools/export_onnx.py` | — | checkpoint → ONNX export + check against PyTorch |
| `tools/make_fixtures.py` | — | runs the reference pipeline and dumps every intermediate |
| `scripts/copy-ort.mjs` | — | copies ORT's WASM runtime into `public/ort/` |

`prepare` runs everything that depends only on the network outputs: keypoints, edge cleanup,
line maps, and the automatic marks and USM. `finish(prepared, marks, usm, options)` runs surgery,
downsampling, grouping and smoothing (~0.2 s). The automatic result is
`finish(prep, prep.autoMarks, prep.autoUsm)`, and an edited result is the same call with the user's
marks and USM.

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
Detected keypoints are flat `[x0, y0, …]` arrays per type. The surgery uses **marks**
`{type, cell, x, y}`: a keypoint bound to the grid cell whose region it counts for, with the position
lines are connected to. The **USM** is a `Uint8Array` on the grid: 0 none, 1 refine, 2 keep detail
only. Segments are flat `[sx, sy, ex, ey, …]`. Strokes are flat polylines.

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

`tests/editing.test.ts` (7 tests) checks the editing semantics on a fixture's network outputs:
- the automatic pass equals `surgery(autoMarks, autoUsm)`;
- erasing a refined region removes its surgery, and restoring the USM gives the automatic result back;
- adding a keypoint to an unrefined region makes it refinable;
- painting a new region over a line rebuilds it;
- painting next to a detail-only region keeps it separate;
- `snapMark` snaps or keeps the position.

`tests/e2e.test.ts` (`npm run test:e2e`) runs the ONNX models in onnxruntime-web (WASM, Node) with
tiling, then the TypeScript pipeline. It passes on all fixtures: UDF and aux within 1e-3 of PyTorch,
no edge-flag flips, and segments identical to the reference's grouping input.

The deployed app was also driven in headless Chrome (Playwright, WebGPU and WASM). Before the bug
fixes, on the procedural sample, the reference produced 54 strokes and the port 55. The stroke fragmentation seen with "Color
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
4. **Grouping keypoints come from the marks.** These are the keypoints the surgery used, one per
   vertex. The reference passes its aligned keypoint lists instead, which can contain duplicates,
   keep sharp turns whose vertex was re-marked as a junction, and miss junctions aligned at
   valence 2. This keeps the automatic and the edited paths identical.
5. **Bézier smoothing works.** The reference's `fitBezier` raises `AttributeError` (it calls
   `.length()` on a list), and `simplify_SVG` swallows it, so `--bezier` is a silent no-op there.
   The port runs the underlying Schneider fit (`fitCurve(points, 0.1)`).

### Reference bugs fixed (2026-10-22)

The fixtures are generated from the reference with the same fixes patched in (`tools/make_fixtures.py`),
so the golden tests still compare like with like. For bug-for-bug parity with the reference, check
out commit `14b4fec`.

1. `downsample_all`: `usm_downsample[r_mask] = False` indexed with a **uint8** array, which NumPy
   treats as integer row indices: it cleared coarse rows 0 and 1 instead of dropping the
   keypoint-free USM regions. Fixed: those regions are dropped. On the four fixtures, 40–65% of the
   downsampling's USM regions had no keypoint, and the fix cuts the segment count by 20–30%.
2. Stroke grouping received keypoints in **1× pixel units** but segments in **2× SVG units**, so its
   keypoint alignment matched the wrong vertices. Fixed: keypoints are scaled to 2×. The alignment
   thresholds (1.5, 1.5, 5) are kept, now in 2× units.

### Remaining reference quirks (kept)

1. `refine_topology` only aligns junction keypoints to valence-3 vertices (plus a pass at valence 2),
   never valence 4.
2. Schneider fitting puts control points far outside the curve for tiny segments: overshooting
   control points that, rendered, would show as loops. It's visible only with Bézier smoothing.
3. Grouping quality: straight edges often split into several strokes, and parallel duplicate segments
   survive along some lines (also in the reference output).

## 8. Interactive refinement

The reference's web UI lets the user edit keypoints and the USM, then re-runs `refine_topology` in
manual mode on the modified regions only (`usm_surgery('modify')`). It keeps a server-side history
in npz files. That incremental bookkeeping has gaps: a region deleted entirely is never un-refined,
and a modified region's original lines are not restored (the restoring code is commented out).

**Design: the edit state determines the result.** The state is (marks, USM). Every edit re-runs
`finish` on the whole state from the cached, unmodified `DCBase`. The automatic pass is the same
function on the automatic state, so with no edits the result is identical, and undo is a plain
snapshot stack on the main thread (60 steps, about 1 MB per USM snapshot at 512 px).

Semantics:
- **Regions** are 4-connected groups of cells with the same USM value. Regions of 1s get surgery.
  Value 2 marks regions the automatic pass kept out of surgery (larger than long side / 50 cells, or
  closing merged several predicted regions). Their geometry is still kept at full resolution by the
  downsampling. **Painting only affects the painted cells:** paint touching a detail-only region
  forms its own region, and polylines entering it from the detail region are reconnected like any
  others. In the automatic state, regions of 1s and 2s never touch, so this labeling equals the
  reference's components of the whole USM.
- **Keypoints inside a region** are the marks whose cell is in it. Marks from the automatic pass sit
  on the dual-contouring vertex they were aligned to. A keypoint the user adds or moves is snapped
  to the nearest vertex within 1.5 px (`snapMark`); with none near, it keeps its position and counts
  for the cell under it. The worker returns snapped positions and the UI adopts them.
- The surgery rules are the reference's `update_line_maps` with `full_auto_mode` branches. For
  example, a region two polylines enter is rebuilt by joining them, and with one junction keypoint
  every entering polyline is connected to it.
- **Status per region**, shown as USM color: refined (orange), no rule applies with the keypoints
  inside, e.g. several junctions (red), not refinable because the region has holes or nothing enters
  it (gray), detail only (purple). Unprocessed edits show as pending (blue).
- Edits survive post-processing option changes. They are cleared when the network outputs change
  (new image, model, or working size), with a status message.

Pointer handling calls `preventDefault` on pointerdown, and the image is `draggable="false"`.
Otherwise Chrome starts a native image drag, which cancels the pointer stream mid-stroke.

UI (`src/editor.ts`): a floating toolbar with these tools:
- **Pan** (V; or hold Space; scroll zooms about the cursor).
- **Keypoints** (K): click to add with the current type (1/2/3), drag to move, double-click to cycle
  the type, Alt-click or Delete to remove.
- **USM brush** (B) and **Eraser** (E), with a size control.
- **Lasso** (L): draw a loop to fill the cells whose centers are inside; hold Alt on release to erase.

Also: undo / redo (⌘Z, ⇧⌘Z / Ctrl+Y), reset to the automatic result (undoable), and zoom buttons
(− / Fit / +, keys −, 0, +). Edits are coalesced: while a run is in flight, further edits trigger
one more run when it returns.

Verified in headless Chrome: painting a disc over the sample's chimney/roof T-junction rebuilt only
the disc, as straight lines meeting at the junction keypoint, and left the adjacent detail-only
strip along the roof with its original lines. Undo, redo, reset, eraser, and keeping edits across an option change all work.

## 9. Next steps

- Editing polish:
  - show which keypoints are inside which region, and why a region is not refined;
  - pinch zoom on touch screens.
- Downloading the edit state (marks + USM) to resume later.
- fp16 models for the full variant (half the download). This needs an accuracy check against fp32,
  since a flipped edge flag changes topology.
- Hosting: the models could be served from Hugging Face (`VITE_MODEL_BASE`) instead of the site.
- Optional pre-processing networks: the line thinner (1 MB) and Anime2Sketch (208 MB, lazy).
