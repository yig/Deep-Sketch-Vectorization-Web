"""Run the reference Python pipeline (predict_s1.py's automatic CLI path) on test images and dump
every intermediate stage as .npy / .json, for golden tests of the TypeScript port.

Usage:
    uv run python make_fixtures.py IMAGE [IMAGE ...] [--variant light|full] [--out ../tests/fixtures]

Each image produces <out>/<stem>-<variant>/; DESIGN.md (Verification) lists the files.

Determinism: the reference stroke grouping picks start points with random.sample(sorted(S), 1).
We patch it to pick min(S), which the TypeScript port also does.
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

REF = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Deep-Sketch-Vectorization"))
sys.path.insert(0, REF)
sys.path.insert(0, os.path.join(REF, "utils"))

import predict_s1 as P  # noqa: E402
import utils.ndc_tools as NT  # noqa: E402
import utils.svg_tools as ST  # noqa: E402
from utils.keypt_tools import udf_filter, extract_keypts  # noqa: E402

ST.sample = lambda population, k: list(population)[:k]


def _fix_downsample_all():
    """Reference bug: `usm_downsample[r_mask] = False` indexes with a uint8 array, which NumPy treats as
    row indices (clearing coarse rows 0 and 1) instead of a boolean mask. The browser port fixes it;
    patch the reference the same way so the fixtures stay comparable."""
    import inspect
    import textwrap
    src = textwrap.dedent(inspect.getsource(NT.downsample_all))
    bad = "usm_downsample[r_mask] = False"
    assert src.count(bad) == 1
    src = src.replace(bad, "usm_downsample[r_mask.astype(bool)] = False")
    exec(compile(src, "<downsample_all, patched by make_fixtures.py>", "exec"), NT.__dict__)


_fix_downsample_all()


def save(d, name, arr):
    np.save(os.path.join(d, name + ".npy"), np.ascontiguousarray(arr))


def save_json(d, name, obj):
    with open(os.path.join(d, name + ".json"), "w") as f:
        json.dump(obj, f)


def keypts_json(k):
    return {key: np.asarray(v, dtype=float).reshape(-1, 2).tolist() for key, v in k.items()}


def linemap_dense(lines_map):
    """numpy object array (h, w) of False | (start, end) -> float32 (h, w, 2, 2) and bool mask (h, w)."""
    h, w = lines_map.shape
    dense = np.zeros((h, w, 2, 2), np.float32)
    mask = np.zeros((h, w), bool)
    for i, j in zip(*np.where(lines_map != False)):  # noqa: E712
        s, e = lines_map[i, j]
        dense[i, j, 0] = s
        dense[i, j, 1] = e
        mask[i, j] = True
    return dense, mask


def paths_json(strokes):
    """svgpathtools Paths -> list of strokes, each a list of segments [[x0,y0],...] (2 pts for Line, 4 for Cubic)."""
    out = []
    for s in strokes:
        segs = []
        for seg in s:
            pts = [seg.start, seg.end] if isinstance(seg, ST.Line) else [seg.start, seg.control1, seg.control2, seg.end]
            segs.append([[p.real, p.imag] for p in pts])
        out.append(segs)
    return out


class Recorder:
    """Wrap a module-level function to record its last call's arguments and result."""

    def __init__(self, module, name):
        self.module, self.name, self.fn = module, name, getattr(module, name)
        self.calls = []
        setattr(module, name, self)

    def __call__(self, *args, **kwargs):
        import copy
        args_copy = copy.deepcopy(args)
        res = self.fn(*args, **kwargs)
        self.calls.append((args_copy, kwargs, res))
        return res


def run(image_path, variant, out_root, resize_to=512, down_rate=8, skip_len=4, eps=0.4):
    stem = os.path.splitext(os.path.basename(image_path))[0].replace(" ", "_")
    d = os.path.join(out_root, f"{stem}-{variant}")
    os.makedirs(d, exist_ok=True)
    pre = os.path.join(os.path.dirname(__file__), "..", "..", "pretrained")
    model_udf, args_udf = P.load_checkpoint_udf(os.path.join(pre, f"udf_{variant}.pth"))
    model_ndc, args_ndc = P.load_checkpoint_ndc(os.path.join(pre, f"ndc_{variant}.pth"))

    # Input: resize the long side to resize_to if larger (as the web UI's open_img does), crop to multiple of 8.
    pil = Image.open(image_path)
    ww, hh = pil.size
    resize = max(ww, hh) > resize_to
    img, img_np, (h, w), _ = P.load_img(image_path, "cpu", True, resize=resize, resize_to=resize_to, ui_mode=False)
    save(d, "image", img[0, 0].numpy().astype(np.float32))
    meta = dict(source=os.path.basename(image_path), variant=variant, h=h, w=w, original_w=ww, original_h=hh,
                resized=bool(resize), resize_to=resize_to, down_rate=down_rate, skip_len=skip_len, eps=eps,
                dist_clip=float(args_udf.dist_clip))

    # ---- Distance field prediction (predict_UDF) ----
    udf_topo, udf_all, udf_end, udf_sharp, udf_junc, usm = P.pre_udf(model_udf, img)
    save(d, "udf", P.D(udf_topo).astype(np.float32))
    save(d, "aux", np.stack([udf_all, udf_end, udf_sharp, udf_junc, usm]).astype(np.float32))
    usm_thr = 0.5
    usm_pre = (usm.clip(0, 1) < usm_thr) | (udf_all.clip(0, 1) < usm_thr / 3)
    save(d, "usm_pre", usm_pre)
    f_all, f_end, f_sharp, f_junc = [udf_filter(u.copy()) for u in (udf_all, udf_end, udf_sharp, udf_junc)]
    save(d, "udf_filtered", np.stack([f_all, f_end, f_sharp, f_junc]).astype(np.float32))
    _, keypts = extract_keypts([f_end, f_sharp, f_junc], 1000)
    save_json(d, "keypts_2x", keypts_json(keypts))
    keypts = {k: v / 2 for k, v in keypts.items()}

    # ---- Line reconstruction (predict_SVG) ----
    udf_in = udf_topo.clip(0, 1)
    edge_logits, pt_map = P.pre_svg(model_ndc, udf_in, [0.5])
    save(d, "edge_logits", P.D(edge_logits)[:4].astype(np.float32))
    with torch.no_grad():
        save(d, "offset", P.D(model_ndc.eval().forward(udf_in, [0.5])[1]).astype(np.float32))
    pt_map = P.D(pt_map.squeeze().permute(1, 2, 0))
    save(d, "pt_map", pt_map.astype(np.float32))
    em = edge_logits.permute(0, 2, 3, 1)[..., :-1].squeeze()
    ex, ey, _ = NT.pre_to_map(em)
    edge_raw = P.D(torch.stack((ex, ey), dim=-1)).astype(bool)
    save(d, "edge_raw", edge_raw)

    # map_to_lines(refine=True) step by step, then check against the real call.
    e = NT.cut_branch(edge_raw.copy())
    save(d, "edge_cut1", e)
    skel = NT.remove_parallel(e[..., 0] | e[..., 1], e)
    save(d, "skel_parallel", skel)
    e2 = NT.fill_edge_hole(e & skel[..., np.newaxis])
    save(d, "edge_filled", e2)
    e3 = NT.cut_branch(e2 & skel[..., np.newaxis])
    lines_raw, lmx, lmy, edge_final = NT.map_to_lines(edge_raw.copy(), pt_map, refine=True)
    assert (edge_final == e3).all()
    save(d, "edge_final", edge_final)
    save(d, "lines_raw", lines_raw.astype(np.float64))
    for name, lm in (("linemap_x", lmx), ("linemap_y", lmy)):
        dense, mask = linemap_dense(lm)
        save(d, name, dense)
        save(d, name + "_mask", mask)

    # ---- USM surgery + dual contouring downsampling (refine_topology, full auto) ----
    rec_down = Recorder(NT, "downsample_all")
    lines_refine, refined_dict, usm_applied, usm_uncertain = NT.refine_topology(
        edge_final, pt_map, usm_pre, lmx, lmy, keypts,
        down_rate=down_rate, downsample=True, full_auto_mode=True)
    NT.downsample_all = rec_down.fn
    save(d, "lines_refine", np.asarray(lines_refine, np.float64))
    save(d, "usm_applied", np.asarray(usm_applied, np.int32))
    save(d, "usm_uncertain", np.asarray(usm_uncertain, np.int32))
    save_json(d, "keypts_aligned", keypts_json(keypts))
    save_json(d, "refined_regions", {str(k): np.asarray(v, float).reshape(-1, 2).tolist() for k, v in refined_dict.items()})
    (args, kwargs, lines_down) = rec_down.calls[-1]
    _, _, _, usm_regions, valence_map, pt_type, lmx_s, lmy_s = args
    save(d, "down_usm_in", np.asarray(usm_regions, bool))
    save(d, "valence_map", np.asarray(valence_map, np.int32))
    save(d, "pt_type", np.asarray(pt_type, np.int32))
    for name, lm in (("linemap_x_surgery", lmx_s), ("linemap_y_surgery", lmy_s)):
        dense, mask = linemap_dense(lm)
        save(d, name, dense)
        save(d, name + "_mask", mask)
    save(d, "lines_down", np.asarray(lines_down, np.float64))

    # ---- Stroke grouping + smoothing (simplify_SVG) ----
    with tempfile.TemporaryDirectory() as tmp:
        svg = os.path.join(tmp, "x_refine.svg")
        NT.lines_to_svg(lines_refine * 2, w * 2, h * 2, svg)
        paths, _ = ST.open_svg_flatten(svg)
        save(d, "grouping_in", np.array([[[p.start.real, p.start.imag], [p.end.real, p.end.imag]] for p in paths], np.float64))
        # Reference bug: simplify_SVG passes 1x keypoints with 2x SVG coordinates. Fixed: scale them.
        keypts2x = {k: np.asarray(v) * 2 for k, v in keypts.items()}
        strokes, _ = ST.simplify_graph(paths, keypts2x, mode="hybird", skip_len=skip_len)
        save_json(d, "strokes", paths_json(strokes))
        save_json(d, "strokes_rdp", paths_json(ST.ramerDouglas(strokes, eps)))
        # ST.fitBezier raises AttributeError (calls .length() on a list); the reference swallows it in
        # simplify_SVG, so --bezier is a no-op there. Fit with the underlying Schneider routine directly.
        beziers = [[np.asarray(b).tolist() for b in ST.fitCurve(ST.flatten_paths_known_to_be_polyline_to_pts_array(s), 0.1)]
                   for s in strokes]
        save_json(d, "strokes_bezier", beziers)

    save_json(d, "meta", meta)
    print("wrote", d)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--variant", default="light", choices=["light", "full"])
    ap.add_argument("--out", default=os.path.join(here, "..", "tests", "fixtures"))
    a = ap.parse_args()
    for im in a.images:
        run(im, a.variant, a.out)


if __name__ == "__main__":
    main()
