"""Golden fixtures for the optional pre-processing (predict_s1.load_img with thin / line_extractor).

Usage:
    uv run python make_preprocess_fixtures.py IMAGE[:x0,y0,x1,y1] ... [--out ../tests/fixtures/preprocess]

The optional crop box selects part of an image. Each input produces <out>/<name>/ with:
    rgb.npy         uint8 [H,W,3]   input (PIL convert("RGB"))
    gray.npy        uint8 [H,W]     PIL convert("L")
    thin.npy        uint8 [H,W]     line thinning of gray (as load_img(thin=...))
    a2s_in.npy      float32 [3,512,512]  Anime2Sketch input (PIL bicubic resize, normalized to [-1,1])
    a2s_out.npy     float32 [512,512]    Anime2Sketch output in [-1,1]
    a2s.npy         uint8 [H,W]     Anime2Sketch result resized back (cv2 INTER_AREA), one channel
    both.npy        uint8 [H,W]     line thinning of the Anime2Sketch result
    load_<mode>.npy float32 [h,w]   load_img's network input for mode in none / thin / a2s / both
Plus resize/<WxH>.npy: cv2.resize(random 512x512x3 uint8, (W, H), INTER_AREA), seed 0, for unit tests.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.abspath(os.path.join(HERE, "..", "..", "Deep-Sketch-Vectorization"))
sys.path.insert(0, REF)
sys.path.insert(0, HERE)

import predict_s1 as P  # noqa: E402
from network.thin import Thinner  # noqa: E402
from export_onnx import load_anime2sketch  # noqa: E402

PRE = os.path.join(HERE, "..", "..", "pretrained")
RESIZE_SIZES = [(300, 200), (512, 512), (700, 900), (256, 640), (1000, 333)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "tests", "fixtures", "preprocess"))
    a = ap.parse_args()
    thinner = Thinner(weightfile=os.path.join(PRE, "line_thin.pth.tar"))
    net = load_anime2sketch(PRE)

    rng = np.random.default_rng(0)
    src = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    os.makedirs(os.path.join(a.out, "resize"), exist_ok=True)
    np.save(os.path.join(a.out, "resize", "src.npy"), src)
    for w, h in RESIZE_SIZES:
        np.save(os.path.join(a.out, "resize", f"{w}x{h}.npy"), cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA))

    for spec in a.images:
        path, box = (spec.rsplit(":", 1) + [None])[:2] if ":" in os.path.basename(spec) else (spec, None)
        img = Image.open(path)
        if box:
            img = img.crop(tuple(int(v) for v in box.split(",")))
        name = os.path.splitext(os.path.basename(path))[0].replace(" ", "_") + (f"-{box.replace(',', '_')}" if box else "")
        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        save = lambda n, arr: np.save(os.path.join(d, n + ".npy"), np.ascontiguousarray(arr))  # noqa: E731

        rgb = img.convert("RGB")
        save("rgb", np.array(rgb))
        gray = np.array(img.convert("L"))
        save("gray", gray)
        save("thin", (thinner(gray).cpu().squeeze().numpy() * 255).astype(int).astype(np.uint8))

        x, size = P.read_img_path(img, 512)
        save("a2s_in", x[0].numpy().astype(np.float32))
        with torch.no_grad():
            y = net(x)
        save("a2s_out", y[0, 0].numpy().astype(np.float32))
        a2s = cv2.resize(P.tensor_to_img(y), (size[0], size[1]), interpolation=cv2.INTER_AREA)
        assert (a2s[..., 0] == a2s[..., 1]).all() and (a2s[..., 0] == a2s[..., 2]).all()
        save("a2s", a2s[..., 0])
        save("both", (thinner(a2s).cpu().squeeze().numpy() * 255).astype(int).astype(np.uint8))

        # The real code path, at full resolution (resize=False) for each mode.
        for mode, thin, ext in [("none", None, None), ("thin", thinner, None), ("a2s", None, net), ("both", thinner, net)]:
            t, _, _, _ = P.load_img(img, "cpu", True, thin=thin, line_extractor=ext, resize=False, img_name="x.png", ui_mode=False)
            save(f"load_{mode}", t[0, 0].numpy().astype(np.float32))
        print("wrote", d, img.size)


if __name__ == "__main__":
    main()
