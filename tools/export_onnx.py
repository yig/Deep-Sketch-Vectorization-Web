"""Export the pretrained UDF (distance field) and NDC (line reconstruction) networks to ONNX.

Usage:
    uv run python export_onnx.py [--pretrained ../../pretrained] [--out ../public/models]

Writes {udf,ndc}_{light,full}.onnx (fp32) and *.fp16.onnx (fp16 weights, fp32 I/O),
then checks each ONNX model against PyTorch on a random input.

Also exports the optional pre-processing networks (--preprocess, default on):
    thin.onnx            Edgar Simo-Serra's line thinning (network/thin.py)
                         image float32 [1,1,H,W] in [0,1] -> thinned float32 [1,1,H,W] in [0,1]; any H, W
    anime2sketch.onnx    Xiaoyu Xiang's Anime2Sketch (network/anime2sketch.py), fixed 512 x 512
                         rgb float32 [1,3,512,512] in [-1,1] -> sketch float32 [1,1,512,512] in [-1,1]
                         (also anime2sketch.w16.onnx: fp16-stored weights, half the size; the app uses it)

Model I/O (N = 1, H and W are dynamic and must be multiples of 8):
    udf:  image  float32 [1,1,H,W] in [0,1] (white background = 1)
          -> udf  float32 [1,1,2H+1,2W+1]  centerline UDF (normalized by dist_clip)
          -> aux  float32 [1,5,2H,2W]      all / end / sharp / junc keypoint UDFs, USM
    ndc:  udf    float32 [1,1,2H+1,2W+1]  centerline UDF; clipped to [0,1] inside the graph
          -> edge   float32 [1,4,2H,2W]   edge-flag logits: none / x / y / both
          -> offset float32 [1,2,2H,2W]   vertex offsets (x, y) relative to the grid-cell base coordinate
"""

import argparse
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch

REF = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Deep-Sketch-Vectorization"))
sys.path.insert(0, REF)

from predict_s1 import load_checkpoint_udf, load_checkpoint_ndc  # noqa: E402
from network.thin import ThinningNet  # noqa: E402
from network.anime2sketch import UnetGenerator  # noqa: E402


class UDFWrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image):
        udf, classes = self.net.forward_with_downscale_outs(image)
        return udf, torch.cat(classes, dim=1)


class NDCWrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, udf):
        edge, offset = self.net.forward(udf.clamp(0, 1), [0.5])
        # Channel 4 of the edge logits is unused by the reference pipeline.
        return edge[:, :4], offset


def explicit_same_padding(module):
    """padding='same' exports as auto_pad=SAME_UPPER, which onnxruntime rejects for dilated convs.
    All 'same' convs here have odd kernels, so the equivalent explicit padding is symmetric."""
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d) and m.padding == "same":
            assert all(k % 2 == 1 for k in m.kernel_size) and m.stride == (1, 1)
            m.padding = tuple(d * (k - 1) // 2 for k, d in zip(m.kernel_size, m.dilation))


def resnext_forward_without_concat(self, x):
    """ResNextModule.forward (cardinality branch) with plusconv(cat(branches)) rewritten as a sum of
    per-branch 1x1 convolutions. Same function, but the (width * C)-channel concatenation is never
    materialized: for the full UDF model at 2x resolution that tensor is 2048 channels."""
    x_skip = self.skipconv(x)
    conv = self.plusconv.conv
    c = conv.out_channels
    acc = None
    for i in range(self.width):
        y = torch.nn.functional.conv2d(self.pathes[i](x), conv.weight[:, i * c:(i + 1) * c])
        acc = y if acc is None else acc + y
    x = acc + conv.bias.view(1, -1, 1, 1)
    if self.plusconv.bn is not None:
        x = self.plusconv.bn(x)
    if self.final_batchnorm:
        return self.activation(self.finalbathcnorm(x + x_skip))
    return self.activation(x + x_skip)


def split_resnext_concat(module):
    import types
    from network.keypoint import ResNextModule

    for m in module.modules():
        if isinstance(m, ResNextModule) and m.cardinality:
            m.forward = types.MethodType(resnext_forward_without_concat, m)


def export(module, example, path, input_names, output_names, dynamic_axes):
    module.eval()
    explicit_same_padding(module)
    split_resnext_concat(module)
    torch.onnx.export(
        module,
        (example,),
        path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    model = onnx.load(path)
    onnx.checker.check_model(model)
    return model


def to_fp16(path, out_path):
    from onnxconverter_common import float16

    model = onnx.load(path)
    model16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model16, out_path)


def fp16_weights(path, out_path):
    """Store float weights as fp16 and cast them back to fp32 inside the graph. Halves the download;
    onnxruntime folds the casts when the session is created, so inference stays fp32 (unlike a true
    fp16 graph, which needs shader-f16 on WebGPU and is poorly supported on the WASM backend)."""
    from onnx import helper, numpy_helper, TensorProto

    model = onnx.load(path)
    g = model.graph
    casts, inits = [], []
    for init in g.initializer:
        if init.data_type != TensorProto.FLOAT or np.prod(init.dims) < 16:
            inits.append(init)
            continue
        arr = numpy_helper.to_array(init).astype(np.float16)
        inits.append(numpy_helper.from_array(arr, init.name + "__fp16"))
        casts.append(helper.make_node("Cast", [init.name + "__fp16"], [init.name], to=TensorProto.FLOAT))
    del g.initializer[:]
    g.initializer.extend(inits)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    onnx.checker.check_model(model)
    onnx.save(model, out_path)


def compare(module, path, example):
    """`module` must be an unpatched reference module (fresh from the checkpoint)."""
    with torch.no_grad():
        out = module.eval()(example)
        ref = [t.numpy() for t in (out if isinstance(out, (tuple, list)) else [out])]
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: example.numpy()})
    errs = [float(np.abs(r - g).max()) for r, g in zip(ref, got)]
    shapes = [g.shape for g in got]
    return errs, shapes


def load_thinner(pretrained):
    net = ThinningNet()
    net.load_state_dict(torch.load(os.path.join(pretrained, "line_thin.pth.tar"), map_location="cpu"))
    return net.eval()


def load_anime2sketch(pretrained):
    """network.anime2sketch.create_model('default') without its hard-coded relative path."""
    import functools
    norm_layer = functools.partial(torch.nn.InstanceNorm2d, affine=False, track_running_stats=False)
    net = UnetGenerator(3, 1, 8, 64, norm_layer=norm_layer, use_dropout=False)
    ckpt = torch.load(os.path.join(pretrained, "netG.pth"), map_location="cpu")
    net.load_state_dict({k.replace("module.", ""): v for k, v in ckpt.items()})
    return net.eval()


def export_preprocess(pretrained, out):
    p = os.path.join(out, "thin.onnx")
    export(load_thinner(pretrained), torch.rand(1, 1, 64, 80), p, ["image"], ["thinned"],
           {"image": {2: "H", 3: "W"}, "thinned": {2: "H", 3: "W"}})
    print(p, "max abs err vs torch:", *compare(load_thinner(pretrained), p, torch.rand(1, 1, 75, 93)))

    p = os.path.join(out, "anime2sketch.onnx")
    x = torch.rand(1, 3, 512, 512) * 2 - 1
    export(load_anime2sketch(pretrained), x, p, ["rgb"], ["sketch"], None)
    q = p.replace(".onnx", ".w16.onnx")
    fp16_weights(p, q)
    x = torch.rand(1, 3, 512, 512) * 2 - 1
    print(p, "max abs err vs torch:", *compare(load_anime2sketch(pretrained), p, x))
    print(q, "(fp16 weights) max abs err vs torch:", *compare(load_anime2sketch(pretrained), q, x))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", default=os.path.join(here, "..", "..", "pretrained"))
    ap.add_argument("--out", default=os.path.join(here, "..", "public", "models"))
    ap.add_argument("--only-preprocess", action="store_true", help="export just the pre-processing networks")
    ap.add_argument("--no-preprocess", action="store_true", help="skip the pre-processing networks")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    if not args.no_preprocess:
        export_preprocess(args.pretrained, args.out)
    if args.only_preprocess:
        return

    for variant in ["light", "full"]:
        udf_net, udf_args = load_checkpoint_udf(os.path.join(args.pretrained, f"udf_{variant}.pth"))
        ndc_net, ndc_args = load_checkpoint_ndc(os.path.join(args.pretrained, f"ndc_{variant}.pth"))
        assert udf_args.up_scale, "the browser pipeline assumes the 2x super-resolution UDF model"
        assert udf_args.dist_clip == ndc_args.dist_clip, "UDF re-normalization between networks is not implemented"

        udf = UDFWrapper(udf_net).eval()
        ndc = NDCWrapper(ndc_net).eval()

        img = torch.rand(1, 1, 64, 96)
        p = os.path.join(args.out, f"udf_{variant}.onnx")
        export(udf, img, p, ["image"], ["udf", "aux"],
               {"image": {2: "H", 3: "W"}, "udf": {2: "H2p1", 3: "W2p1"}, "aux": {2: "H2", 3: "W2"}})
        to_fp16(p, p.replace(".onnx", ".fp16.onnx"))
        ref_udf = UDFWrapper(load_checkpoint_udf(os.path.join(args.pretrained, f"udf_{variant}.pth"))[0])
        print(p, "max abs err vs torch:", *compare(ref_udf, p, torch.rand(1, 1, 80, 72)))

        u = torch.rand(1, 1, 129, 193) * 1.2
        p = os.path.join(args.out, f"ndc_{variant}.onnx")
        export(ndc, u, p, ["udf"], ["edge", "offset"],
               {"udf": {2: "H2p1", 3: "W2p1"}, "edge": {2: "H2", 3: "W2"}, "offset": {2: "H2", 3: "W2"}})
        to_fp16(p, p.replace(".onnx", ".fp16.onnx"))
        ref_ndc = NDCWrapper(load_checkpoint_ndc(os.path.join(args.pretrained, f"ndc_{variant}.pth"))[0])
        print(p, "max abs err vs torch:", *compare(ref_ndc, p, torch.rand(1, 1, 161, 145) * 1.2))


if __name__ == "__main__":
    main()
