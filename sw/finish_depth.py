# ABOUTME: Finishes a depth map from encoder taps the accelerator produced: the DPT head in fp32 on the host.
# ABOUTME: Writes the board's picture beside the fp32 reference and the emulator's, with the numbers between them.

"""The last stage of the pipeline, on the host.

The board runs the encoder and leaves four tap tensors in DDR3; the blob's
sidecar (`X.json`) says where and at what scale. This reads them back -- from
files load_model.py dumped, or from the emulator's `.emu.npz` -- dequantizes,
feeds them to the fp32 graph in place of its own taps, and runs the DPT head
in float. That head is 60% of the model's MACs and the next thing to move
onto the accelerator; until then this is where the picture comes from.

    python -m sw.finish_depth X.bin                 # emulator taps
    python -m sw.finish_depth X.bin --fpga          # X.tap0.bin ... from the board
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .execute import run_float
from .frontend.dav2 import DAV2Config, build_dav2_graph, load_state_dict
from .imageio import load_image, save_depth_png, synthetic_image
from .numerics import sqnr_db


def depth_metrics(ref: np.ndarray, got: np.ndarray) -> dict[str, float]:
    """DA-V2 predicts relative disparity, defined only up to scale and shift,
    so AbsRel and delta are taken after a least-squares affine fit of `got`
    onto `ref` -- the standard protocol for relative-depth models -- and only
    where the reference is meaningfully positive (the sky is ~0 and would
    make any ratio explode). SQNR and Pearson are on the raw outputs."""
    r, g = ref.ravel().astype(np.float64), got.ravel().astype(np.float64)
    A = np.stack([g, np.ones_like(g)], 1)
    scale, shift = np.linalg.lstsq(A, r, rcond=None)[0]
    al = scale * g + shift
    valid = r > 0.05 * r.max()
    rv, av = r[valid], np.maximum(al[valid], 1e-6)
    ratio = np.maximum(av / rv, rv / av)
    return {"sqnr_db": sqnr_db(r, g), "pearson": float(np.corrcoef(r, g)[0, 1]),
            "absrel": float(np.mean(np.abs(av - rv) / rv)), "delta1": float(np.mean(ratio < 1.25))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("blob", type=Path)
    ap.add_argument("--fpga", action="store_true", help="use the taps load_model.py dumped")
    ap.add_argument("--out", type=Path, default=None, help="output stem (default: the blob's)")
    a = ap.parse_args()

    side = json.loads(a.blob.with_suffix(".json").read_text())
    S = side["image"]
    stem = a.out or a.blob.with_suffix("")
    sd = load_state_dict()
    graph = build_dav2_graph(sd, DAV2Config(img_size=S))
    if side["image_path"]:
        image = load_image(side["image_path"], S)
    else:
        image = synthetic_image(S, np.random.default_rng(side["seed"]))

    ref = run_float(graph, {"image": image})[graph.outputs[0]].squeeze()
    save_depth_png(ref, f"{stem}.ref.png")

    emu = np.load(a.blob.with_suffix(".emu.npz"))
    sources = {"emu": {t["name"]: emu[t["name"]] for t in side["taps"]}}
    if a.fpga:
        sources["fpga"] = {}
        for j, t in enumerate(side["taps"]):
            raw = np.fromfile(f"{a.blob}.tap{j}.bin", dtype=np.int8)
            sources["fpga"][t["name"]] = raw[:t["bytes"]].reshape(t["shape"][1:])
        for t in side["taps"]:
            same = np.array_equal(sources["fpga"][t["name"]], sources["emu"][t["name"]])
            print(f"{t['name']}: board {'==' if same else '!='} emulator")

    for src, taps in sources.items():
        over = {t["name"]: taps[t["name"]].astype(np.float32).reshape(t["shape"]) * t["scale"]
                for t in side["taps"]}
        depth = run_float(graph, {"image": image}, overrides=over)[graph.outputs[0]].squeeze()
        save_depth_png(depth, f"{stem}.{src}.png")
        m = depth_metrics(ref, depth)
        print(f"{src}: SQNR {m['sqnr_db']:.1f} dB, Pearson {m['pearson']:.3f}, "
              f"AbsRel {m['absrel'] * 100:.1f}%, delta<1.25 {m['delta1'] * 100:.1f}%  -> {stem}.{src}.png")


if __name__ == "__main__":
    main()
