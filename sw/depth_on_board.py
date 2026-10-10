# ABOUTME: Depth maps from the board for any number of pictures: one program blob, one input per picture.
# ABOUTME: Writes inputs, drives rvlab's load_model.py, checks the board's taps against the emulator, finishes in fp32.

"""Run an image-independent encoder program on the FPGA over a list of pictures.

The program (`python -m sw.export_tpu --calib C.json --program-only -o P.bin`)
carries the weights and every constant; its sidecar names the input's DRAM
address and scale. Per picture, this writes the im2col'd int8 input next to
the picture's outputs, then runs them all in one board session -- the 30 MB
blob is loaded once -- and for each picture:

  - reads back what the board left in DDR3: the four taps, and the depth map
    itself when the program carries the head (`--head`),
  - checks them byte for byte against the emulator at the same fixed scales,
  - writes the depth PNG -- the board's own int8 map, or, for an encoder-only
    program, the DPT head run in fp32 on the board's taps -- beside the fp32
    reference, with the metrics between them.

    python -m sw.depth_on_board P.bin assets/examples/*.jpg --out results/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .execute import run_float
from .export_tpu import encoder_input, run_encoder_emulator
from .finish_depth import depth_metrics
from .frontend.dav2 import DAV2Config, build_dav2_graph, load_state_dict
from .imageio import load_image, save_depth_png

ROOT = Path(__file__).resolve().parents[1]
RVLAB = ROOT / "rvlab"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("program", type=Path)
    ap.add_argument("images", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=Path("."))
    ap.add_argument("--timeout", type=float, default=900.0, help="per run, seconds")
    ap.add_argument("--skip-board", action="store_true", help="reuse taps already on disk")
    ap.add_argument("--cpu-paths", action="store_true",
                    help="the CPU gathers and copies results, not the DMA (for comparison)")
    a = ap.parse_args()

    side = json.loads(a.program.with_suffix(".json").read_text())
    if "input" not in side or not side.get("calib"):
        sys.exit(f"{a.program} is not a program blob (export it with --calib ... --program-only)")
    scales = json.loads(Path(side["calib"]).read_text())["scales"]
    S = side["image"]
    a.out.mkdir(parents=True, exist_ok=True)

    inputs = []
    for img in a.images:
        x = encoder_input(load_image(img, S), side["input"]["scale"])
        assert list(x.shape) == side["input"]["shape"]
        path = a.out / f"{img.stem}.in.bin"
        x.tofile(path)
        inputs.append(path)

    if not a.skip_board:
        cmd = [sys.executable, "-u", "src/sw/project/tools/load_model.py", "--blob",
               str(a.program.resolve()), "--log", str((a.out / "openocd.log").resolve()),
               "--timeout", str(a.timeout), *(["--cpu-paths"] if a.cpu_paths else []),
               "--inputs", *[str(p.resolve()) for p in inputs]]
        rc = subprocess.call(cmd, cwd=RVLAB)
        if rc != 0:
            print(f"board run failed (load_model.py exit {rc})")
            return rc

    sd = load_state_dict()
    graph = build_dav2_graph(sd, DAV2Config(img_size=S))
    out_name = graph.outputs[0]
    worst = 0
    for img, inp in zip(a.images, inputs):
        image = load_image(img, S)
        has_head = "depth" in side
        emu = run_encoder_emulator(sd, image, scales, head=has_head)
        board = {}
        for j, t in enumerate(side["taps"]):
            raw = np.fromfile(f"{inp}.tap{j}.bin", dtype=np.int8)[:t["bytes"]]
            board[t["name"]] = raw.reshape(t["shape"][1:])
        if has_head:
            d = side["depth"]
            board["depth"] = np.fromfile(f"{inp}.depth.bin", dtype=np.int8)[:d["bytes"]].reshape(d["shape"])
        same = all(np.array_equal(board[k], emu[k]) for k in emu)
        worst |= 0 if same else 1

        ref = run_float(graph, {"image": image})[out_name].squeeze()
        if has_head:
            # The board's own answer: channel 0 of the last conv, dequantized.
            depth = board["depth"][..., 0].astype(np.float32) * side["depth"]["scale"]
        else:
            over = {t["name"]: board[t["name"]].astype(np.float32).reshape(t["shape"]) * t["scale"]
                    for t in side["taps"]}
            depth = run_float(graph, {"image": image}, overrides=over)[out_name].squeeze()
        save_depth_png(ref, a.out / f"{img.stem}.ref.png")
        save_depth_png(depth, a.out / f"{img.stem}.fpga.png")
        m = depth_metrics(ref, depth)
        what = "taps+depth" if has_head else "taps"
        print(f"{img.stem}: {what} board {'==' if same else '!='} emulator; vs fp32 Pearson "
              f"{m['pearson']:.3f}, AbsRel {m['absrel'] * 100:.1f}%, delta<1.25 "
              f"{m['delta1'] * 100:.1f}%  -> {a.out / (img.stem + '.fpga.png')}")
    return worst


if __name__ == "__main__":
    sys.exit(main())
