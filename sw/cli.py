# ABOUTME: Command-line entry points: inspect the target, cost the schedule, and score int8 accuracy.
# ABOUTME: Run as `python -m sw <command>`.

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .execute import run_quant
from .frontend import DAV2Config, build_dav2_graph, load_state_dict
from .harness import compare
from .imageio import calibration_batches, find_images, load_image, save_depth_png
from .lower import lower_graph, require_supported
from .machine import MachineSpec, UnsupportedOpError
from .quantize import quantize_graph


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--machine", default="tpu-v2", help="machine description name or path")
    p.add_argument("--img-size", type=int, default=518,
                   help="input resolution; must be a multiple of 14 (default 518)")
    p.add_argument("--checkpoint", default=None, help="path to depth_anything_v2_*.pth")
    p.add_argument("--top", type=int, default=15, help="rows to show in ranked tables")


def _add_quant(p: argparse.ArgumentParser) -> None:
    p.add_argument("--calib-dir", default=None,
                   help="directory of calibration images (strongly recommended)")
    p.add_argument("--calib-count", type=int, default=8, help="number of calibration images")
    # Defaults are the measured optimum on DA-V2 (see README): clipping is worth
    # +12 dB over no clipping, and the optimum is sharp -- 99.99 and 99.5 both
    # collapse. Re-sweep it for a different model or resolution.
    p.add_argument("--calibration", choices=["amax", "percentile", "mse"], default="percentile",
                   help="how to pick each tensor's clipping threshold (default: percentile)")
    p.add_argument("--percentile", type=float, default=99.9,
                   help="clipping percentile for --calibration percentile (default: 99.9)")
    p.add_argument("--per-channel", action="store_true",
                   help="one weight scale and requant multiplier per output channel")
    p.add_argument("--bias-correction", action="store_true",
                   help="fold the mean shift caused by weight quantization into the biases; "
                        "free at inference, costs one extra calibration pass")
    p.add_argument("--residual-int16", action="store_true",
                   help="carry the transformer residual stream at int16 (wider UB entries, "
                        "recovers the accuracy ViT outlier channels cost at int8)")


def _build(args) -> tuple:
    machine = MachineSpec.load(args.machine)
    cfg = DAV2Config(img_size=args.img_size)
    sd = load_state_dict(args.checkpoint) if args.checkpoint else load_state_dict()
    graph = build_dav2_graph(sd, cfg)
    return machine, cfg, graph


def _calib(args, cfg) -> tuple:
    images = None
    if args.calib_dir:
        images = find_images(args.calib_dir)
        if not images:
            raise SystemExit(f"no images found under {args.calib_dir}")
    else:
        print("WARNING: calibrating on synthetic 1/f noise. Activation ranges will not "
              "match real photographs, so the accuracy numbers below are indicative "
              "only. Pass --calib-dir with real images for a meaningful result.",
              file=sys.stderr)
    return calibration_batches(cfg.img_size, images=images, count=args.calib_count)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_info(args) -> int:
    machine, cfg, graph = _build(args)
    print(machine.summary())
    print()
    print(graph.summary())
    print()
    try:
        require_supported(graph, machine)
    except UnsupportedOpError as exc:
        print(f"UNSUPPORTED: {exc}")
        return 1
    sched = lower_graph(graph, machine, {"image": (1, 3, cfg.img_size, cfg.img_size)})
    print(sched.text(top=args.top))
    return 0


def cmd_evaluate(args) -> int:
    machine, cfg, graph = _build(args)
    require_supported(graph, machine)

    t0 = time.time()
    qp = quantize_graph(graph, _calib(args, cfg), machine=machine,
                        method=args.calibration, percentile=args.percentile,
                        residual_int16=args.residual_int16,
                        per_channel_weights=args.per_channel,
                        bias_correction=args.bias_correction)
    print(f"calibrated and quantized in {time.time() - t0:.1f}s; "
          f"{qp.weight_bytes() / (1 << 20):.1f} MiB of int8 weights "
          f"({qp.weight_bytes() / machine.dram['bytes'] * 100:.1f}% of DRAM)")

    if args.image:
        sample = {"image": load_image(args.image, cfg.img_size)}
    else:
        sample = next(iter(calibration_batches(cfg.img_size, count=1, seed=1234)))

    t0 = time.time()
    report = compare(graph, qp, sample, machine=machine, per_tensor=not args.no_per_tensor)
    print(f"compared in {time.time() - t0:.1f}s\n")
    print(report.text(top=args.top))

    if args.save_depth:
        out = Path(args.save_depth)
        save_depth_png(report.depth_reference, out.with_suffix(".fp32.png"))
        save_depth_png(report.depth_quantized, out.with_suffix(".int8.png"))
        print(f"\nwrote {out.with_suffix('.fp32.png')} and {out.with_suffix('.int8.png')}")
    return 0


def cmd_run(args) -> int:
    machine, cfg, graph = _build(args)
    require_supported(graph, machine)
    qp = quantize_graph(graph, _calib(args, cfg), machine=machine,
                        method=args.calibration, percentile=args.percentile,
                        residual_int16=args.residual_int16,
                        per_channel_weights=args.per_channel,
                        bias_correction=args.bias_correction)
    image = load_image(args.image, cfg.img_size)
    t0 = time.time()
    depth = run_quant(graph, qp, {"image": image}, machine=machine)[graph.outputs[0]]
    print(f"int8 emulation: {time.time() - t0:.1f}s, output {depth.shape}, "
          f"range {float(depth.min()):.4f}..{float(depth.max()):.4f}")
    save_depth_png(depth, args.out)
    print(f"wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sw", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="show the target, the graph, and the tiled schedule")
    _add_common(p_info)
    p_info.set_defaults(func=cmd_info)

    p_eval = sub.add_parser("evaluate", help="score the int8 emulator against fp32, per tensor")
    _add_common(p_eval)
    _add_quant(p_eval)
    p_eval.add_argument("--image", default=None, help="image to evaluate on")
    p_eval.add_argument("--no-per-tensor", action="store_true",
                        help="compare outputs only; uses far less memory")
    p_eval.add_argument("--save-depth", default=None, help="write fp32/int8 depth PNGs here")
    p_eval.set_defaults(func=cmd_evaluate)

    p_run = sub.add_parser("run", help="run one image through the integer emulator")
    _add_common(p_run)
    _add_quant(p_run)
    p_run.add_argument("--image", required=True)
    p_run.add_argument("--out", default="depth.png")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    # Fail loudly on NaN/Inf, which always means a bug. Underflow is expected:
    # softmax exponentiates large negative logits to zero on purpose.
    np.seterr(divide="raise", invalid="raise", over="raise", under="ignore")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
