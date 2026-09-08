# ABOUTME: Differential harness comparing the fp32 reference against the integer emulator, per tensor.
# ABOUTME: A single end-to-end number hides which layer broke; this attributes the loss to named ops.

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .execute import run_float, run_quant
from .ir import Graph
from .machine import MachineSpec
from .numerics import cosine_similarity, sqnr_db
from .quantize import QuantParams


@dataclass
class TensorStat:
    name: str
    op_type: str
    shape: tuple[int, ...]
    scale: float
    sqnr: float
    cosine: float
    saturation: float  # fraction of elements pinned at the int8 rail


@dataclass
class Report:
    stats: list[TensorStat] = field(default_factory=list)
    output_sqnr: float = 0.0
    output_cosine: float = 0.0
    depth_reference: np.ndarray | None = None
    depth_quantized: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)

    def worst(self, n: int = 15) -> list[TensorStat]:
        return sorted(self.stats, key=lambda s: s.sqnr)[:n]

    def most_saturated(self, n: int = 10) -> list[TensorStat]:
        return sorted(self.stats, key=lambda s: s.saturation, reverse=True)[:n]

    def depth_metrics(self) -> dict[str, float]:
        """Agreement of the int8 depth map with the fp32 one, in depth-estimation terms.

        Relative metrics are computed only on pixels where the reference is
        meaningfully non-zero. The network ends in a ReLU, so genuine exact
        zeros are common, and dividing by them turns AbsRel into a number in the
        millions that says nothing about the depth map.
        """
        if self.depth_reference is None or self.depth_quantized is None:
            return {}
        ref = self.depth_reference.astype(np.float64).ravel()
        got = self.depth_quantized.astype(np.float64).ravel()
        rmse = float(np.sqrt(np.mean((ref - got) ** 2)))
        corr = float(np.corrcoef(ref, got)[0, 1]) if ref.std() and got.std() else float("nan")

        valid = ref > 1e-3 * max(float(ref.max()), 1e-12)
        if not valid.any():
            return {"abs_rel": float("nan"), "rmse": rmse, "delta1": float("nan"),
                    "corr": corr, "valid_fraction": 0.0}
        r, g = ref[valid], np.maximum(got[valid], 1e-12)
        ratio = np.maximum(g / r, r / g)
        return {
            "abs_rel": float(np.mean(np.abs(r - g) / r)),
            "rmse": rmse,
            "delta1": float(np.mean(ratio < 1.25)),
            "corr": corr,
            "valid_fraction": float(valid.mean()),
        }

    def text(self, top: int = 15) -> str:
        lines = [
            "int8 emulator vs fp32 reference",
            f"  end-to-end SQNR   {self.output_sqnr:.2f} dB",
            f"  end-to-end cosine {self.output_cosine:.6f}",
        ]
        metrics = self.depth_metrics()
        if metrics:
            lines += [
                f"  depth AbsRel      {metrics['abs_rel'] * 100:.2f} %",
                f"  depth RMSE        {metrics['rmse']:.4f}",
                f"  depth delta<1.25  {metrics['delta1'] * 100:.2f} %",
                f"  depth pearson r   {metrics['corr']:.6f}",
            ]
        if self.stats:
            lines += ["", f"  worst {top} tensors by SQNR:",
                      f"  {'tensor':40s} {'op':16s} {'SQNR dB':>8s} {'cosine':>9s} {'sat%':>6s}"]
            for s in self.worst(top):
                lines.append(
                    f"  {s.name[:40]:40s} {s.op_type:16s} {s.sqnr:8.2f} {s.cosine:9.5f} "
                    f"{s.saturation * 100:5.1f}%"
                )
            lines += ["", f"  most saturated tensors:"]
            for s in self.most_saturated(6):
                lines.append(f"  {s.name[:40]:40s} {s.op_type:16s} {s.saturation * 100:5.1f}% "
                             f"at the rail (scale {s.scale:.3e})")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


def compare(graph: Graph, qp: QuantParams, inputs: dict[str, np.ndarray], *,
            machine: MachineSpec, per_tensor: bool = True) -> Report:
    """Run both executors on the same input and attribute the error per tensor.

    ``per_tensor`` retains every fp32 intermediate for the duration of the
    integer run. At 518x518 that is a few gigabytes, mostly attention
    probabilities; turn it off to compare outputs only.
    """
    report = Report()
    producer = {t: op for op in graph.ops for t in op.outputs}

    reference: dict[str, np.ndarray] = {}
    if per_tensor:
        def keep_float(name: str, value: np.ndarray) -> None:
            reference[name] = np.array(value, dtype=np.float32, copy=True)
        float_out = run_float(graph, inputs, observer=keep_float)
    else:
        float_out = run_float(graph, inputs)

    def score(name: str, raw: np.ndarray) -> None:
        ref = reference.get(name)
        if ref is None:
            return
        got = qp.dequantize(name, raw)
        if ref.shape != got.shape:
            report.warnings.append(f"{name}: shape drift {ref.shape} vs {got.shape}")
            return
        op = producer.get(name)
        rail = qp.rails.get(name, machine.act_max)
        sat = float(np.mean(np.abs(raw.astype(np.int32)) >= rail)) if raw.size else 0.0
        report.stats.append(TensorStat(
            name=name,
            op_type=op.type if op else "input",
            shape=tuple(ref.shape),
            scale=qp.scales.get(name, float("nan")),
            sqnr=sqnr_db(ref, got),
            cosine=cosine_similarity(ref, got),
            saturation=sat,
        ))
        reference.pop(name, None)  # release as soon as it has been scored

    quant_out = run_quant(graph, qp, inputs, machine=machine,
                          observer=score if per_tensor else None)

    out_name = graph.outputs[0]
    ref_depth = float_out[out_name]
    got_depth = quant_out[out_name]
    report.output_sqnr = sqnr_db(ref_depth, got_depth)
    report.output_cosine = cosine_similarity(ref_depth, got_depth)
    report.depth_reference = ref_depth
    report.depth_quantized = got_depth
    return report
