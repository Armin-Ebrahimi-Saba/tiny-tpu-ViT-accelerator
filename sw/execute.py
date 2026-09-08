# ABOUTME: Graph executors: an fp32 reference interpreter and the bit-accurate integer emulator.
# ABOUTME: Both walk the same IR, so any divergence is attributable to a named tensor.

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from . import kernels_float as F
from . import kernels_int as Q
from .ir import Graph, Op
from .machine import MachineSpec, UnsupportedOpError
from .numerics import Requant

Observer = Callable[[str, np.ndarray], None]


class _Values:
    """Tensor pool that frees intermediates once their last consumer has run.

    A 518x518 ViT-S holds ~45 MB of attention probabilities per block; keeping
    all of them alive costs more than a gigabyte for no reason.
    """

    def __init__(self, graph: Graph, pinned: Iterable[str] = ()) -> None:
        self._v: dict[str, np.ndarray] = {}
        self._pinned = set(pinned) | set(graph.outputs)
        self._refs: dict[str, int] = {}
        for op in graph.ops:
            for t in op.inputs:
                self._refs[t] = self._refs.get(t, 0) + 1

    def __setitem__(self, name: str, value: np.ndarray) -> None:
        self._v[name] = value

    def __getitem__(self, name: str) -> np.ndarray:
        return self._v[name]

    def __contains__(self, name: str) -> bool:
        return name in self._v

    def take(self, op: Op) -> list[np.ndarray]:
        out = [self._v[t] for t in op.inputs]
        for t in op.inputs:
            self._refs[t] -= 1
            if self._refs[t] <= 0 and t not in self._pinned:
                self._v.pop(t, None)
        return out

    def collect(self, names: Iterable[str]) -> dict[str, np.ndarray]:
        return {n: self._v[n] for n in names}


# ---------------------------------------------------------------------------
# Shape-only ops, shared by both executors
# ---------------------------------------------------------------------------


def _structural(op: Op, args: list[np.ndarray]) -> np.ndarray | None:
    if op.type == "reshape":
        return args[0].reshape(tuple(op.attr("shape")))
    if op.type == "transpose":
        return np.ascontiguousarray(args[0].transpose(tuple(op.attr("perm"))))
    if op.type == "slice":
        axis = op.attr("axis")
        sl = [slice(None)] * args[0].ndim
        sl[axis] = slice(op.attr("start"), op.attr("stop"))
        return np.ascontiguousarray(args[0][tuple(sl)])
    if op.type == "concat":
        return np.concatenate(args, axis=op.attr("axis"))
    return None


# ---------------------------------------------------------------------------
# fp32 reference
# ---------------------------------------------------------------------------


def run_float(graph: Graph, inputs: dict[str, np.ndarray], *,
              observer: Observer | None = None,
              keep: Iterable[str] = ()) -> dict[str, np.ndarray]:
    """Execute in fp32. Returns the graph outputs plus anything named in ``keep``."""
    keep = list(keep)
    vals = _Values(graph, pinned=keep)
    for name, arr in graph.initializers.items():
        vals[name] = arr
    for name in graph.inputs:
        if name not in inputs:
            raise KeyError(f"missing graph input {name!r}")
        vals[name] = np.asarray(inputs[name], dtype=np.float32)
        if observer:
            observer(name, vals[name])

    for op in graph.ops:
        args = vals.take(op)
        out = _structural(op, args)
        if out is None:
            out = _float_op(op, args)
        vals[op.outputs[0]] = out
        if observer:
            observer(op.outputs[0], out)
    return vals.collect(list(graph.outputs) + keep)


def _float_op(op: Op, a: list[np.ndarray]) -> np.ndarray:
    t = op.type
    if t == "conv2d":
        return F.conv2d(a[0], a[1], a[2] if len(a) > 2 else None, op.attr("stride", 1), op.attr("pad", 0))
    if t == "conv_transpose2d":
        return F.conv_transpose2d(a[0], a[1], a[2] if len(a) > 2 else None, op.attr("stride", 1), op.attr("pad", 0))
    if t == "matmul":
        return F.linear(a[0], a[1], a[2] if len(a) > 2 else None)
    if t == "batch_matmul":
        return F.batch_matmul(a[0], a[1])
    if t == "add":
        return (a[0] + a[1]).astype(np.float32)
    if t == "mul":
        return (a[0] * a[1]).astype(np.float32)
    if t == "layernorm":
        return F.layernorm(a[0], a[1], a[2], op.attr("eps", 1e-6))
    if t == "softmax":
        return F.softmax(a[0], op.attr("axis", -1))
    if t == "gelu":
        return F.gelu(a[0])
    if t == "relu":
        return F.relu(a[0])
    if t == "interpolate":
        return F.interpolate_bilinear(a[0], tuple(op.attr("size")), op.attr("align_corners", True))
    raise UnsupportedOpError(f"float executor has no kernel for op type {t!r} (op {op.name})")


# ---------------------------------------------------------------------------
# integer emulator
# ---------------------------------------------------------------------------


def run_quant(graph: Graph, qparams, inputs: dict[str, np.ndarray], *,
              machine: MachineSpec,
              observer: Observer | None = None,
              keep: Iterable[str] = ()) -> dict[str, np.ndarray]:
    """Execute with the integer datapath. Returns dequantized fp32 for comparison.

    ``qparams`` is a :class:`sw.quantize.QuantParams` carrying per-tensor scales,
    quantized constants, and per-op requantization constants. The observer sees
    the *raw* integer tensors, not dequantized ones, so callers can measure
    saturation as well as error.
    """
    keep = list(keep)
    vals = _Values(graph, pinned=keep)
    for name in graph.initializers:
        vals[name] = qparams.const_q[name]
    for name in graph.inputs:
        if name not in inputs:
            raise KeyError(f"missing graph input {name!r}")
        vals[name] = qparams.quantize(name, inputs[name])
        if observer:
            observer(name, vals[name])

    for op in graph.ops:
        args = vals.take(op)
        out = _structural(op, args)
        if out is None:
            out = _quant_op(op, args, qparams, machine)
        vals[op.outputs[0]] = out
        if observer:
            observer(op.outputs[0], out)

    raw = vals.collect(list(graph.outputs) + keep)
    return {k: qparams.dequantize(k, v) for k, v in raw.items()}


def _quant_op(op: Op, a: list[np.ndarray], qp, machine: MachineSpec) -> np.ndarray:
    t = op.type
    machine.require(t)

    # Bilinear resampling is a convex combination, so it carries its input's
    # scale through unchanged and needs no per-op constants.
    if t == "interpolate":
        return Q.qinterpolate_bilinear(a[0], tuple(op.attr("size")), op.attr("align_corners", True))

    p = qp.op_params[op.name]
    if t == "conv2d":
        return Q.qconv2d(a[0], a[1], p.bias_q, op.attr("stride", 1), op.attr("pad", 0), p.requant)
    if t == "conv_transpose2d":
        return Q.qconv_transpose2d(a[0], a[1], p.bias_q, op.attr("stride", 1), op.attr("pad", 0), p.requant)
    if t == "matmul":
        return Q.qlinear(a[0], a[1], p.bias_q, p.requant)
    if t == "batch_matmul":
        return Q.qbatch_matmul(a[0], a[1], p.requant)
    if t == "add":
        return Q.qadd(a[0], a[1], p.requant_a, p.requant_b, p.out_min, p.out_max)
    if t == "mul":
        return Q.qmul_channel(a[0], a[1], p.requant)
    if t == "layernorm":
        return Q.qlayernorm(a[0], a[1], p.bias_q, p.scale_in, p.requant, op.attr("eps", 1e-6))
    if t == "softmax":
        return Q.qsoftmax(a[0], p.exp_lut, p.scale_out, op.attr("axis", -1))
    if t == "gelu":
        return Q.apply_unary_lut(a[0], p.lut)
    if t == "relu":
        return Q.qrelu(a[0], p.requant)
    raise UnsupportedOpError(f"integer emulator has no kernel for op type {t!r} (op {op.name})")


# ---------------------------------------------------------------------------
# Calibration observer
# ---------------------------------------------------------------------------


class RangeObserver:
    """Accumulates per-tensor dynamic range across calibration batches.

    With ``percentile`` set, the clipping threshold is the mean of each batch's
    quantile rather than the global maximum. That is an approximation of a true
    global percentile — exact would need a histogram pass — but it is stable
    across batches and it matters for ViTs, whose activations carry a handful of
    extreme outliers that would otherwise set the scale for every other value.
    """

    def __init__(self, percentile: float | None = None) -> None:
        if percentile is not None and not 0.0 < percentile < 100.0:
            raise ValueError(f"percentile must be in (0, 100), got {percentile}")
        self.percentile = percentile
        self.amax: dict[str, float] = {}
        self._quantile_sum: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def __call__(self, name: str, value: np.ndarray) -> None:
        if value.size == 0:
            return
        av = np.abs(value)
        m = float(av.max())
        if not np.isfinite(m):
            raise ValueError(f"tensor {name!r} contains a non-finite value during calibration")
        self.amax[name] = max(self.amax.get(name, 0.0), m)
        if self.percentile is not None:
            q = float(np.quantile(av.astype(np.float32), self.percentile / 100.0))
            self._quantile_sum[name] = self._quantile_sum.get(name, 0.0) + q
            self._counts[name] = self._counts.get(name, 0) + 1

    def thresholds(self) -> dict[str, float]:
        """Clipping threshold per tensor: the percentile if configured, else the max."""
        if self.percentile is None:
            return dict(self.amax)
        out = {}
        for name, m in self.amax.items():
            n = self._counts.get(name, 0)
            out[name] = (self._quantile_sum[name] / n) if n else m
        return out


class HistogramObserver:
    """Bins |x| per tensor against an already-known range.

    Used as the second calibration pass: with the maximum known from the first
    pass, a fixed-range histogram can be accumulated across batches in bounded
    memory, and the optimal clipping threshold read off it afterwards without
    keeping any activation tensor alive.
    """

    def __init__(self, amax: dict[str, float], bins: int = 2048) -> None:
        if bins < 16:
            raise ValueError(f"need a usable number of bins, got {bins}")
        self.amax = amax
        self.bins = bins
        self.hist: dict[str, np.ndarray] = {}

    def __call__(self, name: str, value: np.ndarray) -> None:
        limit = self.amax.get(name, 0.0)
        if limit <= 0.0 or value.size == 0:
            return
        counts, _ = np.histogram(np.abs(value), bins=self.bins, range=(0.0, limit))
        prev = self.hist.get(name)
        self.hist[name] = counts.astype(np.int64) if prev is None else prev + counts
