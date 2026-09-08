# ABOUTME: Turns a calibrated fp32 graph into per-tensor int8 scales, quantized constants, and requant tables.
# ABOUTME: Every constant the RTL needs at run time is produced here, at compile time.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from . import kernels_float as F
from . import kernels_int as Q
from .execute import HistogramObserver, RangeObserver, run_float
from .ir import Graph, Op
from .machine import MachineSpec, UnsupportedOpError
from .numerics import (
    Requant,
    amax_per_channel,
    dequantize_tensor,
    quantize_bias,
    quantize_per_channel,
    quantize_tensor,
    scale_from_amax,
)

# Which weight axis indexes output channels, per op type. conv_transpose2d
# stores weights as [in, out, kh, kw], so its output axis is 1, not 0.
WEIGHT_OUTPUT_AXIS = {"matmul": 0, "conv2d": 0, "conv_transpose2d": 1}

# Ops whose output is a permutation or selection of their input's values, so
# input and output must share one scale or the values would change meaning.
SCALE_PRESERVING = {"reshape", "transpose", "slice", "concat", "interpolate"}

# Attention logits span far more range than 8 bits can hold once the token
# sequence is long: quantizing them to int8 makes exp() granular enough to
# distort every probability in the row. The machine therefore carries softmax
# inputs at int16, and their scales must use the wider level count to benefit.
WIDE_LEVELS = 32767

# Ops that can consume a 16-bit activation without overflowing an int32
# accumulator. Anything reducing over a long axis (conv2d, matmul) cannot:
# 32767 * 127 * 576 already exceeds int32.
WIDE_SAFE_CONSUMERS = {"layernorm", "add", "mul", "reshape", "transpose", "slice", "concat"}


def wide_tensors(graph: Graph, residual: bool = False) -> set[str]:
    """Tensors the datapath keeps at int16 rather than int8.

    Attention logits always qualify. With ``residual`` set, so does the
    transformer's residual stream: a ViT accumulates a handful of very large
    channels down the residual path, and at int8 those outliers set the scale
    for every other value in the tensor, so ordinary activations end up with
    one or two quantization levels. Clipping them away does not help -- the
    large channels carry real signal -- so the fix is range, not clamping.

    Only adds whose consumers can all take a wide input are widened, which
    naturally selects the transformer stream and leaves the DPT head, whose
    adds feed convolutions, at int8.
    """
    wide = {op.inputs[0] for op in graph.ops if op.type == "softmax"}

    if residual:
        consumers: dict[str, list[str]] = {}
        for op in graph.ops:
            for t in op.inputs:
                consumers.setdefault(t, []).append(op.type)
        for op in graph.ops:
            if op.type != "add":
                continue
            out = op.outputs[0]
            cons = consumers.get(out, [])
            if cons and all(c in WIDE_SAFE_CONSUMERS for c in cons):
                wide.add(out)

    # A scale-preserving group shares one scale, so it must share one width too.
    groups = unify_scale_groups(graph)
    wide_roots = {groups[t] for t in wide if t in groups}
    return wide | {t for t, root in groups.items() if root in wide_roots}


@dataclass
class OpQParams:
    """Per-op run-time constants. Only the fields an op actually uses are set."""

    requant: Requant | None = None
    requant_a: Requant | None = None
    requant_b: Requant | None = None
    bias_q: np.ndarray | None = None
    lut: np.ndarray | None = None
    exp_lut: np.ndarray | None = None
    scale_in: float = 0.0
    scale_out: float = 0.0
    out_min: int = -127
    out_max: int = 127


@dataclass
class QuantParams:
    scales: dict[str, float] = field(default_factory=dict)
    const_q: dict[str, np.ndarray] = field(default_factory=dict)
    op_params: dict[str, OpQParams] = field(default_factory=dict)
    # Saturation rail per tensor; absent means the default int8 rail.
    rails: dict[str, int] = field(default_factory=dict)
    # Per-output-channel weight scales, where per-channel quantization is used.
    weight_scales: dict[str, np.ndarray] = field(default_factory=dict)

    def quantize(self, name: str, x: np.ndarray) -> np.ndarray:
        return quantize_tensor(x, self.scales[name])

    def dequantize(self, name: str, q: np.ndarray) -> np.ndarray:
        return dequantize_tensor(q, self.scales[name]).astype(np.float32)

    def weight_bytes(self) -> int:
        """Total int8 weight footprint, i.e. what has to live in DRAM."""
        return int(sum(v.size for v in self.const_q.values()))


# ---------------------------------------------------------------------------
# Scale unification
# ---------------------------------------------------------------------------


def unify_scale_groups(graph: Graph) -> dict[str, str]:
    """Union tensors that must share a scale; returns tensor -> group representative."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for op in graph.ops:
        if op.type in SCALE_PRESERVING:
            for t in op.inputs:
                union(op.outputs[0], t)
    for t in list(graph.initializers) + graph.inputs:
        find(t)
    for op in graph.ops:
        for t in op.outputs:
            find(t)
    return {t: find(t) for t in parent}


def resolve_scales(graph: Graph, thresholds: dict[str, float],
                   weight_thresholds: dict[str, float] | None = None,
                   wide: set[str] | None = None) -> dict[str, float]:
    """Combine calibrated activation ranges and weight ranges into one scale per tensor."""
    groups = unify_scale_groups(graph)
    amax: dict[str, float] = dict(thresholds)
    for name, arr in graph.initializers.items():
        w = (weight_thresholds or {}).get(name)
        amax[name] = w if w is not None else float(np.abs(arr).max())

    group_amax: dict[str, float] = {}
    for tensor, root in groups.items():
        if tensor in amax:
            group_amax[root] = max(group_amax.get(root, 0.0), amax[tensor])

    wide = wide or set()

    def levels(tensor: str) -> int:
        return WIDE_LEVELS if tensor in wide else 127

    scales: dict[str, float] = {}
    for tensor, root in groups.items():
        scales[tensor] = scale_from_amax(
            group_amax.get(root, amax.get(tensor, 0.0)), levels(tensor))
    for tensor, value in amax.items():
        scales.setdefault(tensor, scale_from_amax(value, levels(tensor)))
    return scales


# ---------------------------------------------------------------------------
# Parameter derivation
# ---------------------------------------------------------------------------


def _gelu_scalar(v: np.ndarray) -> np.ndarray:
    return F.gelu(v.astype(np.float32)).astype(np.float64)


def build_quant_params(graph: Graph, scales: dict[str, float], machine: MachineSpec,
                       residual_int16: bool = False,
                       per_channel_weights: bool = False) -> QuantParams:
    """Derive every run-time integer constant from the resolved scales.

    With ``per_channel_weights``, each output channel of a convolution or matmul
    gets its own weight scale and its own requantization multiplier. One badly
    scaled output channel otherwise drags down every other channel sharing the
    tensor, which is the dominant int8 error source in a ViT.
    """
    if per_channel_weights and not machine.requant.get("per_channel", False):
        raise UnsupportedOpError(
            f"machine {machine.name!r} declares requant.per_channel=false; it has no "
            "per-output-column multiplier storage, so per-channel weights cannot run on it"
        )
    qp = QuantParams(scales=scales)

    for name, arr in graph.initializers.items():
        qp.const_q[name] = quantize_tensor(arr, scales[name])

    shift_min = int(machine.requant["shift_min"])
    shift_max = int(machine.requant["shift_max"])
    wide = wide_tensors(graph, residual=residual_int16)
    for tensor in wide:
        qp.rails[tensor] = WIDE_LEVELS

    def rq(m: float, rail: int = 0) -> Requant:
        rail = rail or machine.act_max
        return Requant.from_real_multiplier(
            m, shift_min=shift_min, shift_max=shift_max,
            out_min=-rail, out_max=rail,
        )

    for op in graph.ops:
        machine.require(op.type)
        if op.type in SCALE_PRESERVING:
            continue
        out = op.outputs[0]
        sy = scales[out]
        sx = scales[op.inputs[0]]
        rail = qp.rails.get(out, machine.act_max)
        p = OpQParams(scale_in=sx, scale_out=sy, out_min=-rail, out_max=rail)

        if op.type in ("conv2d", "conv_transpose2d", "matmul"):
            w_name = op.inputs[1]
            if per_channel_weights:
                axis = WEIGHT_OUTPUT_AXIS[op.type]
                w = graph.initializers[w_name]
                sw = np.array([scale_from_amax(float(a)) for a in amax_per_channel(w, axis)])
                qp.weight_scales[w_name] = sw
                qp.const_q[w_name] = quantize_per_channel(w, sw, axis)
            else:
                sw = scales[w_name]
            p.requant = rq(sx * sw / sy, rail)
            if len(op.inputs) > 2:
                p.bias_q = quantize_bias(graph.initializers[op.inputs[2]], sx * sw)
            _check_conv_limits(op, graph, machine)

        elif op.type == "batch_matmul":
            p.requant = rq(sx * scales[op.inputs[1]] / sy, rail)

        elif op.type == "add":
            p.requant_a = rq(sx / sy, rail)
            p.requant_b = rq(scales[op.inputs[1]] / sy, rail)

        elif op.type == "mul":
            p.requant = rq(sx * scales[op.inputs[1]] / sy)

        elif op.type == "layernorm":
            # The input scale cancels inside the normalization; what remains is
            # gamma's scale, the sqrt(N) folded out of the integer rsqrt, and
            # the fixed-point shift the kernel divides by.
            n = int(graph.initializers[op.inputs[1]].shape[-1])
            sg = scales[op.inputs[1]]
            p.requant = rq(sg * math.sqrt(n) / (float(1 << Q._LN_FRAC_BITS) * sy))
            p.bias_q = quantize_bias(graph.initializers[op.inputs[2]], sy)

        elif op.type == "softmax":
            entries = int(machine.vector_unit["softmax"]["exp_lut_entries"])
            out_bits = int(machine.vector_unit["softmax"]["exp_out_bits"])
            p.exp_lut = Q.build_exp_lut(sx, entries=entries, out_bits=out_bits)

        elif op.type == "gelu":
            p.lut = Q.build_unary_lut(_gelu_scalar, sx, sy)

        elif op.type == "relu":
            p.requant = rq(sx / sy)

        else:
            raise NotImplementedError(f"no quantization rule for op type {op.type!r}")

        qp.op_params[op.name] = p

    return qp


def _check_conv_limits(op: Op, graph: Graph, machine: MachineSpec) -> None:
    w = graph.initializers[op.inputs[1]]
    if op.type in ("conv2d", "conv_transpose2d"):
        machine.check_attr(op.type, "kernel", max(int(w.shape[2]), int(w.shape[3])))
    if op.type == "conv_transpose2d":
        constraints = machine.require(op.type)
        if constraints.get("stride_equals_kernel_only") and int(op.attr("stride", 1)) != int(w.shape[2]):
            raise NotImplementedError(
                f"{op.name}: {machine.name} implements transposed convolution only where "
                f"stride == kernel; got stride {op.attr('stride', 1)} kernel {w.shape[2]}"
            )


# ---------------------------------------------------------------------------
# End-to-end calibration
# ---------------------------------------------------------------------------


DEFAULT_CALIBRATION_NORM = 1.0


def mse_optimal_threshold(hist: np.ndarray, amax: float, n_levels: int = 127,
                          candidates: int = 96, min_fraction: float = 0.02,
                          p: float = DEFAULT_CALIBRATION_NORM) -> float:
    """Pick the clipping threshold minimizing ``E|x - quant(x)|^p``, from a histogram.

    Clipping at the maximum wastes resolution on rare outliers; clipping at a
    fixed percentile is an arbitrary guess. This measures the error instead:
    values below the threshold suffer rounding error, values above it suffer
    saturation error, and the histogram says how many values are where.
    Saturation needs no special case -- clamping the level at ``n_levels``
    reconstructs anything above the threshold as the threshold.

    ``p`` matters more than it looks, and not monotonically. Measured end to end
    on DA-V2 at 518x518: p=2.0 gives 6.2 dB, p=1.0 gives 14.0 dB, p=0.7 gives
    3.6 dB, p=0.3 gives 1.6 dB. Squared error is dominated by the few largest
    outliers and under-clips; a very low exponent stops caring about magnitude
    at all and over-clips. p=1 is the measured optimum, but the sharp cliffs
    either side of it are a warning: the model's accuracy hangs on a handful of
    sensitive tensors, and no single global rule addresses that well.
    """
    if amax <= 0.0 or hist.sum() == 0:
        return amax
    if p <= 0.0:
        raise ValueError(f"norm exponent must be positive, got {p}")
    bins = hist.shape[0]
    centers = (np.arange(bins, dtype=np.float64) + 0.5) * (amax / bins)
    counts = hist.astype(np.float64)

    thresholds = np.linspace(max(min_fraction, 1.0 / bins) * amax, amax, candidates)
    step = thresholds / n_levels                                        # [C]
    levels = np.clip(np.rint(centers[None, :] / step[:, None]), 0, n_levels)
    error = np.abs(centers[None, :] - levels * step[:, None]) ** p       # [C, B]
    return float(thresholds[int(np.argmin(error @ counts))])


def calibrate(graph: Graph, batches: Iterable[dict[str, np.ndarray]], *,
              method: str = "amax",
              percentile: float | None = None,
              bins: int = 2048,
              norm: float = DEFAULT_CALIBRATION_NORM,
              progress: Callable[[int], None] | None = None) -> dict[str, float]:
    """Run the fp32 graph over calibration inputs and choose a clipping threshold per tensor.

    ``method`` is one of:

    - ``"amax"``    -- clip at the observed maximum; no clipping at all.
    - ``"percentile"`` -- clip at ``percentile``, averaged over batches.
    - ``"mse"``     -- two passes: find the range, histogram against it, then pick
      the threshold minimizing quantization error. Costs one extra forward pass
      and is the only one of the three that optimizes anything.
    """
    if method not in ("amax", "percentile", "mse"):
        raise ValueError(f"unknown calibration method {method!r}")
    if method == "percentile" and percentile is None:
        raise ValueError("method='percentile' needs a percentile value")

    batches = list(batches)  # the mse method needs a second pass over the same data
    if not batches:
        raise ValueError("calibration needs at least one batch")

    obs = RangeObserver(percentile=percentile if method == "percentile" else None)
    for n, batch in enumerate(batches, 1):
        run_float(graph, batch, observer=obs)
        if progress:
            progress(n)
    if method != "mse":
        return obs.thresholds()

    hist_obs = HistogramObserver(obs.amax, bins=bins)
    for n, batch in enumerate(batches, 1):
        run_float(graph, batch, observer=hist_obs)
        if progress:
            progress(len(batches) + n)

    thresholds: dict[str, float] = {}
    for name, limit in obs.amax.items():
        counts = hist_obs.hist.get(name)
        thresholds[name] = (
            mse_optimal_threshold(counts, limit, p=norm) if counts is not None else limit
        )
    return thresholds


def _channel_mean(value: np.ndarray, axis: int) -> np.ndarray:
    """Mean over every axis except ``axis``, which may be negative."""
    value = np.asarray(value, dtype=np.float64)
    axis %= value.ndim
    other = tuple(i for i in range(value.ndim) if i != axis)
    return value.mean(axis=other)


def correct_biases(graph: Graph, qp: QuantParams, batches: Iterable[dict[str, np.ndarray]], *,
                   machine: MachineSpec) -> int:
    """Fold the systematic error introduced by weight rounding into the biases.

    Rounding weights does not only add noise, it shifts each output channel's
    mean by ``E[(W - Ŵ)x]``, where Ŵ is the dequantized quantized weight. That
    shift is a constant, so it can be cancelled by a constant -- and the bias is
    already in the datapath, which makes the correction free at inference.

    The expectation is evaluated analytically as ``(W - Ŵ) · E[x]``, needing
    only the per-channel mean of each layer's input from one float pass. That
    matters for more than speed: measuring the *observed* output shift of the
    quantized network instead would fold in the error inherited from every
    earlier layer, so correcting all layers at once would count those inherited
    shifts repeatedly. Measured on DA-V2 that mistake destroys the model
    outright -- 15.9 dB down to 0.05 dB.

    Returns the number of ops corrected.
    """
    from .execute import run_float

    batches = list(batches)
    if not batches:
        raise ValueError("bias correction needs at least one batch")

    targets = [
        op for op in graph.ops
        if op.type in ("conv2d", "conv_transpose2d", "matmul") and len(op.inputs) > 2
    ]
    if not targets:
        return 0
    # Input channel axis: matmul reduces over the last axis, convolutions over NCHW's C.
    input_axis = {"matmul": -1, "conv2d": 1, "conv_transpose2d": 1}
    wanted = {op.inputs[0]: input_axis[op.type] for op in targets}

    sums: dict[str, np.ndarray] = {}

    def observe(name: str, value: np.ndarray) -> None:
        axis = wanted.get(name)
        if axis is None:
            return
        means = _channel_mean(value, axis)
        sums[name] = means if name not in sums else sums[name] + means

    for batch in batches:
        run_float(graph, batch, observer=observe)

    corrected = 0
    n = float(len(batches))
    for op in targets:
        params = qp.op_params[op.name]
        if params.bias_q is None or op.inputs[0] not in sums:
            continue
        mean_in = sums[op.inputs[0]] / n

        w_name = op.inputs[1]
        w = graph.initializers[w_name].astype(np.float64)
        s_w = qp.weight_scales.get(w_name)
        if s_w is None:
            w_hat = qp.const_q[w_name].astype(np.float64) * qp.scales[w_name]
            step_w = qp.scales[w_name]
        else:
            shape = [1] * w.ndim
            shape[WEIGHT_OUTPUT_AXIS[op.type]] = -1
            w_hat = qp.const_q[w_name].astype(np.float64) * s_w.reshape(shape)
            step_w = s_w
        residual = w - w_hat

        if op.type == "matmul":
            delta = residual @ mean_in                       # [out]
        elif op.type == "conv2d":
            delta = np.einsum("ocij,c->o", residual, mean_in)
        else:  # conv_transpose2d stores [in, out, kh, kw]
            delta = np.einsum("iokl,i->o", residual, mean_in)

        # The accumulator is in units of s_x * s_w, so adding delta/(s_x*s_w)
        # here arrives as delta in real units after requantization.
        step = np.asarray(qp.scales[op.inputs[0]] * np.asarray(step_w), dtype=np.float64)
        adjusted = params.bias_q.astype(np.float64) + delta / step
        params.bias_q = np.clip(np.rint(adjusted), -(1 << 31), (1 << 31) - 1).astype(np.int32)
        corrected += 1
    return corrected


def quantize_graph(graph: Graph, batches: Iterable[dict[str, np.ndarray]], *,
                   machine: MachineSpec,
                   method: str = "amax",
                   percentile: float | None = None,
                   norm: float = DEFAULT_CALIBRATION_NORM,
                   residual_int16: bool = False,
                   per_channel_weights: bool = False,
                   bias_correction: bool = False,
                   progress: Callable[[int], None] | None = None) -> QuantParams:
    batches = list(batches)
    thresholds = calibrate(graph, batches, method=method, percentile=percentile,
                           norm=norm, progress=progress)
    wide = wide_tensors(graph, residual=residual_int16)
    scales = resolve_scales(graph, thresholds, wide=wide)
    qp = build_quant_params(graph, scales, machine, residual_int16=residual_int16,
                            per_channel_weights=per_channel_weights)
    if bias_correction:
        correct_biases(graph, qp, batches, machine=machine)
    return qp
