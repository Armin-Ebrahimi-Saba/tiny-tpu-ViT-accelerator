# ABOUTME: Tests the machine description, scale unification, tiling equivalence, and op-support gating.
# ABOUTME: Covers the compiler-side invariants that silently corrupt a model when they break.

from __future__ import annotations

import json

import numpy as np
import pytest

from sw.execute import run_float, run_quant
from sw.ir import Graph, GraphBuilder, Op
from sw.lower import lower_graph, require_supported, run_tiled_gemm
from sw.machine import MachineSpec, UnsupportedOpError
from sw.numerics import Requant, quantize_tensor, scale_from_amax
from sw.quantize import build_quant_params, resolve_scales, unify_scale_groups

RNG = np.random.default_rng(3)


@pytest.fixture
def machine() -> MachineSpec:
    return MachineSpec.load("tpu-v2")


# ---------------------------------------------------------------------------
# Machine description
# ---------------------------------------------------------------------------


def test_machine_loads_and_reports(machine: MachineSpec) -> None:
    assert machine.rows == 16 and machine.cols == 16
    assert machine.macs_per_cycle == 256
    assert machine.peak_macs_per_second == pytest.approx(12.8e9)
    assert "16x16" in machine.summary()


def test_unsupported_op_is_fatal_and_names_alternatives(machine: MachineSpec) -> None:
    with pytest.raises(UnsupportedOpError) as exc:
        machine.require("depthwise_conv2d")
    assert "depthwise_conv2d" in str(exc.value)
    assert "conv2d" in str(exc.value)


def test_machine_validates_its_own_invariants() -> None:
    raw = json.loads((MachineSpec.load("tpu-v2").__class__.__module__ and
                      __import__("pathlib").Path("sw/machines/tpu_v2.json")).read_text())
    raw["requant"]["shift_min"] = raw["requant"]["shift_max"] + 1
    with pytest.raises(ValueError):
        MachineSpec.from_dict(raw)


def test_check_attr_enforces_kernel_bound(machine: MachineSpec) -> None:
    machine.check_attr("conv2d", "kernel", 14)
    with pytest.raises(UnsupportedOpError):
        machine.check_attr("conv2d", "kernel", 32)


# ---------------------------------------------------------------------------
# IR
# ---------------------------------------------------------------------------


def test_graph_rejects_use_before_def() -> None:
    g = Graph(name="bad", inputs=["x"],
              ops=[Op(type="relu", name="r", inputs=["missing"], outputs=["y"])],
              outputs=["y"])
    with pytest.raises(ValueError, match="undefined tensor"):
        g.validate()


def test_graph_rejects_double_assignment() -> None:
    g = Graph(name="bad", inputs=["x"],
              ops=[Op(type="relu", name="a", inputs=["x"], outputs=["y"]),
                   Op(type="relu", name="b", inputs=["y"], outputs=["y"])],
              outputs=["y"])
    with pytest.raises(ValueError, match="more than once"):
        g.validate()


# ---------------------------------------------------------------------------
# Scale unification
# ---------------------------------------------------------------------------


def _reshape_graph() -> Graph:
    b = GraphBuilder("t")
    x = b.input("x")
    w = b.constant("w", np.ones((4, 8), dtype=np.float32))
    h = b.emit("matmul", [x, w], "h")
    r = b.emit("reshape", [h], "r", shape=[2, 2, 4])
    t = b.emit("transpose", [r], "t", perm=[0, 2, 1])
    b.output(t)
    return b.build()


def test_scale_preserving_ops_share_one_scale() -> None:
    g = _reshape_graph()
    groups = unify_scale_groups(g)
    assert groups["h"] == groups["r"] == groups["t"]
    scales = resolve_scales(g, {"x": 1.0, "h": 2.0, "r": 2.0, "t": 2.0})
    assert scales["h"] == scales["r"] == scales["t"]


def test_concat_forces_inputs_onto_a_common_scale() -> None:
    b = GraphBuilder("c")
    x = b.input("x")
    c = b.constant("c", np.full((1, 1, 4), 9.0, dtype=np.float32))
    out = b.emit("concat", [c, x], "out", axis=1)
    b.output(out)
    g = b.build()
    scales = resolve_scales(g, {"x": 1.0, "out": 9.0})
    # The constant's own range (9.0) must win over the activation's (1.0),
    # otherwise concatenating them would silently rescale the constant.
    assert scales["x"] == scales["c"] == scales["out"]
    assert scales["out"] == pytest.approx(scale_from_amax(9.0))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _histogram(values: np.ndarray, amax: float, bins: int = 2048) -> np.ndarray:
    counts, _ = np.histogram(np.abs(values), bins=bins, range=(0.0, amax))
    return counts


def _direct_error(x: np.ndarray, threshold: float, p: float) -> float:
    """Reference objective computed on raw samples rather than a histogram."""
    from sw.numerics import dequantize_tensor, quantize_tensor, scale_from_amax

    s = scale_from_amax(threshold)
    recon = dequantize_tensor(quantize_tensor(x, s), s)
    return float(np.mean(np.abs(recon - x) ** p))


@pytest.mark.parametrize("p", [0.5, 1.0, 2.0])
def test_threshold_search_agrees_with_brute_force(p: float) -> None:
    """The histogram search must find what a direct scan of the samples would."""
    from sw.quantize import mse_optimal_threshold

    x = RNG.standard_normal(100_000)
    x[:300] = RNG.uniform(20, 60, 300)
    amax = float(np.abs(x).max())

    chosen = mse_optimal_threshold(_histogram(x, amax), amax, p=p)
    grid = np.linspace(0.02 * amax, amax, 96)
    brute = grid[int(np.argmin([_direct_error(x, t, p) for t in grid]))]
    # Histogram binning blurs the optimum slightly; agreement to a few bins is exact enough.
    assert abs(chosen - brute) <= 0.05 * amax


def test_norm_exponent_controls_outlier_tolerance() -> None:
    """Why the default is a fractional norm, in one assertion.

    Squared error is dominated by a handful of extreme values, so p=2 refuses to
    clip them and every ordinary value pays for it. A fractional norm counts how
    many values are hurt instead, and clips. Measured on DA-V2 this is the
    difference between 5.8 dB and 17.6 dB end to end.
    """
    from sw.quantize import mse_optimal_threshold

    x = RNG.standard_normal(200_000)
    x[:20] = 400.0                       # 0.01 % of samples, 400x the bulk
    amax = float(np.abs(x).max())
    hist = _histogram(x, amax)

    assert mse_optimal_threshold(hist, amax, p=2.0) > 0.5 * amax
    assert mse_optimal_threshold(hist, amax, p=0.5) < 0.05 * amax


def test_threshold_search_keeps_the_range_on_a_uniform_tensor() -> None:
    """With no outliers there is nothing to gain by clipping, so it should not."""
    from sw.quantize import mse_optimal_threshold

    x = RNG.uniform(-1.0, 1.0, 200_000)
    amax = float(np.abs(x).max())
    assert mse_optimal_threshold(_histogram(x, amax), amax) > 0.9 * amax


def test_threshold_search_rejects_a_nonpositive_norm() -> None:
    from sw.quantize import mse_optimal_threshold

    with pytest.raises(ValueError, match="norm exponent"):
        mse_optimal_threshold(np.ones(64, dtype=np.int64), 1.0, p=0.0)


def test_mse_threshold_survives_degenerate_input() -> None:
    from sw.quantize import mse_optimal_threshold

    assert mse_optimal_threshold(np.zeros(64, dtype=np.int64), 0.0) == 0.0
    assert mse_optimal_threshold(np.zeros(64, dtype=np.int64), 3.0) == 3.0


def test_calibrate_rejects_bad_configuration() -> None:
    from sw.quantize import calibrate

    g = _reshape_graph()
    with pytest.raises(ValueError, match="unknown calibration method"):
        calibrate(g, [{"x": np.zeros((2, 8), np.float32)}], method="magic")
    with pytest.raises(ValueError, match="needs a percentile"):
        calibrate(g, [{"x": np.zeros((2, 8), np.float32)}], method="percentile")
    with pytest.raises(ValueError, match="at least one batch"):
        calibrate(g, [], method="amax")


def test_mse_calibration_runs_two_passes_and_clips() -> None:
    """End-to-end through `calibrate`, on a tensor large enough for outliers to be rare.

    Sample count matters: one extreme value among a few dozen is 3 % of the
    distribution and clipping it is genuinely not worth it, so a toy tensor
    would show no clipping and prove nothing.
    """
    from sw.quantize import calibrate

    b = GraphBuilder("big")
    x = b.input("x")
    b.output(b.emit("relu", [x], "y"))
    g = b.build()

    values = RNG.standard_normal((4096, 8)).astype(np.float32)
    values[0, 0] = 500.0
    batch = {"x": values}

    amax = calibrate(g, [batch], method="amax")
    mse = calibrate(g, [batch], method="mse")
    assert set(mse) == set(amax)
    assert amax["x"] == pytest.approx(500.0)
    assert mse["x"] < 0.1 * amax["x"]


# ---------------------------------------------------------------------------
# Bias correction
# ---------------------------------------------------------------------------


def _biased_graph() -> tuple[Graph, dict[str, np.ndarray]]:
    """A matmul whose weight quantization shifts each output channel's mean."""
    b = GraphBuilder("biased")
    x = b.input("x")
    w = (RNG.standard_normal((24, 32)) * 0.4).astype(np.float32)
    w += 0.11                       # a constant offset quantizes asymmetrically
    wc = b.constant("w", w)
    bc = b.constant("b", np.zeros(24, dtype=np.float32))
    b.output(b.emit("matmul", [x, wc, bc], "y"))
    graph = b.build()
    inputs = {"x": (RNG.standard_normal((256, 32)) + 1.5).astype(np.float32)}
    return graph, inputs


def test_bias_correction_reduces_the_systematic_output_shift(machine: MachineSpec) -> None:
    from sw.quantize import correct_biases

    graph, inputs = _biased_graph()
    ranges: dict[str, float] = {}
    ref = run_float(graph, inputs, observer=lambda n, v: ranges.__setitem__(n, float(np.abs(v).max())))
    scales = resolve_scales(graph, ranges)

    qp = build_quant_params(graph, scales, machine)
    before = run_quant(graph, qp, inputs, machine=machine)["y"]
    shift_before = float(np.abs(ref["y"].mean(axis=0) - before.mean(axis=0)).mean())

    assert correct_biases(graph, qp, [inputs], machine=machine) == 1
    after = run_quant(graph, qp, inputs, machine=machine)["y"]
    shift_after = float(np.abs(ref["y"].mean(axis=0) - after.mean(axis=0)).mean())

    assert shift_after < shift_before


def test_bias_correction_leaves_biasless_ops_alone(machine: MachineSpec) -> None:
    from sw.quantize import correct_biases

    b = GraphBuilder("nobias")
    x = b.input("x")
    wc = b.constant("w", RNG.standard_normal((8, 16)).astype(np.float32))
    b.output(b.emit("matmul", [x, wc], "y"))
    graph = b.build()
    inputs = {"x": RNG.standard_normal((32, 16)).astype(np.float32)}
    scales = resolve_scales(graph, {"x": 3.0, "y": 10.0})
    qp = build_quant_params(graph, scales, machine)
    assert correct_biases(graph, qp, [inputs], machine=machine) == 0


def test_bias_correction_needs_data(machine: MachineSpec) -> None:
    from sw.quantize import correct_biases

    graph, _ = _biased_graph()
    scales = resolve_scales(graph, {"x": 5.0, "y": 20.0})
    qp = build_quant_params(graph, scales, machine)
    with pytest.raises(ValueError, match="at least one batch"):
        correct_biases(graph, qp, [], machine=machine)


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m,k,n", [(7, 16, 16), (33, 40, 20), (1, 384, 1536), (64, 17, 3)])
def test_tiled_gemm_equals_whole_gemm(machine: MachineSpec, m: int, k: int, n: int) -> None:
    a = RNG.integers(-127, 128, size=(m, k)).astype(np.int8)
    b = RNG.integers(-127, 128, size=(k, n)).astype(np.int8)
    tiled = run_tiled_gemm(a, b, machine)
    whole = a.astype(np.int64) @ b.astype(np.int64)
    assert np.array_equal(tiled, whole)


def test_tiled_gemm_rejects_shape_mismatch(machine: MachineSpec) -> None:
    with pytest.raises(ValueError):
        run_tiled_gemm(np.zeros((2, 3), np.int8), np.zeros((4, 5), np.int8), machine)


def test_schedule_costs_a_known_gemm(machine: MachineSpec) -> None:
    b = GraphBuilder("g")
    x = b.input("x")
    w = b.constant("w", np.ones((64, 32), dtype=np.float32))
    b.output(b.emit("matmul", [x, w], "y"))
    g = b.build()
    sched = lower_graph(g, machine, {"x": (10, 32)})
    assert sched.total_macs == 10 * 32 * 64
    assert sched.total_cycles > 0
    assert 0.0 < sched.utilization <= 1.0
    assert sched.total_dram_bytes >= 64 * 32  # weights stream from DRAM at least once


def test_lowering_rejects_a_graph_with_an_unknown_op(machine: MachineSpec) -> None:
    g = Graph(name="x", inputs=["a"],
              ops=[Op(type="fft", name="f", inputs=["a"], outputs=["b"])], outputs=["b"])
    with pytest.raises(UnsupportedOpError, match="fft"):
        require_supported(g, machine)


# ---------------------------------------------------------------------------
# Executor agreement on a small end-to-end graph
# ---------------------------------------------------------------------------


def test_per_channel_weight_quantization_is_far_more_faithful() -> None:
    """One outsized output channel otherwise sets the scale for every other channel.

    This tests the weight quantization itself, not an end-to-end graph: with a
    single per-tensor *output* scale downstream, a dominant channel would swamp
    the result no matter how well the weights were quantized, so an end-to-end
    assertion would measure the wrong thing.
    """
    from sw.numerics import (
        amax_per_channel, dequantize_tensor, quantize_per_channel, quantize_tensor, sqnr_db,
    )

    w = RNG.standard_normal((16, 24)).astype(np.float32)
    w[0] *= 1000.0

    flat = scale_from_amax(float(np.abs(w).max()))
    per_tensor = dequantize_tensor(quantize_tensor(w, flat), flat)

    channel_scales = np.array([scale_from_amax(float(a)) for a in amax_per_channel(w, 0)])
    q = quantize_per_channel(w, channel_scales, 0)
    per_channel = np.asarray(q, dtype=np.float64) * channel_scales.reshape(-1, 1)

    # Score the ordinary channels only. Total SQNR is dominated by the outsized
    # row, which both schemes represent well, and that masks the whole effect:
    # per-tensor quantizes every other row essentially to zero.
    # Per-tensor flushes them all to zero, so the error energy equals the signal
    # energy exactly and SQNR lands on 0 dB: nothing of those channels survives.
    assert np.all(per_tensor[1:] == 0.0)
    assert sqnr_db(w[1:], per_tensor[1:]) <= 0.0
    assert sqnr_db(w[1:], per_channel[1:]) > 35.0


def test_per_channel_uses_the_right_axis_for_transposed_convolution() -> None:
    """conv_transpose2d stores weights [in, out, kh, kw]; output channels are axis 1."""
    from sw.quantize import WEIGHT_OUTPUT_AXIS

    assert WEIGHT_OUTPUT_AXIS["conv2d"] == 0
    assert WEIGHT_OUTPUT_AXIS["matmul"] == 0
    assert WEIGHT_OUTPUT_AXIS["conv_transpose2d"] == 1


def test_per_channel_requant_is_a_vector_of_multipliers(machine: MachineSpec) -> None:
    b = GraphBuilder("pc")
    x = b.input("x")
    wc = b.constant("w", RNG.standard_normal((12, 8)).astype(np.float32))
    b.output(b.emit("matmul", [x, wc], "y"))
    graph = b.build()
    scales = resolve_scales(graph, {"x": 1.0, "y": 3.0})
    qp = build_quant_params(graph, scales, machine, per_channel_weights=True)
    rq = qp.op_params["y/matmul"].requant
    assert rq.is_per_channel
    assert np.asarray(rq.multiplier).shape == (12,)
    assert qp.weight_scales["w"].shape == (12,)


def test_per_channel_is_refused_when_the_machine_forbids_it() -> None:
    import copy
    raw = json.loads(__import__("pathlib").Path("sw/machines/tpu_v2.json").read_text())
    raw = copy.deepcopy(raw)
    raw["requant"]["per_channel"] = False
    strict = MachineSpec.from_dict(raw)

    b = GraphBuilder("pc")
    x = b.input("x")
    wc = b.constant("w", np.ones((4, 4), dtype=np.float32))
    b.output(b.emit("matmul", [x, wc], "y"))
    graph = b.build()
    scales = resolve_scales(graph, {"x": 1.0, "y": 1.0})
    with pytest.raises(UnsupportedOpError, match="per_channel"):
        build_quant_params(graph, scales, strict, per_channel_weights=True)


def test_small_graph_float_and_int_agree(machine: MachineSpec) -> None:
    b = GraphBuilder("tiny")
    x = b.input("x")
    w1 = b.constant("w1", (RNG.standard_normal((32, 16)) * 0.3).astype(np.float32))
    b1 = b.constant("b1", (RNG.standard_normal(32) * 0.1).astype(np.float32))
    h = b.emit("matmul", [x, w1, b1], "h")
    h = b.emit("relu", [h], "act")
    g_ = b.constant("g", np.ones(32, dtype=np.float32))
    be = b.constant("be", np.zeros(32, dtype=np.float32))
    h = b.emit("layernorm", [h, g_, be], "ln", axis=-1, eps=1e-6)
    b.output(h)
    graph = b.build()

    inputs = {"x": RNG.standard_normal((12, 16)).astype(np.float32)}
    ranges = {}
    ref = run_float(graph, inputs, observer=lambda n, v: ranges.__setitem__(n, float(np.abs(v).max())))
    scales = resolve_scales(graph, ranges)
    qp = build_quant_params(graph, scales, machine)
    got = run_quant(graph, qp, inputs, machine=machine)

    from sw.numerics import sqnr_db
    assert sqnr_db(ref["ln"], got["ln"]) > 20
