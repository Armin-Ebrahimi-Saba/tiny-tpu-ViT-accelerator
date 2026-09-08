# ABOUTME: Checks each integer kernel against its fp32 counterpart at an SQNR floor.
# ABOUTME: A sign flip, a bad index, or a wrong shift shows up here as a collapse in SQNR.

from __future__ import annotations

import numpy as np
import pytest

from sw import kernels_float as F
from sw import kernels_int as Q
from sw.machine import MachineSpec
from sw.numerics import (
    Requant,
    dequantize_tensor,
    quantize_bias,
    quantize_tensor,
    scale_from_amax,
    sqnr_db,
)

RNG = np.random.default_rng(7)
# int8 per-tensor quantization tops out near 40 dB; anything below 25 dB means
# the kernel is wrong, not merely coarse.
SQNR_FLOOR = 25.0


def amax(x: np.ndarray) -> float:
    return float(np.abs(x).max())


def scales(*arrays: np.ndarray) -> list[float]:
    return [scale_from_amax(amax(a)) for a in arrays]


def test_qlinear() -> None:
    x = RNG.standard_normal((8, 96)).astype(np.float32)
    w = (RNG.standard_normal((48, 96)) * 0.3).astype(np.float32)
    b = (RNG.standard_normal(48) * 0.1).astype(np.float32)
    ref = F.linear(x, w, b)
    sx, sw, sy = scales(x, w, ref)
    got = Q.qlinear(quantize_tensor(x, sx), quantize_tensor(w, sw),
                    quantize_bias(b, sx * sw), Requant.from_real_multiplier(sx * sw / sy))
    assert sqnr_db(ref, dequantize_tensor(got, sy)) > SQNR_FLOOR


@pytest.mark.parametrize("stride,pad,k", [(1, 1, 3), (2, 1, 3), (1, 0, 1)])
def test_qconv2d(stride: int, pad: int, k: int) -> None:
    x = RNG.standard_normal((1, 8, 14, 14)).astype(np.float32)
    w = (RNG.standard_normal((16, 8, k, k)) * 0.2).astype(np.float32)
    b = (RNG.standard_normal(16) * 0.1).astype(np.float32)
    ref = F.conv2d(x, w, b, stride, pad)
    sx, sw, sy = scales(x, w, ref)
    got = Q.qconv2d(quantize_tensor(x, sx), quantize_tensor(w, sw),
                    quantize_bias(b, sx * sw), stride, pad,
                    Requant.from_real_multiplier(sx * sw / sy))
    assert got.shape == ref.shape
    assert sqnr_db(ref, dequantize_tensor(got, sy)) > SQNR_FLOOR


@pytest.mark.parametrize("stride", [2, 4])
def test_qconv_transpose2d(stride: int) -> None:
    x = RNG.standard_normal((1, 6, 5, 5)).astype(np.float32)
    w = (RNG.standard_normal((6, 4, stride, stride)) * 0.2).astype(np.float32)
    b = (RNG.standard_normal(4) * 0.1).astype(np.float32)
    ref = F.conv_transpose2d(x, w, b, stride, 0)
    sx, sw, sy = scales(x, w, ref)
    got = Q.qconv_transpose2d(quantize_tensor(x, sx), quantize_tensor(w, sw),
                              quantize_bias(b, sx * sw), stride, 0,
                              Requant.from_real_multiplier(sx * sw / sy))
    assert got.shape == ref.shape
    assert sqnr_db(ref, dequantize_tensor(got, sy)) > SQNR_FLOOR


def test_qsoftmax_rows_sum_to_one() -> None:
    x = (RNG.standard_normal((2, 4, 64, 64)) * 6).astype(np.float32)
    ref = F.softmax(x, -1)
    sx = scale_from_amax(amax(x))
    sy = 1.0 / 127
    got = dequantize_tensor(Q.qsoftmax(quantize_tensor(x, sx), Q.build_exp_lut(sx), sy), sy)
    assert sqnr_db(ref, got) > SQNR_FLOOR
    assert np.abs(got.sum(axis=-1) - 1.0).max() < 0.05
    assert (got >= 0).all()


def test_qsoftmax_rejects_non_innermost_axis() -> None:
    with pytest.raises(NotImplementedError):
        Q.qsoftmax(np.zeros((2, 3), dtype=np.int8), Q.build_exp_lut(0.1), 1 / 127, axis=0)


def test_qlayernorm() -> None:
    n = 384
    x = (RNG.standard_normal((16, n)) * 2).astype(np.float32)
    g = (RNG.standard_normal(n) * 0.4 + 1).astype(np.float32)
    b = (RNG.standard_normal(n) * 0.1).astype(np.float32)
    ref = F.layernorm(x, g, b, 1e-6)
    sx, sg, sy = scales(x, g, ref)
    m = sg * np.sqrt(n) / ((1 << Q._LN_FRAC_BITS) * sy)
    got = Q.qlayernorm(quantize_tensor(x, sx), quantize_tensor(g, sg),
                       quantize_bias(b, sy), sx, Requant.from_real_multiplier(m), 1e-6)
    assert sqnr_db(ref, dequantize_tensor(got, sy)) > SQNR_FLOOR


def test_qlayernorm_is_invariant_to_input_scale() -> None:
    """The input scale cancels in the normalization; the kernel must show that."""
    n = 64
    x = (RNG.standard_normal((4, n)) * 2).astype(np.float32)
    g = np.ones(n, dtype=np.float32)
    b = np.zeros(n, dtype=np.float32)
    ref = F.layernorm(x, g, b, 1e-6)
    sy = scale_from_amax(amax(ref))
    rq = Requant.from_real_multiplier(
        scale_from_amax(1.0) * np.sqrt(n) / ((1 << Q._LN_FRAC_BITS) * sy))
    outs = []
    for factor in (1.0, 4.0):
        sx = scale_from_amax(amax(x) * factor)
        outs.append(Q.qlayernorm(quantize_tensor(x, sx), quantize_tensor(g, scale_from_amax(1.0)),
                                 quantize_bias(b, sy), sx, rq, 1e-6))
    # Different scales cost resolution, but the results must stay close.
    assert sqnr_db(dequantize_tensor(outs[0], sy), dequantize_tensor(outs[1], sy)) > 15


def test_qgelu_lut() -> None:
    x = (RNG.standard_normal(20000) * 3).astype(np.float32)
    ref = F.gelu(x)
    sx, sy = scales(x, ref)
    lut = Q.build_unary_lut(lambda v: F.gelu(v.astype(np.float32)).astype(np.float64), sx, sy)
    assert lut.shape == (256,)
    got = Q.apply_unary_lut(quantize_tensor(x, sx), lut)
    assert sqnr_db(ref, dequantize_tensor(got, sy)) > SQNR_FLOOR


def test_qinterpolate_preserves_scale() -> None:
    x = RNG.standard_normal((1, 4, 19, 19)).astype(np.float32)
    ref = F.interpolate_bilinear(x, (37, 37), True)
    sx = scale_from_amax(amax(x))
    got = Q.qinterpolate_bilinear(quantize_tensor(x, sx), (37, 37), True)
    assert sqnr_db(ref, dequantize_tensor(got, sx)) > SQNR_FLOOR


def test_qadd_beats_naive_rescaling() -> None:
    a = RNG.standard_normal(5000).astype(np.float32)
    b = (RNG.standard_normal(5000) * 0.5).astype(np.float32)
    ref = a + b
    sa, sb, sy = scales(a, b, ref)
    got = Q.qadd(quantize_tensor(a, sa), quantize_tensor(b, sb),
                 Requant.from_real_multiplier(sa / sy), Requant.from_real_multiplier(sb / sy))
    assert sqnr_db(ref, dequantize_tensor(got, sy)) > 30


def test_exact_int_matmul_is_exact_for_deep_reductions() -> None:
    a = RNG.integers(-127, 128, size=(31, 4608), dtype=np.int64).astype(np.int8)
    b = RNG.integers(-127, 128, size=(4608, 17), dtype=np.int64).astype(np.int8)
    fast = Q.exact_int_matmul(a, b)
    slow = a.astype(np.int64) @ b.astype(np.int64)
    assert np.array_equal(fast, slow)


def test_accumulator_overflow_is_reported_not_silent() -> None:
    machine = MachineSpec.load("tpu-v2")
    assert machine.accum_max == (1 << 31) - 1
    huge = np.full((1, 1 << 20), 127, dtype=np.int8)
    w = np.full((1, 1 << 20), 127, dtype=np.int8)
    with pytest.raises(Exception):
        Q.qlinear(huge, w, None, Requant.from_real_multiplier(1e-6))
