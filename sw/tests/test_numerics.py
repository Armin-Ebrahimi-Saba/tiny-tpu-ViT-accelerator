# ABOUTME: Tests the quantized multiplier, saturation, and rounding contract the RTL must implement.
# ABOUTME: These are the rules an RTL engineer reads off; if they change here, the hardware is wrong.

from __future__ import annotations

import numpy as np
import pytest

from sw.numerics import (
    INT8_MAX,
    INT8_MIN,
    QuantizationError,
    Requant,
    cosine_similarity,
    dequantize_tensor,
    quantize_bias,
    quantize_tensor,
    scale_from_amax,
    sqnr_db,
)


@pytest.mark.parametrize("m", [1e-6, 0.001, 0.5, 0.999, 1.0, 1.5, 100.0, 1e5])
def test_requant_multiplier_is_normalized_and_accurate(m: float) -> None:
    rq = Requant.from_real_multiplier(m)
    assert (1 << 30) <= rq.multiplier < (1 << 31)
    assert 0 < rq.shift <= 62
    assert rq.real_multiplier == pytest.approx(m, rel=1e-8)


def test_requant_matches_its_integer_definition_exactly() -> None:
    """The contract is the integer expression, evaluated here in Python bignums.

    Comparing against ``acc * 0.37`` instead would disagree at exact ties,
    because 0.37 is not representable and the hardware never sees it -- it sees
    the normalized int32 multiplier.
    """
    rq = Requant.from_real_multiplier(0.37)
    acc = np.arange(-5000, 5000, dtype=np.int32)
    got = rq.apply(acc)
    want = [
        max(INT8_MIN, min(INT8_MAX, (int(a) * rq.multiplier + (1 << (rq.shift - 1))) >> rq.shift))
        for a in acc
    ]
    assert got.tolist() == want


def test_requant_tracks_the_real_multiplier_to_within_one_lsb() -> None:
    rq = Requant.from_real_multiplier(0.37)
    acc = np.arange(-340, 340, dtype=np.int32)
    got = rq.apply(acc).astype(np.float64)
    assert np.abs(got - acc * 0.37).max() <= 0.5 + 1e-6


def test_requant_saturates_both_rails() -> None:
    rq = Requant.from_real_multiplier(1.0)
    out = rq.apply(np.array([-100000, -128, 0, 128, 100000], dtype=np.int32))
    assert out.tolist() == [INT8_MIN, INT8_MIN, 0, INT8_MAX, INT8_MAX]


def test_requant_rejects_nonsense_multipliers() -> None:
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(QuantizationError):
            Requant.from_real_multiplier(bad)


def test_requant_rejects_overflowed_accumulator() -> None:
    rq = Requant.from_real_multiplier(0.5)
    with pytest.raises(QuantizationError):
        rq.apply(np.array([1 << 40], dtype=np.int64))


def test_apply_keep_is_higher_precision_than_apply() -> None:
    """The guard bits must actually buy precision, otherwise qadd's extra work is pointless."""
    rq = Requant.from_real_multiplier(0.1)
    acc = np.arange(-500, 500, dtype=np.int32)
    coarse = rq.apply(acc).astype(np.float64)
    fine = rq.apply_keep(acc, 12).astype(np.float64) / 4096.0
    exact = acc * 0.1
    assert np.abs(fine - exact).max() < np.abs(coarse - exact).max()
    assert np.abs(fine - exact).max() < 1e-3


def test_quantize_round_trip_is_within_half_an_lsb() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(10000).astype(np.float32) * 3
    s = scale_from_amax(float(np.abs(x).max()))
    back = dequantize_tensor(quantize_tensor(x, s), s)
    assert np.abs(back - x).max() <= s / 2 + 1e-9


def test_quantize_saturates_beyond_the_scale() -> None:
    q = quantize_tensor(np.array([-10.0, 10.0], dtype=np.float32), scale_from_amax(1.0))
    assert q.tolist() == [INT8_MIN, INT8_MAX]


def test_scale_from_degenerate_amax_stays_positive() -> None:
    assert scale_from_amax(0.0) > 0
    assert scale_from_amax(float("nan")) > 0


def test_quantize_bias_rejects_int32_overflow() -> None:
    with pytest.raises(QuantizationError):
        quantize_bias(np.array([1e9], dtype=np.float32), 1e-6)


def test_sqnr_and_cosine_edge_cases() -> None:
    x = np.array([1.0, 2.0, 3.0])
    assert sqnr_db(x, x) == float("inf")
    assert cosine_similarity(x, x) == pytest.approx(1.0)
    assert cosine_similarity(np.zeros(3), np.zeros(3)) == 1.0
    with pytest.raises(ValueError):
        sqnr_db(x, np.array([1.0]))
