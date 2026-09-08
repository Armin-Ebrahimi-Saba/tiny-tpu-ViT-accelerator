# ABOUTME: Bit-accurate symmetric int8 quantization primitives shared by the emulator and compiler.
# ABOUTME: Every function here is integer-only so the RTL can mirror it exactly; floats appear only in offline constant derivation.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INT8_MIN = -127  # see machines/tpu_v2.json: -128 is excluded to keep negation safe
INT8_MAX = 127
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


class QuantizationError(ValueError):
    """Raised when a scale or multiplier cannot be represented by the datapath."""


# ---------------------------------------------------------------------------
# Scales
# ---------------------------------------------------------------------------


def scale_from_amax(amax: float, n_levels: int = INT8_MAX) -> float:
    """Symmetric scale mapping +/-amax onto +/-n_levels.

    A zero or non-finite amax degenerates the tensor to all zeros, which would
    make every downstream multiplier zero too; we clamp to a tiny positive
    scale instead so the graph stays numerically well-formed.
    """
    if not np.isfinite(amax) or amax <= 0.0:
        return float(np.finfo(np.float32).tiny)
    return float(amax) / float(n_levels)


def quantize_tensor(x: np.ndarray, scale: float) -> np.ndarray:
    """Real -> int8 with round-half-away-from-zero and saturation."""
    if scale <= 0.0:
        raise QuantizationError(f"scale must be positive, got {scale}")
    q = np.rint(np.asarray(x, dtype=np.float64) / scale)
    return np.clip(q, INT8_MIN, INT8_MAX).astype(np.int8)


def dequantize_tensor(q: np.ndarray, scale: float) -> np.ndarray:
    return np.asarray(q, dtype=np.float64) * scale


def amax_per_channel(w: np.ndarray, axis: int) -> np.ndarray:
    """Absolute maximum of every slice along ``axis``, for per-channel weight scales."""
    other = tuple(i for i in range(w.ndim) if i != axis)
    return np.abs(np.asarray(w, dtype=np.float64)).max(axis=other)


def quantize_per_channel(w: np.ndarray, scales: np.ndarray, axis: int) -> np.ndarray:
    """Real -> int8 with one scale per slice along ``axis``."""
    shape = [1] * w.ndim
    shape[axis] = -1
    s = np.asarray(scales, dtype=np.float64).reshape(shape)
    if np.any(s <= 0.0):
        raise QuantizationError("per-channel scales must all be positive")
    q = np.rint(np.asarray(w, dtype=np.float64) / s)
    return np.clip(q, INT8_MIN, INT8_MAX).astype(np.int8)


def quantize_bias(b: np.ndarray, scale) -> np.ndarray:
    """Bias is quantized at the accumulator scale (s_w * s_x) as int32."""
    scale = np.asarray(scale, dtype=np.float64)
    if np.any(scale <= 0.0):
        raise QuantizationError(f"bias scale must be positive, got {scale}")
    q = np.rint(np.asarray(b, dtype=np.float64) / scale)
    if np.any(np.abs(q) > INT32_MAX):
        raise QuantizationError("bias does not fit in int32 at the accumulator scale")
    return q.astype(np.int32)


# ---------------------------------------------------------------------------
# Requantization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Requant:
    """acc_int32 -> int8 as ``sat(round_half_up(acc * multiplier / 2**shift))``.

    One 32x32->64 multiply plus an arithmetic shift, which is a single DSP
    slice and a shifter in hardware. ``multiplier`` is normalized into
    [2**30, 2**31) so the shift alone carries the magnitude.
    """

    multiplier: int
    shift: int
    out_min: int = INT8_MIN
    out_max: int = INT8_MAX

    @classmethod
    def from_real_multiplier(cls, m: float | np.ndarray, shift_min: int = 0, shift_max: int = 62,
                             out_min: int = INT8_MIN, out_max: int = INT8_MAX) -> Requant:
        """Normalize one multiplier, or a vector of them for per-channel requantization.

        ``frexp`` splits m into a mantissa in [0.5, 1) and an exponent, which is
        exactly the normalization wanted: ``mult = mantissa * 2**31`` lands in
        [2**30, 2**31) and ``shift = 31 - exponent`` carries the magnitude, with
        no iteration and no rounding drift.
        """
        arr = np.asarray(m, dtype=np.float64)
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
            raise QuantizationError("real multipliers must all be finite and positive")

        mantissa, exponent = np.frexp(arr)
        mult = np.rint(mantissa * float(1 << 31)).astype(np.int64)
        shift = (31 - exponent).astype(np.int64)
        # Rounding can push a mantissa of nearly 1.0 onto exactly 2**31.
        overflow = mult >= (1 << 31)
        mult = np.where(overflow, mult >> 1, mult)
        shift = np.where(overflow, shift - 1, shift)

        if np.any(shift < shift_min) or np.any(shift > shift_max):
            raise QuantizationError(
                f"multiplier needs a shift outside [{shift_min}, {shift_max}] "
                f"(got {int(shift.min())}..{int(shift.max())}); the two tensors' scales "
                "differ by more than the datapath can express"
            )
        if arr.ndim == 0:
            return cls(multiplier=int(mult), shift=int(shift), out_min=out_min, out_max=out_max)
        return cls(multiplier=mult, shift=shift, out_min=out_min, out_max=out_max)

    @property
    def is_per_channel(self) -> bool:
        return isinstance(self.multiplier, np.ndarray)

    @property
    def real_multiplier(self):
        return np.asarray(self.multiplier) / np.exp2(np.asarray(self.shift, dtype=np.float64))

    def broadcast_to(self, ndim: int, axis: int) -> Requant:
        """Reshape a per-channel multiplier so it broadcasts along ``axis`` of an ndim tensor."""
        if not self.is_per_channel:
            return self
        shape = [1] * ndim
        shape[axis] = -1
        return Requant(
            multiplier=np.asarray(self.multiplier).reshape(shape),
            shift=np.asarray(self.shift).reshape(shape),
            out_min=self.out_min, out_max=self.out_max,
        )

    def apply(self, acc: np.ndarray) -> np.ndarray:
        """Integer-only requantization. Input must already fit in int32."""
        acc = np.asarray(acc)
        if acc.size and (acc.min() < INT32_MIN or acc.max() > INT32_MAX):
            raise QuantizationError(
                "accumulator overflowed int32 before requantization; "
                "reduce the reduction depth or lower the input scale"
            )
        prod = acc.astype(np.int64) * np.asarray(self.multiplier, dtype=np.int64)
        # round-half-up on a two's-complement arithmetic shift
        shift = np.asarray(self.shift, dtype=np.int64)
        half = np.int64(1) << (shift - 1)
        shifted = (prod + half) >> shift
        clipped = np.clip(shifted, self.out_min, self.out_max)
        return clipped.astype(np.int8 if self.out_max <= INT8_MAX else np.int16)

    def apply_keep(self, acc: np.ndarray, keep_bits: int) -> np.ndarray:
        """Rescale but stop ``keep_bits`` short of the output LSB, without saturating.

        The result is int64 at scale ``s_out / 2**keep_bits``. Used where several
        rescaled terms are summed before a single final rounding, so that each
        term does not contribute its own half-LSB of rounding error.
        """
        if keep_bits < 0:
            raise QuantizationError("keep_bits must be non-negative")
        prod = np.asarray(acc).astype(np.int64) * np.int64(self.multiplier)
        sh = self.shift - keep_bits
        if sh > 0:
            half = np.int64(1) << np.int64(sh - 1)
            return (prod + half) >> np.int64(sh)
        return prod << np.int64(-sh)


def saturate_accum(acc: np.ndarray, accum_bits: int = 32) -> np.ndarray:
    lo = -(1 << (accum_bits - 1))
    hi = (1 << (accum_bits - 1)) - 1
    return np.clip(acc, lo, hi).astype(np.int64)


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------


def sqnr_db(reference: np.ndarray, actual: np.ndarray) -> float:
    """Signal-to-quantization-noise ratio in dB. Higher is better; >20 dB is usually usable."""
    ref = np.asarray(reference, dtype=np.float64).ravel()
    act = np.asarray(actual, dtype=np.float64).ravel()
    if ref.shape != act.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {act.shape}")
    noise = float(np.sum((ref - act) ** 2))
    signal = float(np.sum(ref**2))
    if noise == 0.0:
        return float("inf")
    if signal == 0.0:
        return float("-inf")
    return 10.0 * np.log10(signal / noise)


def cosine_similarity(reference: np.ndarray, actual: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64).ravel()
    act = np.asarray(actual, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(ref) * np.linalg.norm(act))
    if denom == 0.0:
        return 1.0 if np.allclose(ref, act) else 0.0
    return float(np.dot(ref, act) / denom)
