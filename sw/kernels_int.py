# ABOUTME: Integer-only kernels defining exactly what the tpu-v2 RTL must compute, bit for bit.
# ABOUTME: No float arithmetic occurs on any datapath value; floats appear only in compile-time constant tables.

from __future__ import annotations

import numpy as np

from .numerics import INT8_MAX, INT8_MIN, QuantizationError, Requant

# int8 products summed over K fit exactly in a float64 mantissa while
# K * 127 * 127 < 2**53, which holds for every layer in DA-V2 by a wide margin.
_EXACT_FLOAT_MATMUL_LIMIT = (1 << 53) // (127 * 127)


def exact_int_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Integer matmul with no rounding, using BLAS when it is provably exact.

    numpy has no BLAS path for integer dtypes, and an int64 einsum over a ViT
    is unusably slow. float64 products of int8 values are exact, and so are
    their sums while the accumulator stays under 2**53, so the fast path is
    bit-identical to the slow one rather than an approximation of it.
    """
    k = a.shape[-1]
    if k <= _EXACT_FLOAT_MATMUL_LIMIT:
        out = np.matmul(a.astype(np.float64), b.astype(np.float64))
        return out.astype(np.int64)
    return np.matmul(a.astype(np.int64), b.astype(np.int64))


def _check_accum(acc: np.ndarray, what: str, accum_bits: int = 32) -> np.ndarray:
    lo, hi = -(1 << (accum_bits - 1)), (1 << (accum_bits - 1)) - 1
    if acc.size and (int(acc.min()) < lo or int(acc.max()) > hi):
        raise QuantizationError(
            f"{what}: accumulator exceeded int{accum_bits} "
            f"(range {int(acc.min())}..{int(acc.max())}); the reduction is too deep "
            "for this datapath at these scales"
        )
    return acc


# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------


def qlinear(x_q: np.ndarray, w_q: np.ndarray, bias_q: np.ndarray | None,
            requant: Requant) -> np.ndarray:
    """x_q [..., K] int8, w_q [N, K] int8, bias_q [N] int32 at scale sx*sw."""
    acc = exact_int_matmul(x_q, w_q.T)
    if bias_q is not None:
        acc = acc + bias_q.astype(np.int64)
    _check_accum(acc, "qlinear")
    # Output channels are the innermost axis, where a per-channel vector already
    # broadcasts correctly.
    return requant.apply(acc.astype(np.int32))


def qbatch_matmul(a_q: np.ndarray, b_q: np.ndarray, requant: Requant) -> np.ndarray:
    acc = exact_int_matmul(a_q, b_q)
    _check_accum(acc, "qbatch_matmul")
    return requant.apply(acc.astype(np.int32))


def qconv2d(x_q: np.ndarray, w_q: np.ndarray, bias_q: np.ndarray | None,
            stride: int, pad: int, requant: Requant) -> np.ndarray:
    """x_q [N,C,H,W] int8, w_q [O,C,kh,kw] int8. Lowered exactly as the array runs it."""
    from .kernels_float import im2col  # layout helper is dtype-agnostic

    n, _, h, w = x_q.shape
    o, c, kh, kw = w_q.shape
    cols = im2col(x_q, kh, kw, stride, pad)             # [N, C*kh*kw, L] int8
    mat = w_q.reshape(o, c * kh * kw)                   # [O, C*kh*kw]
    acc = exact_int_matmul(mat, cols)                   # [N, O, L] via broadcast
    if bias_q is not None:
        acc = acc + bias_q.reshape(1, o, 1).astype(np.int64)
    _check_accum(acc, "qconv2d")
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (w + 2 * pad - kw) // stride + 1
    # acc is [N, O, L]: output channels sit on axis 1.
    out = requant.broadcast_to(3, 1).apply(acc.astype(np.int32))
    return out.reshape(n, o, oh, ow)


def qconv_transpose2d(x_q: np.ndarray, w_q: np.ndarray, bias_q: np.ndarray | None,
                      stride: int, pad: int, requant: Requant) -> np.ndarray:
    """x_q [N,I,H,W] int8, w_q [I,O,kh,kw] int8. Scatter-accumulate into int32."""
    n, i, h, w = x_q.shape
    wi, o, kh, kw = w_q.shape
    if wi != i:
        raise ValueError(f"qconv_transpose2d channel mismatch: {i} vs {wi}")
    oh = (h - 1) * stride - 2 * pad + kh
    ow = (w - 1) * stride - 2 * pad + kw
    canvas = np.zeros((n, o, oh + 2 * pad, ow + 2 * pad), dtype=np.int64)
    rows = np.arange(h) * stride
    cols = np.arange(w) * stride
    xi = x_q.astype(np.float64)
    for kr in range(kh):
        for kc in range(kw):
            contrib = np.einsum("nihw,io->nohw", xi, w_q[:, :, kr, kc].astype(np.float64))
            canvas[:, :, rows[:, None] + kr, cols[None, :] + kc] += contrib.astype(np.int64)
    acc = canvas[:, :, pad:pad + oh, pad:pad + ow]
    if bias_q is not None:
        acc = acc + bias_q.reshape(1, o, 1, 1).astype(np.int64)
    _check_accum(acc, "qconv_transpose2d")
    return requant.broadcast_to(4, 1).apply(acc.astype(np.int32))


# ---------------------------------------------------------------------------
# Elementwise
# ---------------------------------------------------------------------------

_ADD_GUARD_BITS = 12  # keep sub-LSB precision through the residual chain


def qadd(a_q: np.ndarray, b_q: np.ndarray, requant_a: Requant, requant_b: Requant,
         out_min: int = INT8_MIN, out_max: int = INT8_MAX) -> np.ndarray:
    """Residual add of two tensors on different scales onto a third scale.

    Each operand is rescaled while retaining ``_ADD_GUARD_BITS`` fractional
    bits, summed, then rounded once. Rounding each operand to int8 first would
    inject up to half an LSB per add, and DA-V2 has 24 of them in series.

    ``out_min``/``out_max`` widen the result when the residual stream is carried
    at int16 rather than int8.
    """
    wide_a = requant_a.apply_keep(a_q.astype(np.int32), _ADD_GUARD_BITS)
    wide_b = requant_b.apply_keep(b_q.astype(np.int32), _ADD_GUARD_BITS)
    acc = wide_a + wide_b
    half = np.int64(1) << np.int64(_ADD_GUARD_BITS - 1)
    out = (acc + half) >> np.int64(_ADD_GUARD_BITS)
    clipped = np.clip(out, out_min, out_max)
    return clipped.astype(np.int8 if out_max <= INT8_MAX else np.int16)


def qmul_channel(x_q: np.ndarray, gamma_q: np.ndarray, requant: Requant) -> np.ndarray:
    """Per-channel scalar multiply (LayerScale). gamma_q broadcasts over the last axis."""
    acc = x_q.astype(np.int32) * gamma_q.astype(np.int32)
    return requant.apply(acc)


def qrelu(x_q: np.ndarray, requant: Requant | None = None) -> np.ndarray:
    out = np.maximum(x_q.astype(np.int32), 0)
    if requant is None:
        return np.clip(out, INT8_MIN, INT8_MAX).astype(np.int8)
    return requant.apply(out)


# ---------------------------------------------------------------------------
# LUT-based nonlinearities
# ---------------------------------------------------------------------------


def build_unary_lut(fn, scale_in: float, scale_out: float) -> np.ndarray:
    """256-entry int8->int8 table for an arbitrary scalar function.

    Indexed by ``x_q + 128`` so the table is a flat ROM addressed by the raw
    two's-complement byte. Built once at compile time.
    """
    q = np.arange(-128, 128, dtype=np.int32)
    real = fn(q.astype(np.float64) * scale_in)
    out = np.rint(real / scale_out)
    return np.clip(out, INT8_MIN, INT8_MAX).astype(np.int8)


def apply_unary_lut(x_q: np.ndarray, lut: np.ndarray) -> np.ndarray:
    if lut.shape != (256,):
        raise ValueError(f"unary LUT must have 256 entries, got {lut.shape}")
    return lut[x_q.astype(np.int32) + 128]


def build_exp_lut(scale_in: float, entries: int = 256, out_bits: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Two cascaded tables giving exp(-scale_in * d) for d in [0, entries**2).

    Softmax subtracts the row max first, so the argument is always <= 0 and a
    one-sided table suffices. A single 256-entry table is enough for int8
    logits but not for int16 ones, where the useful range of d spans tens of
    thousands. Splitting d into a high and low byte turns that into two 256-entry
    ROMs and one multiply -- ``exp(-s*(256*hi + lo)) = exp(-s*256*hi) *
    exp(-s*lo)`` -- which is cheap in hardware and exact to the output LSB.

    Both tables are scaled by ``2**(out_bits-1) - 1``. The resulting uniform
    factor cancels in softmax's own normalization, so it costs no accuracy.
    """
    one = (1 << (out_bits - 1)) - 1
    idx = np.arange(entries, dtype=np.float64)
    lut_lo = np.rint(np.exp(-scale_in * idx) * one).astype(np.int32)
    lut_hi = np.rint(np.exp(-scale_in * idx * entries) * one).astype(np.int32)
    return lut_lo, lut_hi


_SOFTMAX_FRAC_BITS = 15
_EXP_LUT_SHIFT = 15


def qsoftmax(x_q: np.ndarray, exp_lut: tuple[np.ndarray, np.ndarray], scale_out: float,
             axis: int = -1) -> np.ndarray:
    """Integer softmax over the innermost axis. Output is int8 at ``scale_out``.

    Accepts int8 or int16 logits; attention logits need the wider input because
    their spread across a long token sequence does not fit in 8 bits.
    """
    if axis not in (-1, x_q.ndim - 1):
        raise NotImplementedError("tpu-v2 softmax is defined on the innermost axis only")
    lut_lo, lut_hi = exp_lut
    entries = lut_lo.shape[0]
    xi = x_q.astype(np.int32)
    m = xi.max(axis=-1, keepdims=True)
    d = np.clip(m - xi, 0, entries * entries - 1)       # >= 0 by construction
    hi, lo = d // entries, d % entries
    e = (lut_lo[lo].astype(np.int64) * lut_hi[hi].astype(np.int64)) >> np.int64(_EXP_LUT_SHIFT)
    total = e.sum(axis=-1, keepdims=True)
    if np.any(total <= 0):
        raise QuantizationError("softmax denominator underflowed to zero; exp LUT is too coarse")
    probs = (e << np.int64(_SOFTMAX_FRAC_BITS)) // total  # fixed-point in [0, 2**15]
    # scale_out maps [0,1] onto int8; the reciprocal is a compile-time constant.
    requant = Requant.from_real_multiplier(1.0 / (float(1 << _SOFTMAX_FRAC_BITS) * scale_out))
    return requant.apply(probs.astype(np.int32))


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------


_LN_FRAC_BITS = 14


def _isqrt(x: np.ndarray) -> np.ndarray:
    """Exact integer square root for any non-negative int64.

    A float64 sqrt seeds the value, then Newton iterations refine it in pure
    integer arithmetic -- the same structure the RTL uses, and the reason it
    stays exact past 2**53 where float64 alone would not be. Two final
    comparisons pin down the off-by-one.
    """
    if np.any(x < 0):
        raise ValueError("isqrt of a negative value")
    if x.size and int(x.max()) >= (1 << 62):
        raise QuantizationError("LayerNorm variance accumulator exceeded 2**62")
    x = x.astype(np.int64)
    nonzero = x > 0
    r = np.maximum(np.sqrt(x.astype(np.float64)).astype(np.int64), 1)
    for _ in range(4):
        r = np.where(nonzero, (r + np.where(nonzero, x, 0) // r) // 2, r)
        r = np.maximum(r, 1)
    r = np.where(r * r > x, r - 1, r)
    r = np.where((r + 1) * (r + 1) <= x, r + 1, r)
    return np.where(nonzero, r, 0)


def qlayernorm(x_q: np.ndarray, gamma_q: np.ndarray, beta_q32: np.ndarray,
               scale_in: float, requant: Requant, eps: float = 1e-6) -> np.ndarray:
    """Integer LayerNorm over the innermost axis.

    The input scale cancels inside ``(x - mean) / sqrt(var)``, so the whole
    normalization runs in the raw quantized domain and only the affine tail
    needs a scale. ``requant`` must have been built with the ``sqrt(N) /
    2**FRAC`` constant folded in (see quantize.py).
    """
    n = x_q.shape[-1]
    xi = x_q.astype(np.int64)
    total = xi.sum(axis=-1, keepdims=True)
    d = n * xi - total                                   # == N * (x - mean), exact
    sum_d2 = (d * d).sum(axis=-1, keepdims=True)
    # eps lives in the real domain; convert once to the squared quantized domain.
    eps_q = int(round(eps / (scale_in * scale_in) * n**3))
    denom = sum_d2 + eps_q
    r = _isqrt(denom)
    r = np.maximum(r, 1)
    norm = (d << np.int64(_LN_FRAC_BITS)) // r            # ~ normalized * 2**FRAC / sqrt(N)
    acc = norm * gamma_q.astype(np.int64)
    _check_accum(acc, "qlayernorm", accum_bits=40)
    out = requant.apply_keep(acc, 0) + beta_q32.astype(np.int64)
    return np.clip(out, INT8_MIN, INT8_MAX).astype(np.int8)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

_INTERP_FRAC_BITS = 8


def qinterpolate_bilinear(x_q: np.ndarray, size: tuple[int, int],
                          align_corners: bool = True) -> np.ndarray:
    """Fixed-point bilinear resize. Output shares the input's scale, so no requant.

    Weights are quantized to ``_INTERP_FRAC_BITS`` fractional bits, matching the
    machine description's ``bilinear_upsample.weight_frac_bits``.
    """
    from .kernels_float import _sample_coords

    n, c, h, w = x_q.shape
    oh, ow = size
    one = 1 << _INTERP_FRAC_BITS
    sy = _sample_coords(oh, h, align_corners)
    sx = _sample_coords(ow, w, align_corners)
    y0 = np.floor(sy).astype(np.int64)
    x0 = np.floor(sx).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    ty = np.rint((sy - y0) * one).astype(np.int64)
    tx = np.rint((sx - x0) * one).astype(np.int64)
    xi = x_q.astype(np.int64)
    top = xi[:, :, y0, :]
    bot = xi[:, :, y1, :]
    rows = top * (one - ty)[None, None, :, None] + bot * ty[None, None, :, None]
    left = rows[:, :, :, x0]
    right = rows[:, :, :, x1]
    acc = left * (one - tx)[None, None, None, :] + right * tx[None, None, None, :]
    half = np.int64(1) << np.int64(2 * _INTERP_FRAC_BITS - 1)
    out = (acc + half) >> np.int64(2 * _INTERP_FRAC_BITS)
    return np.clip(out, INT8_MIN, INT8_MAX).astype(np.int8)
