#!/usr/bin/env python3
# ABOUTME: Generates bit-exact stimulus/expected vectors for the tpu-v2 RTL from the sw/ reference kernels.
# ABOUTME: The emulator is the specification, so every expected value here comes from sw.numerics / sw.kernels_int.

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from sw.kernels_int import apply_unary_lut, build_unary_lut, exact_int_matmul  # noqa: E402
from sw.numerics import INT8_MAX, INT8_MIN, Requant  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "vec"
LANES_TR = 16   # buffer lanes, and the square transpose's side


def _hex(value: int, bits: int) -> str:
    """Two's-complement hex, exactly ceil(bits/4) nibbles."""
    return f"{value & ((1 << bits) - 1):0{(bits + 3) // 4}x}"


# ---------------------------------------------------------------------------
# requant_unit
# ---------------------------------------------------------------------------

def gen_requant(rng: np.random.Generator, n: int) -> int:
    """One packed word per vector: {acc[31:0], mult[31:0], shift[5:0], expect[15:0]}."""
    accs: list[int] = []
    mults: list[int] = []
    shifts: list[int] = []

    # Real multipliers spanning the range a scale ratio can plausibly take, plus
    # the extremes of the accumulator so saturation is exercised on both sides.
    reals = np.exp2(rng.uniform(-24.0, -1.0, size=n))
    for i, real in enumerate(reals):
        rq = Requant.from_real_multiplier(float(real))
        mults.append(int(rq.multiplier))
        shifts.append(int(rq.shift))
        if i % 8 == 0:
            # Force ties and saturation cases rather than hoping they turn up.
            # A tie is acc == 2**(shift-1) / mult scaled, but the cheap
            # constructions below already land on exact powers of two, which is
            # where round-half-up and round-half-away disagree.
            step = 1 << min(rq.shift, 30)
            candidates = [0, 1, -1, (1 << 31) - 1, -(1 << 31), step, -step, step >> 1]
            accs.append(int(np.clip(rng.choice(candidates), -(1 << 31), (1 << 31) - 1)))
        else:
            accs.append(int(rng.integers(-(1 << 31), 1 << 31, dtype=np.int64)))

    lines = []
    for acc, mult, shift in zip(accs, mults, shifts):
        expect = int(Requant(multiplier=mult, shift=shift).apply(np.int32(acc)))
        lines.append(_hex(acc, 32) + _hex(mult, 32) + _hex(shift, 8) + _hex(expect, 16))

    (OUT / "requant.hex").write_text("\n".join(lines) + "\n")
    return len(lines)


# ---------------------------------------------------------------------------
# systolic_int8
# ---------------------------------------------------------------------------

def gen_gemm(rng: np.random.Generator, m: int, k: int, n: int) -> None:
    """Exact int32 GEMM: acc[m][n] = bias[n] + sum_k x[m][k] * w[k][n]."""
    x = rng.integers(INT8_MIN, INT8_MAX + 1, size=(m, k), dtype=np.int64)
    w = rng.integers(INT8_MIN, INT8_MAX + 1, size=(k, n), dtype=np.int64)
    # Push a couple of lanes to the corners of the int8 range so the widest
    # product and the deepest column sum both appear in the trace.
    #
    # Careful: a whole column of constant weights makes that column's result
    # invariant to any permutation of k, which masks weight-load misalignment.
    # Keep the saturation cases to single columns and leave the rest random --
    # that asymmetry is exactly what caught the original shift-chain bug.
    x[0, :] = INT8_MIN
    w[:, 0] = INT8_MIN
    if m > 1:
        x[1, :] = INT8_MAX
    if n > 1:
        w[:, 1] = INT8_MAX
    bias = rng.integers(-(1 << 20), 1 << 20, size=n, dtype=np.int64)

    acc = exact_int_matmul(x.astype(np.int8), w.astype(np.int8)) + bias

    # Per-output-channel requantization, sized so most channels land inside int8
    # while the saturating columns above still clip.
    reals = np.exp2(rng.uniform(-1.0, 4.0, size=n)) / max(float(np.abs(acc).max()), 1.0)
    rq = Requant.from_real_multiplier(reals)
    out = rq.apply(acc.astype(np.int32))

    (OUT / "gemm_x.hex").write_text(
        "\n".join(" ".join(_hex(int(v), 8) for v in row) for row in x) + "\n")
    (OUT / "gemm_w.hex").write_text(
        "\n".join(" ".join(_hex(int(v), 8) for v in row) for row in w) + "\n")
    (OUT / "gemm_bias.hex").write_text(
        "\n".join(_hex(int(v), 32) for v in bias) + "\n")
    (OUT / "gemm_acc.hex").write_text(
        "\n".join(" ".join(_hex(int(v), 32) for v in row) for row in acc) + "\n")
    (OUT / "gemm_mult.hex").write_text(
        "\n".join(_hex(int(v), 32) for v in np.asarray(rq.multiplier).ravel()) + "\n")
    (OUT / "gemm_shift.hex").write_text(
        "\n".join(_hex(int(v), 8) for v in np.asarray(rq.shift).ravel()) + "\n")
    (OUT / "gemm_out.hex").write_text(
        "\n".join(" ".join(_hex(int(v), 8) for v in row) for row in out) + "\n")


# ---------------------------------------------------------------------------
# unary_lut (GELU)
# ---------------------------------------------------------------------------

def gen_gelu(rng: np.random.Generator, n: int) -> int:
    """A real GELU table plus random int8 probes through it."""
    from sw.kernels_float import gelu

    # Scales chosen so the table spans the interesting part of the curve rather
    # than saturating: +/-8 in, +/-4 out.
    lut = build_unary_lut(gelu, scale_in=8.0 / 127.0, scale_out=4.0 / 127.0)
    x = rng.integers(INT8_MIN, INT8_MAX + 1, size=n, dtype=np.int64).astype(np.int8)
    # Pin the endpoints and zero, which is where an index-offset bug would show.
    x[:5] = np.array([INT8_MIN, -1, 0, 1, INT8_MAX], dtype=np.int8)
    y = apply_unary_lut(x, lut)

    (OUT / "gelu_lut.hex").write_text(
        "\n".join(_hex(int(v), 8) for v in lut) + "\n")
    (OUT / "gelu_x.hex").write_text(
        "\n".join(_hex(int(v), 8) for v in x) + "\n")
    (OUT / "gelu_y.hex").write_text(
        "\n".join(_hex(int(v), 8) for v in y) + "\n")
    return len(x)


# ---------------------------------------------------------------------------
# divider
# ---------------------------------------------------------------------------

DIV_NUM_BITS = 64
DIV_DEN_BITS = 32


def gen_div(rng: np.random.Generator, n: int) -> int:
    """One packed word per vector: {num[63:0], den[31:0], expect[63:0]}.

    Expected values come from Python's ``//`` because that is literally what
    qsoftmax and qlayernorm use -- floor division, not truncation.
    """
    num_lim = 1 << (DIV_NUM_BITS - 1)
    den_lim = 1 << DIV_DEN_BITS

    cases: list[tuple[int, int]] = []

    # Directed cases first. The negative-inexact ones are the whole reason the
    # divider carries a floor correction, and exact negatives are the trap right
    # next to them: -8 // 2 must stay -4, not become -5.
    cases += [
        (0, 1), (1, 1), (-1, 1), (7, 2), (-7, 2), (8, 2), (-8, 2),
        (-1, 2), (-1, den_lim - 1), (num_lim - 1, 1), (-num_lim + 1, 1),
        (num_lim - 1, den_lim - 1), (-num_lim + 1, den_lim - 1),
        (12345, 12345), (-12345, 12345), (12344, 12345), (-12344, 12345),
    ]

    # Softmax shape: (e << 15) // total, both non-negative.
    for _ in range(n // 4):
        e = int(rng.integers(0, 1 << 15))
        total = int(rng.integers(1, 1 << 26))
        cases.append((e << 15, max(total, e)))

    # LayerNorm shape: (d << 14) // r, signed numerator, r from a real isqrt.
    for _ in range(n // 4):
        d = int(rng.integers(-(1 << 17), 1 << 17))
        r = int(rng.integers(1, 1 << 27))
        cases.append((d << 14, r))

    # Fully random over the declared widths.
    while len(cases) < n:
        num = int(rng.integers(-num_lim, num_lim, dtype=np.int64))
        den = int(rng.integers(1, den_lim, dtype=np.int64))
        cases.append((num, den))

    cases = cases[:n]
    lines = [_hex(num, DIV_NUM_BITS) + _hex(den, DIV_DEN_BITS) + _hex(num // den, DIV_NUM_BITS)
             for num, den in cases]
    (OUT / "div.hex").write_text("\n".join(lines) + "\n")
    return len(lines)


# ---------------------------------------------------------------------------
# softmax_int
# ---------------------------------------------------------------------------

SM_SCALE_IN = 0.02
SM_SCALE_OUT = 1.0 / 127.0


def gen_softmax(rng: np.random.Generator, rows: int, length: int) -> tuple[int, int, int, int]:
    """Rows of int16 logits plus the exact qsoftmax() output for each."""
    from sw.kernels_int import build_exp_lut, qsoftmax

    lut_lo, lut_hi = build_exp_lut(SM_SCALE_IN)
    x = rng.integers(-200, 201, size=(rows, length), dtype=np.int64)
    # Degenerate rows the random draw would not produce: a flat row (every term
    # identical, so the quotient must land exactly on 2**15 / length), and a row
    # with one dominant logit (every other term exp-underflows toward zero).
    x[0, :] = 7
    if rows > 1:
        x[1, :] = -200
        x[1, length // 2] = 200
    y = qsoftmax(x.astype(np.int16), (lut_lo, lut_hi), SM_SCALE_OUT)

    rq = Requant.from_real_multiplier(1.0 / (float(1 << 15) * SM_SCALE_OUT))

    (OUT / "sm_lut_lo.hex").write_text(
        "\n".join(_hex(int(v), 16) for v in lut_lo) + "\n")
    (OUT / "sm_lut_hi.hex").write_text(
        "\n".join(_hex(int(v), 16) for v in lut_hi) + "\n")
    (OUT / "sm_x.hex").write_text(
        "\n".join(" ".join(_hex(int(v), 16) for v in row) for row in x) + "\n")
    (OUT / "sm_y.hex").write_text(
        "\n".join(" ".join(_hex(int(v), 8) for v in row) for row in y) + "\n")
    return rows, length, int(rq.multiplier), int(rq.shift)


# ---------------------------------------------------------------------------
# isqrt
# ---------------------------------------------------------------------------

ISQRT_BITS = 64


def gen_isqrt(rng: np.random.Generator, n: int) -> int:
    """One packed word per vector: {x[63:0], root[31:0]}."""
    from sw.kernels_int import _isqrt

    cases = [0, 1, 2, 3, 4, 5, 8, 9, 15, 16, 17, (1 << 62) - 1]
    # Perfect squares and their immediate neighbours: floor(sqrt) steps exactly
    # at k**2, so k**2 - 1 and k**2 are where an off-by-one shows up.
    for _ in range(n // 3):
        k = int(rng.integers(1, 1 << 30))
        cases += [k * k - 1, k * k, k * k + 1]
    while len(cases) < n:
        cases.append(int(rng.integers(0, 1 << 62)))
    cases = [c for c in cases[:n] if 0 <= c < (1 << 62)]

    roots = _isqrt(np.array(cases, dtype=np.int64))
    lines = [_hex(x, ISQRT_BITS) + _hex(int(r), 32) for x, r in zip(cases, roots)]
    (OUT / "isqrt.hex").write_text("\n".join(lines) + "\n")
    return len(lines)


# ---------------------------------------------------------------------------
# layernorm_int
# ---------------------------------------------------------------------------

LN_SCALE_IN = 1.0 / 64.0
LN_SCALE_GAMMA = 1.0 / 100.0
LN_SCALE_OUT = 1.0 / 127.0
LN_EPS = 1e-6


def gen_layernorm(rng: np.random.Generator, rows: int, length: int) -> tuple:
    """Rows of int8 activations plus the exact qlayernorm() output for each."""
    import math

    from sw.kernels_int import qlayernorm

    x = rng.integers(INT8_MIN, INT8_MAX + 1, size=(rows, length), dtype=np.int64)
    # A constant row drives the variance to zero, which is the only path that
    # reaches the max(r, 1) clamp -- worth pinning rather than hoping for.
    x[0, :] = 42
    if rows > 1:
        x[1, :] = INT8_MIN
        x[1, 0] = INT8_MAX

    gamma = rng.integers(INT8_MIN, INT8_MAX + 1, size=length, dtype=np.int64).astype(np.int8)
    beta = rng.integers(-40, 41, size=length, dtype=np.int64).astype(np.int32)

    # Same construction quantize.py uses: gamma's scale, the sqrt(N) folded out
    # of the integer rsqrt, and the output scale, all in one multiplier.
    real = LN_SCALE_GAMMA * math.sqrt(length) / (float(1 << 14) * LN_SCALE_OUT)
    rq = Requant.from_real_multiplier(real)

    y = qlayernorm(x.astype(np.int8), gamma, beta, LN_SCALE_IN, rq, eps=LN_EPS)
    eps_q = int(round(LN_EPS / (LN_SCALE_IN * LN_SCALE_IN) * length**3))

    (OUT / "ln_x.hex").write_text(
        "\n".join(" ".join(_hex(int(v), 8) for v in row) for row in x) + "\n")
    (OUT / "ln_y.hex").write_text(
        "\n".join(" ".join(_hex(int(v), 8) for v in row) for row in y) + "\n")
    (OUT / "ln_gamma.hex").write_text(
        "\n".join(_hex(int(v), 8) for v in gamma) + "\n")
    (OUT / "ln_beta.hex").write_text(
        "\n".join(_hex(int(v), 32) for v in beta) + "\n")
    return rows, length, eps_q, int(rq.multiplier), int(rq.shift)


# ---------------------------------------------------------------------------
# qadd_unit
# ---------------------------------------------------------------------------

def gen_qadd(rng: np.random.Generator, n: int) -> int:
    """Packed: {a[7:0], b[7:0], mult_a[31:0], sh_a[7:0], mult_b[31:0], sh_b[7:0], expect[7:0]}."""
    from sw.kernels_int import qadd

    a = rng.integers(INT8_MIN, INT8_MAX + 1, size=n, dtype=np.int64).astype(np.int8)
    b = rng.integers(INT8_MIN, INT8_MAX + 1, size=n, dtype=np.int64).astype(np.int8)
    # Both extremes on both operands, so the saturation branch is exercised in
    # each direction rather than left to chance.
    a[:4] = np.array([INT8_MIN, INT8_MAX, INT8_MIN, INT8_MAX], dtype=np.int8)
    b[:4] = np.array([INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN], dtype=np.int8)

    # Residual scale ratios sit near 1, which is the regime the RTL will see.
    lines = []
    for i in range(n):
        rq_a = Requant.from_real_multiplier(float(np.exp2(rng.uniform(-2.0, 1.0))))
        rq_b = Requant.from_real_multiplier(float(np.exp2(rng.uniform(-2.0, 1.0))))
        y = int(qadd(np.array([a[i]]), np.array([b[i]]), rq_a, rq_b)[0])
        lines.append(
            _hex(int(a[i]), 8) + _hex(int(b[i]), 8)
            + _hex(int(rq_a.multiplier), 32) + _hex(int(rq_a.shift), 8)
            + _hex(int(rq_b.multiplier), 32) + _hex(int(rq_b.shift), 8)
            + _hex(y, 8))
    (OUT / "qadd.hex").write_text("\n".join(lines) + "\n")
    return len(lines)


# ---------------------------------------------------------------------------
# gemm_seq
# ---------------------------------------------------------------------------

SEQ_ROWS = 16
SEQ_COLS = 16


def _word_lines(arr: np.ndarray, lanes: int, bits: int) -> str:
    """One packed word per line, lane 0 in the low bits (matching out_data[8*c+:8])."""
    out = []
    for row in arr:
        out.append("".join(_hex(int(v), bits) for v in reversed(row[:lanes])))
    return "\n".join(out) + "\n"


def gen_gemm_seq(rng: np.random.Generator, m: int, k: int, n: int,
                 rows: int, cols: int) -> tuple:
    """A GEMM larger than the array in every dimension, plus its qlinear() result.

    K and N are padded up to whole tiles with zeros, which is what the host does:
    zero weights contribute nothing to the sum, so every tile in the sweep stays
    a full ROWS x COLS tile and the sequencer needs no partial-tile case.
    """
    from sw.kernels_int import qlinear

    kt = (k + rows - 1) // rows
    nt = (n + cols - 1) // cols
    kp, np_ = kt * rows, nt * cols

    x = rng.integers(INT8_MIN, INT8_MAX + 1, size=(m, k), dtype=np.int64).astype(np.int8)
    w = rng.integers(INT8_MIN, INT8_MAX + 1, size=(n, k), dtype=np.int64).astype(np.int8)
    bias = rng.integers(-(1 << 18), 1 << 18, size=n, dtype=np.int64).astype(np.int32)

    acc_max = max(int(np.abs(exact_int_matmul(x, w.T) + bias).max()), 1)
    rq = Requant.from_real_multiplier(np.exp2(rng.uniform(-1.0, 3.0, size=n)) / acc_max)
    y = qlinear(x, w, bias, rq)

    xp = np.zeros((m, kp), dtype=np.int8);   xp[:, :k] = x
    wp = np.zeros((kp, np_), dtype=np.int8); wp[:k, :n] = w.T
    biasp = np.zeros(np_, dtype=np.int64);   biasp[:n] = bias
    multp = np.ones(np_, dtype=np.int64) << 30
    shiftp = np.full(np_, 30, dtype=np.int64)
    multp[:n] = np.asarray(rq.multiplier).ravel()
    shiftp[:n] = np.asarray(rq.shift).ravel()

    # Activation words are (row, k tile); weight words are (k, n tile).
    act = xp.reshape(m * kt, rows)
    wgt = wp.reshape(kp * nt, cols)

    (OUT / "seq_act.hex").write_text(_word_lines(act, rows, 8))
    (OUT / "seq_wgt.hex").write_text(_word_lines(wgt, cols, 8))
    (OUT / "seq_out.hex").write_text(
        "\n".join(" ".join(_hex(int(v), 8) for v in row) for row in y) + "\n")
    (OUT / "seq_bias.hex").write_text("\n".join(_hex(int(v), 32) for v in biasp) + "\n")
    (OUT / "seq_mult.hex").write_text("\n".join(_hex(int(v), 32) for v in multp) + "\n")
    (OUT / "seq_shift.hex").write_text("\n".join(_hex(int(v), 8) for v in shiftp) + "\n")
    return m, k, n, kt, nt


# ---------------------------------------------------------------------------
# vpu_seq
# ---------------------------------------------------------------------------

VPU_LANES = 16


def _pack_stream(vals: np.ndarray, path: pathlib.Path) -> int:
    """Write a flat int8 stream as 128-bit buffer words, zero-padding the tail."""
    flat = np.asarray(vals).ravel()
    words = (len(flat) + VPU_LANES - 1) // VPU_LANES
    padded = np.zeros(words * VPU_LANES, dtype=np.int64)
    padded[:len(flat)] = flat
    path.write_text(_word_lines(padded.reshape(words, VPU_LANES), VPU_LANES, 8))
    return words


def gen_vpu(rng: np.random.Generator, sm_rows: int, sm_len: int,
            ln_rows: int, ln_len: int, n: int,
            tr_rows: int = 13, tr_cols: int = 19) -> dict:
    """Operand streams and exact results for every op vpu_seq dispatches.

    Deliberately int8 in and int8 out throughout, including softmax: that is
    what a requantized GEMM produces and what the next op in a transformer block
    consumes, so it is the only width the SoC seam ever sees. softmax_int's
    int16 input port stays covered by tb_softmax_int.

    The layernorm case is sized so rows*len is NOT a multiple of 16, which is
    the only way the destination's tail-word byte mask gets exercised.
    """
    import math

    from sw.kernels_int import (apply_unary_lut, build_exp_lut, qadd,
                                qlayernorm, qsoftmax)
    from sw.kernels_float import gelu

    # -- softmax ------------------------------------------------------------
    lut_lo, lut_hi = build_exp_lut(SM_SCALE_IN)
    sx = rng.integers(INT8_MIN, INT8_MAX + 1, size=(sm_rows, sm_len), dtype=np.int64)
    sx[0, :] = 5                                  # flat row: exact 2**15 / len
    if sm_rows > 1:
        sx[1, :] = INT8_MIN
        sx[1, sm_len // 2] = INT8_MAX             # one dominant logit
    sy = qsoftmax(sx.astype(np.int16), (lut_lo, lut_hi), SM_SCALE_OUT)
    sm_rq = Requant.from_real_multiplier(1.0 / (float(1 << 15) * SM_SCALE_OUT))

    _pack_stream(sx, OUT / "vpu_sm_x.hex")
    (OUT / "vpu_sm_y.hex").write_text(
        "\n".join(_hex(int(v), 8) for v in np.asarray(sy).ravel()) + "\n")

    # -- layernorm ----------------------------------------------------------
    lx = rng.integers(INT8_MIN, INT8_MAX + 1, size=(ln_rows, ln_len), dtype=np.int64)
    lx[0, :] = 42                                 # zero variance: the max(r,1) clamp
    lgamma = rng.integers(INT8_MIN, INT8_MAX + 1, size=ln_len, dtype=np.int64).astype(np.int8)
    lbeta = rng.integers(-40, 41, size=ln_len, dtype=np.int64).astype(np.int32)
    ln_rq = Requant.from_real_multiplier(
        LN_SCALE_GAMMA * math.sqrt(ln_len) / (float(1 << 14) * LN_SCALE_OUT))
    ly = qlayernorm(lx.astype(np.int8), lgamma, lbeta, LN_SCALE_IN, ln_rq, eps=LN_EPS)
    ln_eps_q = int(round(LN_EPS / (LN_SCALE_IN * LN_SCALE_IN) * ln_len**3))

    _pack_stream(lx, OUT / "vpu_ln_x.hex")
    (OUT / "vpu_ln_y.hex").write_text(
        "\n".join(_hex(int(v), 8) for v in np.asarray(ly).ravel()) + "\n")
    (OUT / "vpu_ln_gamma.hex").write_text(
        "\n".join(_hex(int(v), 8) for v in lgamma) + "\n")
    (OUT / "vpu_ln_beta.hex").write_text(
        "\n".join(_hex(int(v), 32) for v in lbeta) + "\n")

    # -- residual add -------------------------------------------------------
    # One scale pair for the whole run: vpu_seq applies a single requant config
    # to the stream, unlike tb_qadd_unit which redraws it per vector.
    qa = rng.integers(INT8_MIN, INT8_MAX + 1, size=n, dtype=np.int64).astype(np.int8)
    qb = rng.integers(INT8_MIN, INT8_MAX + 1, size=n, dtype=np.int64).astype(np.int8)
    qa[:4] = np.array([INT8_MIN, INT8_MAX, INT8_MIN, INT8_MAX], dtype=np.int8)
    qb[:4] = np.array([INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN], dtype=np.int8)
    rq_a = Requant.from_real_multiplier(0.75)
    rq_b = Requant.from_real_multiplier(1.25)
    qy = qadd(qa, qb, rq_a, rq_b)

    _pack_stream(qa, OUT / "vpu_qa_a.hex")
    _pack_stream(qb, OUT / "vpu_qa_b.hex")
    (OUT / "vpu_qa_y.hex").write_text(
        "\n".join(_hex(int(v), 8) for v in np.asarray(qy).ravel()) + "\n")

    # -- activation table ---------------------------------------------------
    ulut = build_unary_lut(gelu, scale_in=8.0 / 127.0, scale_out=4.0 / 127.0)
    ux = rng.integers(INT8_MIN, INT8_MAX + 1, size=n, dtype=np.int64).astype(np.int8)
    ux[:5] = np.array([INT8_MIN, -1, 0, 1, INT8_MAX], dtype=np.int8)
    uy = apply_unary_lut(ux, ulut)

    _pack_stream(ux, OUT / "vpu_un_x.hex")
    (OUT / "vpu_un_y.hex").write_text(
        "\n".join(_hex(int(v), 8) for v in np.asarray(uy).ravel()) + "\n")
    (OUT / "vpu_un_lut.hex").write_text(
        "\n".join(_hex(int(v), 8) for v in ulut) + "\n")

    # -- transpose ----------------------------------------------------------
    # The one op with no kernel behind it: only the read address pattern
    # changes, so what is under test is purely the index generator. Neither
    # dimension is a multiple of the 16 lanes, and rows != cols, so a
    # sequencer that swapped the two counters or walked the source in
    # write order cannot pass.
    tx = rng.integers(INT8_MIN, INT8_MAX + 1, size=(tr_rows, tr_cols),
                      dtype=np.int64).astype(np.int8)
    ty = tx.T.copy()

    _pack_stream(tx, OUT / "vpu_tr_x.hex")
    (OUT / "vpu_tr_y.hex").write_text(
        "\n".join(_hex(int(v), 8) for v in np.asarray(ty).ravel()) + "\n")

    # A square case whose side is exactly the lane count, which is what
    # attention actually asks for (K-transpose is head_dim x tokens, and
    # head_dim is 16). It is the degenerate geometry for the index generator:
    # the source lane stops changing within an output row, so a stride bug that
    # the ragged case above catches by landing in the wrong lane instead lands
    # in the right lane of the wrong word.
    t2 = rng.integers(INT8_MIN, INT8_MAX + 1, size=(LANES_TR, LANES_TR),
                      dtype=np.int64).astype(np.int8)
    _pack_stream(t2, OUT / "vpu_tr2_x.hex")
    (OUT / "vpu_tr2_y.hex").write_text(
        "\n".join(_hex(int(v), 8) for v in np.asarray(t2.T.copy()).ravel()) + "\n")

    return {
        "VPU_TR_ROWS": tr_rows, "VPU_TR_COLS": tr_cols,
        "VPU_TR2_N": LANES_TR,
        "VPU_SM_ROWS": sm_rows, "VPU_SM_LEN": sm_len,
        "VPU_SM_MULT": int(sm_rq.multiplier), "VPU_SM_SHIFT": int(sm_rq.shift),
        "VPU_LN_ROWS": ln_rows, "VPU_LN_LEN": ln_len,
        "VPU_LN_MULT": int(ln_rq.multiplier), "VPU_LN_SHIFT": int(ln_rq.shift),
        "VPU_QA_N": n,
        "VPU_QA_MULT_A": int(rq_a.multiplier), "VPU_QA_SHIFT_A": int(rq_a.shift),
        "VPU_QA_MULT_B": int(rq_b.multiplier), "VPU_QA_SHIFT_B": int(rq_b.shift),
        "VPU_UN_N": n,
        "_eps": ln_eps_q,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--sm-rows", type=int, default=8)
    ap.add_argument("--sm-len", type=int, default=64)
    ap.add_argument("--ln-rows", type=int, default=8)
    ap.add_argument("--ln-len", type=int, default=48)
    ap.add_argument("--seq-m", type=int, default=40)
    ap.add_argument("--seq-k", type=int, default=45)
    ap.add_argument("--seq-n", type=int, default=30)
    ap.add_argument("--vpu-sm-rows", type=int, default=3)
    ap.add_argument("--vpu-sm-len", type=int, default=32)
    ap.add_argument("--vpu-ln-rows", type=int, default=3)
    ap.add_argument("--vpu-ln-len", type=int, default=40)
    ap.add_argument("--vpu-n", type=int, default=200)
    ap.add_argument("--vpu-tr-rows", type=int, default=13)
    ap.add_argument("--vpu-tr-cols", type=int, default=19)
    ap.add_argument("--requant-vectors", type=int, default=512)
    ap.add_argument("-m", type=int, default=8, help="activation rows per GEMM pass")
    ap.add_argument("-k", type=int, default=16, help="reduction depth == array ROWS")
    ap.add_argument("-n", type=int, default=16, help="output channels == array COLS")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    n_requant = gen_requant(rng, args.requant_vectors)
    gen_gemm(rng, args.m, args.k, args.n)
    n_gelu = gen_gelu(rng, args.requant_vectors)
    n_div = gen_div(rng, args.requant_vectors)
    sm_rows, sm_len, sm_mult, sm_shift = gen_softmax(rng, args.sm_rows, args.sm_len)
    n_isqrt = gen_isqrt(rng, args.requant_vectors)
    ln_rows, ln_len, ln_eps, ln_mult, ln_shift = gen_layernorm(rng, args.ln_rows, args.ln_len)
    n_qadd = gen_qadd(rng, args.requant_vectors)
    # gemm_seq tiles onto a fixed ROWS x COLS array. That is deliberately NOT
    # the -k/-n sweep, which resizes the bare systolic array: mixing the two
    # would tile the vectors to one geometry and run them on another.
    sq_m, sq_k, sq_n, sq_kt, sq_nt = gen_gemm_seq(
        rng, args.seq_m, args.seq_k, args.seq_n, SEQ_ROWS, SEQ_COLS)
    vpu = gen_vpu(rng, args.vpu_sm_rows, args.vpu_sm_len,
                  args.vpu_ln_rows, args.vpu_ln_len, args.vpu_n,
                  args.vpu_tr_rows, args.vpu_tr_cols)
    vpu_eps = vpu.pop("_eps")

    (OUT / "params.svh").write_text(
        "// generated by test/v2/gen_vectors.py -- do not edit\n"
        f"localparam int N_REQUANT = {n_requant};\n"
        f"localparam int N_GELU = {n_gelu};\n"
        f"localparam int N_DIV = {n_div};\n"
        f"localparam int DIV_NUM_BITS = {DIV_NUM_BITS};\n"
        f"localparam int DIV_DEN_BITS = {DIV_DEN_BITS};\n"
        f"localparam int SM_ROWS = {sm_rows};\n"
        f"localparam int SM_LEN = {sm_len};\n"
        f"localparam int SM_MULT = {sm_mult};\n"
        f"localparam int SM_SHIFT = {sm_shift};\n"
        f"localparam int N_ISQRT = {n_isqrt};\n"
        f"localparam int ISQRT_BITS = {ISQRT_BITS};\n"
        f"localparam int LN_ROWS = {ln_rows};\n"
        f"localparam int LN_LEN = {ln_len};\n"
        f"localparam longint LN_EPS_Q = {ln_eps};\n"
        f"localparam int LN_MULT = {ln_mult};\n"
        f"localparam int LN_SHIFT = {ln_shift};\n"
        f"localparam int N_QADD = {n_qadd};\n"
        f"localparam int SEQ_M = {sq_m};\n"
        f"localparam int SEQ_K = {sq_k};\n"
        f"localparam int SEQ_N = {sq_n};\n"
        f"localparam int SEQ_KT = {sq_kt};\n"
        f"localparam int SEQ_NT = {sq_nt};\n"
        f"localparam int GEMM_M = {args.m};\n"
        f"localparam int GEMM_K = {args.k};\n"
        f"localparam int GEMM_N = {args.n};\n"
        + "".join(f"localparam int {k} = {v};\n" for k, v in vpu.items())
        + f"localparam longint VPU_LN_EPS_Q = {vpu_eps};\n"
    )
    print(f"wrote {n_requant} requant vectors and a {args.m}x{args.k}x{args.n} GEMM to {OUT}")


if __name__ == "__main__":
    main()
