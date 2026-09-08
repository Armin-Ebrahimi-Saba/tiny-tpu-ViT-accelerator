#!/usr/bin/env python3
# ABOUTME: Emits a whole quantized transformer block as a descriptor table the SoC driver walks.
# ABOUTME: Every intermediate is computed by sw/kernels_int.py, so the CV32E40P checks each op bit-exactly.

"""One DINOv2-shaped transformer block, lowered to the ops tiny-tpu actually has.

The block is deliberately small -- 16 tokens, 32 channels, 2 heads -- because
what is under test is the *sequence*, not the arithmetic. Every kernel is
already checked element-by-element by the Verilator testbenches; nothing here
would find a bug in one. What it can find is everything between them: whether
an op can read what the previous op wrote, whether the scales compose, and
whether the register map can express a real graph rather than one op at a time.

Three lowering decisions are forced by the hardware, and all three are the kind
of thing a compiler does rather than a kernel:

1. **q/k/v are six GEMMs, not three.** The array's activation port reads
   `[M][K]` contiguously, so a head's slice of a `[T][E]` projection -- 16 of
   every 32 columns -- is not addressable. Projecting each head separately
   costs exactly the same multiplies and makes every operand contiguous.

2. **The output projection is split over heads and summed with qadd.** The
   concatenation `[ctx_0 | ctx_1] @ Wo` is the same arithmetic as
   `ctx_0 @ Wo[:D] + ctx_1 @ Wo[D:]`, and the second form needs no strided
   scatter. gemm_seq cannot accumulate across two calls, so the residual adder
   does it -- which is what that unit is for.

3. **LayerScale folds into the preceding GEMM.** ls1/ls2 are per-channel
   constants and the projection already has a per-channel multiplier, so the
   two multiply together at compile time and the op disappears.

The one thing that does *not* match the tpu-v2 spec: attention logits are int8
here, not int16. The GEMM's output path requantizes to int8 and the vector
unit's softmax is fed int8, so int8 is what the seam can carry today. It costs
accuracy, not correctness, and the reference below is computed the same way.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from sw.kernels_float import gelu  # noqa: E402
from sw.kernels_int import (apply_unary_lut, build_exp_lut, build_unary_lut,  # noqa: E402
                            qadd, qbatch_matmul, qlayernorm, qlinear, qsoftmax)
from sw.numerics import INT8_MAX, INT8_MIN, Requant  # noqa: E402

LANES = 16
EPS = 1e-6


def qscale(x: np.ndarray) -> float:
    return max(float(np.abs(x).max()), 1e-9) / INT8_MAX


def quant(x: np.ndarray, s: float) -> np.ndarray:
    return np.clip(np.rint(x / s), INT8_MIN, INT8_MAX).astype(np.int8)


class Slots:
    """Word allocator for the result region, which is where the block lives.

    Only the vector unit takes a destination base; the GEMM always lands at word
    zero. So the low end of the region is the landing zone every GEMM writes
    into, and everything above it is the block's register file.
    """

    def __init__(self, base: int, limit: int):
        self.next = base
        self.limit = limit
        self.names: dict[str, int] = {}

    def alloc(self, name: str, elems: int) -> int:
        words = (elems + LANES - 1) // LANES
        if self.next + words > self.limit:
            raise SystemExit(f"result region overflow allocating {name}")
        at = self.next
        self.next += words
        self.names[name] = at
        return at


class Program:
    """The op list, in the order the driver will execute it."""

    def __init__(self):
        self.ops: list[dict] = []
        self.arrays: dict[str, tuple[str, list[int]]] = {}

    def array(self, name: str, kind: str, vals: list[int]) -> str:
        self.arrays[name] = (kind, vals)
        return name


def u32_words(packed: np.ndarray) -> list[int]:
    """Pack int8 lanes little-endian into the 32-bit words the CPU stores."""
    flat = np.asarray(packed).ravel()
    pad = (-len(flat)) % 4
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=flat.dtype)])
    b = flat.astype(np.uint8).reshape(-1, 4)
    return [int(b[i, 0]) | (int(b[i, 1]) << 8)
            | (int(b[i, 2]) << 16) | (int(b[i, 3]) << 24)
            for i in range(b.shape[0])]


def pad_words(packed: np.ndarray) -> list[int]:
    """Same, but padded out to a whole 128-bit buffer word."""
    flat = np.asarray(packed).ravel()
    pad = (-len(flat)) % LANES
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=flat.dtype)])
    return u32_words(flat)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("-t", "--tokens", type=int, default=16)
    ap.add_argument("-e", "--embed", type=int, default=32)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--out-words", type=int, default=512)
    ap.add_argument("--landing", type=int, default=64,
                    help="words reserved at the bottom of the result region")
    ap.add_argument("-o", type=pathlib.Path, required=True)
    args = ap.parse_args()

    T, E, H = args.tokens, args.embed, args.heads
    D, HID = E // H, args.hidden
    for name, v in (("tokens", T), ("embed", E), ("head_dim", D), ("hidden", HID)):
        if v % LANES:
            raise SystemExit(f"{name}={v} must be a multiple of {LANES}")

    rng = np.random.default_rng(args.seed)
    prog = Program()
    slots = Slots(args.landing, args.out_words)

    def randf(*shape, scale=1.0):
        return (rng.standard_normal(shape) * scale).astype(np.float64)

    # ---------------------------------------------------------------- helpers
    def emit_expect(name: str, y_q: np.ndarray) -> tuple[str, int]:
        arr = prog.array(f"exp_{name}", "u32", pad_words(y_q))
        return arr, int(np.asarray(y_q).size)

    def gemm(name, x_q, s_x, w_f, b_f, s_out, a_slot, d_slot, per_chan_scale=None):
        """One `[M][K] @ [K][N]` with per-channel weights, as gemm_seq runs it.

        `per_chan_scale` is LayerScale: a per-output-channel float folded into
        the requantization multiplier, which is exact and costs no op.
        """
        m, k = x_q.shape
        n = w_f.shape[1]
        s_w = np.maximum(np.abs(w_f).max(axis=0), 1e-12) / INT8_MAX
        w_q = quant(w_f, s_w)                       # [K][N], the buffer layout
        b_q = np.rint(b_f / (s_x * s_w)).astype(np.int64)
        real = s_x * s_w / s_out
        if per_chan_scale is not None:
            real = real * per_chan_scale
        rq = Requant.from_real_multiplier(real)
        y_q = qlinear(x_q, w_q.T, b_q.astype(np.int32), rq)
        exp, n_exp = emit_expect(name, y_q)
        prog.ops.append(dict(
            name=name, op=0, a_slot=a_slot, a_words=(m * k) // LANES,
            b_slot=0, b_words=0,
            wgt=prog.array(f"w_{name}", "u32", u32_words(w_q.ravel())),
            wgt_words=(k * n) // LANES,
            d_slot=d_slot, d_words=(m * n) // LANES,
            m=m, kt=k // LANES, nt=n // LANES,
            bias=prog.array(f"b_{name}", "u32", [int(v) & 0xFFFFFFFF for v in b_q]),
            mult=prog.array(f"m_{name}", "u32",
                            [int(v) & 0xFFFFFFFF for v in np.asarray(rq.multiplier).ravel()]),
            shift=prog.array(f"s_{name}", "u32", [int(v) for v in np.asarray(rq.shift).ravel()]),
            expect=exp, expect_n=n_exp))
        return y_q

    def bmm(name, a_q, s_a, b_q, s_b, s_out, a_slot, b_slot, d_slot):
        """`[M][K] @ [K][N]` where both operands are activations: no weights, no bias."""
        m, k = a_q.shape
        n = b_q.shape[1]
        rq = Requant.from_real_multiplier(s_a * s_b / s_out)
        y_q = qbatch_matmul(a_q, b_q, rq)
        exp, n_exp = emit_expect(name, y_q)
        prog.ops.append(dict(
            name=name, op=0, a_slot=a_slot, a_words=(m * k) // LANES,
            b_slot=b_slot, b_words=(k * n) // LANES, wgt=None, wgt_words=0,
            d_slot=d_slot, d_words=(m * n) // LANES,
            m=m, kt=k // LANES, nt=n // LANES,
            bias=prog.array(f"b_{name}", "u32", [0] * n),
            mult=prog.array(f"m_{name}", "u32", [int(rq.multiplier) & 0xFFFFFFFF] * n),
            shift=prog.array(f"s_{name}", "u32", [int(rq.shift)] * n),
            expect=exp, expect_n=n_exp))
        return y_q

    def vec(name, opcode, y_q, rows, length, a_slot, a_words, d_slot, **kw):
        exp, n_exp = emit_expect(name, y_q)
        prog.ops.append(dict(
            name=name, op=opcode, a_slot=a_slot, a_words=a_words,
            b_slot=kw.pop("b_slot", 0), b_words=kw.pop("b_words", 0),
            wgt=None, wgt_words=0, d_slot=d_slot,
            d_words=(rows * length + LANES - 1) // LANES,
            m=0, kt=0, nt=0, rows=rows, len=length,
            expect=exp, expect_n=n_exp, **kw))
        return y_q

    # ------------------------------------------------------------- allocation
    # Slots are reused the way a register allocator reuses them: `H` carries
    # norm1 and then norm2, the per-head q/k/v/logits/ctx slots are rewritten
    # once per head, and the attention sum's slot becomes the MLP's second GEMM
    # output once the residual has consumed it. Without that the block does not
    # fit in the 8 KB result region, which is itself the finding: 16 tokens of a
    # 32-channel block already fills it.
    S_X = slots.alloc("x", T * E)
    S_H = slots.alloc("h", T * E)
    S_Q = slots.alloc("q", T * D)
    S_K = slots.alloc("k", T * D)
    S_V = slots.alloc("v", T * D)
    S_KT = slots.alloc("kt", D * T)
    S_P = slots.alloc("p", T * T)
    S_C = slots.alloc("c", T * D)
    S_A = [slots.alloc(f"a{h}", T * E) for h in range(H)]
    S_ATT = slots.alloc("att", T * E)
    S_X1 = slots.alloc("x1", T * E)
    S_F1 = slots.alloc("f1", T * HID)
    S_G = slots.alloc("g", T * HID)
    S_Y = slots.alloc("y", T * E)
    S_F2 = S_ATT                       # dead once the residual has been formed

    # -------------------------------------------------------------- the block
    x_f = randf(T, E)
    s_x = qscale(x_f)
    x_q = quant(x_f, s_x)

    def layernorm(name, xq, sx, xf, gamma_f, beta_f, a_slot, d_slot):
        h_f = ((xf - xf.mean(-1, keepdims=True))
               / np.sqrt(xf.var(-1, keepdims=True) + EPS)) * gamma_f + beta_f
        s_out = qscale(h_f)
        s_g = qscale(gamma_f)
        g_q = quant(gamma_f, s_g)
        b_q = np.rint(beta_f / s_out).astype(np.int32)
        rq = Requant.from_real_multiplier(
            s_g * math.sqrt(xq.shape[-1]) / (float(1 << 14) * s_out))
        y_q = qlayernorm(xq, g_q, b_q, sx, rq, eps=EPS)
        eps_q = int(round(EPS / (sx * sx) * xq.shape[-1] ** 3))
        vec(name, 2, y_q, xq.shape[0], xq.shape[1], a_slot,
            xq.size // LANES, d_slot,
            vmult=int(rq.multiplier), vshift=int(rq.shift),
            vmult_b=0, vshift_b=0, eps=eps_q,
            ln_gamma=prog.array(f"g_{name}", "u32", [int(v) & 0xFF for v in g_q]),
            ln_beta=prog.array(f"bt_{name}", "u32", [int(v) & 0xFFFFFFFF for v in b_q]),
            ln_len=int(xq.shape[-1]))
        return y_q, s_out, h_f

    # -- norm1 -> q, k, v, one GEMM per head --------------------------------
    h1_q, s_h1, h1_f = layernorm("norm1", x_q, s_x, x_f,
                                 1.0 + randf(E, scale=0.1), randf(E, scale=0.1),
                                 S_X, S_H)

    heads = []
    for h in range(H):
        # DINOv2 folds the 1/sqrt(head_dim) attention scale into the q
        # projection, which is exact and removes an op.
        wq = randf(E, D, scale=0.3) * (D ** -0.5)
        wk, wv = randf(E, D, scale=0.3), randf(E, D, scale=0.3)
        bq, bk, bv = randf(D, scale=0.1), randf(D, scale=0.1), randf(D, scale=0.1)

        q_f, k_f, v_f = h1_f @ wq + bq, h1_f @ wk + bk, h1_f @ wv + bv
        # Only the float projections are computed here. The GEMMs themselves
        # are emitted inside the attention loop below, because q/k/v share one
        # slot triple across heads: emitting all six up front would have head 1
        # overwrite head 0's operands before head 0's attention ever reads
        # them. The one thing this loop must run ahead for is the logit scale,
        # which is shared by both heads and needs both heads' floats.
        heads.append(dict(w=(wq, wk, wv), b=(bq, bk, bv),
                          qf=q_f, kf=k_f, vf=v_f))

    # -- attention ----------------------------------------------------------
    # One logit scale for both heads so the exp table is loaded once. Per-head
    # scales would be more accurate and would cost 512 register writes a head.
    logit_amax = max(float(np.abs(hd["qf"] @ hd["kf"].T).max()) for hd in heads)
    s_logit = max(logit_amax, 1e-9) / INT8_MAX
    s_prob = 1.0 / INT8_MAX

    exp_lo, exp_hi = build_exp_lut(s_logit)
    sm_rq = Requant.from_real_multiplier(1.0 / (float(1 << 15) * s_prob))

    attn_parts = []
    for h, hd in enumerate(heads):
        wq, wk, wv = hd["w"]
        bq, bk, bv = hd["b"]
        s_q, s_k, s_v = qscale(hd["qf"]), qscale(hd["kf"]), qscale(hd["vf"])
        q_q = gemm(f"q{h}", h1_q, s_h1, wq, bq, s_q, S_H, S_Q)
        k_q = gemm(f"k{h}", h1_q, s_h1, wk, bk, s_k, S_H, S_K)
        v_q = gemm(f"v{h}", h1_q, s_h1, wv, bv, s_v, S_H, S_V)

        kt_q = k_q.T.copy()
        vec(f"kt{h}", 5, kt_q, T, D, S_K, (T * D) // LANES, S_KT,
            vmult=0, vshift=0, vmult_b=0, vshift_b=0, eps=0,
            ln_gamma=None, ln_beta=None, ln_len=0)

        l_q = bmm(f"logit{h}", q_q, s_q, kt_q, s_k, s_logit, S_Q, S_KT, S_P)

        p_q = qsoftmax(l_q, (exp_lo, exp_hi), s_prob)
        vec(f"prob{h}", 1, p_q, T, T, S_P, (T * T) // LANES, S_P,
            vmult=int(sm_rq.multiplier), vshift=int(sm_rq.shift),
            vmult_b=0, vshift_b=0, eps=0,
            ln_gamma=None, ln_beta=None, ln_len=0)

        ctx_f = (p_q.astype(np.float64) * s_prob) @ (v_q.astype(np.float64) * s_v)
        s_ctx = qscale(ctx_f)
        c_q = bmm(f"ctx{h}", p_q, s_prob, v_q, s_v, s_ctx, S_P, S_V, S_C)

        # The output projection is split across heads and the parts summed, so
        # neither the concatenation nor a strided write is ever needed.
        wo = randf(D, E, scale=0.3)
        bo = randf(E, scale=0.05) if h == 0 else np.zeros(E)
        ls1 = 1.0 + randf(E, scale=0.05)          # LayerScale, folded below
        part_f = ctx_f @ wo + bo
        s_part = qscale(part_f * ls1)
        a_q = gemm(f"proj{h}", c_q, s_ctx, wo, bo, s_part, S_C, S_A[h],
                   per_chan_scale=ls1)
        attn_parts.append((a_q, s_part, part_f * ls1))

    # -- residual 1 ---------------------------------------------------------
    (a0_q, s_a0, a0_f), (a1_q, s_a1, a1_f) = attn_parts
    attn_f = a0_f + a1_f
    s_attn = qscale(attn_f)
    attn_q = qadd(a0_q, a1_q, Requant.from_real_multiplier(s_a0 / s_attn),
                  Requant.from_real_multiplier(s_a1 / s_attn))
    rq_a0 = Requant.from_real_multiplier(s_a0 / s_attn)
    rq_a1 = Requant.from_real_multiplier(s_a1 / s_attn)
    vec("attn", 3, attn_q, T, E, S_A[0], (T * E) // LANES, S_ATT,
        b_slot=S_A[1], b_words=(T * E) // LANES,
        vmult=int(rq_a0.multiplier), vshift=int(rq_a0.shift),
        vmult_b=int(rq_a1.multiplier), vshift_b=int(rq_a1.shift), eps=0,
        ln_gamma=None, ln_beta=None, ln_len=0)

    x1_f = x_f + attn_f
    s_x1 = qscale(x1_f)
    rq_x = Requant.from_real_multiplier(s_x / s_x1)
    rq_at = Requant.from_real_multiplier(s_attn / s_x1)
    x1_q = qadd(x_q, attn_q, rq_x, rq_at)
    vec("res1", 3, x1_q, T, E, S_X, (T * E) // LANES, S_X1,
        b_slot=S_ATT, b_words=(T * E) // LANES,
        vmult=int(rq_x.multiplier), vshift=int(rq_x.shift),
        vmult_b=int(rq_at.multiplier), vshift_b=int(rq_at.shift), eps=0,
        ln_gamma=None, ln_beta=None, ln_len=0)

    # -- norm2 -> MLP -------------------------------------------------------
    h2_q, s_h2, h2_f = layernorm("norm2", x1_q, s_x1, x1_f,
                                 1.0 + randf(E, scale=0.1), randf(E, scale=0.1),
                                 S_X1, S_H)

    w1, b1 = randf(E, HID, scale=0.3), randf(HID, scale=0.1)
    f1_f = h2_f @ w1 + b1
    s_f1 = qscale(f1_f)
    f1_q = gemm("fc1", h2_q, s_h2, w1, b1, s_f1, S_H, S_F1)

    # The GELU table is built at the fc1 output scale, so the activation is a
    # pure table lookup with no requantization around it.
    s_g1 = qscale(gelu(f1_f))
    un_lut = build_unary_lut(gelu, scale_in=s_f1, scale_out=s_g1)
    g1_q = apply_unary_lut(f1_q, un_lut)
    vec("gelu", 4, g1_q, T, HID, S_F1, (T * HID) // LANES, S_G,
        vmult=0, vshift=0, vmult_b=0, vshift_b=0, eps=0,
        ln_gamma=None, ln_beta=None, ln_len=0)

    w2, b2 = randf(HID, E, scale=0.3), randf(E, scale=0.05)
    ls2 = 1.0 + randf(E, scale=0.05)
    g1_f = apply_unary_lut(f1_q, un_lut).astype(np.float64) * s_g1
    f2_f = (g1_f @ w2 + b2) * ls2
    s_f2 = qscale(f2_f)
    f2_q = gemm("fc2", g1_q, s_g1, w2, b2, s_f2, S_G, S_F2, per_chan_scale=ls2)

    y_f = x1_f + f2_f
    s_y = qscale(y_f)
    rq_x1 = Requant.from_real_multiplier(s_x1 / s_y)
    rq_f2 = Requant.from_real_multiplier(s_f2 / s_y)
    y_q = qadd(x1_q, f2_q, rq_x1, rq_f2)
    vec("res2", 3, y_q, T, E, S_X1, (T * E) // LANES, S_Y,
        b_slot=S_F2, b_words=(T * E) // LANES,
        vmult=int(rq_x1.multiplier), vshift=int(rq_x1.shift),
        vmult_b=int(rq_f2.multiplier), vshift_b=int(rq_f2.shift), eps=0,
        ln_gamma=None, ln_beta=None, ln_len=0)

    # ------------------------------------------------------------ emit header
    prog.array("blk_x", "u32", pad_words(x_q))
    prog.array("blk_exp_lo", "u32", [int(v) & 0xFFFF for v in exp_lo])
    prog.array("blk_exp_hi", "u32", [int(v) & 0xFFFF for v in exp_hi])
    prog.array("blk_un_lut", "u32", [int(v) & 0xFF for v in un_lut])

    def carr(name: str, vals: list[int]) -> str:
        body = ",\n    ".join(", ".join(f"0x{v:08x}" for v in vals[i:i + 6])
                              for i in range(0, len(vals), 6))
        # 16-byte alignment: the DMA reads whole 128-bit words and drops the low
        # four source address bits, so an unaligned array is read from the wrong
        # offset.
        return (f"static const uint32_t __attribute__((aligned(16)))\n"
                f"{name}[{len(vals)}] = {{\n    {body}\n}};\n")

    out = [
        "/* Generated by test/v2/gen_block_vectors.py -- do not edit. */",
        "#pragma once",
        "#include <stdint.h>",
        "",
        f"#define BLK_T {T}",
        f"#define BLK_E {E}",
        f"#define BLK_H {H}",
        f"#define BLK_D {D}",
        f"#define BLK_HID {HID}",
        f"#define BLK_X_SLOT {S_X}",
        f"#define BLK_Y_SLOT {S_Y}",
        f"#define BLK_X_WORDS {(T * E) // LANES}",
        f"#define BLK_N_OPS {len(prog.ops)}",
        f"#define BLK_HIGH_WATER {slots.next}",
        "",
    ]
    for name, (_kind, vals) in prog.arrays.items():
        out.append(carr(name, vals))

    out += [
        "",
        "/* One entry per op, in execution order. `wgt` non-null means operand B",
        " * is a weight array in main memory; otherwise B is a result-region slot",
        " * (or unused). Everything the driver needs to run and then check an op",
        " * is here, so main.c is an interpreter rather than a transcription. */",
        "typedef struct {",
        "    const char *name;",
        "    uint8_t  op;                 /* 0 = GEMM, else the vector opcode */",
        "    uint16_t a_slot, a_words;",
        "    uint16_t b_slot, b_words;",
        "    const uint32_t *wgt;",
        "    uint16_t wgt_words;",
        "    uint16_t d_slot, d_words;",
        "    uint16_t m, kt, nt;          /* GEMM shape */",
        "    uint16_t rows, len;          /* vector shape */",
        "    const uint32_t *bias, *mult, *shift;",
        "    uint32_t vmult, vmult_b;",
        "    uint8_t  vshift, vshift_b;",
        "    uint32_t eps_lo, eps_hi;",
        "    const uint32_t *ln_gamma, *ln_beta;",
        "    uint16_t ln_len;",
        "    const uint32_t *expect;",
        "    uint16_t expect_n;",
        "} blk_op_t;",
        "",
        "static const blk_op_t blk_ops[BLK_N_OPS] = {",
    ]

    def ref(v):
        return v if v else "0"

    for o in prog.ops:
        eps = int(o.get("eps", 0) or 0)
        out.append(
            "    {{ \"{name}\", {op}, {a_slot}, {a_words}, {b_slot}, {b_words}, "
            "{wgt}, {wgt_words}, {d_slot}, {d_words}, {m}, {kt}, {nt}, "
            "{rows}, {len}, {bias}, {mult}, {shift}, {vmult}u, {vmult_b}u, "
            "{vshift}, {vshift_b}, {eps_lo}u, {eps_hi}u, {ln_gamma}, {ln_beta}, "
            "{ln_len}, {expect}, {expect_n} }},".format(
                name=o["name"], op=o["op"],
                a_slot=o["a_slot"], a_words=o["a_words"],
                b_slot=o.get("b_slot", 0), b_words=o.get("b_words", 0),
                wgt=ref(o.get("wgt")), wgt_words=o.get("wgt_words", 0),
                d_slot=o["d_slot"], d_words=o["d_words"],
                m=o.get("m", 0), kt=o.get("kt", 0), nt=o.get("nt", 0),
                rows=o.get("rows", 0), len=o.get("len", 0),
                bias=ref(o.get("bias")), mult=ref(o.get("mult")),
                shift=ref(o.get("shift")),
                vmult=o.get("vmult", 0) & 0xFFFFFFFF,
                vmult_b=o.get("vmult_b", 0) & 0xFFFFFFFF,
                vshift=o.get("vshift", 0), vshift_b=o.get("vshift_b", 0),
                eps_lo=eps & 0xFFFFFFFF, eps_hi=(eps >> 32) & 0xFFFFFFFF,
                ln_gamma=ref(o.get("ln_gamma")), ln_beta=ref(o.get("ln_beta")),
                ln_len=o.get("ln_len", 0),
                expect=o["expect"], expect_n=o["expect_n"]))
    out += ["};", ""]

    args.o.write_text("\n".join(out) + "\n")

    n_gemm = sum(1 for o in prog.ops if o["op"] == 0)
    sat = float(np.mean(np.abs(y_q) == INT8_MAX)) * 100.0
    print(f"wrote {args.o}: {len(prog.ops)} ops ({n_gemm} GEMM, "
          f"{len(prog.ops) - n_gemm} vector), T={T} E={E} H={H} HID={HID}, "
          f"result-region high water {slots.next}/{args.out_words} words, "
          f"output saturation {sat:.1f}%")


if __name__ == "__main__":
    main()
