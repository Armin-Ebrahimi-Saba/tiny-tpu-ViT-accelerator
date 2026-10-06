# ABOUTME: Lowers Depth Anything V2's DPT head onto tiny-tpu's six ops, continuing the encoder's blob.
# ABOUTME: Convs gather their im2col in the driver; upsampling is a GEMM per row plus a qadd per row.

"""The DPT depth head, on the accelerator.

Tensors are [H][W][C] int8 in the arena -- pixel-major, channels contiguous --
which is what the encoder's tokens already are once the cls row is dropped,
and what a GEMM's [M][K] activation operand wants when M is pixels.

Every op is built from what the hardware has:

    conv k x k      a GEMM whose activation tiles the driver gathers as im2col
                    rows straight into the buffer (descriptor g_* fields); K is
                    k*k*C in (ky, kx, c) order. Deeper than the weight buffer
                    (the two 384-channel 3x3s) it is split by taps and summed
                    with a qadd.
    conv-transpose  kernel == stride, so outputs never overlap: one GEMM per
                    output sub-row dy, [HW][Ci] x [Ci][(dx, co)], and the pixel
                    shuffle is a two-level strided result copy.
    bilinear        separable. Width: per input row, a GEMM of the constant
                    [Wo][W] interpolation matrix (int8, each row summing to 127)
                    against the row's [W][C]. Height: per output row, a qadd of
                    the two source rows with multipliers 1-t and t.
    relu            the unary LUT.
    add             qadd.

Scales go through export_tpu.act_scale like the encoder's, so the head is
calibrated with it. The float path alongside follows the reference graph
(checked against run_float in the tests), and is only used for scales.
"""

from __future__ import annotations

import numpy as np

from . import export_tpu as ex
from .export_tpu import (BUFFERS, CFG_UN_LUT, INT8_MAX, LANES, OP_QADD, OP_UNARY, Blob, Ref,
                         Requant, act_scale, emit_config, emit_flat, emit_gemm, emit_qadd,
                         linear_q, quant, ref_add)
from .kernels_float import _sample_coords
from .kernels_int import apply_unary_lut, build_unary_lut, qadd, qlinear

WGT_ROWS = BUFFERS.wgt_words            # deepest K one call can take, at 16 output channels


class T:
    """A tensor of the head: int8 [H][W][C], where it lives, its scale, and
    the float path's value (for scales only)."""

    def __init__(self, q: np.ndarray, ref: Ref, s: float, f: np.ndarray):
        assert q.ndim == 3 and q.shape == f.shape
        self.q, self.ref, self.s, self.f = q, ref, s, f

    @property
    def shape(self):
        return self.q.shape


def im2col(x: np.ndarray, k: int, stride: int, pad: int, t0: int = 0, t1: int | None = None):
    """[H][W][C] -> [Ho*Wo][(t1-t0)*C], taps in (ky, kx) row-major order, zero padded.
    Exactly the rows the driver's gather builds."""
    H, W, C = x.shape
    t1 = k * k if t1 is None else t1
    Ho, Wo = (H + 2 * pad - k) // stride + 1, (W + 2 * pad - k) // stride + 1
    xp = np.zeros((H + 2 * pad, W + 2 * pad, C), dtype=x.dtype)
    xp[pad:pad + H, pad:pad + W] = x
    cols = []
    for t in range(t0, t1):
        ky, kx = divmod(t, k)
        cols.append(xp[ky:ky + stride * Ho:stride, kx:kx + stride * Wo:stride].reshape(Ho * Wo, C))
    return np.concatenate(cols, axis=1), Ho, Wo


def conv_weight(w: np.ndarray, t0: int = 0, t1: int | None = None) -> np.ndarray:
    """torch [Co][Ci][k][k] -> [(ky, kx, ci)][Co] over taps t0..t1."""
    co, ci, k, _ = w.shape
    t1 = k * k if t1 is None else t1
    m = w.astype(np.float64).transpose(2, 3, 1, 0).reshape(k * k, ci, co)
    return m[t0:t1].reshape(-1, co)


class Head:
    def __init__(self, blob: Blob, sd: dict[str, np.ndarray], check: bool):
        self.blob, self.sd, self.check = blob, sd, check
        self.lut = None                  # the unary table last loaded, by name

    def w(self, key):
        return self.sd["depth_head." + key].astype(np.float64)

    # ------------------------------------------------------------- ops
    def conv(self, key: str, x: T, k: int, stride: int = 1, pad: int = 0,
             bias: bool = True, prefix: str | None = None) -> T:
        prefix = prefix or key
        w = self.w(prefix + ".weight")
        b = self.w(prefix + ".bias") if bias else np.zeros(w.shape[0])
        H, W, C = x.shape
        Co = w.shape[0]
        n_pad = (Co + LANES - 1) // LANES * LANES       # the 1-channel output conv
        if n_pad != Co:
            w = np.concatenate([w, np.zeros((n_pad - Co,) + w.shape[1:])])
            b = np.concatenate([b, np.zeros(n_pad - Co)])
        a_f, Ho, Wo = im2col(x.f, k, stride, pad)
        y_full = (a_f @ conv_weight(w) + b).reshape(Ho, Wo, n_pad)
        s_y = act_scale(key, y_full[..., :Co])
        out = self.blob.scratch(Ho * Wo * n_pad)

        # Taps per call: as many as the weight buffer takes at 16 output columns.
        per = max(1, min(k * k, WGT_ROWS // C))
        groups = [(t, min(t + per, k * k)) for t in range(0, k * k, per)]
        parts = []
        for gi, (t0, t1) in enumerate(groups):
            a_q, _, _ = im2col(x.q, k, stride, pad, t0, t1)
            a_fp, _, _ = im2col(x.f, k, stride, pad, t0, t1)
            bb = b if gi == 0 else np.zeros(n_pad)
            last = len(groups) == 1
            kk = f"{key}" if last else f"{key}.part{gi}"
            w_q, b_q, rq, y_f, s_p = linear_q(kk, a_fp, x.s, conv_weight(w, t0, t1), bb,
                                              s_y=s_y if last else None)
            ref = out if last else self.blob.scratch(Ho * Wo * n_pad)
            y_q = emit_gemm(self.blob, kk, a_q, None, w_q, b_q, rq, out_ref=ref,
                            check=self.check,
                            gather=dict(src=x.ref, H=H, W=W, C=C, Wo=Wo, k=k, stride=stride,
                                        pad=pad, t0=t0, t1=t1))
            parts.append(T(y_q.reshape(Ho, Wo, n_pad), ref, s_p, y_f.reshape(Ho, Wo, n_pad)))
        if len(parts) == 1:
            p = parts[0]
            return T(p.q, out, s_y, y_full)
        acc = parts[0]
        for gi, p in enumerate(parts[1:], 1):
            last = gi == len(parts) - 1
            s_o = s_y if last else act_scale(f"{key}.sum{gi}", acc.f + p.f)
            ref = out if last else self.blob.scratch(Ho * Wo * n_pad)
            q = emit_qadd(self.blob, f"{key}.sum{gi}", acc.q, acc.ref, acc.s, p.q, p.ref, p.s,
                          s_o, ref, check=self.check)
            acc = T(q, ref, s_o, acc.f + p.f)
        return T(acc.q, out, s_y, y_full)

    def conv_t(self, key: str, x: T, k: int) -> T:
        """ConvTranspose2d with kernel == stride, pad 0: one GEMM per dy."""
        w = self.w(key + ".weight")                      # torch [Ci][Co][k][k]
        b = self.w(key + ".bias")
        H, W, Ci = x.shape
        Co = w.shape[1]
        Ho, Wo = H * k, W * k
        y_full = np.zeros((Ho, Wo, Co))
        mats = []
        for dy in range(k):
            m = w[:, :, dy, :].transpose(0, 2, 1).reshape(Ci, k * Co)    # [Ci][(dx, co)]
            mats.append(m)
            y = x.f.reshape(H * W, Ci) @ m + np.tile(b, k)
            y_full[dy::k] = y.reshape(H, W, k, Co).reshape(H, W * k, Co)
        s_y = act_scale(key, y_full)
        out = self.blob.scratch(Ho * Wo * Co)
        y_q = np.zeros((Ho, Wo, Co), dtype=np.int8)
        for dy, m in enumerate(mats):
            w_q, b_q, rq, _, _ = linear_q(key, x.f.reshape(H * W, Ci), x.s, m, np.tile(b, k),
                                          s_y=s_y)
            q = emit_gemm(self.blob, f"{key}.dy{dy}", x.q.reshape(H * W, Ci), x.ref, w_q, b_q,
                          rq, out_ref=ref_add(out, dy * Wo * Co), out_stride=k * Co,
                          out_grp=W, out_grp_stride=k * Wo * Co, m_align=W, check=self.check)
            y_q[dy::k] = q.reshape(H, W, k, Co).reshape(H, Wo, Co)
        return T(y_q, out, s_y, y_full)

    def relu(self, key: str, x: T) -> T:
        lut = build_unary_lut(lambda v: np.maximum(v, 0), scale_in=x.s, scale_out=x.s)
        if self.lut != "relu":
            emit_config(self.blob, key + ".lut", CFG_UN_LUT, lut.astype(np.int64) & 0xFF)
            self.lut = "relu"
        y_q = apply_unary_lut(x.q, lut)
        ref = self.blob.scratch(x.q.size)
        emit_flat(self.blob, key, OP_UNARY, x.q, x.ref, y_q, ref, check=self.check)
        return T(y_q, ref, x.s, np.maximum(x.f, 0))

    def add(self, key: str, a: T, b: T) -> T:
        f = a.f + b.f
        s = act_scale(key, f)
        ref = self.blob.scratch(a.q.size)
        q = emit_qadd(self.blob, key, a.q, a.ref, a.s, b.q, b.ref, b.s, s, ref, check=self.check)
        return T(q, ref, s, f)

    def interp(self, key: str, x: T, Ho: int, Wo: int) -> T:
        """Bilinear, align_corners=True, separable. Keeps the input's scale:
        a convex combination cannot leave the input's range."""
        H, W, C = x.shape
        # width pass: per input row, [Wo][Wk] x [W][C]; K padded to a word
        # with zero columns, so whatever the weight buffer holds past row W
        # is multiplied by zero.
        sx = _sample_coords(Wo, W, True)
        x0 = np.floor(sx).astype(int)
        x1 = np.minimum(x0 + 1, W - 1)
        tx = sx - x0
        Wk = (W + LANES - 1) // LANES * LANES
        mx = np.zeros((Wo, Wk))
        np.add.at(mx, (np.arange(Wo), x0), 1 - tx)
        np.add.at(mx, (np.arange(Wo), x1), tx)
        mq = np.rint(mx * INT8_MAX).astype(np.int64)
        # each row sums to exactly 127, so a constant passes through unchanged
        fix = INT8_MAX - mq.sum(1)
        mq[np.arange(Wo), np.argmax(mq, 1)] += fix
        mq = mq.astype(np.int8)
        rq = Requant.from_real_multiplier(1.0 / INT8_MAX)
        mid = self.blob.scratch(H * Wo * C)
        mid_q = np.zeros((H, Wo, C), dtype=np.int8)
        a_h = self.blob.data(mq)
        for y in range(H):
            row = np.zeros((Wk, C), dtype=np.int8)
            row[:W] = x.q[y]
            # The row is the GEMM's B operand ([K][N] = [W][C] row-major,
            # which is the tensor's own layout). The DMA loads Wk rows, so
            # the rows past W are whatever follows in DRAM -- multiplied by
            # the interpolation matrix's zero columns.
            q = emit_gemm(self.blob, f"{key}.w{y}", mq, a_h, row, np.zeros(C, dtype=np.int64),
                          rq, w_ref=ref_add(x.ref, y * W * C), out_ref=ref_add(mid, y * Wo * C),
                          check=self.check)
            mid_q[y] = q
        # height pass: one qadd per output row
        sy = _sample_coords(Ho, H, True)
        y0 = np.floor(sy).astype(int)
        y1 = np.minimum(y0 + 1, H - 1)
        ty = sy - y0
        out = self.blob.scratch(Ho * Wo * C)
        y_q = np.zeros((Ho, Wo, C), dtype=np.int8)
        rowb = Wo * C
        for oy in range(Ho):
            t = float(ty[oy])
            a, b = int(y0[oy]), int(y1[oy])
            # A sample a rounding error off a source row is that row: a
            # 1e-16 weight is not a multiplier the requantizer can hold.
            if t > 1.0 - 1e-6:
                t, a = 0.0, b
            if t < 1e-6 or a == b:
                wa, wb, b = 0.5, 0.5, a          # exact copy: half plus half
            else:
                wa, wb = 1.0 - t, t
            rq_a = Requant.from_real_multiplier(wa)
            rq_b = Requant.from_real_multiplier(wb)
            q = qadd(mid_q[a], mid_q[b], rq_a, rq_b)
            emit_flat(self.blob, f"{key}.h{oy}", OP_QADD, mid_q[a], ref_add(mid, a * rowb), q,
                      ref_add(out, oy * rowb), b_ref=ref_add(mid, b * rowb), check=self.check,
                      vmult=int(rq_a.multiplier), vmult_b=int(rq_b.multiplier),
                      vshift=int(rq_a.shift) | (int(rq_b.shift) << ex.VEC_SHIFT_B_LSB))
            y_q[oy] = q
        from .kernels_float import interpolate_bilinear
        f = interpolate_bilinear(x.f.transpose(2, 0, 1)[None], (Ho, Wo), True)[0].transpose(1, 2, 0)
        return T(y_q, out, x.s, f.astype(np.float64))

    # ----------------------------------------------------------- blocks
    def rcu(self, key: str, prefix: str, x: T) -> T:
        h = self.relu(key + ".relu1", x)
        h = self.conv(key + ".conv1", h, 3, pad=1, prefix=prefix + ".conv1")
        h = self.relu(key + ".relu2", h)
        h = self.conv(key + ".conv2", h, 3, pad=1, prefix=prefix + ".conv2")
        return self.add(key + ".out", h, x)

    def fusion(self, key: str, i: int, deep: T, skip: T | None, size) -> T:
        prefix = f"scratch.refinenet{i}"
        out = deep
        if skip is not None:
            res = self.rcu(key + ".rcu1", prefix + ".resConfUnit1", skip)
            out = self.add(key + ".fuse", out, res)
        out = self.rcu(key + ".rcu2", prefix + ".resConfUnit2", out)
        out = self.interp(key + ".up", out, *size)
        return self.conv(key + ".out_conv", out, 1, prefix=prefix + ".out_conv")


def build_head(blob: Blob, sd: dict[str, np.ndarray], taps: list[dict], grid: int,
               out_size: int, check: bool = True) -> dict:
    """`taps` are the encoder's four normalized token tensors ({q, ref, s, f}
    over [T][E], cls row first). Returns the sidecar entry for the depth map."""
    hd = Head(blob, sd, check)
    feats = []
    for j, t in enumerate(taps):
        E = t["q"].shape[1]
        q = t["q"][1:].reshape(grid, grid, E)
        f = t["f"][1:].reshape(grid, grid, E)
        feats.append(T(q, ref_add(t["ref"], E), t["s"], f))   # drop cls: one row in

    rs = []
    for i, x in enumerate(feats):
        h = hd.conv(f"head.proj{i}", x, 1, prefix=f"projects.{i}")
        if i == 0:
            h = hd.conv_t("resize_layers.0", h, 4)
        elif i == 1:
            h = hd.conv_t("resize_layers.1", h, 2)
        elif i == 3:
            h = hd.conv("head.down3", h, 3, stride=2, pad=1, prefix="resize_layers.3")
        rs.append(hd.conv(f"head.rn{i + 1}", h, 3, pad=1, bias=False,
                          prefix=f"scratch.layer{i + 1}_rn"))

    sizes = [r.shape[:2] for r in rs]
    out = hd.fusion("head.refine4", 4, rs[3], None, sizes[2])
    out = hd.fusion("head.refine3", 3, out, rs[2], sizes[1])
    out = hd.fusion("head.refine2", 2, out, rs[1], sizes[0])
    out = hd.fusion("head.refine1", 1, out, rs[0], (sizes[0][0] * 2, sizes[0][1] * 2))

    out = hd.conv("head.conv1", out, 3, pad=1, prefix="scratch.output_conv1")
    out = hd.interp("head.up", out, out_size, out_size)
    out = hd.conv("head.conv2", out, 3, pad=1, prefix="scratch.output_conv2.0")
    out = hd.relu("head.relu", out)
    out = hd.conv("head.conv3", out, 1, prefix="scratch.output_conv2.2")    # 1 -> 16 channels
    out = hd.relu("head.depth", out)
    return {"name": "depth", "ref": out.ref, "shape": list(out.shape), "scale": float(out.s),
            "emu": out.q, "f": out.f}
