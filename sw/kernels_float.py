# ABOUTME: Pure-numpy fp32 reference kernels; this is the golden model the integer emulator is measured against.
# ABOUTME: Each kernel is validated against torch's independent implementation in sw/tests/test_kernels_float.py.

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------


def pad_nchw(x: np.ndarray, pad_h: int, pad_w: int, value: float = 0.0) -> np.ndarray:
    if pad_h == 0 and pad_w == 0:
        return x
    return np.pad(
        x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=value
    )


def im2col(x: np.ndarray, kh: int, kw: int, stride: int, pad: int) -> np.ndarray:
    """[N,C,H,W] -> [N, C*kh*kw, OH*OW]. This is the exact layout the array consumes."""
    n, c, h, w = x.shape
    xp = pad_nchw(x, pad, pad)
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (w + 2 * pad - kw) // stride + 1
    # Strided view avoids materializing the patch copy twice.
    s = xp.strides
    view = np.lib.stride_tricks.as_strided(
        xp,
        shape=(n, c, kh, kw, oh, ow),
        strides=(s[0], s[1], s[2], s[3], s[2] * stride, s[3] * stride),
        writeable=False,
    )
    return view.reshape(n, c * kh * kw, oh * ow)


# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------


def conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None,
           stride: int = 1, pad: int = 0) -> np.ndarray:
    """x [N,C,H,W], w [O,C,kh,kw] -> [N,O,OH,OW]."""
    n, _, h, ww = x.shape
    o, c, kh, kw = w.shape
    if x.shape[1] != c:
        raise ValueError(f"conv2d channel mismatch: x has {x.shape[1]}, w expects {c}")
    cols = im2col(x, kh, kw, stride, pad)                       # [N, C*kh*kw, L]
    mat = w.reshape(o, c * kh * kw).astype(np.float32)          # [O, C*kh*kw]
    out = np.matmul(mat, cols.astype(np.float32))               # BLAS, broadcast over N
    if b is not None:
        out += b.reshape(1, o, 1).astype(np.float32)
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (ww + 2 * pad - kw) // stride + 1
    return out.reshape(n, o, oh, ow)


def conv_transpose2d(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None,
                     stride: int = 1, pad: int = 0) -> np.ndarray:
    """x [N,I,H,W], w [I,O,kh,kw] -> [N,O,OH,OW]. Matches torch's ConvTranspose2d."""
    n, i, h, ww = x.shape
    wi, o, kh, kw = w.shape
    if wi != i:
        raise ValueError(f"conv_transpose2d channel mismatch: x has {i}, w expects {wi}")
    oh = (h - 1) * stride - 2 * pad + kh
    ow = (ww - 1) * stride - 2 * pad + kw
    # Scatter into a padded canvas, then crop, so `pad` needs no index masking.
    canvas = np.zeros((n, o, oh + 2 * pad, ow + 2 * pad), dtype=np.float32)
    rows = np.arange(h) * stride
    cols = np.arange(ww) * stride
    xf = x.astype(np.float32)
    wf = w.astype(np.float32)
    for kr in range(kh):
        for kc in range(kw):
            contrib = np.einsum("nihw,io->nohw", xf, wf[:, :, kr, kc])
            canvas[:, :, rows[:, None] + kr, cols[None, :] + kc] += contrib
    out = canvas[:, :, pad:pad + oh, pad:pad + ow]
    if b is not None:
        out = out + b.reshape(1, o, 1, 1).astype(np.float32)
    return out


def linear(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """x [..., in], w [out, in] -> [..., out]. Same convention as torch.nn.Linear."""
    out = np.matmul(x.astype(np.float32), w.astype(np.float32).T)
    if b is not None:
        out = out + b.astype(np.float32)
    return out


def batch_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.matmul(a.astype(np.float32), b.astype(np.float32))


# ---------------------------------------------------------------------------
# Normalization and activations
# ---------------------------------------------------------------------------


def layernorm(x: np.ndarray, gamma: np.ndarray | None, beta: np.ndarray | None,
              eps: float = 1e-6) -> np.ndarray:
    xf = x.astype(np.float32)
    mean = xf.mean(axis=-1, keepdims=True)
    centered = xf - mean
    # Biased variance, matching torch.nn.LayerNorm.
    var = np.mean(centered * centered, axis=-1, keepdims=True)
    out = centered / np.sqrt(var + eps)
    if gamma is not None:
        out = out * gamma.astype(np.float32)
    if beta is not None:
        out = out + beta.astype(np.float32)
    return out


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    xf = x.astype(np.float32)
    xf = xf - xf.max(axis=axis, keepdims=True)
    e = np.exp(xf)
    return e / e.sum(axis=axis, keepdims=True)


def _erf(x: np.ndarray) -> np.ndarray:
    """Abramowitz & Stegun 7.1.26; |error| < 1.5e-7, below fp32 resolution."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = np.sign(x)
    ax = np.abs(x).astype(np.float64)
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-ax * ax)
    return (sign * y).astype(np.float32)


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact (erf) GELU, matching torch.nn.GELU()'s default."""
    xf = x.astype(np.float32)
    return 0.5 * xf * (1.0 + _erf(xf / np.float32(np.sqrt(2.0))))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def _sample_coords(out_size: int, in_size: int, align_corners: bool) -> np.ndarray:
    if align_corners:
        if out_size == 1:
            return np.zeros(1, dtype=np.float64)
        scale = (in_size - 1) / (out_size - 1)
        return np.arange(out_size, dtype=np.float64) * scale
    scale = in_size / out_size
    return np.maximum((np.arange(out_size, dtype=np.float64) + 0.5) * scale - 0.5, 0.0)


def interpolate_bilinear(x: np.ndarray, size: tuple[int, int],
                         align_corners: bool = True) -> np.ndarray:
    """[N,C,H,W] -> [N,C,size[0],size[1]], matching F.interpolate(mode='bilinear')."""
    n, c, h, w = x.shape
    oh, ow = size
    xf = x.astype(np.float32)
    sy = _sample_coords(oh, h, align_corners)
    sx = _sample_coords(ow, w, align_corners)
    y0 = np.floor(sy).astype(np.int64)
    x0 = np.floor(sx).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    ty = (sy - y0).astype(np.float32)
    tx = (sx - x0).astype(np.float32)
    top = xf[:, :, y0, :]
    bot = xf[:, :, y1, :]
    rows = top * (1 - ty)[None, None, :, None] + bot * ty[None, None, :, None]
    left = rows[:, :, :, x0]
    right = rows[:, :, :, x1]
    return left * (1 - tx)[None, None, None, :] + right * tx[None, None, None, :]


def _cubic_weights(t: np.ndarray, a: float = -0.75) -> np.ndarray:
    """Torch's cubic convolution coefficients, returned as [4, len(t)]."""
    t = t.astype(np.float64)
    w0 = ((a * (t + 1) - 5 * a) * (t + 1) + 8 * a) * (t + 1) - 4 * a
    w1 = ((a + 2) * t - (a + 3)) * t * t + 1
    s = 1.0 - t
    w2 = ((a + 2) * s - (a + 3)) * s * s + 1
    w3 = ((a * (s + 1) - 5 * a) * (s + 1) + 8 * a) * (s + 1) - 4 * a
    return np.stack([w0, w1, w2, w3])


def interpolate_bicubic(x: np.ndarray, size: tuple[int, int],
                        align_corners: bool = False) -> np.ndarray:
    """[N,C,H,W] -> [N,C,*size], matching F.interpolate(mode='bicubic'). Used only for
    positional-embedding resampling, which happens once at compile time."""
    n, c, h, w = x.shape
    oh, ow = size
    xf = x.astype(np.float64)

    def coords(out_n: int, in_n: int) -> np.ndarray:
        # Unlike bilinear, torch does not clamp bicubic source coords at zero;
        # out-of-range taps are handled by clamping the *index* below instead.
        if align_corners:
            if out_n == 1:
                return np.zeros(1, dtype=np.float64)
            return np.arange(out_n, dtype=np.float64) * ((in_n - 1) / (out_n - 1))
        return (np.arange(out_n, dtype=np.float64) + 0.5) * (in_n / out_n) - 0.5

    sy, sx = coords(oh, h), coords(ow, w)
    y0 = np.floor(sy).astype(np.int64)
    x0 = np.floor(sx).astype(np.int64)
    wy = _cubic_weights(sy - y0)                                      # [4, oh]
    wx = _cubic_weights(sx - x0)                                      # [4, ow]
    ys = np.clip(y0[None, :] + np.arange(-1, 3)[:, None], 0, h - 1)   # [4, oh]
    xs = np.clip(x0[None, :] + np.arange(-1, 3)[:, None], 0, w - 1)   # [4, ow]
    rows = np.einsum("nctyw,ty->ncyw", xf[:, :, ys, :], wy)           # reduce y taps
    out = np.einsum("ncytx,tx->ncyx", rows[:, :, :, xs], wx)          # reduce x taps
    return out.astype(np.float32)
