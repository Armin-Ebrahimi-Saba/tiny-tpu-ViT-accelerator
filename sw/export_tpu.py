# ABOUTME: Writes a tiny-tpu program blob: descriptors the on-board driver interprets, plus every byte they name.
# ABOUTME: All decisions -- tiling, layout, scales, requant constants -- are made here so the driver makes none.

"""tiny-tpu blob exporter.

The split is the one `test/v2/gen_block_vectors.py` and `run_block()` already
use at T=16, generalised: the exporter decides everything and the driver on the
CV32E40P is an interpreter. Every loop is unrolled here into one descriptor per
hardware op, so a tiling bug is a Python bug, and every operand is laid out
here exactly as the buffer wants it, so the driver never transposes and never
sees a float.

Blob layout (little-endian, loaded at TPU_BLOB_ADDR in DDR3, all offsets
relative to the blob's first byte):

    header      64 B    see Header
    descriptors n x 96 B, see Desc; the driver walks them in order
    data        16-byte aligned; weights, config tables, expected results
    arena       not in the file: activations the ops leave for each other,
                allocated by the exporter directly above `total`; the header
                gives its size and the driver zero-fills it, since padding
                rows are read by ops that never wrote them

A descriptor names DRAM addresses, not offsets, because the driver would only
add the base anyway and a wrong base is easier to see in a hex dump this way.

Descriptor fields (each a u32):

    op          0..5 the hardware op (TPU_OP_*); 0x10 config-only; 0x11 end
    a_dram      DMA source for the activation buffer, 0 = nothing to load
    a_words     128-bit words to load
    b_dram      DMA source for the weight buffer, 0 = nothing to load
    b_words
    src_a       value written to the src_a register: word | region << 16
    src_b       likewise for src_b
    dst         value written to dst: a result-region word
    shape       GEMM: m | k_tiles << 12 | n_tiles << 20; vector: len | rows << 16
    cfg_dram    u32 words to copy into the config region, 0 = none
    cfg_dst     config-region word index they go to
    cfg_n       how many
    vmult, vmult_b, vshift, eps_lo, eps_hi     the vector unit's registers
    out_dram    where the result is copied after the op, 0 = leave it
    out_cols    bytes per result row (a multiple of 4)
    out_rows    result rows; rows are packed in the result region
    out_stride  bytes between rows in DRAM, so a tile lands inside a tensor
    check_dram  expected int8 elements of the result, 0 = do not check
    check_n     how many elements
    name_off    offset of a NUL-terminated name, for the driver's messages
    g_src       GEMM only: gather operand A as an im2col instead of DMA'ing
                a_dram. g_src is an [H][W][C] int8 tensor in DRAM
    g_hw        H | W << 16
    g_cw        C (bytes per pixel, a multiple of 16) | Wo << 16
    g_k         k | stride << 8 | pad << 16 | t0 << 20 | t1 << 26: kernel
                taps t0..t1-1 of the k*k, row-major, each C bytes in a row
    g_p0        first output pixel of this m tile (raster order over Wo)
    out_grp     result rows come in groups of this many (0 = one group);
    out_grp_stride  bytes between groups' first rows in DRAM -- a strided
                copy one level deeper, for a transposed conv's pixel shuffle
    pad

Config-region layout is the driver's (main.c): per-channel GEMM constants at
word 4*c + {0 bias, 1 mult, 2 shift}; layernorm gamma/beta pairs at
TPU_LN_PAR; the exp table halves and the unary LUT above that. The exporter
lays tables out in exactly that shape so a config load is a plain copy.
"""

from __future__ import annotations

import argparse
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .kernels_float import gelu
from .kernels_int import (apply_unary_lut, build_exp_lut, build_unary_lut, qadd, qlayernorm,
                          qlinear, qsoftmax)
from .numerics import INT8_MAX, INT8_MIN, Requant
from .tiling import LANES, Buffers, tile_gemm

MAGIC = b"TPU1"
VERSION = 3

TPU_BLOB_ADDR = 0x8000_0000

# Fast-aperture regions, as tinytpu.sv decodes them.
R_ACT, R_WGT, R_OUT, R_CFG = 0, 1, 2, 3

# Hardware ops, as tinytpu.hjson's op.code has them, plus the driver's own.
OP_GEMM, OP_SOFTMAX, OP_LAYERNORM, OP_QADD, OP_UNARY, OP_TRANSPOSE = 0, 1, 2, 3, 4, 5
OP_CONFIG, OP_END = 0x10, 0x11

# Config-region word indices, matching main.c.
CFG_TAB = 12288
CFG_LN_PAR = CFG_TAB
CFG_EXP_LO = CFG_TAB + 2048
CFG_EXP_HI = CFG_TAB + 2048 + 256
CFG_UN_LUT = CFG_TAB + 2048 + 512

# vec_shift packs the residual add's second shift above the first (tinytpu.hjson).
VEC_SHIFT_B_LSB = 8

# The RTL's buffers (tinytpu.sv). sw/tiling.py sized these.
BUFFERS = Buffers(act_words=512, wgt_words=3072, out_words=512)

DESC_WORDS = 32
DESC_FIELDS = ("op", "a_dram", "a_words", "b_dram", "b_words", "src_a", "src_b",
               "dst", "shape", "cfg_dram", "cfg_dst", "cfg_n", "vmult", "vmult_b",
               "vshift", "eps_lo", "eps_hi", "out_dram", "out_cols", "out_rows",
               "out_stride", "check_dram", "check_n", "name_off",
               "g_src", "g_hw", "g_cw", "g_k", "g_p0", "out_grp", "out_grp_stride", "pad")
HEADER_BYTES = 64

# A data handle is a chunk index; an arena handle has this bit set.
ARENA = 1 << 30

# The largest flat vector op: the whole activation buffer.
VEC_MAX_ELEMS = BUFFERS.act_words * LANES

Ref = int | tuple[int, int]


def ref_add(ref: Ref, delta: int) -> Ref:
    h, off = ref if isinstance(ref, tuple) else (ref, 0)
    return (h, off + delta)


def src(region: int, word: int) -> int:
    return (word & 0xFFFF) | (region << 16)


def gemm_shape(m: int, k_tiles: int, n_tiles: int) -> int:
    return m | (k_tiles << 12) | (n_tiles << 20)


def vec_shape(length: int, rows: int) -> int:
    return length | (rows << 16)


def quant(x: np.ndarray, s: float | np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x / s), INT8_MIN, INT8_MAX).astype(np.int8)


def qscale(x: np.ndarray) -> float:
    return max(float(np.abs(x).max()), 1e-9) / INT8_MAX


class Calibration:
    """Where activation scales come from.

    By default each scale is taken from the tensor being quantized -- the
    image's own fp32 forward, which is exact for that image and is what the
    increments up to 6.19 did. That makes the program image-specific and puts
    the fp32 model on the host for every picture.

    `record()` runs over a calibration set and keeps, per named activation,
    the largest |x| (or its `percentile`); `fixed` then returns those scales
    whatever the input. With fixed scales the whole program is independent of
    the image and only the input tensor changes between pictures.
    """

    def __init__(self, fixed: dict[str, float] | None = None, percentile: float = 100.0):
        self.fixed = fixed
        self.percentile = percentile
        self.amax: dict[str, float] = {}

    def __call__(self, key: str, x: np.ndarray) -> float:
        a = np.abs(np.asarray(x, dtype=np.float64))
        amax = float(a.max() if self.percentile >= 100 else np.percentile(a, self.percentile))
        self.amax[key] = max(self.amax.get(key, 0.0), amax)
        if self.fixed is not None:
            return self.fixed[key]
        return max(amax, 1e-9) / INT8_MAX

    def scales(self) -> dict[str, float]:
        return {k: max(v, 1e-9) / INT8_MAX for k, v in self.amax.items()}


CALIB = Calibration()


def act_scale(key: str, x: np.ndarray) -> float:
    """The scale for activation `key`, per the current Calibration."""
    return CALIB(key, x)


@dataclass
class Desc:
    op: int
    name: str = ""
    a_dram: int = 0
    a_words: int = 0
    b_dram: int = 0
    b_words: int = 0
    src_a: int = 0
    src_b: int = src(R_WGT, 0)
    dst: int = 0
    shape: int = 0
    cfg_dram: int = 0
    cfg_dst: int = 0
    cfg_n: int = 0
    vmult: int = 0
    vmult_b: int = 0
    vshift: int = 0
    eps_lo: int = 0
    eps_hi: int = 0
    out_dram: int = 0
    out_cols: int = 0
    out_rows: int = 0
    out_stride: int = 0
    check_dram: int = 0
    check_n: int = 0
    name_off: int = 0
    g_src: int = 0
    g_hw: int = 0
    g_cw: int = 0
    g_k: int = 0
    g_p0: int = 0
    out_grp: int = 0
    out_grp_stride: int = 0
    pad: int = 0

    def pack(self) -> bytes:
        vals = [getattr(self, f) & 0xFFFFFFFF for f in DESC_FIELDS]
        return struct.pack(f"<{DESC_WORDS}I", *vals)


@dataclass
class Blob:
    """Descriptors plus a data area, resolved to DRAM addresses at write time.

    Data is appended while descriptors are being built, but its final address
    depends on how many descriptors there are; so `data()` hands back an
    opaque handle and `write()` patches every descriptor field that holds one.
    """

    base: int = TPU_BLOB_ADDR
    descs: list[Desc] = field(default_factory=list)
    _chunks: list[bytes] = field(default_factory=list)
    _chunk_off: list[int] = field(default_factory=list)
    _data_len: int = 0
    _names: bytearray = field(default_factory=bytearray)
    _name_off: dict[str, int] = field(default_factory=dict)
    _refs: list[tuple[int, str, int, int]] = field(default_factory=list)  # (desc, field, handle, +bytes)
    _arena_off: list[int] = field(default_factory=list)
    _arena_len: int = 0
    _layout: tuple[int, int] | None = None     # (data_off, total) once written

    def data(self, raw: bytes | np.ndarray) -> int:
        """Append bytes, 16-byte aligned so the DMA can start on them. Returns a handle."""
        b = np.ascontiguousarray(raw).tobytes() if isinstance(raw, np.ndarray) else bytes(raw)
        pad = (-len(b)) % 16
        self._chunk_off.append(self._data_len)
        self._chunks.append(b + b"\0" * pad)
        self._data_len += len(b) + pad
        return len(self._chunks) - 1

    def scratch(self, nbytes: int) -> int:
        """Reserve DRAM above the blob for an activation. Returns a handle."""
        self._arena_off.append(self._arena_len)
        self._arena_len += (nbytes + 15) & ~15
        return ARENA | (len(self._arena_off) - 1)

    def resolve(self, ref: Ref, data_off: int, total: int) -> int:
        h, off = ref if isinstance(ref, tuple) else (ref, 0)
        if h & ARENA:
            return self.base + total + self._arena_off[h ^ ARENA] + off
        return self.base + data_off + self._chunk_off[h] + off

    def name(self, s: str) -> int:
        if s not in self._name_off:
            self._name_off[s] = len(self._names)
            self._names += s.encode() + b"\0"
        return self._name_off[s]

    def add(self, d: Desc, **dram_refs: Ref) -> Desc:
        """Append a descriptor.

        `dram_refs` maps a field to a data or arena handle, or to (handle,
        byte offset) for a slice -- an m tile's rows inside a whole tensor.
        """
        d.name_off = self.name(d.name) if d.name else 0
        self.descs.append(d)
        for fld, ref in dram_refs.items():
            h, off = ref if isinstance(ref, tuple) else (ref, 0)
            self._refs.append((len(self.descs) - 1, fld, h, off))
        return d

    def write(self, path: Path) -> int:
        self.add(Desc(op=OP_END, name="end"))
        n = len(self.descs)
        desc_off = HEADER_BYTES
        names_off = desc_off + n * DESC_WORDS * 4
        data_off = (names_off + len(self._names) + 15) & ~15
        total = data_off + self._data_len

        for idx, fld, h, off in self._refs:
            setattr(self.descs[idx], fld, self.resolve((h, off), data_off, total))
        for d in self.descs:
            d.name_off = self.base + names_off + d.name_off if d.name else 0

        hdr = struct.pack("<4s8I", MAGIC, VERSION, n, desc_off, names_off, data_off,
                          total, self.base, self._arena_len)
        hdr += b"\0" * (HEADER_BYTES - len(hdr))
        out = bytearray(hdr)
        for d in self.descs:
            out += d.pack()
        out += self._names
        out += b"\0" * (data_off - len(out))
        for c in self._chunks:
            out += c
        assert len(out) == total
        path.write_bytes(out)
        self._layout = (data_off, total)
        return total

    def address(self, ref: Ref) -> int:
        """The DRAM address a ref resolved to. Valid after write()."""
        assert self._layout is not None, "address() before write()"
        return self.resolve(ref, *self._layout)


# --------------------------------------------------------------------- GEMMs

def per_channel_config(bias_q: np.ndarray, rq: Requant, n: int) -> np.ndarray:
    """[n][4] u32: bias, mult, shift, pad -- the config region's own layout."""
    mult = np.broadcast_to(np.asarray(rq.multiplier), (n,))
    shift = np.broadcast_to(np.asarray(rq.shift), (n,))
    cfg = np.zeros((n, 4), dtype=np.uint32)
    cfg[:, 0] = np.asarray(bias_q, dtype=np.int64).astype(np.uint32)
    cfg[:, 1] = mult.astype(np.int64).astype(np.uint32)
    cfg[:, 2] = shift.astype(np.uint32)
    return cfg


def _requant_slice(rq: Requant, n0: int, n1: int) -> Requant:
    if rq.is_per_channel:
        return Requant(multiplier=np.asarray(rq.multiplier)[n0:n1],
                       shift=np.asarray(rq.shift)[n0:n1])
    return rq


def emit_gemm(blob: Blob, name: str, x_q: np.ndarray, x_ref: Ref | None,
              w_q: np.ndarray, bias_q: np.ndarray, rq: Requant,
              w_ref: Ref | None = None, out_ref: Ref | None = None,
              out_stride: int | None = None, check: bool = True,
              gather: dict | None = None, out_grp: int = 0, out_grp_stride: int = 0,
              m_align: int = 1) -> np.ndarray:
    """`y = requant(x_q @ w_q + bias)` as the tiles gemm_seq can run.

    x_q is [M][K] int8 row-major -- the activation buffer's layout, so an m
    tile is a contiguous slice of rows. w_q is [K][N] int8, the weight
    buffer's layout for one n tile; wider N is cut into n tiles here and each
    tile's [K][Nt] is stored contiguously, since the buffer wants it that way
    and the weights are ours to arrange.

    x_ref / w_ref name where the operands already are in DRAM (a tensor an
    earlier op left in the arena); None places them in the blob's data area.
    A w_ref must be one whole [K][N] tile, since an earlier op wrote it that
    way and nothing here can re-lay it out.

    out_ref is where the result goes, row-major with `out_stride` bytes per
    row (default N): a tile lands at (m0, n0) inside the tensor. None leaves
    each tile in the result region, for a probe.

    The tiling is sw/tiling.py's. The weight tile and its config load once
    per n tile and the m tiles under it name only their activations, which is
    how weight reuse shows up in the descriptor stream. Returns the emulator's
    y_q, which is what the driver checks each tile against.
    """
    m, k = x_q.shape
    k2, n = w_q.shape
    assert k == k2 and k % LANES == 0 and n % LANES == 0
    t = tile_gemm(name, m, k, n, BUFFERS)
    if not t.fits:
        raise ValueError(f"{name}: {t.reason}")
    m_tile = t.m_tile
    if m_align > 1 and m_tile < m:
        m_tile = m_tile // m_align * m_align
        assert m_tile > 0, f"{name}: an m tile cannot hold one group of {m_align} rows"
    if w_ref is not None and t.n_tile < n:
        raise ValueError(f"{name}: an operand already in DRAM cannot be split into n tiles")
    stride = n if out_stride is None else out_stride

    y_q = qlinear(x_q, w_q.T, bias_q.astype(np.int32), rq)
    if x_ref is None and gather is None:
        x_ref = blob.data(x_q)
    k_words = k // LANES

    for n0 in range(0, n, t.n_tile):
        nt = min(t.n_tile, n - n0)
        w_h = w_ref if w_ref is not None else blob.data(np.ascontiguousarray(w_q[:, n0:n0 + nt]))
        cfg_h = blob.data(per_channel_config(bias_q[n0:n0 + nt], _requant_slice(rq, n0, n0 + nt), nt))

        for i, m0 in enumerate(range(0, m, m_tile)):
            mt = min(m_tile, m - m0)
            d = Desc(op=OP_GEMM, name=f"{name}[{m0}:{m0 + mt},{n0}:{n0 + nt}]",
                     a_words=mt * k_words,
                     src_a=src(R_ACT, 0), src_b=src(R_WGT, 0), dst=0,
                     shape=gemm_shape(mt, k_words, nt // LANES))
            if gather is None:
                refs: dict[str, Ref] = {"a_dram": ref_add(x_ref, m0 * k)}
            else:
                # The driver builds this tile's im2col rows in the buffer itself.
                g = gather
                d.a_words = 0
                d.g_hw = g["H"] | g["W"] << 16
                d.g_cw = g["C"] | g["Wo"] << 16
                d.g_k = g["k"] | g["stride"] << 8 | g["pad"] << 16 | g["t0"] << 20 | g["t1"] << 26
                d.g_p0 = m0
                refs = {"g_src": g["src"]}
            if i == 0:
                d.b_words = k * (nt // LANES)
                refs["b_dram"] = w_h
                d.cfg_dst, d.cfg_n = 0, nt * 4
                refs["cfg_dram"] = cfg_h
            if out_ref is not None:
                d.out_cols, d.out_rows, d.out_stride = nt, mt, stride
                if out_grp:
                    assert m0 % out_grp == 0
                    d.out_grp, d.out_grp_stride = out_grp, out_grp_stride
                    refs["out_dram"] = ref_add(out_ref, (m0 // out_grp) * out_grp_stride + n0)
                else:
                    refs["out_dram"] = ref_add(out_ref, m0 * stride + n0)
            if check:
                exp = np.ascontiguousarray(y_q[m0:m0 + mt, n0:n0 + nt])
                d.check_n = exp.size
                refs["check_dram"] = blob.data(exp)
            blob.add(d, **refs)
    return y_q


# --------------------------------------------------------------- vector ops

def emit_config(blob: Blob, name: str, cfg_word: int, words: np.ndarray) -> None:
    """A table load with no op: the exp pair, the unary LUT."""
    d = Desc(op=OP_CONFIG, name=name, cfg_dst=cfg_word, cfg_n=int(words.size))
    blob.add(d, cfg_dram=blob.data(np.asarray(words, dtype=np.uint32)))


def _emit_vec(blob: Blob, name: str, op: int, rows: int, length: int,
              a_ref: Ref, y_q: np.ndarray, out_ref: Ref | None, check: bool,
              b_ref: Ref | None = None, cfg: tuple[int, np.ndarray] | None = None,
              **regs: int) -> None:
    """One vector op over `rows` x `length` elements, A (and B) DMA'd in."""
    elems = rows * length
    assert elems % LANES == 0, f"{name}: {rows}x{length} is not word-aligned"
    words = elems // LANES
    assert words <= BUFFERS.act_words and words <= BUFFERS.out_words, f"{name}: {words} words"
    d = Desc(op=op, name=name, a_words=words, src_a=src(R_ACT, 0), dst=0,
             shape=vec_shape(length, rows), **regs)
    refs: dict[str, Ref] = {"a_dram": a_ref}
    if b_ref is not None:
        assert words <= BUFFERS.wgt_words
        d.b_words, d.src_b = words, src(R_WGT, 0)
        refs["b_dram"] = b_ref
    if cfg is not None:
        d.cfg_dst, d.cfg_n = cfg[0], int(cfg[1].size)
        refs["cfg_dram"] = blob.data(np.asarray(cfg[1], dtype=np.uint32))
    if out_ref is not None:
        d.out_cols, d.out_rows, d.out_stride = elems, 1, elems
        refs["out_dram"] = out_ref
    if check:
        d.check_n = elems
        refs["check_dram"] = blob.data(np.ascontiguousarray(y_q))
    blob.add(d, **refs)


def emit_rows(blob: Blob, name: str, op: int, x_q: np.ndarray, x_ref: Ref,
              y_q: np.ndarray, out_ref: Ref | None, check: bool = True,
              cfg: tuple[int, np.ndarray] | None = None, **regs: int) -> None:
    """A row-wise op (softmax, layernorm) over [rows][len], cut into what the
    activation buffer holds. The config, if any, rides on the first cut."""
    rows, length = x_q.shape
    assert length % LANES == 0
    per = min(rows, VEC_MAX_ELEMS // length)
    for r0 in range(0, rows, per):
        rt = min(per, rows - r0)
        _emit_vec(blob, f"{name}[{r0}:{r0 + rt}]", op, rt, length,
                  ref_add(x_ref, r0 * length), y_q[r0:r0 + rt],
                  None if out_ref is None else ref_add(out_ref, r0 * length),
                  check, cfg=cfg if r0 == 0 else None, **regs)


def emit_flat(blob: Blob, name: str, op: int, x_q: np.ndarray, x_ref: Ref,
              y_q: np.ndarray, out_ref: Ref | None, b_ref: Ref | None = None,
              check: bool = True, **regs: int) -> None:
    """An elementwise op (unary LUT, qadd) over a whole tensor, in buffer-sized
    flat chunks. The tensor's shape is nothing to the hardware."""
    n = x_q.size
    assert n % LANES == 0
    per = VEC_MAX_ELEMS
    yf = y_q.reshape(-1)
    for e0 in range(0, n, per):
        et = min(per, n - e0)
        _emit_vec(blob, f"{name}[{e0}:{e0 + et}]", op, et // LANES, LANES,
                  ref_add(x_ref, e0), yf[e0:e0 + et],
                  None if out_ref is None else ref_add(out_ref, e0), check,
                  b_ref=None if b_ref is None else ref_add(b_ref, e0), **regs)


def emit_transpose(blob: Blob, name: str, x_q: np.ndarray, x_ref: Ref,
                   out_ref: Ref | None, check: bool = True) -> np.ndarray:
    """[rows][len] -> [len][rows], one op; both must fit their buffers."""
    rows, length = x_q.shape
    y_q = np.ascontiguousarray(x_q.T)
    _emit_vec(blob, name, OP_TRANSPOSE, rows, length, x_ref, y_q, out_ref, check)
    return y_q


def emit_layernorm(blob: Blob, name: str, x_q: np.ndarray, x_ref: Ref, s_x: float,
                   x_f: np.ndarray, gamma_f: np.ndarray, beta_f: np.ndarray,
                   out_ref: Ref | None, eps: float = 1e-6, check: bool = True):
    """Returns (y_q, s_out, y_f). The requant constant carries sqrt(len)/2^14
    as qlayernorm() expects; eps is pre-scaled into the squared int domain."""
    length = x_q.shape[-1]
    h_f = ((x_f - x_f.mean(-1, keepdims=True))
           / np.sqrt(x_f.var(-1, keepdims=True) + eps)) * gamma_f + beta_f
    s_out = act_scale(name, h_f)
    s_g = qscale(gamma_f)
    g_q = quant(gamma_f, s_g)
    b_q = np.rint(beta_f / s_out).astype(np.int32)
    rq = Requant.from_real_multiplier(s_g * math.sqrt(length) / (float(1 << 14) * s_out))
    y_q = qlayernorm(x_q, g_q, b_q, s_x, rq, eps=eps)
    eps_q = int(round(eps / (s_x * s_x) * length ** 3))
    par = np.zeros((length, 2), dtype=np.uint32)
    par[:, 0] = g_q.astype(np.int64) & 0xFF
    par[:, 1] = b_q.astype(np.int64).astype(np.uint32)
    emit_rows(blob, name, OP_LAYERNORM, x_q, x_ref, y_q, out_ref, check,
              cfg=(CFG_LN_PAR, par), vmult=int(rq.multiplier), vshift=int(rq.shift),
              eps_lo=eps_q & 0xFFFFFFFF, eps_hi=(eps_q >> 32) & 0xFFFFFFFF)
    return y_q, s_out, h_f


def emit_qadd(blob: Blob, name: str, a_q, a_ref, s_a, b_q, b_ref, s_b, s_out,
              out_ref: Ref | None, check: bool = True) -> np.ndarray:
    rq_a = Requant.from_real_multiplier(s_a / s_out)
    rq_b = Requant.from_real_multiplier(s_b / s_out)
    y_q = qadd(a_q, b_q, rq_a, rq_b)
    emit_flat(blob, name, OP_QADD, a_q, a_ref, y_q, out_ref, b_ref=b_ref, check=check,
              vmult=int(rq_a.multiplier), vmult_b=int(rq_b.multiplier),
              vshift=int(rq_a.shift) | (int(rq_b.shift) << VEC_SHIFT_B_LSB))
    return y_q


# ---------------------------------------------------------------- programs

def build_gemm_probe(blob: Blob, sd: dict[str, np.ndarray], tokens: int, seed: int) -> None:
    """Increment 1: one real projection of block 0, head 0's q, at real shape.

    The activations are synthetic -- the check is that the GEMM over the real
    weights, tiled the real way, streamed from DDR3, is bit-exact against the
    emulator, not that the numbers mean anything yet.
    """
    rng = np.random.default_rng(seed)
    e = 384
    d = 64
    w_qkv = sd["pretrained.blocks.0.attn.qkv.weight"]        # [3E][E], torch [out][in]
    b_qkv = sd["pretrained.blocks.0.attn.qkv.bias"]
    w_f = w_qkv[:d, :].T.astype(np.float64) * (d ** -0.5)    # [K=E][N=D], with 1/sqrt(d) folded
    b_f = b_qkv[:d].astype(np.float64) * (d ** -0.5)

    x_f = rng.standard_normal((tokens, e)) * 0.5
    s_x = qscale(x_f)
    x_q = quant(x_f, s_x)

    s_w = np.maximum(np.abs(w_f).max(axis=0), 1e-12) / INT8_MAX
    w_q = quant(w_f, s_w)
    b_q = np.rint(b_f / (s_x * s_w)).astype(np.int64)
    y_f = x_f @ w_f + b_f
    s_y = qscale(y_f)
    rq = Requant.from_real_multiplier(s_x * s_w / s_y)

    emit_gemm(blob, "blk0.q0", x_q, None, w_q, b_q, rq)


def linear_q(key: str, x_f: np.ndarray, s_x: float, w_f: np.ndarray, b_f: np.ndarray,
             per_chan: np.ndarray | None = None, s_y: float | None = None):
    """Quantize a [K][N] weight per output channel and derive the requant for
    the float output's own scale. `per_chan` (LayerScale) folds into the
    multiplier -- its sign into the weight column, since the requantizer's
    multiplier is unsigned. Returns (w_q, b_q, rq, y_f, s_y)."""
    if per_chan is not None:
        sign = np.where(per_chan < 0, -1.0, 1.0)
        w_f, b_f = w_f * sign, b_f * sign
        per_chan = np.maximum(np.abs(per_chan), 1e-12)
    # An all-zero column (an output channel padded to a whole word) takes the
    # layer's widest scale: its output is 0 whatever the multiplier, and a
    # 1e-12 scale would need a shift the requantizer does not have.
    amax = np.abs(w_f).max(axis=0)
    amax = np.where(amax > 0, amax, max(float(amax.max()), 1e-12))
    s_w = amax / INT8_MAX
    w_q = quant(w_f, s_w)
    b_q = np.rint(b_f / (s_x * s_w)).astype(np.int64)
    y_f = x_f @ w_f + b_f
    if per_chan is not None:
        y_f = y_f * per_chan
    if s_y is None:
        s_y = act_scale(key, y_f)
    real = s_x * s_w / s_y
    if per_chan is not None:
        real = real * per_chan
    return w_q, b_q, Requant.from_real_multiplier(real), y_f, s_y


def build_block(blob: Blob, sd: dict[str, np.ndarray], tokens: int, seed: int,
                blk: int = 0, x_f: np.ndarray | None = None, x_ref: Ref | None = None,
                s_x: float | None = None, x_q: np.ndarray | None = None, check: bool = True):
    """Increment 2: one transformer block of the real model, at real width,
    activations living in the DDR3 arena between ops.

    The lowering is test/v2/gen_block_vectors.py's, with two things the
    arena makes possible: each head's context is written straight into its
    columns of one [T][E] tensor, so the output projection is one GEMM over
    the real weight instead of a per-head split summed with qadd; and the
    token count need not be a multiple of 16 -- only the key axis of the
    logits is, and it is padded to T_pad with a bias that saturates the
    padded columns to INT8_MIN, which the exp table sends to zero.

    Scales come from a float forward of the same input, as the T=16 block's
    did. Returns (y_q, y_ref, s_y, y_f) so blocks can chain.
    """
    pre = f"pretrained.blocks.{blk}."
    E, H, D, HID = 384, 6, 64, 1536
    T = tokens
    T_pad = (T + LANES - 1) // LANES * LANES
    rng = np.random.default_rng(seed)
    f64 = lambda k: sd[pre + k].astype(np.float64)  # noqa: E731

    if x_f is None:
        x_f = rng.standard_normal((T, E)) * 0.5
        s_x = qscale(x_f)
        x_q = quant(x_f, s_x)
        x_ref = blob.data(x_q)
    else:
        # The integer tensor the previous op produced, not a re-quantization
        # of the float one: the two differ by an LSB here and there, and the
        # hardware only ever sees the former.
        assert x_ref is not None and s_x is not None and x_q is not None
    nm = f"blk{blk}."

    # -- norm1 -> q, k, v per head, each a contiguous [T][D] --------------
    h1_ref = blob.scratch(T * E)
    h1_q, s_h1, h1_f = emit_layernorm(blob, nm + "norm1", x_q, x_ref, s_x, x_f,
                                      f64("norm1.weight"), f64("norm1.bias"), h1_ref, check=check)

    w_qkv, b_qkv = f64("attn.qkv.weight"), f64("attn.qkv.bias")   # [3E][E] torch
    ctx_ref = blob.scratch(T * E)
    ctx_q = np.zeros((T, E), dtype=np.int8)
    ctx_f = np.zeros((T, E))
    s_ctx = np.zeros(H)
    for h in range(H):
        rows = {"q": slice(h * D, (h + 1) * D), "k": slice(E + h * D, E + (h + 1) * D),
                "v": slice(2 * E + h * D, 2 * E + (h + 1) * D)}
        proj = {}
        for which, sl in rows.items():
            w_f, b_f = w_qkv[sl, :].T.copy(), b_qkv[sl].copy()
            if which == "q":                      # DINOv2 folds 1/sqrt(d) into q
                w_f, b_f = w_f * D ** -0.5, b_f * D ** -0.5
            w_q, b_q, rq, y_f, s_y = linear_q(f"{nm}{which}{h}", h1_f, s_h1, w_f, b_f)
            # k and v are padded to T_pad rows: the transpose wants whole
            # words per output row, and P.V's reduction runs over T_pad.
            t_rows = T if which == "q" else T_pad
            ref = blob.scratch(t_rows * D)
            y_q = emit_gemm(blob, f"{nm}{which}{h}", h1_q, h1_ref, w_q, b_q, rq,
                            out_ref=ref, check=check)
            if t_rows > T:
                y_q = np.concatenate([y_q, np.zeros((t_rows - T, D), dtype=np.int8)])
                y_f = np.concatenate([y_f, np.zeros((t_rows - T, D))])
            proj[which] = (y_q, ref, s_y, y_f)

        q_q, q_ref, s_q, q_f = proj["q"]
        k_q, k_ref, s_k, k_f = proj["k"]
        v_q, v_ref, s_v, v_f = proj["v"]

        kt_ref = blob.scratch(D * T_pad)
        kt_q = emit_transpose(blob, f"{nm}kt{h}", k_q, k_ref, kt_ref, check=check)

        # logits [T][T_pad]: the GEMM's per-column bias saturates the padded
        # keys to INT8_MIN, so softmax sees them as -inf as near as the exp
        # table can say.
        s_logit = act_scale(f"{nm}logit{h}", q_f @ k_f.T)
        rq_l = Requant.from_real_multiplier(s_q * s_k / s_logit)
        bias_l = np.zeros(T_pad, dtype=np.int64)
        bias_l[T:] = -int(math.ceil(2 * INT8_MAX / rq_l.real_multiplier))
        l_ref = blob.scratch(T * T_pad)
        l_q = emit_gemm(blob, f"{nm}logit{h}", q_q, q_ref, kt_q, bias_l, rq_l,
                        w_ref=kt_ref, out_ref=l_ref, check=check)
        assert (l_q[:, T:] == INT8_MIN).all()

        exp_lo, exp_hi = build_exp_lut(s_logit)
        emit_config(blob, f"{nm}exp{h}", CFG_EXP_LO,
                    np.concatenate([exp_lo & 0xFFFF, exp_hi & 0xFFFF]))
        s_prob = 1.0 / INT8_MAX
        sm_rq = Requant.from_real_multiplier(1.0 / (float(1 << 15) * s_prob))
        p_q = qsoftmax(l_q, (exp_lo, exp_hi), s_prob)
        p_ref = blob.scratch(T * T_pad)
        emit_rows(blob, f"{nm}prob{h}", OP_SOFTMAX, l_q, l_ref, p_q, p_ref, check,
                  vmult=int(sm_rq.multiplier), vshift=int(sm_rq.shift))

        # ctx_h = P.V, written into its head's columns of the [T][E] context.
        c_f = (p_q.astype(np.float64) * s_prob) @ (v_q.astype(np.float64) * s_v)
        s_c = act_scale(f"{nm}ctx{h}", c_f)
        rq_c = Requant.from_real_multiplier(s_prob * s_v / s_c)
        c_q = emit_gemm(blob, f"{nm}ctx{h}", p_q, p_ref, v_q, np.zeros(D, dtype=np.int64),
                        rq_c, w_ref=v_ref, out_ref=ref_add(ctx_ref, h * D), out_stride=E,
                        check=check)
        ctx_q[:, h * D:(h + 1) * D] = c_q
        ctx_f[:, h * D:(h + 1) * D] = c_f
        s_ctx[h] = s_c

    # -- output projection, LayerScale folded; residual 1 -----------------
    # The heads' contexts carry six scales but the projection's activation
    # input has one: bring them to a common scale by folding each head's
    # ratio into its rows of the weight, exactly, before quantizing.
    s_cx = float(s_ctx.max())
    w_o = f64("attn.proj.weight").T.copy()                     # [E][E], [K][N]
    for h in range(H):
        w_o[h * D:(h + 1) * D, :] *= s_ctx[h] / s_cx
    ctx_f_common = ctx_q.astype(np.float64) * s_cx           # what the GEMM sees
    w_oq, b_oq, rq_o, a_f, s_a = linear_q(nm + "proj", ctx_f_common, s_cx, w_o,
                                          f64("attn.proj.bias"),
                                          per_chan=f64("ls1.gamma"))
    a_ref = blob.scratch(T * E)
    a_q = emit_gemm(blob, nm + "proj", ctx_q, ctx_ref, w_oq, b_oq, rq_o, out_ref=a_ref,
                    check=check)

    x1_f = x_f + a_f
    s_x1 = act_scale(nm + "res1", x1_f)
    x1_ref = blob.scratch(T * E)
    x1_q = emit_qadd(blob, nm + "res1", x_q, x_ref, s_x, a_q, a_ref, s_a, s_x1, x1_ref,
                     check=check)

    # -- norm2 -> MLP; residual 2 ------------------------------------------
    h2_ref = blob.scratch(T * E)
    h2_q, s_h2, h2_f = emit_layernorm(blob, nm + "norm2", x1_q, x1_ref, s_x1, x1_f,
                                      f64("norm2.weight"), f64("norm2.bias"), h2_ref, check=check)
    w1q, b1q, rq1, f1_f, s_f1 = linear_q(nm + "fc1", h2_f, s_h2, f64("mlp.fc1.weight").T.copy(),
                                         f64("mlp.fc1.bias"))
    f1_ref = blob.scratch(T * HID)
    f1_q = emit_gemm(blob, nm + "fc1", h2_q, h2_ref, w1q, b1q, rq1, out_ref=f1_ref, check=check)

    s_g1 = act_scale(nm + "gelu", gelu(f1_f))
    un_lut = build_unary_lut(gelu, scale_in=s_f1, scale_out=s_g1)
    emit_config(blob, nm + "gelu_lut", CFG_UN_LUT, un_lut.astype(np.int64) & 0xFF)
    g1_q = apply_unary_lut(f1_q, un_lut)
    g1_ref = blob.scratch(T * HID)
    emit_flat(blob, nm + "gelu", OP_UNARY, f1_q, f1_ref, g1_q, g1_ref, check=check)

    g1_f = g1_q.astype(np.float64) * s_g1
    w2q, b2q, rq2, f2_f, s_f2 = linear_q(nm + "fc2", g1_f, s_g1, f64("mlp.fc2.weight").T.copy(),
                                         f64("mlp.fc2.bias"), per_chan=f64("ls2.gamma"))
    f2_ref = blob.scratch(T * E)
    f2_q = emit_gemm(blob, nm + "fc2", g1_q, g1_ref, w2q, b2q, rq2, out_ref=f2_ref, check=check)

    y_f = x1_f + f2_f
    s_y = act_scale(nm + "res2", y_f)
    y_ref = blob.scratch(T * E)
    y_q = emit_qadd(blob, nm + "res2", x1_q, x1_ref, s_x1, f2_q, f2_ref, s_f2, s_y, y_ref,
                    check=check)
    return y_q, y_ref, s_y, y_f


def encoder_im2col(image: np.ndarray, patch: int = 14) -> np.ndarray:
    """[1][3][S][S] -> [T][K] float: one row per patch in (c, kh, kw) order, as
    the conv weight flattens, K padded to a whole word; row 0 is the cls slot
    and stays zero."""
    _, C, S, _ = image.shape
    g = S // patch
    x = image[0].astype(np.float64)
    patches = x.reshape(C, g, patch, g, patch).transpose(1, 3, 0, 2, 4).reshape(g * g, -1)
    K = (patches.shape[1] + LANES - 1) // LANES * LANES
    a_f = np.zeros((g * g + 1, K))
    a_f[1:, :patches.shape[1]] = patches
    return a_f


def encoder_input(image: np.ndarray, scale: float) -> np.ndarray:
    """The int8 bytes a program blob expects at its input address."""
    return quant(encoder_im2col(image), scale)


def build_encoder(blob: Blob, sd: dict[str, np.ndarray], image: np.ndarray,
                  check: bool = True, taps: tuple[int, ...] = (2, 5, 8, 11),
                  dump: list[str] = (), input_in_arena: bool = False,
                  head: bool = False) -> dict:
    """Increment 3: the whole ViT-S encoder over a preprocessed [1][3][S][S]
    image, S a multiple of 14. Returns the sidecar the host needs to finish
    the picture: where each tap's normalized tokens are, and their scales.

    Patch embedding is the 14x14 stride-14 convolution as a GEMM over an
    im2col the host does -- every patch is a row of 3*14*14 = 588 pixels,
    padded to 592 -- with row 0 all zeros for the cls token, so the GEMM
    gives it the bias alone. One qadd then adds a constant that is the
    position embedding for every patch and (cls + pos[0] - bias) for row 0,
    which is the cls token and its position in one op.

    DINOv2 resamples the position grid bicubically for a non-518 input; the
    frontend's _resample_pos_embed does that in float here.
    """
    from .frontend.dav2 import DAV2Config, _resample_pos_embed
    E, P = 384, 14
    _, C, S, S2 = image.shape
    assert S == S2 and S % P == 0
    assert not (input_in_arena and check), "a program's expected values would be one image's"
    g = S // P
    T = g * g + 1
    cfg = DAV2Config(img_size=S)
    f64 = lambda k: sd["pretrained." + k].astype(np.float64)  # noqa: E731

    a_f = encoder_im2col(image)
    K = a_f.shape[1]
    s_a = act_scale("embed.in", a_f)
    a_q = quant(a_f, s_a)
    a_ref = blob.scratch(a_q.size) if input_in_arena else blob.data(a_q)

    w_f = np.zeros((K, E))
    w_f[:C * P * P, :] = f64("patch_embed.proj.weight").reshape(E, -1).T
    b_f = f64("patch_embed.proj.bias")
    w_q, b_q, rq, tok_f, s_tok = linear_q("embed", a_f, s_a, w_f, b_f)
    tok_ref = blob.scratch(T * E)
    tok_q = emit_gemm(blob, "embed", a_q, a_ref, w_q, b_q, rq, out_ref=tok_ref, check=check)

    pos = _resample_pos_embed(sd["pretrained.pos_embed"], cfg)[0].astype(np.float64)   # [T][E]
    const_f = pos.copy()
    const_f[0] += f64("cls_token")[0, 0] - b_f
    s_const = qscale(const_f)
    const_q = quant(const_f, s_const)
    x_f = tok_f + const_f
    s_x = act_scale("tokens", x_f)
    x_ref = blob.scratch(T * E)
    x_q = emit_qadd(blob, "tokens", tok_q, tok_ref, s_tok, const_q, blob.data(const_q), s_const,
                    s_x, x_ref, check=check)

    # -- twelve blocks, four taps through the final norm ------------------
    side = {"image": S, "tokens": T, "taps": [], "dumps": []}
    if input_in_arena:
        # A program: the host writes each picture's im2col here before a run.
        side["input"] = {"name": "embed.in", "ref": a_ref, "shape": list(a_q.shape),
                         "scale": float(s_a), "emu": a_q}
    # Debugging: record the input and output of the layernorms named in
    # `dump` so load_model.py reads them back in the run's own session.
    global emit_layernorm
    plain = emit_layernorm

    def recording(blob, name, x_q, x_ref, *args, **kw):
        out = plain(blob, name, x_q, x_ref, *args, **kw)
        if name in dump:
            for tag, ref, q in (("in", x_ref, x_q), ("out", args[-1], out[0])):
                side["dumps"].append({"name": f"{name}/{tag}", "ref": ref,
                                      "shape": list(q.shape), "emu": q})
        return out
    emit_layernorm = recording
    try:
        for i in range(cfg.depth):
            x_q, x_ref, s_x, x_f = build_block(blob, sd, T, 0, blk=i, x_f=x_f, x_ref=x_ref,
                                               s_x=s_x, x_q=x_q, check=check)
            if i in taps:
                j = len(side["taps"])
                t_ref = blob.scratch(T * E)
                t_q, s_t, t_f = emit_layernorm(blob, f"tap{j}.norm", x_q, x_ref, s_x, x_f,
                                               f64("norm.weight"), f64("norm.bias"), t_ref,
                                               check=check)
                side["taps"].append({"name": f"tap{j}/norm", "block": i, "ref": t_ref,
                                     "scale": float(s_t), "shape": [1, T, E], "emu": t_q,
                                     "f": t_f})
        if head:
            from .export_head import build_head
            side["depth"] = build_head(blob, sd, [{"q": t["emu"], "ref": t["ref"], "s": t["scale"],
                                                   "f": t["f"]} for t in side["taps"]],
                                       g, S, check=check)
    finally:
        emit_layernorm = plain
    return side


CALIB_IMAGES = ("skimage:astronaut", "skimage:coffee", "skimage:rocket",
                "skimage:chelsea", "skimage:cat")


def calibrate(sd: dict[str, np.ndarray], images, size: int, percentile: float) -> dict:
    """Record every activation's range over `images` and return the scales.

    Each image is lowered with its own scales (as an image-specific program
    would be) on a throwaway blob; the recorder keeps the widest range seen
    per activation. Scales chain -- a block's input scale shapes its outputs
    -- so this is the same approximation static calibration always makes."""
    global CALIB
    from .imageio import load_image
    saved, CALIB = CALIB, Calibration(percentile=percentile)
    try:
        for img in images:
            build_encoder(Blob(), sd, load_image(str(img), size), check=False, head=True)
        rec = CALIB
    finally:
        CALIB = saved
    return {"size": size, "percentile": percentile, "images": [str(i) for i in images],
            "scales": rec.scales()}


def run_encoder_emulator(sd: dict[str, np.ndarray], image: np.ndarray, scales: dict[str, float],
                         taps: tuple[int, ...] = (2, 5, 8, 11),
                         head: bool = False) -> dict[str, np.ndarray]:
    """What a fixed-scale program computes for `image`, bit-exact: the taps,
    and with `head` the int8 depth map ([S][S][16], channel 0 is depth)."""
    global CALIB
    saved, CALIB = CALIB, Calibration(fixed=scales)
    try:
        side = build_encoder(Blob(), sd, image, check=False, taps=taps, head=head)
    finally:
        CALIB = saved
    out = {t["name"]: t["emu"] for t in side["taps"]}
    if head:
        out["depth"] = side["depth"]["emu"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="write a tiny-tpu program blob")
    ap.add_argument("-o", type=Path, required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--tokens", type=int, default=82, help="82 = 126x126 input")
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--program", choices=("gemm-probe", "block", "encoder"), default="encoder")
    ap.add_argument("--image", type=Path, default=None, help="encoder: the picture")
    ap.add_argument("--size", type=int, default=126, help="encoder: input side, a multiple of 14")
    ap.add_argument("--no-check", action="store_true", help="no expected results in the blob")
    ap.add_argument("--dump", action="append", default=[], metavar="OP",
                    help="encoder: also read back this layernorm's input and output, e.g. blk0.norm1")
    ap.add_argument("--calibrate", action="store_true",
                    help="record activation scales over --calib-images and write them to -o (JSON)")
    ap.add_argument("--calib-images", nargs="+", default=list(CALIB_IMAGES))
    ap.add_argument("--percentile", type=float, default=100.0)
    ap.add_argument("--calib", type=Path, default=None,
                    help="encoder: fixed scales from --calibrate instead of the image's own")
    ap.add_argument("--head", action="store_true",
                    help="encoder: run the DPT head on the accelerator too; the blob ends in a depth map")
    ap.add_argument("--program-only", action="store_true",
                    help="encoder: image-independent program; the host writes each input (needs --calib)")
    a = ap.parse_args()

    from .frontend.dav2 import load_state_dict
    sd = load_state_dict(a.checkpoint) if a.checkpoint else load_state_dict()

    import json
    if a.calibrate:
        cal = calibrate(sd, a.calib_images, a.size, a.percentile)
        a.o.write_text(json.dumps(cal, indent=1))
        print(f"wrote {a.o}: {len(cal['scales'])} activation scales over "
              f"{len(cal['images'])} images at {a.size}x{a.size}, percentile {a.percentile}")
        return
    if a.program_only and a.calib is None:
        ap.error("--program-only needs --calib: a program cannot take scales from an image")
    if a.calib is not None:
        global CALIB
        CALIB = Calibration(fixed=json.loads(a.calib.read_text())["scales"])

    blob = Blob()
    if a.program == "gemm-probe":
        build_gemm_probe(blob, sd, a.tokens, a.seed)
    elif a.program == "block":
        build_block(blob, sd, a.tokens, a.seed, check=not a.no_check)
    elif a.program == "encoder":
        from .imageio import load_image, synthetic_image
        if a.image is not None:
            image = load_image(a.image, a.size)
        else:
            image = synthetic_image(a.size, np.random.default_rng(a.seed))
        side = build_encoder(blob, sd, image, check=not (a.no_check or a.program_only),
                             dump=a.dump, input_in_arena=a.program_only, head=a.head)
    total = blob.write(a.o)
    if a.program == "encoder":
        # The sidecar: what the host needs to read the taps back and finish
        # the picture. The emulator's own taps ride along as .npz so the
        # board's can be compared to them byte for byte.
        emu = {}
        for t in side["taps"]:
            t.pop("f", None)
        if "depth" in side:
            side["depth"].pop("f")
        for t in (side["taps"] + side["dumps"] + ([side["input"]] if "input" in side else [])
                  + ([side["depth"]] if "depth" in side else [])):
            t["addr"] = blob.address(t.pop("ref"))
            t["bytes"] = int(np.prod(t["shape"]))
            emu[t["name"]] = t.pop("emu")
        side["image_path"] = str(a.image) if a.image else None
        side["seed"] = a.seed
        side["calib"] = str(a.calib.resolve()) if a.calib else None
        a.o.with_suffix(".json").write_text(json.dumps(side, indent=1))
        np.savez(a.o.with_suffix(".emu.npz"), **emu)
    n_ops = sum(d.op < 0x10 for d in blob.descs)
    print(f"wrote {a.o}: {total} bytes, {n_ops} hardware ops, "
          f"{len(blob.descs)} descriptors")


if __name__ == "__main__":
    # Run as the canonical module, not as __main__: export_head imports
    # sw.export_tpu, and the calibration state must be the one it sees.
    from sw import export_tpu as _canonical
    _canonical.main()
