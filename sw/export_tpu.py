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
    out_words   128-bit words to copy
    check_dram  expected int8 elements of the result, 0 = do not check
    check_n     how many elements
    name_off    offset of a NUL-terminated name, for the driver's messages
    pad x 2

Config-region layout is the driver's (main.c): per-channel GEMM constants at
word 4*c + {0 bias, 1 mult, 2 shift}; layernorm gamma/beta pairs at
TPU_LN_PAR; the exp table halves and the unary LUT above that. The exporter
lays tables out in exactly that shape so a config load is a plain copy.
"""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .kernels_int import qlinear
from .numerics import INT8_MAX, INT8_MIN, Requant
from .tiling import LANES, Buffers, tile_gemm

MAGIC = b"TPU1"
VERSION = 1

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

# The RTL's buffers (tinytpu.sv). sw/tiling.py sized these.
BUFFERS = Buffers(act_words=512, wgt_words=3072, out_words=512)

DESC_WORDS = 24
DESC_FIELDS = ("op", "a_dram", "a_words", "b_dram", "b_words", "src_a", "src_b",
               "dst", "shape", "cfg_dram", "cfg_dst", "cfg_n", "vmult", "vmult_b",
               "vshift", "eps_lo", "eps_hi", "out_dram", "out_words", "check_dram",
               "check_n", "name_off", "pad0", "pad1")
HEADER_BYTES = 64


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
    out_words: int = 0
    check_dram: int = 0
    check_n: int = 0
    name_off: int = 0

    def pack(self) -> bytes:
        vals = [getattr(self, f) & 0xFFFFFFFF for f in DESC_FIELDS[:-2]] + [0, 0]
        return struct.pack("<24I", *vals)


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
    _refs: list[tuple[int, str, int, int]] = field(default_factory=list)  # (desc, field, chunk, +bytes)

    def data(self, raw: bytes | np.ndarray) -> int:
        """Append bytes, 16-byte aligned so the DMA can start on them. Returns a handle."""
        b = np.ascontiguousarray(raw).tobytes() if isinstance(raw, np.ndarray) else bytes(raw)
        pad = (-len(b)) % 16
        self._chunk_off.append(self._data_len)
        self._chunks.append(b + b"\0" * pad)
        self._data_len += len(b) + pad
        return len(self._chunks) - 1

    def name(self, s: str) -> int:
        if s not in self._name_off:
            self._name_off[s] = len(self._names)
            self._names += s.encode() + b"\0"
        return self._name_off[s]

    def add(self, d: Desc, **dram_refs: int | tuple[int, int]) -> Desc:
        """Append a descriptor.

        `dram_refs` maps a field to a data handle, or to (handle, byte offset)
        for a slice of a chunk -- an m tile's rows inside a whole tensor.
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
            setattr(self.descs[idx], fld, self.base + data_off + self._chunk_off[h] + off)
        for d in self.descs:
            d.name_off = self.base + names_off + d.name_off if d.name else 0

        hdr = struct.pack("<4s7I", MAGIC, VERSION, n, desc_off, names_off, data_off,
                          total, self.base)
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
        return total


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


def emit_gemm(blob: Blob, name: str, x_q: np.ndarray, x_dram: int | None,
              w_q: np.ndarray, bias_q: np.ndarray, rq: Requant,
              dst_word: int = 0, check: bool = True) -> np.ndarray:
    """`y = requant(x_q @ w_q + bias)` as the tiles gemm_seq can run.

    x_q is [M][K] int8 row-major -- the activation buffer's layout, so an m
    tile is a contiguous slice of rows. w_q is [K][N] int8, the weight
    buffer's layout for one n tile; wider N is cut into n tiles here and each
    tile's [K][Nt] is stored contiguously, since the buffer wants it that way
    and the weights are ours to arrange.

    If x_dram is None the activations are placed in the blob's data area (a
    probe, or a first layer). Otherwise it is the DRAM address of the
    row-major tensor an earlier op left behind.

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

    y_q = qlinear(x_q, w_q.T, bias_q.astype(np.int32), rq)
    x_h = blob.data(x_q) if x_dram is None else None
    k_words = k // LANES

    for n0 in range(0, n, t.n_tile):
        nt = min(t.n_tile, n - n0)
        w_h = blob.data(np.ascontiguousarray(w_q[:, n0:n0 + nt]))
        if rq.is_per_channel:
            rq_tile = Requant(multiplier=np.asarray(rq.multiplier)[n0:n0 + nt],
                              shift=np.asarray(rq.shift)[n0:n0 + nt])
        else:
            rq_tile = rq
        cfg_h = blob.data(per_channel_config(bias_q[n0:n0 + nt], rq_tile, nt))

        for i, m0 in enumerate(range(0, m, t.m_tile)):
            mt = min(t.m_tile, m - m0)
            d = Desc(op=OP_GEMM, name=f"{name}[{m0}:{m0 + mt},{n0}:{n0 + nt}]",
                     a_words=mt * k_words,
                     src_a=src(R_ACT, 0), src_b=src(R_WGT, 0), dst=dst_word,
                     shape=gemm_shape(mt, k_words, nt // LANES))
            refs: dict[str, int | tuple[int, int]] = {}
            if x_h is None:
                d.a_dram = x_dram + m0 * k
            else:
                refs["a_dram"] = (x_h, m0 * k)
            if i == 0:
                d.b_words = k * (nt // LANES)
                refs["b_dram"] = w_h
                d.cfg_dst, d.cfg_n = 0, nt * 4
                refs["cfg_dram"] = cfg_h
            if check:
                exp = np.ascontiguousarray(y_q[m0:m0 + mt, n0:n0 + nt])
                d.check_n = exp.size
                refs["check_dram"] = blob.data(exp)
            blob.add(d, **refs)
    return y_q


# --------------------------------------------------------------------- probes

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


def main() -> None:
    ap = argparse.ArgumentParser(description="write a tiny-tpu program blob")
    ap.add_argument("-o", type=Path, required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--tokens", type=int, default=82, help="82 = 126x126 input")
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--program", choices=("gemm-probe",), default="gemm-probe")
    a = ap.parse_args()

    from .frontend.dav2 import load_state_dict
    sd = load_state_dict(a.checkpoint) if a.checkpoint else load_state_dict()

    blob = Blob()
    if a.program == "gemm-probe":
        build_gemm_probe(blob, sd, a.tokens, a.seed)
    total = blob.write(a.o)
    n_ops = sum(d.op < 0x10 for d in blob.descs)
    print(f"wrote {a.o}: {total} bytes, {n_ops} hardware ops, "
          f"{len(blob.descs)} descriptors")


if __name__ == "__main__":
    main()
