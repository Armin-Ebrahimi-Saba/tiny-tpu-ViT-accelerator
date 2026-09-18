# ABOUTME: Reads a written blob back the way the on-board driver does and checks every address it names.
# ABOUTME: The exporter and rvlab's tpu_runtime.c share this format; a drift shows up here before the board.

import struct

import numpy as np
import pytest

from sw.export_tpu import (BUFFERS, DESC_FIELDS, DESC_WORDS, HEADER_BYTES, MAGIC, OP_END,
                           OP_GEMM, OP_QADD, OP_TRANSPOSE, R_ACT, R_WGT, TPU_BLOB_ADDR, VERSION,
                           Blob, emit_gemm, emit_qadd, emit_transpose, gemm_shape, src)
from sw.kernels_int import qadd
from sw.kernels_int import qlinear
from sw.numerics import Requant
from sw.tiling import LANES


def read_blob(raw: bytes):
    """The driver's view: header, descriptors as dicts, and a byte fetch by DRAM address."""
    magic, version, n, desc_off, names_off, data_off, total, base, arena = struct.unpack_from("<4s8I", raw)
    assert magic == MAGIC and version == VERSION and total == len(raw) and base == TPU_BLOB_ADDR
    descs = []
    for i in range(n):
        vals = struct.unpack_from(f"<{DESC_WORDS}I", raw, desc_off + i * DESC_WORDS * 4)
        descs.append(dict(zip(DESC_FIELDS, vals)))

    def fetch(addr: int, nbytes: int) -> bytes:
        off = addr - base
        assert data_off <= off and off + nbytes <= total, "descriptor points outside the data area"
        return raw[off:off + nbytes]

    def name(d) -> str:
        off = d["name_off"] - base
        return raw[off:raw.index(b"\0", off)].decode()

    return descs, fetch, name


def small_gemm(m=40, k=64, n=32, seed=1):
    rng = np.random.default_rng(seed)
    x_q = rng.integers(-127, 128, (m, k), dtype=np.int8)
    w_q = rng.integers(-127, 128, (k, n), dtype=np.int8)
    bias_q = rng.integers(-1000, 1000, n).astype(np.int32)
    rq = Requant.from_real_multiplier(np.full(n, 1 / 300.0))
    return x_q, w_q, bias_q, rq


def test_layout_and_end_marker(tmp_path):
    blob = Blob()
    x_q, w_q, bias_q, rq = small_gemm()
    emit_gemm(blob, "g", x_q, None, w_q, bias_q, rq)
    total = blob.write(tmp_path / "b.bin")
    raw = (tmp_path / "b.bin").read_bytes()
    assert len(raw) == total
    descs, _, name = read_blob(raw)
    assert descs[-1]["op"] == OP_END and name(descs[-1]) == "end"
    _, _, _, _, _, data_off, _, _ = struct.unpack_from("<4s7I", raw)
    assert data_off % 16 == 0 and data_off >= HEADER_BYTES + len(descs) * DESC_WORDS * 4
    for d in descs[:-1]:
        for f in ("a_dram", "b_dram", "cfg_dram", "check_dram"):
            assert d[f] == 0 or d[f] % 16 == 0, f"{f} not DMA-aligned"


def test_every_tile_names_the_bytes_the_hardware_wants(tmp_path):
    """Activation rows, [K][Nt] weights, [c][4] config, and the emulator's expected tile."""
    blob = Blob()
    x_q, w_q, bias_q, rq = small_gemm(m=40, k=64, n=32)
    y_q = emit_gemm(blob, "g", x_q, None, w_q, bias_q, rq)
    assert np.array_equal(y_q, qlinear(x_q, w_q.T, bias_q, rq))
    blob.write(tmp_path / "b.bin")
    descs, fetch, name = read_blob((tmp_path / "b.bin").read_bytes())
    gemms = [d for d in descs if d["op"] == OP_GEMM]
    assert gemms, "no GEMM descriptors"
    m, k = x_q.shape
    covered = np.zeros(m, dtype=bool)
    for d in gemms:
        mt, kt, nt = d["shape"] & 0xFFF, (d["shape"] >> 12) & 0xFF, d["shape"] >> 20
        assert kt == k // LANES and nt == 32 // LANES
        assert d["a_words"] == mt * kt and d["src_a"] == src(R_ACT, 0) and d["src_b"] == src(R_WGT, 0)
        assert d["shape"] == gemm_shape(mt, kt, nt)
        a = np.frombuffer(fetch(d["a_dram"], mt * k), dtype=np.int8).reshape(mt, k)
        rows = [m0 for m0 in range(m) if np.array_equal(x_q[m0:m0 + mt], a)]
        assert rows, f"{name(d)}: activation bytes are not a row slice of x"
        m0 = rows[0]
        covered[m0:m0 + mt] = True
        if d["b_dram"]:
            assert d["b_words"] == k * nt
            b = np.frombuffer(fetch(d["b_dram"], k * 32), dtype=np.int8).reshape(k, 32)
            assert np.array_equal(b, w_q)
            cfg = np.frombuffer(fetch(d["cfg_dram"], d["cfg_n"] * 4), dtype=np.uint32).reshape(-1, 4)
            assert d["cfg_dst"] == 0 and cfg.shape[0] == 32
            assert np.array_equal(cfg[:, 0].astype(np.int32), bias_q)
            assert np.array_equal(cfg[:, 1], np.asarray(rq.multiplier))
            assert np.array_equal(cfg[:, 2], np.asarray(rq.shift))
        exp = np.frombuffer(fetch(d["check_dram"], d["check_n"]), dtype=np.int8).reshape(mt, 32)
        assert np.array_equal(exp, y_q[m0:m0 + mt])
    assert covered.all()
    # Weights and config load once per n tile: the first tile only.
    assert [bool(d["b_dram"]) for d in gemms] == [True] + [False] * (len(gemms) - 1)


def test_weights_split_into_n_tiles(tmp_path):
    blob = Blob()
    x_q, w_q, bias_q, rq = small_gemm(m=8, k=32, n=128)
    emit_gemm(blob, "wide", x_q, None, w_q, bias_q, rq)
    blob.write(tmp_path / "b.bin")
    descs, fetch, name = read_blob((tmp_path / "b.bin").read_bytes())
    loads = [d for d in descs if d["op"] == OP_GEMM and d["b_dram"]]
    seen = np.zeros(128, dtype=bool)
    for d in loads:
        nt = (d["shape"] >> 20) * LANES
        b = np.frombuffer(fetch(d["b_dram"], 32 * nt), dtype=np.int8).reshape(32, nt)
        n0 = next(n0 for n0 in range(0, 128, LANES) if np.array_equal(w_q[:, n0:n0 + nt], b))
        assert not seen[n0:n0 + nt].any()
        seen[n0:n0 + nt] = True
    assert seen.all()


def test_arena_tensors_sit_above_the_blob_and_tiles_land_in_place(tmp_path):
    """An activation the ops leave for each other is addressed above `total`,
    an m tile reads its rows of it, and a result tile is written back at
    (m0, n0) with the tensor's row stride."""
    blob = Blob()
    m, k, n = 50, 384, 256          # 21-row m tiles, 128-wide n tiles
    x_q, w_q, bias_q, rq = small_gemm(m=m, k=k, n=n)
    x_ref = blob.scratch(m * k)
    y_ref = blob.scratch(m * n)
    emit_gemm(blob, "ext", x_q, x_ref, w_q, bias_q, rq, out_ref=y_ref, check=False)
    total = blob.write(tmp_path / "b.bin")
    assert struct.unpack_from("<I", (tmp_path / "b.bin").read_bytes(), 32)[0] == m * k + m * n
    gemms = [d for d in blob.descs if d.op == OP_GEMM]
    assert len(gemms) >= 4
    x_base, y_base = TPU_BLOB_ADDR + total, TPU_BLOB_ADDR + total + m * k
    seen = set()
    for d in gemms:
        mt, nt = d.shape & 0xFFF, (d.shape >> 20) * LANES
        m0 = (d.a_dram - x_base) // k
        assert d.a_dram == x_base + m0 * k
        n0 = d.out_dram - y_base - m0 * n
        assert 0 <= n0 < n and n0 % LANES == 0
        assert (d.out_cols, d.out_rows, d.out_stride) == (nt, mt, n)
        seen.add((m0, n0))
    assert len(seen) == len(gemms)
    assert {m0 for m0, _ in seen} == set(range(0, m, gemms[0].shape & 0xFFF))


def test_vector_ops_chunk_to_the_buffers_and_check_the_emulator(tmp_path):
    blob = Blob()
    rng = np.random.default_rng(3)
    n = BUFFERS.act_words * LANES * 2 + 256          # three chunks
    a = rng.integers(-127, 128, n, dtype=np.int8)
    b = rng.integers(-127, 128, n, dtype=np.int8)
    a_ref, b_ref, y_ref = blob.data(a), blob.data(b), blob.scratch(n)
    y = emit_qadd(blob, "add", a, a_ref, 0.5, b, b_ref, 0.25, 0.7, y_ref)
    assert np.array_equal(y, qadd(a, b, Requant.from_real_multiplier(0.5 / 0.7),
                                  Requant.from_real_multiplier(0.25 / 0.7)))
    kt = emit_transpose(blob, "kt", a[:96 * 64].reshape(96, 64), a_ref, blob.scratch(96 * 64))
    assert kt.shape == (64, 96)
    blob.write(tmp_path / "b.bin")
    descs, fetch, name = read_blob((tmp_path / "b.bin").read_bytes())
    adds = [d for d in descs if d["op"] == OP_QADD]
    assert [d["a_words"] for d in adds] == [BUFFERS.act_words, BUFFERS.act_words, 16]
    e0 = 0
    for d in adds:
        cnt = d["a_words"] * LANES
        assert d["src_b"] == src(R_WGT, 0) and d["b_words"] == d["a_words"]
        assert np.array_equal(np.frombuffer(fetch(d["a_dram"], cnt), dtype=np.int8), a[e0:e0 + cnt])
        assert np.array_equal(np.frombuffer(fetch(d["check_dram"], cnt), dtype=np.int8), y[e0:e0 + cnt])
        assert (d["out_cols"], d["out_rows"]) == (cnt, 1)
        e0 += cnt
    tr = next(d for d in descs if d["op"] == OP_TRANSPOSE)
    assert tr["shape"] == 64 | (96 << 16)
    assert np.array_equal(np.frombuffer(fetch(tr["check_dram"], 96 * 64), dtype=np.int8).reshape(64, 96), kt)


def test_too_large_for_the_buffers_is_refused():
    blob = Blob()
    x_q, w_q, bias_q, rq = small_gemm(m=8, k=16 * (BUFFERS.wgt_words + 16), n=16)
    with pytest.raises(ValueError):
        emit_gemm(blob, "huge", x_q, None, w_q, bias_q, rq)
