# ABOUTME: Checks the buffer-aware tiler against hand-computable cases and its own monotonicity.
# ABOUTME: The buffer sizes in the RTL are chosen from this model, so it has to be right about capacity.

import pytest

from sw.tiling import LANES, Buffers, tile_gemm, vits_block_gemms


def test_small_gemm_fits_in_one_call():
    """Everything resident: one call, and every byte crosses the bus exactly once."""
    t = tile_gemm("tiny", 32, 32, 32, Buffers(4096, 4096, 4096))
    assert t.fits and t.calls == 1
    assert t.m_tile == 32 and t.n_tile == 32
    assert t.weight_bytes == 32 * 32
    assert t.act_bytes == 32 * 32
    assert t.out_bytes == 32 * 32
    assert t.weight_passes == 1.0


def test_weight_buffer_is_indexed_by_k_row():
    """K words per 16 output channels, not K/16 -- the thing fc2 runs into."""
    assert tile_gemm("ok", 16, 512, 16, Buffers(4096, 512, 4096)).fits
    bad = tile_gemm("too deep", 16, 513, 16, Buffers(4096, 512, 4096))
    assert not bad.fits and "513 weight words" in bad.reason


def test_activation_buffer_must_hold_one_row():
    bad = tile_gemm("wide k", 16, 4096, 16, Buffers(16, 4096, 4096))
    assert not bad.fits and "activation words" in bad.reason


def test_more_buffer_never_costs_more_traffic():
    """A larger buffer's set of legal tilings contains the smaller one's."""
    prev = None
    for w in (2048, 3072, 4096, 8192):
        t = tile_gemm("fc2", 1370, 1536, 384, Buffers(512, w, 512))
        assert t.fits
        if prev is not None:
            assert t.dram_bytes <= prev
        prev = t.dram_bytes


def test_tiles_are_whole_lanes_and_cover_the_shape():
    for name, m, k, n in vits_block_gemms(tokens=137):
        t = tile_gemm(name, m, k, n, Buffers(512, 3072, 512))
        assert t.fits, f"{name}: {t.reason}"
        assert t.n_tile % LANES == 0
        assert 1 <= t.m_tile <= m and t.n_tile <= n
        assert t.m_tile * (k // LANES) <= 512 or k < LANES
        assert k * (t.n_tile // LANES) <= 3072


@pytest.mark.parametrize("wgt,unfit", [(512, 7), (3072, 0)])
def test_vits_block_needs_a_bigger_weight_buffer(wgt, unfit):
    """The sizing decision behind WGT_WORDS in tinytpu.sv, kept honest."""
    tiles = [tile_gemm(*g, Buffers(512, wgt, 512)) for g in vits_block_gemms()]
    assert sum(not t.fits for t in tiles) == unfit
