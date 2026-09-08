# ABOUTME: Fits a GEMM into fixed on-chip buffers and reports the tiling and the DRAM traffic it costs.
# ABOUTME: Answers "what buffer geometry does ViT-S need", which lower.py's roofline cost model cannot see.

"""Buffer-aware tiling.

`lower.py` costs a schedule against peak MACs and DRAM bandwidth. It cannot say
whether an op *fits*, because it has no model of the on-chip buffers. That is
the question that decides everything about a real-sized block: with 8 KB
buffers a ViT-S projection's weight matrix is 18x too large to be resident, so
the loop order stops being an optimisation and starts being the difference
between running and not running.

The geometry the buffers actually impose, in 128-bit words of LANES=16 int8:

    activations   Mt * (K / LANES)     one word per (row, k tile)
    weights       K  * (Nt / LANES)     one word per (k, n tile)
    results       Mt * (Nt / LANES)     one word per (row, n tile)

The weight buffer is the asymmetric one: it is indexed by k *row*, not by k
tile, so it holds K words per 16 output channels. That is 16x the activation
buffer's appetite for the same K, and it is what fc2 runs into.

`gemm_seq` accumulates over the k tiles that are resident, so a K larger than
the weight buffer needs partial sums carried between calls -- which the engine
cannot do, because it requantizes to int8 on the way out. So Kc = K is a hard
constraint here, not a choice, and a shape that cannot meet it is reported as
not fitting rather than silently tiled.

Traffic assumes weights and activations both start in DRAM and the result is
written back once. Two loop orders are available and they trade which operand
is re-read:

    N outer, M inner   weights resident per n tile, activations re-read per n tile
    M outer, N inner   activations resident per m tile, weights re-read per m tile
"""

from __future__ import annotations

from dataclasses import dataclass

LANES = 16


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class Buffers:
    """Capacities in 128-bit words. The RTL parameters of the same name."""

    act_words: int
    wgt_words: int
    out_words: int

    @property
    def bytes(self) -> int:
        return (self.act_words + self.wgt_words + self.out_words) * LANES


@dataclass(frozen=True)
class Tiling:
    name: str
    m: int
    k: int
    n: int
    m_tile: int
    n_tile: int
    order: str          # "M-inner" or "N-inner"
    calls: int
    weight_bytes: int   # read from DRAM, counting re-reads
    act_bytes: int
    out_bytes: int
    fits: bool
    reason: str = ""

    @property
    def dram_bytes(self) -> int:
        return self.weight_bytes + self.act_bytes + self.out_bytes

    @property
    def weight_passes(self) -> float:
        """How many times the whole weight matrix crosses the bus."""
        return self.weight_bytes / (self.k * self.n)


def tile_gemm(name: str, m: int, k: int, n: int, buf: Buffers) -> Tiling:
    """Pick the tiling with the least DRAM traffic that fits, or say why none does."""
    k_words = _ceil_div(k, LANES)

    # The weight buffer has to hold a whole K for one n tile, at minimum.
    if k > buf.wgt_words:
        return Tiling(name, m, k, n, 0, 0, "-", 0, 0, 0, 0, False,
                      f"K={k} needs {k} weight words for even one n tile, "
                      f"buffer holds {buf.wgt_words}")
    if k_words > buf.act_words:
        return Tiling(name, m, k, n, 0, 0, "-", 0, 0, 0, 0, False,
                      f"K={k} needs {k_words} activation words for even one row, "
                      f"buffer holds {buf.act_words}")

    best: Tiling | None = None
    # n tiles are whole LANES-wide columns; m tiles are rows, blocked by MBLK
    # inside the sequencer, so any count is legal.
    for nt in range(LANES, n + 1, LANES):
        n_words = nt // LANES
        if k * n_words > buf.wgt_words:
            break
        mt = min(m, buf.act_words // k_words, buf.out_words // n_words)
        if mt < 1:
            continue

        n_tiles, m_tiles = _ceil_div(n, nt), _ceil_div(m, mt)
        calls = n_tiles * m_tiles
        w_once, a_once, o_once = k * nt * n_tiles, m * k, m * n

        for order, w_bytes, a_bytes in (
            ("M-inner", w_once, a_once * n_tiles),
            ("N-inner", w_once * m_tiles, a_once),
        ):
            cand = Tiling(name, m, k, n, mt, nt, order, calls,
                          w_bytes, a_bytes, o_once, True)
            # Least traffic wins; ties go to fewer calls, because per-op setup
            # is not free -- report.md 6.12 measured it at 27% of a block.
            if best is None or (cand.dram_bytes, cand.calls) < (best.dram_bytes, best.calls):
                best = cand

    if best is None:
        return Tiling(name, m, k, n, 0, 0, "-", 0, 0, 0, 0, False,
                      "no n tile leaves room for a single row of activations")
    return best


def vits_block_gemms(tokens: int = 1370, e: int = 384, heads: int = 6,
                     hid: int = 1536) -> list[tuple[str, int, int, int]]:
    """The GEMMs of one DINOv2 ViT-S block, as [M,K,N].

    q/k/v are one GEMM each per head rather than a fused projection, and the
    output projection is split across heads and summed -- both forced by the
    array's operand ports, and both recorded in report.md 6.11.
    """
    d = e // heads
    gemms = []
    for h in range(heads):
        for t in ("q", "k", "v"):
            gemms.append((f"{t}{h}", tokens, e, d))
        gemms.append((f"logit{h}", tokens, d, tokens))
        gemms.append((f"ctx{h}", tokens, tokens, d))
        gemms.append((f"proj{h}", tokens, d, e))
    gemms.append(("fc1", tokens, e, hid))
    gemms.append(("fc2", tokens, hid, e))
    return gemms


def block_report(buf: Buffers, tokens: int = 1370, depth: int = 12,
                 macs_per_cycle: int = 256, clock_mhz: float = 50.0,
                 dram_bytes_per_cycle: int = 8) -> str:
    """One ViT-S block against one buffer geometry, with both bounds named.

    The two numbers that matter are at the bottom: the DRAM time the tiling
    costs, and the compute floor the array imposes no matter what the tiling
    does. Buffers only need to be large enough that the first is below the
    second; past that, growing them buys nothing.
    """
    gemms = vits_block_gemms(tokens=tokens)
    tiles = [tile_gemm(*g, buf) for g in gemms]
    unfit = [t for t in tiles if not t.fits]

    out = [f"buffers: act {buf.act_words} / wgt {buf.wgt_words} / out {buf.out_words} "
           f"words = {buf.bytes // 1024} KB",
           "",
           f"{'op':8s} {'M':>5s} {'K':>5s} {'N':>5s} {'m tile':>7s} {'n tile':>7s} "
           f"{'order':9s} {'calls':>6s} {'DRAM MB':>8s}"]
    for t in tiles:
        if not t.fits:
            out.append(f"{t.name:8s} {t.m:5d} {t.k:5d} {t.n:5d}   does not fit: {t.reason}")
        else:
            out.append(f"{t.name:8s} {t.m:5d} {t.k:5d} {t.n:5d} {t.m_tile:7d} {t.n_tile:7d} "
                       f"{t.order:9s} {t.calls:6d} {t.dram_bytes / 1e6:8.2f}")

    if unfit:
        out += ["", f"{len(unfit)} of {len(tiles)} ops do not fit; totals below are "
                    "for the rest and mean nothing"]

    dram = sum(t.dram_bytes for t in tiles if t.fits)
    macs = sum(t.m * t.k * t.n for t in tiles)
    dram_s = depth * dram / (dram_bytes_per_cycle * clock_mhz * 1e6)
    comp_s = depth * macs / (macs_per_cycle * clock_mhz * 1e6)
    bound = "DRAM" if dram_s > comp_s else "compute"

    out += ["",
            f"per block: {dram / 1e6:.1f} MB of DRAM traffic, {macs / 1e9:.2f} GMAC",
            f"{depth} blocks at {clock_mhz:.0f} MHz:",
            f"  DRAM    {dram_s:5.2f} s  ({dram_bytes_per_cycle} B/cycle)",
            f"  compute {comp_s:5.2f} s  ({macs_per_cycle} MAC/cycle)",
            f"  -> {bound}-bound at {max(dram_s, comp_s):.2f} s/image"]
    return "\n".join(out)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--act", type=int, default=512, help="activation buffer, 128-bit words")
    ap.add_argument("--wgt", type=int, default=3072, help="weight buffer, 128-bit words")
    ap.add_argument("--out", type=int, default=512, help="result buffer, 128-bit words")
    ap.add_argument("--tokens", type=int, default=1370)
    ap.add_argument("--macs-per-cycle", type=int, default=256)
    ap.add_argument("--clock-mhz", type=float, default=50.0)
    a = ap.parse_args()
    print(block_report(Buffers(a.act, a.wgt, a.out), tokens=a.tokens,
                       macs_per_cycle=a.macs_per_cycle, clock_mhz=a.clock_mhz))
