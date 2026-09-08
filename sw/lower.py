# ABOUTME: Lowers graph ops to array-sized tiles, checks them against the machine limits, and costs them.
# ABOUTME: Answers "does this fit and how long does it take" before any RTL exists to measure.

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ir import Graph, Op
from .machine import MachineSpec, UnsupportedOpError


@dataclass(frozen=True)
class GemmShape:
    """Every array op reduces to one of these: [M,K] x [K,N] -> [M,N], batched."""

    m: int
    k: int
    n: int
    batch: int = 1

    @property
    def macs(self) -> int:
        return self.m * self.k * self.n * self.batch


@dataclass
class TiledOp:
    name: str
    op_type: str
    gemm: GemmShape | None
    macs: int
    cycles: int
    weight_bytes: int
    activation_bytes: int
    dram_bytes: int
    notes: str = ""


@dataclass
class Schedule:
    machine: MachineSpec
    ops: list[TiledOp] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_macs(self) -> int:
        return sum(o.macs for o in self.ops)

    @property
    def total_cycles(self) -> int:
        return sum(o.cycles for o in self.ops)

    @property
    def total_dram_bytes(self) -> int:
        return sum(o.dram_bytes for o in self.ops)

    @property
    def seconds(self) -> float:
        return self.total_cycles / (self.machine.clock_mhz * 1e6)

    @property
    def utilization(self) -> float:
        """Fraction of peak MAC throughput the schedule achieves."""
        ideal = self.total_macs / self.machine.macs_per_cycle
        return ideal / self.total_cycles if self.total_cycles else 0.0

    @property
    def dram_seconds(self) -> float:
        rate = float(self.machine.dram["read_bytes_per_cycle"]) * self.machine.clock_mhz * 1e6
        return self.total_dram_bytes / rate if rate else float("inf")

    def hottest(self, n: int = 10) -> list[TiledOp]:
        return sorted(self.ops, key=lambda o: o.cycles, reverse=True)[:n]

    def text(self, top: int = 10) -> str:
        lines = [
            f"schedule for {self.machine.name} ({self.machine.rows}x{self.machine.cols} array "
            f"@ {self.machine.clock_mhz:g} MHz)",
            f"  ops              {len(self.ops)}",
            f"  MACs             {self.total_macs / 1e9:.2f} G",
            f"  cycles           {self.total_cycles / 1e9:.3f} G",
            f"  compute time     {self.seconds:.1f} s/frame",
            f"  array efficiency {self.utilization * 100:.1f} % of peak",
            f"  DRAM traffic     {self.total_dram_bytes / (1 << 20):.1f} MiB/frame "
            f"({self.dram_seconds:.1f} s at {self.machine.dram['read_bytes_per_cycle']} B/cycle)",
            f"  bound by         {'DRAM' if self.dram_seconds > self.seconds else 'compute'}",
            "",
            f"  {'op':38s} {'type':18s} {'MACs':>10s} {'cycles':>12s}  {'M x K x N':>18s}",
        ]
        for o in self.hottest(top):
            g = f"{o.gemm.m}x{o.gemm.k}x{o.gemm.n}" if o.gemm else "-"
            lines.append(
                f"  {o.name[:38]:38s} {o.op_type:18s} {o.macs / 1e6:9.1f}M {o.cycles:12,d}  {g:>18s}"
            )
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


def _gemm_cycles(g: GemmShape, machine: MachineSpec) -> int:
    """Cycles for a weight-stationary tiled GEMM.

    Weights are tiled [rows x cols] over the K and N axes. For each tile the M
    activation rows stream through, plus a fill/drain of rows+cols. Weight loads
    overlap with compute when the array is double buffered, which the machine
    description declares; otherwise they add rows cycles per tile.
    """
    rows, cols = machine.rows, machine.cols
    k_tiles = -(-g.k // rows)
    n_tiles = -(-g.n // cols)
    fill = rows + cols
    per_tile = g.m + fill
    if not machine.array.get("double_buffered_weights", False):
        per_tile += rows
    return int(g.batch * k_tiles * n_tiles * per_tile)


def _vector_cycles(elements: int, machine: MachineSpec, per_element: int = 1) -> int:
    """Vector unit processes one array-column of lanes per cycle."""
    lanes = machine.cols
    return int(-(-elements // lanes) * per_element)


def _shape_of(graph: Graph, tensor: str, shapes: dict[str, tuple[int, ...]]) -> tuple[int, ...]:
    if tensor in shapes:
        return shapes[tensor]
    if tensor in graph.initializers:
        return graph.initializers[tensor].shape
    raise KeyError(f"unknown shape for tensor {tensor!r}")


def infer_shapes(graph: Graph, input_shapes: dict[str, tuple[int, ...]]) -> dict[str, tuple[int, ...]]:
    """Propagate shapes by running the graph on zero-filled arrays.

    Cheaper to be exact than to reimplement shape rules per op, and it cannot
    disagree with the executor because it *is* the executor.
    """
    from .execute import run_float

    shapes: dict[str, tuple[int, ...]] = {}

    def record(name: str, value: np.ndarray) -> None:
        shapes[name] = tuple(value.shape)

    inputs = {k: np.zeros(v, dtype=np.float32) for k, v in input_shapes.items()}
    run_float(graph, inputs, observer=record)
    for name, arr in graph.initializers.items():
        shapes[name] = tuple(arr.shape)
    return shapes


def lower_graph(graph: Graph, machine: MachineSpec,
                input_shapes: dict[str, tuple[int, ...]]) -> Schedule:
    """Tile every op onto the machine and cost it. Raises on anything unsupported."""
    shapes = infer_shapes(graph, input_shapes)
    sched = Schedule(machine=machine)
    ub_bytes = int(machine.unified_buffer["bytes"])

    for op in graph.ops:
        machine.require(op.type)
        sched.ops.append(_lower_op(op, graph, shapes, machine, ub_bytes, sched))
    return sched


def _lower_op(op: Op, graph: Graph, shapes: dict[str, tuple[int, ...]],
              machine: MachineSpec, ub_bytes: int, sched: Schedule) -> TiledOp:
    out_shape = shapes[op.outputs[0]]
    out_elems = int(np.prod(out_shape))
    in_shape = _shape_of(graph, op.inputs[0], shapes)
    gemm: GemmShape | None = None
    weight_bytes = 0
    notes = ""

    if op.type == "conv2d":
        w = graph.initializers[op.inputs[1]]
        o, c, kh, kw = w.shape
        oh, ow = out_shape[2], out_shape[3]
        gemm = GemmShape(m=oh * ow, k=c * kh * kw, n=o)
        weight_bytes = int(w.size)
        notes = f"im2col {kh}x{kw} stride {op.attr('stride', 1)}"

    elif op.type == "conv_transpose2d":
        w = graph.initializers[op.inputs[1]]
        i, o, kh, kw = w.shape
        h, ww = in_shape[2], in_shape[3]
        # stride == kernel: each input pixel produces one disjoint kh*kw output
        # block, so it is a GEMM into O*kh*kw columns followed by a pure scatter.
        gemm = GemmShape(m=h * ww, k=i, n=o * kh * kw)
        weight_bytes = int(w.size)
        notes = f"scatter {kh}x{kw}"

    elif op.type == "matmul":
        w = graph.initializers[op.inputs[1]]
        n_out, k = w.shape
        m = int(np.prod(in_shape[:-1]))
        gemm = GemmShape(m=m, k=k, n=n_out)
        weight_bytes = int(w.size)

    elif op.type == "batch_matmul":
        b_shape = _shape_of(graph, op.inputs[1], shapes)
        batch = int(np.prod(in_shape[:-2]))
        gemm = GemmShape(m=in_shape[-2], k=in_shape[-1], n=b_shape[-1], batch=batch)
        notes = "activation x activation (no DRAM weights)"

    if gemm is not None:
        cycles = _gemm_cycles(gemm, machine)
        macs = gemm.macs
    else:
        per_element = {"softmax": 3, "layernorm": 4, "gelu": 1, "relu": 1,
                       "add": 1, "mul": 1, "interpolate": 2}.get(op.type)
        if per_element is None:
            # Pure data movement: reshape/transpose/slice/concat cost a UB pass.
            per_element = 1
            notes = "data movement"
        cycles = _vector_cycles(out_elems, machine, per_element)
        macs = 0

    act_bytes = out_elems  # int8

    # The unified buffer holds two weight tiles (double buffered) plus one block
    # of activation rows in and out. Whatever is left sets how many rows of the
    # M axis can be in flight; each extra block re-streams the op's weights.
    dram_bytes = weight_bytes
    if gemm is not None:
        tile_bytes = 2 * machine.rows * machine.cols
        per_row = machine.rows + machine.cols
        m_block = (ub_bytes - tile_bytes) // per_row
        if m_block < 1:
            sched.warnings.append(
                f"{op.name}: the {ub_bytes} B unified buffer cannot hold two "
                f"{machine.rows}x{machine.cols} weight tiles plus a single activation row"
            )
            m_block = 1
        m_blocks = -(-gemm.m // m_block)
        dram_bytes = weight_bytes * m_blocks
        if m_blocks > 1:
            notes = (notes + "; " if notes else "") + f"M blocked {m_blocks}x ({m_block} rows)"

    return TiledOp(
        name=op.name, op_type=op.type, gemm=gemm, macs=macs, cycles=cycles,
        weight_bytes=weight_bytes, activation_bytes=act_bytes,
        dram_bytes=dram_bytes, notes=notes,
    )


def run_tiled_gemm(a_q: np.ndarray, b_q: np.ndarray, machine: MachineSpec) -> np.ndarray:
    """Execute [M,K] x [K,N] the way the array does: K/N tiles accumulated in int32.

    This exists so the tiling model can be *proved* equivalent to the whole-op
    result rather than assumed to be. Partial sums across K tiles accumulate in
    the int32 accumulator; if that ordering ever stopped being exact, this would
    diverge from the reference and the test would catch it.
    """
    m, k = a_q.shape
    k2, n = b_q.shape
    if k != k2:
        raise ValueError(f"inner dimensions disagree: {k} vs {k2}")
    rows, cols = machine.rows, machine.cols
    out = np.zeros((m, n), dtype=np.int64)
    for k0 in range(0, k, rows):
        a_tile = a_q[:, k0:k0 + rows].astype(np.int64)
        for n0 in range(0, n, cols):
            b_tile = b_q[k0:k0 + rows, n0:n0 + cols].astype(np.int64)
            out[:, n0:n0 + cols] += a_tile @ b_tile
    lo, hi = machine.accum_min, machine.accum_max
    if out.size and (int(out.min()) < lo or int(out.max()) > hi):
        raise OverflowError("tiled GEMM overflowed the declared accumulator width")
    return out


def check_supported(graph: Graph, machine: MachineSpec) -> list[str]:
    """Return the op types this machine cannot run. Empty means the graph compiles."""
    missing = sorted({op.type for op in graph.ops if not machine.supports(op.type)})
    return missing


def require_supported(graph: Graph, machine: MachineSpec) -> None:
    missing = check_supported(graph, machine)
    if missing:
        raise UnsupportedOpError(
            f"machine {machine.name!r} cannot run this graph; missing ops: {missing}"
        )
