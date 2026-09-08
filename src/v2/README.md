# `src/v2` — tpu-v2 inference RTL

New RTL implementing the contract in [`sw/machines/tpu_v2.json`](../../sw/machines/tpu_v2.json).
It lives alongside the existing `src/*.sv` rather than replacing it: those files are a
16-bit fixed-point *training* accelerator (loss, gradient descent, leaky-ReLU
derivative) with its own SVA and `mnist_demo` flow, and nothing here disturbs them.

The reference implementation is the emulator in [`sw/`](../../sw). Every module below
is checked bit-for-bit against it by [`test/v2/run.sh`](../../test/v2/run.sh) — the Python
is the specification, the Verilog is the thing being verified.

## Modules

| File | What it is | Verified against |
|---|---|---|
| `pe_int8.sv` | Weight-stationary PE: `int8 × int8 → int32` MAC, double-buffered weight | — |
| `systolic_int8.sv` | Parameterized `ROWS × COLS` array, exact int32 accumulation | `exact_int_matmul` |
| `requant_unit.sv` | `sat(round_half_up(acc · mult / 2^shift))`, per-channel constants | `Requant.apply` |
| `gemm_tile_int8.sv` | Array + per-column requant: one complete GEMM tile | `qlinear` |
| `gemm_seq.sv` | Block sequencer: tiles a full M×K×N GEMM onto the array | `qlinear` |
| `divider.sv` / `divider_pipe.sv` | Floor division; the pipelined one is 1 result/cycle | Python `//` |
| `isqrt.sv` | Exact `floor(sqrt(x))`, restoring digit-by-digit | `_isqrt` |
| `unary_lut.sv` | 256-entry int8→int8 activation table | `apply_unary_lut` |
| `softmax_int.sv` | Row max, two-stage exp LUT, sum, divide, requant | `qsoftmax` |
| `layernorm_int.sv` | Exact centering, variance, rsqrt, affine tail | `qlayernorm` |
| `qadd_unit.sv` | Residual add with 12 guard bits | `qadd` |
| `vpu_seq.sv` | Vector sequencer: streams buffer words through any of the four kernels, plus a transpose that is address generation only | `qsoftmax`, `qlayernorm`, `qadd`, `apply_unary_lut`, `.T` |

## Feeding contract

The array is pure structure: every timing decision belongs to `gemm_seq.sv`, which is
now the control unit. The sequence it implements is below, and
`test/v2/tb_systolic_int8.sv` remains the executable description of it for the bare
array:

1. **Weight load.** Hold `accept_w[c]` for `ROWS` cycles while driving `weight_in[c]`
   in **reverse k order** (`w[ROWS-1][c]` first, `w[0][c]` last). `accept_w` is a
   column-wide *broadcast*, not a propagating pulse: the shadow registers themselves
   are the shift chain, each PE handing its own shadow value south.
2. **Switch.** Pulse `switch_in` one cycle *after* the last weight beat. PE(r,c) is
   still writing `w[r]` into its shadow on cycle `ROWS+r`, so a switch landing on that
   cycle promotes the previous k's weight. The pulse then walks the diagonal, reaching
   PE(r,c) at `SWITCH_CYC + r + c`.
3. **Activations.** Present `x[m][k]` on `data_in[k]` at cycle `SWITCH_CYC + 1 + k + m`
   — row k lags row 0 by k cycles, which is what lines each activation up with the
   partial sum descending its column.
4. **Bias.** Hold `bias_in[n]` for the whole pass; it enters the top of column n and
   rides down with the partial sum, so no separate adder stage is needed.

Results leave column `c` in `m` order, `psum_valid[c]` marking each one. `active_cols`
holds columns `>= active_cols` in reset so a wide array can run a narrow tile.

### Three bugs this contract encodes

All three were found by the vector tests, and all three are easy to reintroduce:

- **Weight fan-out.** If a PE passes its *input* south instead of its own shadow
  register, every PE in the column sees the whole weight stream and they all end the
  load holding `w[0]`. The array still produces plausible numbers — and still passes
  any test whose weights happen to be constant along k, since a permutation of k
  cannot change a sum of identical terms.
- **Switch/load race.** Switching on the final load beat is off by one weight, again
  invisible under constant-along-k weights.
- **Synchronous-read skew.** When activations come from a buffer rather than a
  testbench, the data trails its address by a cycle. Raising the valid flag with the
  address instead of with the data puts every activation one cycle early, against the
  wrong partial sum. The array's own tests cannot catch this — they drive it directly,
  with no memory in the loop.

`test/v2/gen_vectors.py` therefore keeps its saturation cases confined to single rows
and columns and leaves everything else random.

## Reset

Every register here resets synchronously: `always_ff @(posedge clk)` with `if (rst)`
inside, never `@(posedge clk or posedge rst)`. On the FPGA the reset is released
synchronously after the MMCM locks, and Xilinx flops power up to their INIT value
anyway, so an asynchronous reset buys nothing — and it costs a great deal. DSP48 and
BRAM registers have synchronous reset only, so a register with an asynchronous one
cannot be absorbed into the block it feeds. Ten of these modules originally had async
resets, and Vivado reported 612 DRC and 374 methodology violations for them, every one
of the form "this is preventing the possibility of merging these registers into the DSP
block".

## The two sequencers

`gemm_seq.sv` and `vpu_seq.sv` are the only modules here that decide *when* anything
happens; everything else is pure arithmetic with a handshake. Both present the same
shape to the SoC — a start pulse, a few config words, `busy`/`done` — which is what lets
`rvlab/src/rtl/student/tinytpu.sv` dispatch between them on one register write.

Both read operand A, read operand B, and write a destination, and all three ports carry a
base word index (`cfg_a_word` / `cfg_b_word` / `cfg_d_word`) that the sequencer adds at the
port while its own counters stay relative. Placement is therefore not a property of the
tiling, which is what lets an intermediate stay where the op that produced it left it
instead of being copied back into an operand buffer between ops — the difference between
26 KB and 8 KB of bus traffic for one transformer block. The SoC wrapper adds a region
select on the two read ports, so "where it already is" can be the result buffer.

The three ports are muxed rather than arbitrated, because `op.code` guarantees only one
sequencer runs at a time. Its fifth
opcode, the transpose, has no kernel behind it at all — it is a read address pattern and
nothing else, walking a column of the source while the write side walks a row of the
destination. It exists because attention needs Kᵀ and the array reads its stationary
operand row-major, so without it a transformer block cannot be expressed. Its element
rate is one per two cycles — the buffer read is registered, and a new read is only
issued once the outstanding element is known to be consumed. Overlapping the two needs
the read pipeline to know a kernel's back-pressure a cycle early, which softmax's
three-phase `in_ready` does not offer.

## Not yet built

`tpu_v2.json` still describes more than this: im2col address generation, bilinear
upsample, and any instruction encoding at all. Weight streaming from DRAM and the
register map both landed (`tinytpu.hjson`, `tinytpu_wdma.sv`); what remains on the path
to a whole network is address generation for convolutions and a way to express a
sequence of ops other than one register write per op.
