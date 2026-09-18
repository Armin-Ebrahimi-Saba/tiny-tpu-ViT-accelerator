# tiny-tpu → Depth Anything V2 on the Nexys Video (XC7A200T)

Working log of the reasoning behind each step, per `task.md`. Newest section last.

---

## 1. Goal and constraints

From `task.md`:

- Run **Depth Anything V2 inference**, smallest variant, on **Xilinx Artix-7 XC7A200T**
- **Model weights live in DDR3**
- tiny-tpu is eventually **integrated into the rvlab project**
- Reason about each step; document it here

The model is **DA-V2-Small**: a DINOv2 **ViT-S/14** encoder (`embed_dim=384`, `num_heads=6`,
`head_dim=64`, `patch_size=14`) plus a DPT depth head — 24,785,089 parameters, confirmed
in `sw/frontend/dav2.py`. At 518×518 the patch grid is 37×37, giving **1370 tokens**.

**First milestone (agreed): one transformer block, end-to-end in SoC simulation.**
Not the whole model. One block exercises every subsystem except the DPT head — GEMM,
softmax, LayerNorm, GELU, residual adds, DDR3 weight streaming, and the CPU/accelerator
handshake — at 7 % of the weight volume. Building the ISA and memory system against a
real, complete workload beats guessing at them from the full model.

---

## 2. Where the work stood

`sw/` is a complete, tested int8 toolchain (3,835 lines, 105 tests): a bit-accurate
quantized emulator, a DA-V2 importer, calibration, graph lowering, and an accuracy
harness. The fp32 graph matches HuggingFace DA-V2 at **131 dB SQNR**; the int8 pipeline
reaches **15.90 dB SQNR / Pearson 0.937 / δ<1.25 = 81.4 %** with 99.9-percentile
calibration.

`sw/machines/tpu_v2.json` is the hardware contract, and it says so itself: *"The RTL in
src/ does NOT implement this yet; this file is the requirements contract the RTL must
satisfy, and the emulator in sw/ is the bit-exact reference it must match."*

The pre-existing `src/*.sv` is a **2×2, 16-bit fixed-point training accelerator** (loss,
gradient descent, leaky-ReLU derivative). It is not a step toward int8 inference, so it
is left untouched — along with its SVA and the `mnist_demo` flow — and the new work goes
in `src/v2/`.

---

## 3. Step 1 — the GEMM tile (done, verified, synthesized)

**Reasoning.** Everything else in the accelerator is plumbing around a correct integer
MAC array. Build that first, prove it bit-exact against the emulator, and the rest can be
developed against a foundation that is known-good rather than suspected-good.

Built in `src/v2/`:

| Module | Verified bit-exactly against |
|---|---|
| `pe_int8.sv` | — (int8 × int8 → int32 weight-stationary PE) |
| `systolic_int8.sv` | `exact_int_matmul` |
| `requant_unit.sv` | `Requant.apply` |
| `gemm_tile_int8.sv` | `qlinear` |

`test/v2/run.sh` regenerates vectors from the emulator and runs three Verilator
testbenches over seven geometries (16×16, 4×7, 7×3, degenerate K=1 and N=1, M=1, M=16).
All pass. **The Python is the specification; the Verilog is what is under test.**

**Synthesis, out-of-context on `xc7a200tfbg484-1`, one 16×16 tile:**

| Metric | Value | Of device |
|---|---|---|
| LUTs | 38,489 | 28.6 % |
| Flip-flops | 15,553 | 5.8 % |
| DSPs | 64 | 8.7 % |
| BRAM | 0 | 0 % |
| WNS @ 20 ns | +8.567 ns → **~87 MHz** | clears the 50 MHz target |

The critical path is the requant unit's 65-bit rounding shift and saturate (28 logic
levels, 21 CARRY4) — pipelinable if it ever binds. Note the 256 MACs mapped entirely to
LUTs, which is nearly all of the 38 k; a DSP48E1 packs two int8 products, so DSP mapping
should cut LUTs several-fold. Deferred until the array competes with the buffer for area.

### Two bugs found, both in the RTL

Both are recorded in `src/v2/README.md` because they are easy to reintroduce:

1. **Weight fan-out.** Each PE passed its *input* south instead of its own shadow
   register, so every PE in a column consumed the whole weight stream and all ended the
   load holding `w[0]`. Fixed by making the shadow registers the shift chain and
   broadcasting `accept_w` across the column.
2. **Switch/load race.** The switch pulse landed on the final weight beat, promoting the
   previous k's weight.

Both produce plausible numbers, and **both are invisible to any test whose weights are
constant along k** — a permutation of k cannot change a sum of identical terms. The first
vector set forced whole columns to `INT8_MIN`/`INT8_MAX` for saturation coverage, and
exactly those columns passed while every random column failed. `gen_vectors.py` now
confines saturation cases to single lanes. *Coverage that looks adversarial can be the
least sensitive part of the test.*

---

## 4. Step 2 — reading the target platform

Before writing integration RTL, the actual constraints. From `rvlab/docs/design_ref/`:

**Board and SoC.** Nexys Video, XC7A200T. CV32E40P (RV32IMC), TL-UL bus, split into
`xbar_main` (fast) and `xbar_peri` (slow). **System clock 50 MHz**, from an MMCM off a
100 MHz crystal.

**DDR3.** 512 MB, via the open-source **UberDDR3** controller, behind a 256-bit-wide
512-entry (16 KB) direct-mapped last-level cache, wrapped by a TL-UL adapter. The docs
note the adapter *"can be removed"* if a project needs more bandwidth.

Two numbers in `tpu_v2.json` were already written to this board and match exactly:
`clock_mhz: 50` and `dram.bytes: 536870912`. Good — the spec needs no revision here.

**Memory map** (`docs/design_ref/memory_map.rst`):

| Start | Size | Device |
|---|---|---|
| `0x10000000` | 16 MB | `student_device_peri` |
| `0x20000000` | 256 MB | `student_device_fast` |
| `0x80000000` | 512 MB | External DDR3 |

**The `student` module already has exactly the three ports tiny-tpu needs:**

- `tl_device_peri` — device port on `xbar_peri`, for control/status registers
- `tl_device_fast` — device port on `xbar_main`, for activation traffic
- **`tl_host` — host port on `xbar_main`**, letting the student module *master* the bus
  and read DDR3 itself. This is the weight-streaming path, and `student_dma.sv` is a
  working example of using it.

**Pinout — already solved by the platform.** `src/design/xdc/rvlab_fpga_top.xdc` carries
the 100 MHz oscillator (`clk_100mhz_i` on R4, with `create_clock`), the full JTAG group
with its input/output delays, and the LEDs; `src/design/xdc/rvlab_ddr.xdc` carries the
complete DDR3 pin map, SSTL15 standards, terminations, and generated clocks. A
`src/design/pincheck/pincheck.csv` gate validates I/O before bitstream generation. This
retires what was previously the largest board bring-up unknown: **tiny-tpu needs no pin
constraints of its own**, because it is an internal peripheral and every external pin
already belongs to rvlab.

**Build flow.** PyDesignFlow. Two facts that shape the integration: RTL is auto-globbed
from `rtl/*/*.sv` (one level deep — files must sit directly in `src/rtl/student/`), and
reggen auto-discovers `src/design/reggen/*.hjson`. **Adding a peripheral requires no flow
edits.** `sim_rtl_xsim` exists, so Questa is not required.

### Bandwidth sizing — the decision this drove

One ViT-S/14 block:

| Sub-layer | MACs |
|---|---|
| qkv projection (384→1152) | 606 M |
| attention logits (6 heads) | 721 M |
| attention × V | 721 M |
| output projection (384→384) | 202 M |
| MLP fc1 (384→1536) | 808 M |
| MLP fc2 (1536→384) | 808 M |
| **total** | **≈ 3.87 GMAC** |

At 256 MACs/cycle and 50 MHz that is **~300 ms per block**. Its weights are
384·1152 + 384·384 + 2·384·1536 ≈ **1.77 M int8 ≈ 1.8 MB**, which at 32 bits/beat over
TL-UL streams in **~9 ms**.

**Compute-bound by ~33×.** So the TL-UL host port is entirely adequate and there is no
need to bypass the adapter for the 256-bit LLC. That removes the single largest piece of
risk from the memory system, and it is worth stating plainly because the docs invite the
opposite choice.

Two more sizing facts that constrain the design:

- 1.8 MB of weights per block **cannot** live on-chip (365 BRAM tiles ≈ 1.6 MB total).
  DDR3 streaming is mandatory, not an optimization — which is exactly what `task.md`
  requires anyway.
- Activations are 1370 × 384 int8 = **526 KB**, against a 64 KB unified buffer. So
  **M-axis blocking is exercised for real** by this milestone, not deferred.

### Array size

Kept at **16×16**, matching `tpu_v2.json` and the already-verified tile. A 32×32 array
would cut the 300 ms to ~75 ms, but LUT-mapped it needs ~154 k LUTs (114 % of the device)
— it requires DSP-mapped PEs first, and it changes the buffer and DDR3 budget throughout.
Get the memory system and control path correct at 16×16, then scale the datapath.

---

## 5. Planned architecture

Given permission to remove unneeded rvlab modules, tiny-tpu takes **all three** student
ports outright:

```
student.sv
  ├── tinytpu_regs   (peri @0x10000000)  control / status / descriptors, via reggen
  ├── tinytpu_core   (fast @0x20000000)  activation SRAM + 16x16 array + VPU
  └── tinytpu_wdma   (tl_host)  ────────► DDR3 @0x80000000   weight streaming
```

`student_rlight` (an LED exercise) and `student_dma` (a memset demo) are removed. This is
not just tidying: **it deletes a prerequisite.** Both tiny-tpu and `student_dma` want the
single `tl_host` port, which would have forced implementing `student_tlul_mux.sv` — today
an empty exercise stub — as an arbiter first. With the DMA gone, tiny-tpu owns `tl_host`
outright and no mux is needed. The `student_rlight`/`student_tlul_mux` testbenches and
their flow targets go with them.

The CPU orchestrates: it writes a descriptor naming the DDR3 address of a layer's
weights, the tile shape, and the requant constants; the accelerator streams weights,
runs, and raises `irq_o`. `src/sw/project/main.c` drives one transformer block and checks
the result against vectors exported from the emulator.

---

## 6. Status and open items

**Done:** the datapath, the SoC seam and one whole transformer block, every intermediate
bit-exact against the emulator, running on the CV32E40P over the real bus; and a routed
bitstream for the Nexys Video that closes timing at 50 MHz using half the XC7A200T's
LUTs and a quarter of its registers. Platform analysis complete; bandwidth question
settled. rvlab Python toolchain installed (`pydesignflow`, `hjson`, `mako`, `lxml`,
`notcl`); `flow` runs and lists all targets.

**Not done:** the network. One block is not 12 blocks, and the DPT head needs im2col
address generation that does not exist yet. Weights still come from BRAM in simulation
because the fast batch flow builds without the DDR3 model; the descriptor for a
`0x8xxxxxxx` source is identical and the crossbar routes it without the DMA knowing, but
that path has not been run.

### 6.1 Toolchain resolved, integration proven

**RISC-V GCC installed.** xPack `riscv-none-elf-gcc` 15.2.0 (v15.2.0-1), extracted to
`/home/armin/Public/xpack-riscv-none-elf-gcc-15.2.0-1` and symlinked into
`~/.local/bin`. Note `/home/armin/Public/riscv-none-elf-gcc-xpack` was already on `PATH`
but is a clone of the xPack *build-scripts* repo with no `bin/` — which is why the
toolchain appeared configured yet never resolved.

**rvlab pruned.** Removed `student_rlight`, `student_dma`, `student_tlul_mux`, their two
testbenches, `student_dma.hjson`, the `rlight`/`dma`/`coremark` programs, and the
`student_rlight_tb` wave configs. Updated `flow/__init__.py`, `src/sw/include/rvlab.h`,
and rewrote `student.sv` around the tiny-tpu port plan. `test_rvlab` was kept — `Sources`
needs it for BRAM init (`swinit`).

**Two fixes this required:**

- The unused device ports are terminated with `tlul_err_resp`, not tied off. An
  unanswered TL-UL request would hang the CPU forever; an error response fails visibly.
- xsim rejects `'{default: '0}` on a struct containing an enum (`tl_a_op_e`), so the idle
  `tl_host_o` literal lists every field explicitly.

**Added `sim_rtl_xsim_batch`.** The existing `sim_rtl_xsim` task launches xsim with
`--gui`, so it never self-terminates and cannot be scripted. The batch variant makes
automated verification (and CI) possible; it was needed immediately to verify the pruning.

**Result — the pruned SoC boots and passes.** `flow systb_minimal.sim_rtl_xsim_batch`:
`test_idcode` pass, `test_dtmcs` pass, `test_sw` pass, `hostio: Hello!`, return value 0.
Ten "Simulation object ... not found" warnings from wave configs still naming `rlight_i`
were removed. The three remaining warnings are pre-existing CV32E40P `unique case`
warnings at time 0 (before reset) — third-party, left alone per `task.md`.

**GELU landed.** `src/v2/unary_lut.sv`, a 256-entry int8→int8 table verified bit-exact
against `apply_unary_lut()` over 512 probes including both int8 endpoints. Indexing is
`x_q + 128`, which on a two's-complement byte is just the sign bit inverted — no adder.
Contents load at runtime, so one instance serves GELU, leaky-ReLU, or any other
elementwise function without respinning the bitstream.

### 6.2 Step 3 — the divider, and a throughput trap

Softmax and LayerNorm both reduce to one integer division per element, so the divider was
built and verified first. Two properties of it needed deciding, and both were decided by
reading the reference rather than by intuition.

**Floor, not truncation.** Python's `//` rounds toward −infinity; SystemVerilog's `/`, like
C's, truncates toward zero. They differ on every inexact negative: `-7 // 2` is `-4`, but
`-7 / 2` is `-3`. This is not an edge case in LayerNorm — its numerator is
`d = N·x − Σx`, which is negative for every below-average channel, so roughly half of all
elements would be off by one. `divider.sv` therefore divides magnitudes and applies an
explicit correction: negate, and add one more only when the remainder is non-zero.

**Pipelined, not sequential.** The plan called for a shared *sequential* divider. Sizing it
against one ViT-S/14 block shows why that was wrong:

| | divisions per block | at 32 cycles each | at 1/cycle |
|---|---|---|---|
| softmax `(e << 15) // total` | 11,261,400 | 7.2 s | 0.225 s |
| LayerNorm `(d << 14) // r` | 1,052,160 | 0.67 s | 0.021 s |

All the GEMMs in the same block take ~300 ms. A sequential divider would have made
division ~95% of the runtime — the array would spend its time waiting on the vector unit.
Unrolling the restoring loop into one stage per quotient bit gives one result per cycle for
roughly `NUM_BITS × (DEN_BITS+1)` LUTs, negligible next to the 38 k the array already uses.

The cheap alternative — reciprocate once per row, then multiply — was rejected deliberately.
It does not reproduce a per-element floor division, and the whole method here is that
`sw/kernels_int.py` is the specification, not something to approximate.

Both modules are in the sweep: `divider.sv` (sequential, small, for low-rate config math)
and `divider_pipe.sv` (the datapath one). Each passes 512/512 vectors against Python `//`,
including exact and inexact negatives, both int64 endpoints, division by one and by self,
and softmax/LayerNorm-shaped operands. The pipelined testbench drives back to back with no
gaps, so a stage leaking state between operands would show up. Division by zero reports on
a `div_zero` flag instead of hanging.

### 6.3 Step 4 — integer softmax

`src/v2/softmax_int.sv` reproduces `qsoftmax()` exactly, verified over 8 rows of 64 int16
logits including two rows the random draw would never produce: a flat row, where every
term is identical and the quotient must land exactly on `2**15 / length`, and a one-hot
row, where every other term exp-underflows toward zero.

Three design points, all forced by the reference rather than chosen:

**The exp table is split in two.** A single 256-entry ROM covers int8 logits but not the
int16 attention logits, whose useful range of `d = max - x` spans tens of thousands.
Factoring `exp(-s(256·hi + lo))` into `exp(-s·256·hi) · exp(-s·lo)` replaces one
impossible 65536-entry ROM with two 256-entry ROMs and one multiply. Both tables carry the
same uniform scale factor and it cancels in the normalization, so the split is free of
accuracy cost.

**Subtracting the row max first** is what makes the table one-sided: the exponent argument
is always ≤ 0, so `d` is unsigned. It is also what bounds `total` — every term is ≤ 2¹⁵,
so a 1370-token row needs 27 bits, not a guess.

**Three passes, and the row must be buffered.** Max, then exp-and-sum, then divide-and-
requant; each pass needs the previous one's reduction, and `total` is not known until the
last element has been seen, so the row cannot stream straight through. At 1370 tokens the
two buffers are ~2.7 KB and ~5.5 KB — a couple of BRAMs. That costs ~3 cycles per element,
about 0.68 s per block for all six heads, against ~300 ms for the block's GEMMs. It is not
yet the bottleneck the sequential divider would have been, but it is the next thing worth
overlapping once the sequencer exists, since the exp pass of one row can run under the
divide pass of the previous one.

A zero denominator means the exp table underflowed for the entire row. The reference raises
there, so the RTL asserts in simulation rather than silently emitting zeros.

### 6.4 Step 5 — integer LayerNorm, and the rsqrt that could not be copied

`src/v2/layernorm_int.sv` reproduces `qlayernorm()` exactly over 8 rows of 48 int8
activations, including a constant row — the only input that drives the variance to zero and
so the only one that reaches the `max(r, 1)` clamp.

**The square root is where the reference could not be transcribed.** `_isqrt()` seeds Newton
from a float64 `sqrt` and then refines in integer arithmetic. RTL has no float64, and
substituting a cheap power-of-two seed does not work either: Newton's convergence depends on
the seed, and from `2**31` a 62-bit input can need ~30 iterations to settle — as expensive as
just doing it properly.

What makes a substitution legal is the *tail* of the reference, not its seed:

```python
r = r - 1  if  r*r > x
r = r + 1  if  (r+1)*(r+1) <= x
```

Those two comparisons pin the exact `floor(sqrt(x))` for any `r` already within one of it.
The reference's actual contract is therefore "exact floor sqrt" — the float seed is an
implementation detail of reaching it. `isqrt.sv` computes that directly by the classic
restoring digit-by-digit method: two bits of radicand per step, one bit of root, 32 steps,
no seed and no convergence question. 512/512 vectors match, clustered on `k²−1 / k² / k²+1`
where `floor(sqrt)` steps.

It stays sequential deliberately. LayerNorm needs one root per *row* — 2740 per transformer
block, against ~1.05 M divisions — so the sqrt is three orders of magnitude rarer than the
divide and does not justify unrolling.

**Two details of the formulation matter and are easy to get wrong.** The mean is never
computed: multiplying through by N keeps `d = N·x − Σx` an exact integer, where a rounded
mean would inject an error into every element before the variance is even taken. And the
input scale cancels inside `(x − mean)/sqrt(var)`, so the entire normalization runs in the
raw quantized domain — only the affine tail needs a scale, with `sqrt(N)/2¹⁴` folded into
that multiplier by the host.

Saturation happens exactly once, after beta is added. `requant_unit` is instantiated with a
full int32 output range, which turns it into `apply_keep(acc, 0)` — rescale and round, do
not clip. Clipping the intermediate would change the result.

Gamma and beta ride a delay line alongside their element rather than being re-read at the
far end of the divider. The latency is fixed, so this is exact and needs no reasoning about
which cycle a synchronous read would land on.

### 6.5 Step 6 — residual add

`qadd_unit.sv` matches `qadd()` over 512 vectors including both int8 extremes on both
operands. The 12 guard bits are the entire point: rescaling each operand to int8 before
adding would inject up to half an LSB per add, and DA-V2 Small has **24 residual adds in
series**, where the errors accumulate down the skip chain rather than cancelling. Keeping
sub-LSB precision through the sum and rounding once costs a wider adder and nothing else.

`apply_keep(x, 12)` shifts by `shift − 12`, a *signed* amount — a real multiplier above 1
gives a small shift and the reference then shifts left. Residual scale ratios sit near 1 and
the multiplier is normalized to [2³⁰, 2³¹), so `shift` lands near 30 and that branch never
fires in practice; it is implemented anyway, because the reference has it.

### 6.6 Vector unit status

Every kernel one transformer block needs is now in RTL and bit-exact against
`sw/kernels_int.py`. The sweep runs 7 geometries × 9 testbenches:

| module | checks against | status |
|---|---|---|
| `pe_int8` / `systolic_int8` | `exact_int_matmul` | 7 shapes |
| `requant_unit` | `Requant.apply` | 512 vectors |
| `gemm_tile_int8` | `qlinear` | 7 shapes |
| `unary_lut` | `apply_unary_lut` (GELU) | 512 lookups |
| `divider` / `divider_pipe` | Python `//` | 512 each |
| `isqrt` | `_isqrt` | 512 |
| `softmax_int` | `qsoftmax` | 8 rows × 64 |
| `layernorm_int` | `qlayernorm` | 8 rows × 48 |
| `qadd_unit` | `qadd` | 512 |

### 6.7 Step 7 — the block sequencer

`src/v2/gemm_seq.sv` runs a full M×K×N GEMM on the fixed 16×16 array, replacing the
hand-written schedule that until now existed only inside `test/v2/tb_systolic_int8.sv`. It
is checked against `qlinear()` across the sweep, on shapes that exercise every loop —
40×45×30 (3 k tiles, 2 n tiles, 3 m blocks), 33×48×48, 64×64×16, and the degenerate
1×16×16 and 17×1×1.

**Why M is blocked.** The reduction over k spans several tiles, so int32 partial sums must
survive between them. Holding them for every row at once would need `M × COLS × 4` bytes —
87 KB at M=1370, which does not fit on chip. Blocking M to MBLK rows bounds that to
`MBLK × COLS × 4` (2 KB at MBLK=32) and costs only re-loading the same weight tile once per
block, amortized over MBLK rows of work. This is the M-axis blocking the plan called for,
and it is not decorative: activations for one block are 526 KB against a 64 KB buffer.

**Why n is the outer loop.** The array is weight-stationary and weights are the expensive
operand to fetch from DRAM. With n outermost and k innermost, each weight tile is loaded
once per m block; with m outermost it would be loaded once per (m block, n tile) pair. The
activations get re-read instead, and those are already on chip.

**Bias enters the array, not the accumulator** — it rides down the column with the partial
sum, so it is driven on the first k tile and zeroed afterwards. Driving it every tile would
add it once per tile.

Padding is the host's job: K and N are padded up to whole tiles with zero weights, which
contribute nothing to the sum. That is what a compiler does anyway, and it means every tile
in the sweep is a full 16×16 tile with no partial-tile case in the RTL.

**The bug worth recording.** The first version failed on every shape, including a single
tile — off by small amounts, which pointed at the schedule rather than the tiling. The
activation buffer is read *synchronously*, so data trails its address by one cycle, but the
`av` valid flag was being raised in the same cycle the address went out. Row 0 therefore
reached lane 0 one cycle early and every activation was skewed against the wrong partial
sum. The weight path had been written correctly (`wv` already trailed by one); only the
activation path was wrong, which is why the array's own tests never caught it — they drive
the array directly, with no memory in the loop.

A second failure during the sweep turned out to be in the *test generator*, not the RTL:
`gen_gemm_seq` was inheriting the sweep's `-k/-n` array geometry, so vectors tiled for a
4×7 array were being run on the 16×16 sequencer. Worth stating plainly because the symptom —
a stable ~15% of outputs off by one — looks exactly like a rounding bug in hardware.

### 6.8 Step 8 — the SoC seam: register map, peripheral, driver

tiny-tpu is now a live peripheral on the rvlab SoC, and the CV32E40P runs a GEMM on it and
checks the result against the emulator:

```
[pass] test_idcode
[pass] test_dtmcs
hostio: tinytpu: id ok
hostio: tinytpu: PASS 24x32x32 GEMM, 192 words bit-exact vs qlinear() (39 poll spins)
Execution finished. Return value: 0
[pass] test_sw
```

**Two ports, two jobs.** `tinytpu.hjson` puts only control and status on the slow
peripheral port (`0x10000000`): an id magic, a command register, status, an opcode, and the
GEMM shape. Everything bulk — activations, weights, results, and the per-output-channel
requantization constants — is a memory window on the fast port (`0x20000000`), split into
four regions by the top two address bits.

That split is not tidiness, it is setup cost. A 1536-channel GEMM needs a bias, a multiplier
and a shift per output channel. As MMIO register writes that is ~4600 accesses over the slow
bus before any work can start; as a memory window the CPU just stores them, and later the
weight DMA can fill the same window straight from DDR3 with the CPU out of the loop
entirely.

**The width change.** `tlul_adapter_sram` speaks 32 bits and only 32 bits, but the array
wants a whole 16-byte tile row per cycle. `tinytpu_buf.sv` resolves that by building each
buffer from four 32-bit banks on a shared word address: the CPU addresses `{word, lane}`
and hits one bank, the engine reads or writes all four and sees 128 bits. The two ports are
independent — a true dual-port BRAM — and nothing arbitrates between them, because software
fills the buffers while the engine is idle and reads results after `done`.

**Integration, not duplication.** `rvlab/src/rtl/tinytpu` is a symlink to `src/v2`, so the
verified RTL is compiled into the SoC from its own source rather than copied. Only the two
SoC-coupled files (`tinytpu.sv`, `tinytpu_buf.sv`) live under `rvlab/src/rtl/student/`,
which keeps `src/v2` free of any `tlul_pkg` dependency and so still buildable by
`test/v2/run.sh` on its own.

#### The bug that mattered: a test that passed for the wrong reason

The first end-to-end run failed on exactly one word of 192 — row 0, columns 0–3. Adding two
diagnostic reads of that word made the comparison pass, which is the shape of an
environmental problem rather than a compute one. Probing eight reads showed all eight
returning zero while the very same addresses read back correctly moments later in the
comparison loop.

The cause was in the driver. reggen emits `*_MASK` as the **field width**, not a mask in
place: `TINYTPU_STATUS_DONE_MASK` is `0x1` with `TINYTPU_STATUS_DONE_LSB` `0x1`. So
`status & TINYTPU_STATUS_DONE_MASK` reads *busy*, and the poll fell through the instant the
engine started. The comparison then raced the engine — and the eight slow `printf` calls in
the diagnostic version gave the engine enough time to finish, which is precisely why adding
probes "fixed" it.

Worth recording because of the failure mode, not the fix: with the probes in place the test
reported **PASS while never actually waiting for the hardware**. The driver now extracts the
field properly and asserts that the spin count is non-zero, so a poll that does not wait
fails loudly instead of passing silently.

A second, smaller issue on the way: reading a never-written buffer location put X on the
bus, and TL-UL does not contain it — the adapters assert on unknown response data and the X
propagated through the crossbar FIFOs, killing the simulation 53 assertions deep. The
buffers now zero-initialize (an INIT string on real BRAM, so free) and default their read
registers, matching what `rvlab_bram_main.sv` already does.

### 6.9 The weight DMA

Until now every weight byte reached the array through a CPU load/store pair. That is fine
for a 24x32x32 test and hopeless for the real thing: one ViT-S block holds about 1.8 MB of
int8 weights, which is both far more than the on-chip buffers hold and far more than a
50 MHz in-order core should be copying by hand. `tinytpu_wdma.sv` closes that gap by taking
over the student host port on the main crossbar.

**The descriptor.** Three registers -- `dma_src` (a byte address on the main crossbar),
`dma_dst` (a 128-bit word index plus a two-bit region select) and `dma_len` (a count of
128-bit words) -- then `ctrl.dma_start`. The destination is deliberately *not* an address:
it names a word inside one of the fast-port regions, so the descriptor says nothing about
the SoC memory map on the write side and the same encoding survives a change to the
aperture. In the driver this replaced 256 stores with five register writes.

**Why the source side is just an address.** The DMA does not know DDR3 exists. It issues
TL-UL `Get`s and the crossbar routes them by address, so `0x80000000` reaches DDR3,
`0x00000000` reaches main BRAM, and both work without a line of difference in the engine.
That is what makes the end-to-end test possible at all: the fast batch simulation builds
without the DDR3 model (`srcs_noddr` leaves `WITH_EXT_DRAM` undefined, and
`rvlab_tlul_ddr.sv` puts a `tlul_err_resp` in the controller's place), so the test sources
its weights from BRAM. The descriptor for a DDR3 source is identical.

**Width, again.** TL-UL carries 32 bits; the buffers store 128. The DMA reads four words,
assembles them, and commits one buffer word -- five cycles of state per word, of which four
are bus round trips. It shares the engine-side write port with the array through a plain
mux rather than an arbiter, on the same reasoning the buffers already document: software
runs the DMA while the engine is idle, and `status.busy` / `status.dma_busy` make that
checkable rather than assumed.

**The throughput this does not fix.** One request is outstanding at a time.
`tlul_adapter_host` drives `a_source` to a constant zero, and TL-UL permits only one
in-flight request per source id, so pipelining needs a source counter and a reorder buffer
in the DMA. At 50 MHz that currently means four bytes per bus round trip rather than four
bytes per cycle -- 95 poll spins to move 1 KB in the test. It is the obvious next
optimisation, and it changes throughput without changing behaviour, which is why it waited.

**Errors do not hang.** An error response still advances the transfer and sets a sticky
`status.dma_err`. Stalling on it would leave `dma_busy` asserted forever and hang the
driver's poll -- the failure mode would be a dead SoC rather than a diagnosable one. (In
simulation `tlul_adapter_host` also carries its own assertion against error responses, so
a DMA from an unmapped address is loud twice over.)

**One extra check in the driver.** After the DMA reports done, the CPU reads
`TPU_WGT[0]` back through its own aperture and compares it. The GEMM result would catch a
DMA that did nothing, but not a DMA and a CPU that disagree about how the buffer is
addressed -- that shows up as a wrong answer with no indication of which side is wrong.
The same reasoning gives the `dma_spins == 0` guard as the one on the GEMM poll.

The full run:

```
hostio: tinytpu: id ok
hostio: tinytpu: PASS 24x32x32 GEMM, 192 words bit-exact vs qlinear() (95 DMA spins, 39 poll spins)
Execution finished. Return value: 0
[pass] test_sw
```

### 6.10 The vector unit at the register map

Every kernel a transformer block needs was already built and verified in isolation. What
was missing was a way to *run* one: softmax, layernorm, residual add and the activation
LUT are all one-element-per-cycle stream processors, and the buffers hand out 128 bits at
a time. `src/v2/vpu_seq.sv` is the serializer on one side and the packer on the other,
plus the state machine that counts rows.

**One dispatch point.** `ctrl.start` now branches on `op.code`: 0 is the GEMM, 1-4 are the
vector ops. `status.busy` and `status.done` are the OR of both engines, so the driver's
poll is unchanged and a caller that only ever runs GEMMs cannot tell the difference. That
single-dispatch rule is also what makes the buffer ports a mux instead of an arbiter --
only one engine can be running, by construction.

**Three fixed ports, and why there is no region select.** The vector unit reads operand A
from the activation buffer, operand B from the weight buffer, and writes to the result
buffer -- exactly the ports the GEMM uses. There is deliberately no way to point an op at
an arbitrary region, because the weight DMA already does that job better: the fast
aperture is itself a device on the main crossbar, so a DMA whose source is `0x2xxxxxxx`
copies one region into another with the CPU out of the loop. That is how a transformer
block will pass results from one op into the next, and it means the vector unit needs no
addressing logic of its own.

**Tables are bulk data, so they go where bulk data goes.** A 1024-entry gamma/beta pair
and three 256-entry lookup tables would be 1792 MMIO register writes. Instead the config
region is split again: the top quarter holds the vector unit's tables (gamma/beta in the
lower half, the two exp-LUT halves and the activation LUT in the upper), and the GEMM's
per-channel constants keep the lower three quarters -- 3072 channels against a MAX_N of
2048, so nothing was given up. gamma and beta commit on the beta write, the same contract
the GEMM's shift write already had.

**A byte mask on the write port.** Rows are packed flat, so a run of `rows * len` int8
elements almost never ends on a 16-byte boundary. Writing the tail word whole would
clobber whatever followed it in the buffer. `tinytpu_buf` therefore gained a per-byte
engine-side write mask; the GEMM and the DMA drive it all-ones.

**Testing the mask specifically.** The testbench pre-fills the destination with `0xAA` and
checks that every byte past the last element still reads `0xAA`. Without that, a mask
stuck at all-ones passes every element comparison -- the corruption is entirely outside
the range anyone was comparing. The vector geometries are chosen so `rows * len` is *not*
a multiple of 16 for three of the four ops.

**Distinct base words per op.** Each op in the testbench gets its own source and
destination word, because a sequencer that ignored the base register would still pass on
whichever op happened to run first. Same class of mistake as the polling bug in 6.8: the
test has to be able to fail.

Softmax is fed sign-extended int8 rather than the int16 its port allows. int8 is what a
requantized GEMM produces and what the next op consumes, so it is the only width the SoC
seam ever sees; `tb_softmax_int` keeps the int16 path covered.

Element rate is one per two cycles, for the same structural reason the DMA moves four
bytes per round trip: the buffer read is registered, and a new read is only issued once
the outstanding element is known to be consumed. Overlapping them needs the read pipeline
to know a kernel's back-pressure a cycle early, which softmax's three-phase `in_ready`
does not offer.

Verilator, all 12 testbenches across 7 geometries:

```
tb_vpu_seq: PASS (200 unary, 200 qadd, 3x32 softmax, 3x40 layernorm)
...
all tpu-v2 testbenches passed          (84 PASS)
```

and the same four ops driven by the CV32E40P over the real bus:

```
hostio: tinytpu: PASS 24x32x32 GEMM, 192 words bit-exact vs qlinear() (93 DMA spins, 39 poll spins)
hostio: tinytpu: PASS vector ops -- 64 unary, 64 qadd, 2x32 softmax, 2x40 layernorm, all bit-exact
Execution finished. Return value: 0
[pass] test_sw
```

### 6.11 Step 9 — a whole transformer block

The first thing that runs on this hardware that is not a unit test: one DINOv2
transformer block, 24 ops, driven end to end by the CV32E40P over the real bus, with
every intermediate checked bit-exactly against the Python emulator.

```
hostio: tinytpu: PASS transformer block -- 24 ops (16x32 tokens, 2 heads),
                 every intermediate bit-exact (2563 DMA spins, 1202 engine spins)
```

The block is `T=16` tokens, `E=32` channels, 2 heads of `D=16`, `HID=64` — a scale model
of ViT-S/14's `E=384`, 6 heads, `HID=1536`. Small on purpose: it is the *sequencing* that
is under test, and at 16 tokens the intermediates already fill the 8 KB result region
exactly (512/512 words), which is itself the finding.

`test/v2/gen_block_vectors.py` is the reference. It runs the block in the emulator and
emits `block_vectors.h`: the weights, the requantization constants, the exp/GELU tables,
the expected value of every intermediate, and a `blk_op_t blk_ops[24]` descriptor table.
`main.c` is then an interpreter over that table — DMA operand A into the activation
region, DMA the weights, write the shape or the vector config, start, poll, DMA the
result to its slot, compare. That structure is the point: the driver has no knowledge of
attention at all.

#### Four things the lowering forced

1. **q/k/v are six GEMMs, not three.** The array's activation port reads `[M][K]`
   contiguously, so one head's slice of a `[T][E]` projection is not an addressable
   operand. Per-head projections are the only shape that fits the port.
2. **The output projection is split across heads and summed.** `[ctx₀|ctx₁] @ Wo` equals
   `ctx₀ @ Wo[:D] + ctx₁ @ Wo[D:]`, and `gemm_seq` cannot accumulate across calls — so the
   concatenation never has to exist, and the residual add does the summing.
3. **LayerScale folds into the preceding GEMM's per-channel multiplier** and the op
   disappears entirely.
4. **Attention needs a transpose, and nothing could produce one.** See below.

#### The transpose op

`Qᵀ·K` needs Kᵀ, and the GEMM reads its stationary operand row-major. Doing the
transpose on the CPU is 8 bits per bus round trip — the one place that cost is
unarguable — so it became `op.code = 5`.

It is the odd one out in the vector unit: it has no kernel at all, only a read address
pattern. The write side walks a row of the destination while the read side walks a column
of the source, which is one stride of `len` per element and a step back to the next source
column at the end of each output row. Everything else in the datapath — the serializer,
the packer, the byte-masked tail write — is unchanged. The whole op is one index generator
and a passthrough register that gives it the same latency as the LUT.

Verified at two geometries in `tb_vpu_seq`, and the second one matters more than it looks:

- **13×19**, where neither dimension is a multiple of the 16 lanes and rows ≠ cols. A
  stride error here lands in the wrong *lane* and is loud.
- **16×16**, which is what attention actually asks for (Kᵀ is `head_dim × tokens`, and
  `head_dim` is 16). This is the degenerate case for the index generator: the source lane
  stops changing within an output row, so the same stride error lands in the right lane of
  the *wrong word* and is silent. The ragged case does not catch it.

#### The bug this found, which was not in the hardware

The block failed on its fifth op, `kt0`, with 255 of 256 elements wrong — while the four
ops before it, including two GEMMs whose results had been DMA'd between regions, passed
bit-exactly. The transpose RTL passed in Verilator at both geometries, so the fault had to
be in the SoC path.

It was not. Dumping the staged operand showed the activation buffer held `exp_k1` — head
*one's* K — at the moment head *zero's* transpose ran. The generator emitted all six
q/k/v GEMMs in one loop before the attention loop, and q/k/v share one slot triple across
heads, so head 1 overwrote head 0's operands before head 0 ever read them. The fix is
op ordering, not hardware: the projections now emit inside the attention loop, and only
the float math runs ahead (the logit scale is shared by both heads, so it needs both
heads' values before either head's ops are emitted).

Worth recording because the failure signature pointed the wrong way. A brand-new op
failing in the SoC while passing in isolation reads as an integration bug; it was a
register-allocation bug in the *reference generator*, and the hardware was correct the
whole time. The thing that settled it was making the driver dump what it had actually
staged rather than continuing to reason backwards from three wrong bytes.

#### What it costs

2563 DMA spins against 1202 engine spins: the CPU spends roughly twice as long moving
data around the ops as the ops spend computing. Three bus transfers surround every op that
does real work — operand in, weights in, result out — and the DMA has one outstanding
request. That is the measured target for the next step, and it is now a number rather
than a suspicion.

### 6.12 Step 10 — where the cycles actually go

The block gave step 2 a target, but "2563 DMA spins" answers the wrong question: a spin
is one look at a status register, which conflates the bus round trip of the poll with the
work being polled for. `mcycle` answers the right one. Instrumenting the driver split the
block into the three costs a real inference would still pay:

| | cycles | share |
|---|---|---|
| DMA | 45,255 | 58% |
| Engine | 20,356 | 26% |
| Per-op config writes | 12,336 | 16% |
| **total** | **77,947** | |

(The bit-exact check is 331,636 cycles more, but that is the testbench reading results out
a byte at a time, not the block.)

So the DMA was the bottleneck — but not for the reason the code comment predicted. Of the
26,624 bytes it moved, only 8,192 were weights. The rest was the chip copying data to
itself:

| | bytes | |
|---|---|---|
| Weights from BRAM | 8,192 | irreducible — the chip does not have them yet |
| Operand A staging, `OUT → ACT` | 10,752 | bookkeeping |
| GEMM result copy, `OUT[0] → OUT[d]` | 5,120 | bookkeeping |
| Operand B staging, `OUT → WGT` | 2,560 | bookkeeping |

**69% of all bus traffic was data that was already on the chip, in the right buffer, being
moved because the engines could not name where it was.** Pipelining the DMA would have
made the wrong 69% faster.

#### Operand bases and a region select

Three changes, all of them addressing rather than arithmetic:

- `gemm_seq` gained `cfg_a_word` / `cfg_b_word` / `cfg_d_word`. Its internal counters stay
  relative and the base is added at the port, so nothing about the tiling depends on
  placement. The destination base alone removes every result copy.
- `tinytpu_buf`'s engine port split into an independently addressed read port and write
  port, and the result buffer got a second read port. Two read ports on results is not
  luxury: a GEMM whose stationary operand is another op's output — `Q·Kᵀ` is exactly that
  — reads both operands from there at once.
- `src_a` and `src_b` gained a region field. Operand A can be read from the activation
  region or the result region, operand B from the weight region or the result region.
  Both candidate buffers get the same address and the data is muxed on the way back, which
  costs a mux and no cycles, because every buffer read is registered with the same latency.

The register map changed shape to match. `vec_src`/`vec_srcb`/`vec_dst` are now
`src_a`/`src_b`/`dst`, because both engines use them — leaving the `vec_` prefix on
registers the GEMM also reads would have been a lie. `vec_mult`, `vec_shift` and `vec_eps`
keep it, because they really are vector-only.

The cost is one extra copy of each buffer array, which Vivado infers for the third port.
24 KB of buffers becomes 48 KB on a part with 365 BRAM36s.

#### The result

Same 24 ops, same bit-exact check on every intermediate:

| | before | after |
|---|---|---|
| DMA bytes | 26,624 | 8,192 |
| DMA cycles | 45,255 | 13,654 |
| Engine cycles | 20,356 | 20,373 |
| Config cycles | 12,336 | 12,376 |
| **total** | **77,947** | **46,403** |

1.68× on the whole block, and the DMA now moves exactly the bytes that have to move. The
profile inverted: the engine is now the largest single cost at 44%, the DMA is 29%, and
the per-op constant writes are 27%.

Two things follow. The DMA's remaining 13,654 cycles are 1.67 cycles per byte for traffic
that is now entirely unavoidable, so its single outstanding request is worth fixing — but
it is a 22% win, not a 58% one. And the per-op config writes, which were noise at 16%, are
now within a factor of two of the DMA: 1,500-odd stores of bias, multiplier and shift per
output channel, rewritten for every GEMM. In a real network those constants change per
layer, not per op, so most of that is re-uploading what is already there.

#### The verification that mattered

The GEMM bases are checked in `tb_gemm_seq` by running the same 48×16×64 GEMM twice —
once at base zero, once at bases 13/37/5, unequal and not multiples of anything the
sequencer counts in — with the operand memories cleared and reloaded between runs so the
second cannot pass on what the first left behind. The region select is checked by the
block itself, which now exercises every combination: A from results with B from weights
(the projections), both operands from results (`Q·Kᵀ`, `P·V`, the residual adds), and an
in-place vector op reading and writing the same slot (softmax over the attention
probabilities). That last one is only safe because the reader leads the packer by a whole
word, which is worth knowing rather than assuming — a transpose done in place would not be.

### 6.13 Step 11 — the board

The first synthesis of the whole SoC with tiny-tpu in it, on `xc7a200tsbg484-1` at
50 MHz. It did not fit, twice, for two different reasons, and both are worth recording
because neither is visible in simulation.

#### It did not fit: 362,917 LUTs on a 134,600-LUT part

`place_design` failed outright. 270% of the LUTs and 113% of the registers, with 70,915
F7 and 33,281 F8 muxes — the signature of a memory implemented as flip-flops behind a
mux tree. `WARNING: [Synth 8-11357] ... for RAM bank_reg with 65536 registers` named it:
none of the three data buffers had become memory at all.

The cause was the second read port added in §6.12. **Vivado does not replicate an array
to serve a third port; it dissolves the whole array into registers.** The comment I wrote
when adding that port claimed the opposite, and simulation had no opinion either way.

The fix is to do the replication in RTL: one copy of the array per reader, every write
mirrored into all of them, so each copy has exactly one write access and one read
access — the shape a true dual-port BRAM actually has. That took it to 113,004 LUTs
(84%) with the SoC block still bit-exact.

#### It fitted but wastefully: 24,082 LUTs of distributed RAM

The copies were memory now, but LUT memory. `(* ram_style = "block" *)` came back
`WARNING: [Synth 8-6849] Infeasible attribute ram_style = "block"` — because the read
sat in its own `always_ff`. Vivado's byte-write-enable BRAM template wants one process
holding one write and one read, with the address and byte enables already resolved.
Rewritten that way the buffers became block RAM: 71 → 85 tiles, and 113,004 → 88,120
LUTs.

That is a real finding about this codebase's style. Splitting a memory's read out of its
write process reads better and infers worse, and nothing short of synthesis will say so.

#### 612 DRC violations from one habit

With it fitting, the warning sweep `task.md` asks for. 813 DRC violations and 458
methodology violations, and — unusually — 812 of the 813 were tiny-tpu's own, not the
CPU's or the DDR3 controller's. One rule dominated:

```
DPOR-1  Asynchronous load check  612
DSP .../u_gemm/g_requant[0].u_rq/prod output is connected to registers with an
asynchronous reset ... This is preventing the possibility of merging these
registers in to the DSP Block since the DSP block registers only possess
synchronous reset capability.
```

Ten of the eighteen modules in `src/v2/` had been written `always_ff @(posedge clk or
posedge rst)` and the rest `always_ff @(posedge clk)`. The inconsistency was invisible
in simulation and cost 612 DRC plus 374 methodology violations, because DSP48 and BRAM
registers have synchronous reset only: a register with an asynchronous one cannot be
absorbed into the block it feeds.

Converting all of them to synchronous reset is safe here and not a compromise. `sys_rst_n`
in `rvlab_fpga_top.sv` is released synchronously after the MMCM locks, and Xilinx flops
power up to their INIT value, so the design is already in a reset state before the clock
that would apply the reset. The asynchronous reset was buying nothing at all.

#### Where it landed

| | first synth | replicated | block RAM | sync reset |
|---|---|---|---|---|
| LUTs | 362,917 (270%) | 113,004 (84%) | 88,120 (65%) | **68,133 (51%)** |
| Registers | 304,734 (113%) | 106,042 (39%) | 105,142 (39%) | **62,576 (23%)** |
| BRAM tiles | 71 | 71 | 85 | **87 (24%)** |
| DSPs | 91 | 91 | 91 | **91 (12%)** |
| DRC violations | — | 813 | 813 | **197** |
| Methodology | — | 458 | 458 | **298** |
| `place_design` | failed | — | — | **routed** |

Timing is met at 50 MHz with margin on both edges: WNS +0.160 ns, WHS +0.017 ns, TNS and
THS zero, no failing endpoints of 178,645. The bitstream is 9,730,763 bytes.
`rvlab_fpga_top.io_report.txt` is clean — no pins missing in either direction.

The last column is the honest one to read: half the LUTs, a quarter of the registers,
and the arithmetic living in DSPs and block RAM rather than in fabric. Two of the three
remaining resources are under a quarter used, which is what makes the 32×32 array a
question worth asking rather than a fantasy.

#### What is left, and why

197 DRC and 298 methodology violations remain, all Warning or Advisory, none blocking:

- **DPOP-1/DPOP-2/DPIP-1 (175)** and **DPIR-1 (214)** — "pipelining this multiplier would
  improve performance", and "this DSP input is driven by a register with an asynchronous
  reset". The asynchronous reset is now always OpenTitan's `prim_subreg.sv`, which is
  what every configuration register in the map is built from, so every constant that
  reaches a multiplier arrives through one. `task.md` asks for these fixed *without
  changing third party libraries if possible*, and this is the case where it is not
  possible. The pipelining advisories are the same story from the other side: the
  registers that would have been absorbed cannot be.
- **REQP-1840 (20)** — the buffers' BRAM address is driven by a TL-UL FIFO pointer with an
  asynchronous reset, again rvlab's. Benign here: the write enable is gated on a bus
  request, which cannot be asserted during reset, so there is nothing to corrupt.
- **SYNTH-5 / RTGT-1 (8)** — `gemm_seq`'s per-channel bias, multiplier and shift memories
  are still LUT RAM, about 3k LUTs. Same cause as the buffers, but the read sits inside
  the block sequencer's FSM and moving it into the write process shifts the constant
  fetch by a cycle. That is a change to a verified sequencer for 2% of the LUTs, so it
  is written down rather than done.
- **SYNTH-10 (75)** — wide multipliers, which is what a requantizer is.

### 6.14 Step 12 — what a real block costs, and what the bottleneck actually is

The block in §6.11 is T=16 tokens, E=32 channels. ViT-S/14 at 518×518 is T=1370, E=384,
6 heads of 64, HID=1536. The question this step asks is whether the design scales to
that, and the answer arrived in an order I did not expect.

`sw/tiling.py` is the new tool. `lower.py` costs a schedule against peak MACs and DRAM
bandwidth, but it has no model of the on-chip buffers, so it cannot say whether an op
*fits* — which at real dimensions is the question that decides everything. The tiler
takes the buffer capacities as they are in the RTL and, for each GEMM, picks the tiling
with the least DRAM traffic that fits, or says why none does.

#### Seven of a block's 38 GEMMs could not run at all

The buffer geometry, in 128-bit words of 16 int8 lanes:

```
activations   Mt * (K / 16)     one word per (row, k tile)
weights       K  * (Nt / 16)    one word per (k, n tile)
results       Mt * (Nt / 16)    one word per (row, n tile)
```

The weight buffer is the asymmetric one, and I had not noticed how asymmetric. It is
indexed by k *row*, not by k tile: it holds K words per 16 output channels, sixteen times
the activation buffer's appetite for the same K. With 512 words that caps K at 512, and a
ViT-S block has seven GEMMs deeper than that — `fc2` at K=1536, and the six per-head
`P·V` at K=1370, where the "stationary operand" is the attention probability matrix.
Nothing tiles around it: `gemm_seq` accumulates over the k tiles that are resident and
requantizes to int8 on the way out, so a K that does not fit needs int32 partial sums
carried between calls, which the engine cannot do.

Growing the weight buffer to 3072 words (48 KB) fixes all seven, and nothing else needs
to change — activations and results stay at 8 KB. Total 64 KB, which is exactly what
`sw/machines/tpu_v2.json` has specified as `unified_buffer.bytes` since before any of this
RTL existed. The contract was right and the RTL had simply never been sized against it.

#### The bottleneck is the array, not the bus

The number that reorders the roadmap:

```
per block: 87.1 MB of DRAM traffic, 3.87 GMAC
12 blocks at 50 MHz:
  DRAM     2.61 s  (8 B/cycle)
  compute  3.62 s  (256 MAC/cycle)
  -> compute-bound at 3.62 s/image
```

**The design is compute-bound at every buffer size that fits.** 46.4 GMAC per image
against 256 MACs per cycle is 3.62 seconds, and no tiling changes that. The best tiling's
DRAM traffic is 2.61 s, comfortably underneath. Growing the buffers past 64 KB moves the
DRAM number and nothing else: 160 KB of buffers buys 2.45 s, 320 KB buys 2.45 s, and the
image still takes 3.62 seconds.

That contradicts the ordering §6.13 left behind, which put throughput — the DMA's single
outstanding request, the vector unit's two cycles per element — ahead of the array. Those
were measured on a 16-token block where the buffers held everything and the bus was 58%
of the time. At real dimensions the bus has slack and the array does not. §6.12's
measurement was correct about the block it measured and misleading about the model.

At 1024 MACs per cycle — a 32×32 array — compute drops to 0.91 s and DRAM becomes the
binding constraint at 2.61 s. That is the point at which the DMA's outstanding-request
limit and the buffer sizes start to matter, and not before.

#### Attention is 42% of the traffic

Worth recording separately, because it is a scheduling problem rather than a hardware
one. `logit` and `ctx` account for 36.4 MB of the 87.1 MB. At 1370 tokens the per-head
probability matrix is 1370² = 1.88 MB, six of them per block, and the schedule as written
materializes each one in full before consuming it. Tiling attention over query blocks —
computing a band of rows of `P`, normalizing it, and multiplying it into `V` before moving
on — removes almost all of that, and needs no new hardware, only a different order in the
generator. It is not urgent while the array is the bottleneck, which is precisely the kind
of thing worth writing down now and not doing yet.

#### What was verified

The tiler is checked by `sw/tests/test_tiling.py` — hand-computable single-call cases, the
weight buffer's k-row indexing at the exact boundary, and monotonicity (a larger buffer
can never cost more traffic, since its set of legal tilings contains the smaller one's).

The RTL is checked at the two deepest tile shapes a real block asks for, added to the
`tb_gemm_seq` sweep:

```
PASS gemm_seq 21x1536x32 (96 k tiles, 2 n tiles, MBLK=16): 1344/1344 bit-exact vs qlinear()
PASS gemm_seq 5x1370x32  (86 k tiles, 2 n tiles, MBLK=16):  320/320 bit-exact vs qlinear()
```

96 k tiles against the 1 to 4 every previous case used. Both at base zero and at
non-zero operand bases.

The 48 KB weight buffer still routes, which was the other thing that had to be true.
Rerunning §6.13's flow with `WGT_WORDS = 3072`:

| | 8 KB weights | 48 KB weights |
|---|---|---|
| LUTs | 68,133 (50.9%) | 68,158 (50.9%) |
| Registers | 62,576 (23.3%) | 62,582 (23.3%) |
| BRAM tiles | 87 (23.8%) | 115 (31.5%) |
| WNS / WHS | +0.160 / +0.017 ns | +0.104 / +0.024 ns |
| DRC violations | 197 | 197 |

Six times the weight buffer costs 28 block RAMs and 25 LUTs. That is what the §6.13
inference work bought: at distributed RAM prices this would have been ~18,000 LUTs and
would not have fitted.

What is *not* verified is a whole block at 1370 tokens, and it will not be by simulation:
6,428 GEMM calls at these shapes is more cycles than xsim will deliver in a working day.
The 24-op block in §6.11 stays the end-to-end check on the SoC seam, `tb_gemm_seq` covers
the tile shapes, and the tiler covers the arithmetic that connects them. Closing the last
gap needs the tiled program emitted and run on hardware, not in a simulator.

### 6.15 Step 13 — what the sibling project found in the DDR3 path

A second project on the same rvlab SoC —
[RISC-V-ViT-accelerator](https://github.com/Armin-Ebrahimi-Saba/RISC-V-ViT-accelerator) —
got a full Depth Anything V2 Small inference running on the board, and in doing so found
three defects in the platform's DDR3 path, none in its own accelerator. This project has
not exercised that path at all yet (§6.14: the batch simulation has no DDR3), so those
defects are still ahead of it. This step brings over what transfers, and records what
does not.

#### What was taken

- **`docs/LESSONS.md`** — the portable rules, symptom → cause → rule, copied unchanged.
  The single most useful file in that repository.
- **`docs/DEBUGGING.md`** — the full account, wrong theories included, copied unchanged
  under a provenance note. Worth reading for the instruments alone: a watchdog register
  readable over JTAG on a wedged core, accelerator-vs-CPU comparison at model shapes in
  DDR3, a testbench driver that pipelines requests the way a CPU does, and the negative
  control every time.
- **`CLAUDE.md`** — the build pipeline, the register-offset trap, the `pkill` trap, and
  "define a hardware register when the debugger can't answer", rewritten for this tree's
  layout and `flow` syntax.
- **`student_tl_watch.sv`** and its two `ddr_ctrl` registers — the stalled-transaction
  watchdog, the instrument that found the hang the debugger structurally could not. A
  dozen flops snooping the DDR3 port, latching the oldest unanswered request, its master,
  and a saturating stall count. It drives nothing on the bus.
- **The `a_ready` fix in `rvlab_tlul_ddr.sv`.** The response mux started from the error
  responder and overrode it when the cache had data — so `a_ready` came from the error
  responder, which holds it high whenever idle, while its `a_valid` was forced low after
  calibration. Any request issued while the cache was not ready was handshaked away and
  accepted by nobody. The CPU rarely hits the window; an accelerator with several writes
  in flight hits it every run and wedges the CPU unhaltably.
- **The prefetcher bypass switch**, default bypassed. Their alias test showed the
  prefetcher returning the wrong line under set aliasing, 65 of 256 reads.

#### What was not taken, and why

**The write-back fix was already here.** `rvlab_ddr_block_cache.sv`'s eviction path sent
the RAM's raw output, one cycle stale when the line had been written the cycle before —
one word per tile lost, deterministically, the defect behind their wrong depth map. Their
fix is `a_data: data_rdata`. Our `rvlab/` submodule is on an upstream that merged the same
one-line change on 17 July 2026 (commit `ff12462`, "additional minor ddr patches"),
before this project started. Checked by reading the line, not by assuming.

**Their other cache change was not.** Their tree also gates the data RAM's read index on
`use_be_port || stall` instead of `stall`, with a comment citing "111 words in 65536
lost". That measurement is the one their own §8 retracts — it was the checker, not the
memory — and their §7 records that this gating passed a test which also passed unfixed.
In this tree's cache `stall` is still high on the cycle `use_be_port` fires, so the
change is a no-op. Bringing in an unverified change with a retracted justification is
what `LESSONS.md` warns against, so it stays out.

**The prefetcher they bypassed is not the prefetcher we have.** Their tree carries the
pre-refactor module; upstream refactored it on 17 July (`951fa84`, "cache bugfix, ddr3
prefetcher bugfix + refactor"), and their prefetcher is byte-identical to the one that
commit replaced. Whether the aliasing defect survived the refactor is untested. The
bypass therefore stays on as insurance — §6.14 says this design's DRAM time is a full
second under its compute floor, so the bandwidth is affordable — and comes off only with
their alias testbench ported and passing against the new module. The switch and the
reason are in the RTL.

**The skid buffer and the behavioural DDR3 model** (`student_tl_rsp_hold.sv`,
`ddr3_blk_model.sv`, the `RVLAB_DDR_BEHAVIOURAL` back end) were not requested and were
left. The first works around the cache pulsing `d_valid` for one cycle without looking at
`d_ready` — a real defect, and the one to remember when this project's DMA grows a second
outstanding request. The second is what their alias and `d_ready` testbenches run on, in
seconds rather than hours, and is the right way to port those tests when the time comes.

#### The register-offset trap, avoided by taking it seriously

Their watchdog registers went in *ahead of* `ctrl`, moving it from `+0x4` to `+0xc`, and
software built against the old map wrote the DDR3 reset bit into a read-only address —
their §4. Here they are appended after `ctrl`, so `status` and `ctrl` keep their offsets
and nothing already built is invalidated. The comment in the `.hjson` says why.

#### Verified

The batch simulation, which has no DDR3, is unaffected by construction; it was rerun to
prove the register map and driver still build, and every op is still bit-exact. The DDR3
path exists only in the FPGA build, so the check on the watchdog, the `a_ready` mux and
the prefetcher bypass is synthesis, place-and-route and the bitstream, with CLAUDE.md's
warning sweep:

| | §6.14 | with the DDR3 fixes |
|---|---|---|
| LUTs | 68,158 (50.9%) | 67,061 (50.1%) |
| Registers | 62,582 | 62,616 |
| WNS / WHS | +0.104 / +0.024 ns | +0.218 / +0.020 ns |
| DRC | 197 | 197 |
| Methodology | 298 | 300 |
| `io_report` | clean | clean |

The bypassed prefetcher is ~1,100 LUTs of logic gone; the watchdog is 34 registers. The two
new methodology advisories are RTGT-1 on the block manager's request buffer — a
third-party LUT RAM that Vivado now thinks could be retargeted — not on anything added.
`ctrl` is still at `+0x4`; `wdog_addr` and `wdog_stat` are at `+0x8` and `+0xc`.

### 6.16 Step 14 — silicon

Everything above this line was simulation. The board was connected, so this step is
the first time any of it ran on the XC7A200T.

#### The self-test on hardware

`flow rvlab_fpga_top program`, then the driver over JTAG:

```
tinytpu: id ok
tinytpu: PASS 24x32x32 GEMM, 192 words bit-exact vs qlinear() (96 DMA spins, 40 poll spins)
tinytpu: PASS vector ops -- 64 unary, 64 qadd, 2x32 softmax, 2x40 layernorm, all bit-exact
tinytpu: PASS transformer block -- 24 ops (16x32 tokens, 2 heads), every intermediate bit-exact (686 DMA spins, 1203 engine spins)
tinytpu: cycles: dma 13674, engine 20352, config 12358 (sum 46384)
execution finished in 0.1 s, return value 0
```

The GEMM, all five vector ops, and the 24-op transformer block, every intermediate
bit-exact on silicon. The cycle count is 46,384 against the simulation's 46,403 — the
difference is the CPU's own poll timing, not the engine's.

Two things had to be built to get that line. `flow sw_project run` puts the terminal in
raw mode and starts OpenOCD inside an xterm, so it dies with `Inappropriate ioctl` under
any redirection; `tools/run_fpga.py` is the non-interactive version, and it prints the
`downloaded ... verified ...` lines from OpenOCD's own log because `load_image` fails
silently otherwise — the first run of it did fail silently, and the board ran the course's
`test_rvlab` out of the bitstream's BRAM init instead, which passed 4/4 and looked exactly
like a run. (A useful accident: that was a full 512 MB DDR3 memtest through the `a_ready`
fix and the prefetcher bypass, 4.60 cycles/byte read, clean.) The cause was OpenOCD's view
of the target lagging the DMI reset-halt; the runner now asks for `halt` through OpenOCD
too and refuses to load into anything but `halted`. On timeout it prints the §6.15
watchdog registers before anything else.

#### The model in DDR3

`tools/load_model.py` with the sibling project's exported blob:

```
tinytpu: BLOB_READY
writing 24871428 bytes to 0x80000000 ...
  downloaded 24871428 bytes in 67.410912s (360.305 KiB/s)
tinytpu: blob magic 32564144 version 2 tensors 299 total 24871428 bytes
tinytpu: blob checksum (device) c9fad970 over 24871428 bytes, 89276668 cycles (3675 cycles/KB)
model loaded: 24871428 bytes in DDR3, checksum c9fad970 matches the file
```

The driver brings DDR3 up, prints `BLOB_READY`, and spins on a flag **in BRAM** — not in
DDR3, where the CPU would read it stale through the cache for as long as the line stayed
resident; the sibling project lost a day to that. The host writes the blob with
`load_image ... bin` at 360 KiB/s, sets the flag, and compares a rotate-xor over every
word, computed on the device through the CPU's own cached path, against the same sum over
the file. 24.87 MB of Depth Anything V2 Small — all 299 tensors — are in DRAM on the board
and read back correctly. The flag's address is taken from the ELF's symbol table, so a
relink cannot move it out from under the script.

#### Why there is no depth map yet, precisely

The ask was to run images. That needs a runtime: something that walks the graph, tiles
every GEMM per §6.14, drives the vector unit for softmax, layernorm, GELU and the residual
adds, does patch embedding and the DPT head, and moves activations between DRAM and the
buffers. This project has the emulator that specifies all of that (`sw/`) and the
hardware that executes each op; it does not have the program in between.

The sibling project has such a runtime — `dav2_engine.c`, 2,500 lines, 94 s a frame at
126×126 — and the tempting route is to put tiny-tpu underneath it. Its accelerator
interface is what decides that:

```
acc[m][n] = Σ_k a[n][k] · w[m][k]      a: int16, w: int8, acc: int32, all in DDR3
```

`int16` activations in, `int32` accumulators out, float requantization in software.
tiny-tpu takes `int8` activations and emits `int8` results by design (§6.12 is built on
it). A shim would have to requantize their activations to `int8` on the way in and
expand tiny-tpu's `int8` results back to a coarse `int32` on the way out — which puts the
whole model into the int8-everything regime `sw/` measured at 15.9 dB SQNR / Pearson
0.937 (§2), against their 0.9998. Recognisable depth maps, not their depth maps. It would
also need their `[M][K]` weights turned into tiny-tpu's `[K][N]` layout — the transpose
op can do that on the fly, 16 output channels at a time, and `src_b.region` lets the GEMM
read the result in place — and a per-GEMM output shift chosen without seeing the
accumulator, which is the part with no clean answer.

The other route is this project's own runtime, from `sw/lower.py`'s graph, with the
numerics it was designed for. Bigger, and the right one.

Either is days, not a session. What this step leaves behind is the two halves that both
routes need and that did not exist this morning: the hardware proven on silicon, and the
weights in DRAM with a way to prove they got there.

### 6.17 Step 15 — the runtime, increment 1: one real GEMM out of DDR3

§6.16 ended with the weights in DRAM and no program to use them. The choice in front
of the runtime was where the decisions live: on the CV32E40P, where a tiling bug is a
C bug found through a 30-byte-a-second hostio ring, or in `sw/`, where it is a Python
bug found by reading a file. The split `gen_block_vectors.py` and `run_block()` already
use at T=16 (§6.11) answers it, generalised: **the exporter decides everything, the
driver interprets.**

`sw/export_tpu.py` writes a blob: a 64-byte header, one 96-byte descriptor per
hardware op, then a 16-byte-aligned data area holding every byte the descriptors name —
weights already in the buffer's `[K][Nt]` layout, per-channel `(bias, mult, shift)`
already in the config region's `[c][4]` layout, the emulator's expected result for each
tile. Descriptors carry absolute DRAM addresses, the register values verbatim
(`src_a`, `src_b`, `dst`, `shape`, the vector unit's five), DMA sources and lengths,
and an optional result-copy target and check. `emit_gemm()` tiles with `sw/tiling.py`
(§6.14): the weight tile and its config are named once per n tile, the m tiles under it
name only their rows, which is weight reuse made visible in the stream.

`rvlab/src/sw/project/tpu_runtime.c` is the other half: 150 lines that walk the list
and do, per descriptor, what `run_block()` does per op — DMA A and B if named, copy the
config words, set the registers, start, poll, copy the result out, check. It makes no
decision and sees no float. `tpu.h` holds the register macros both drivers share;
`load_model.py` waits for the verdict after the checksum. `sw/tests/test_export_tpu.py`
reads a written blob back the way the C does and checks that every address it names
points at the bytes the hardware wants.

The first program is block 0's q projection for head 0 at real width — `[384]×[64]`
with `1/√d` folded into the weights, as `sw/` does — over 82 synthetic tokens. The tiler
cuts it into m tiles of 21, 21, 21, 19 under one 64-wide n tile: 4 hardware calls,
62,976-byte blob. On the board:

```
tinytpu: blob v1, 5 descriptors, 62976 bytes
tinytpu: PASS blob -- 4 ops, 0 failed
tinytpu: cycles: dma 105719 (56064 bytes), config 5900, engine 48186, result copy 0; verify 149124
```

Bit-exact against `qlinear()` over real weights, for every one of 82×64 outputs. This is
the first time the DMA has read operands out of DDR3 rather than the CPU's BRAM, and
the number to keep is **1.9 cycles/byte** — twice the BRAM-sourced figure in §6.12,
which is the cache-line-at-a-time TL-UL port with the prefetcher off (§6.15), and the
first datum for item 3 of Remaining.

The first run of the day read `id = 0xaffe`: the board had been power-cycled and was
running whatever the configuration flash holds. `flow rvlab_fpga_top program` first.

**Known costs, on purpose.** Results leave the chip through the CPU, one aperture load
and one DRAM store per word — the DMA only fills buffers. Config loads are CPU word
copies. Both are the price of the first picture, not the design; both show up as their
own cycle counters so they can be priced when they matter.

### 6.18 Step 16 — the runtime, increment 2: a whole block at real width

One GEMM proved the DMA path; a block proves the *stream* — that every op can find what
the previous one left, at the real shapes, through the real buffers. Block 0 of the
model at full width: 384 channels, 6 heads of 64, a 1536-wide MLP, over 82 tokens (a
126×126 image). Nothing of that fits in an 8 KB result region, so the T=16 block's
"everything in the result region" lowering (§6.11) does not survive; what replaces it
is an **arena** in DDR3 above the blob, where every op leaves its output for the next.

Three things the arena made possible, and one it forced:

- **Head concatenation for free.** Format v2's result copy takes a row stride, so
  each head's `P·V` is written straight into its 64 columns of one `[82][384]` context
  tensor. The output projection is then one GEMM over the real `proj.weight` — not six
  K=64 GEMMs summed with five qadds as at T=16. The six heads carry six context scales
  and the GEMM has one activation scale; each head's ratio is folded into its rows of
  the weight before quantization, which is exact.
- **Any token count.** 82 is not a multiple of 16 and needs to be one only along the
  key axis of the logits, which is a GEMM's N. Keys and values are padded to 96 rows,
  and the logit GEMM's per-column bias — a register the hardware already has — pushes
  the 14 padded columns to −127, which the exp table sends to zero. Softmax then runs
  at length 96 and P·V reduces over 96 with zero rows contributing nothing. No masking
  op, no new hardware.
- **LayerScale with negative gammas.** 177 of block 0's 384 `ls1` gammas are negative,
  and the requantizer's multiplier is unsigned; the sign folds into the weight column
  and the bias, exactly.
- **Padding is read that nothing wrote.** The first board run failed exactly at columns
  82–95 of every `Kᵀ`: the padded rows of `K` were uninitialized DDR3. The header now
  carries the arena's size and the driver zero-fills it before the first op — 0.7 MB
  for this block. A simulation with no DDR3 could not have shown this.

The lowering is checked against an independent float implementation of the block:
30.4 dB SQNR at the block output over the int8-everything path, in line with §2. On the
board:

```
tinytpu: blob v2, 400 descriptors, 2636432 bytes
tinytpu: PASS blob -- 392 ops, 0 failed
tinytpu: cycles: dma 9331101 (4871424 bytes), config 504274, engine 5568363, result copy 2611350; verify 19133873
```

392 hardware ops — 348 GEMM tiles, 16 GELU chunks, 8 layernorm, 8 qadd, 6 transposes,
6 softmax — every intermediate bit-exact against `sw/kernels_int.py`, over the real
weights. 18.0 M cycles, **0.36 s per block** at 50 MHz with verification excluded
(the 19 M cycles of byte-compares are the testbench, not the block). Where they go:

| | cycles | share | note |
|---|---|---|---|
| DMA | 9.33 M | 52% | 4.87 MB at 1.9 cycles/byte; activations reloaded per n tile (fc2: 12 n tiles × 17 m tiles) |
| engine | 5.57 M | 31% | 5.4× the 1.03 M MACs/256 that the array would take at full rate |
| result copy | 2.61 M | 15% | CPU, one aperture load + one DRAM store per word |
| config | 0.50 M | 3% | per-channel tables per n tile, exp pair per head |

Twelve such blocks are 4.3 s per 126×126 image before the patch embedding and the head,
against §6.14's 3.62 s/image compute-bound estimate for a 518×518 image — the model
still holds, and the two gaps it names are visible in the table: the DMA's 1.9
cycles/byte (§6.15's cache-line port with the prefetcher off) and the result path
through the CPU. Both are runtime costs, not array costs; the array is 31% busy.

The sequencer testbench (`tb_vpu_seq`) has a fixed 128-word memory with hard-coded
slot offsets and cannot hold a 21×384 layernorm or an 82×96 softmax; the kernel-level
`tb_layernorm_int` now runs at 384 in the sweep, and the sequencer at those shapes was
proven on the board instead.

### 6.19 Remaining

Only `verible-verilog-lint` is still missing, which makes `srcs.lint` unavailable. It is
optional — a code-quality check, not a build step — so it is not blocking.

**Next, in order:**

1. **The array**, which §6.14 says is the only thing standing between here and a
   reasonable frame time: 3.62 s/image at 16×16 and 50 MHz, against 2.61 s of DRAM
   traffic that is already comfortably underneath it. 32×32 takes compute to 0.91 s.
   DSP-mapped PEs come first — §6.13 leaves 51% of the LUTs free, which 1024 LUT-mapped
   MACs would not fit into, and 88% of the DSPs are idle.
2. **The runtime**, continued from §6.18. Next the encoder: patch embedding as a GEMM
   over a host-side im2col, the position add, twelve blocks chained through the arena,
   the four taps copied out — with the DPT head on the CPU in float, which is the first
   depth image beside PyTorch's. Then the head on the accelerator, and 518×518.
3. **Throughput**, which §6.12 measured and §6.14 demoted: the engine's 44%, the DMA's
   29%, the config writes' 27%. These bind only once the array is faster than the bus,
   which is to say after item 1.

**Deferred, with reasons:** query-block tiling for attention (§6.14: 42% of a block's
DRAM traffic, but the array is the bottleneck, so it buys nothing yet); im2col address
generation (the DPT head needs it, one transformer block does not);
instruction encoding proper (`sw/lower.py`'s `TiledOp` is a cost model — `macs`, `cycles`,
`dram_bytes` — with no addresses or loop bounds; the `blk_op_t` table in §6.11 is the
first real step toward it); softmax accuracy at 1.4 dB SQNR, the worst op in the graph.
