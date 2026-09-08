# tiny-tpu on Xilinx Artix-7 XC7A200T — Synthesis, Implementation, Bitstream & Simulation Report

**Last updated:** 2026-08-11 (rev 2 — RTL fixes applied, bitstream generated)
**Original investigation:** 2026-08-10
**Target part:** `xc7a200tfbg484-1` (Artix-7 XC7A200T, speed grade -1)
**Toolchain:** Vivado 2022.2 (`synth_design` / `place_design` / `route_design` / `write_bitstream`, `xvlog` / `xelab` / `xsim`)
**Design under test:** `src/*.sv`, top = `tpu`, `SYSTOLIC_ARRAY_WIDTH = 2`, `UNIFIED_BUFFER_WIDTH = 128`

---

## 1. Executive summary

| Question | Answer |
|---|---|
| Does the design fit on an XC7A200T? | **Yes, comfortably** — 16.6% LUTs, 1.2% FFs, 1.8% DSPs, 0% BRAM |
| Does it close timing? | **Yes at 62.5 MHz** (16 ns, +0.163 ns). **No at 100 MHz** (−6.081 ns) |
| Does it synthesize as-written? | **Yes** — as of rev 2. Previously blocked (Issue #1) |
| Does a bitstream build? | **Yes** — 9,730,752 bytes, 0 errors, all timing met |
| Does behavioral simulation pass? | **Yes** — clean, zero X. The 207 assertion failures are an xsim bug (Issue #7) |
| Does post-route timing simulation pass? | **Partially** — runs to completion, but residual X remains (Issue #6) |
| Ready for a real board? | **Not yet** — needs a pin XDC + host interface, and Issue #6 closed |

Area was never the constraint. Five of the seven issues found are now fixed; one is a tool bug in xsim, and one remains open.

---

## 2. Issues register

The core reference table. Issues #1–#5 were found during the rev 1 investigation and fixed in rev 2.

| # | Issue | Severity | Status |
|---|---|---|---|
| 1 | `` `default_nettype none `` + `input logic` blocks all Xilinx tools | Blocker | ✅ **Fixed** (rev 2) |
| 2 | `tpu` has no output ports → entire design optimizes away | Blocker | ✅ **Fixed** (rev 2) |
| 3 | `rd_weight_skip_size` missing from async-reset branch | High | ✅ **Fixed** (rev 2) |
| 4 | Seven `*_ptr_next` module-scope temporaries infer real registers | High | ✅ **Fixed** (rev 2) |
| 5 | Dead `grad_descent_in_reg` declaration, permanently X | Low | ✅ **Fixed** (rev 2) |
| 6 | Gate-level X-propagation in `bias_parent/column_2` | High | ⚠️ **Open** — improved, not eliminated |
| 7 | 207 spurious assertion failures under xsim | Informational | ➖ **Not a design bug** — xsim tool bug |
| 8 | Neither testbench checks numerical correctness | High (process) | ⚠️ **Open** |
| 9 | Unified buffer consumes 97% of design area | Optimization | ⚠️ **Open** |
| 10 | VPU loss→activation path is combinational, caps Fmax at ~63 MHz | Optimization | ⚠️ **Open** |

---

## 3. Issues #1–#5: what was wrong and what changed

### Issue #1 — `default_nettype none` blocked every Xilinx tool ✅ Fixed

Every file in `src/` and `sva/` opened with `` `default_nettype none `` and then declared ports as `input logic clk`. Vivado 2022.2 rejects this in both synthesis and simulation:

```
ERROR: [Synth 8-6735] net type must be explicitly specified for 'clk'
                      when default_nettype is none [src/bias_child.sv:5]
ERROR: [VRFC 10-1103] net type must be explicitly specified for 'clk'
                      when default_nettype is none [src/bias_child.sv:5]
```

Vivado wants `input wire logic`. Questa and Icarus accept the original code, so this was Vivado-specific strictness — but it meant the design could not be built with Xilinx tools at all.

**Fix:** the directive was dropped from all 25 files (15 in `src/`, 10 in `sva/`). `sva/` was included because it hits the identical `xvlog` error; leaving it would keep the assertion suite unbuildable in Vivado.

### Issue #2 — no output ports, so the design vanished ✅ Fixed

`src/tpu.sv` declared only inputs. The first synthesis run produced a completely empty netlist:

```
| Slice LUTs      |    0 |  134600 | 0.00 |
| Slice Registers |    0 |  269200 | 0.00 |
```

**Fix:** the four VPU result signals — which already existed as internal wires — were promoted to module ports:

```systemverilog
output logic [15:0] vpu_data_out_1,
output logic [15:0] vpu_data_out_2,
output logic vpu_valid_out_1,
output logic vpu_valid_out_2
```

The `ub_wr_data_in` writeback assigns are unchanged. This removed the need for the `DONT_TOUCH` scaffolding used to measure rev 1, and it is what a usable design needs anyway — a host must be able to read results back.

### Issue #3 — `rd_weight_skip_size` never reset ✅ Fixed

`src/unified_buffer.sv:84`, assigned only in the `ub_rd_start_in` case block. The async-reset branch cleared every other pointer, size, and counter — this one alone was omitted. Vivado flagged it by name and predicted the exact consequence:

```
WARNING: [Synth 8-] Register rd_weight_skip_size_reg in module unified_buffer
has both Set and reset with same priority. This may cause simulation mismatches.
Consider rewriting code
```

A genuine RTL defect: it also meant undefined power-up behaviour in real silicon.

**Fix:** added `rd_weight_skip_size <= '0;` to the reset branch alongside the other `rd_weight_*` resets.

### Issue #4 — `*_ptr_next` temporaries inferred real storage ✅ Fixed

Seven signals — `wr_ptr_next`, `grad_descent_ptr_next`, `rd_input_ptr_next`, `rd_weight_ptr_next`, `rd_Y_ptr_next`, `rd_H_ptr_next`, `rd_grad_weight_ptr_next` — were declared at module scope but used as blocking within-cycle temporaries inside `always_ff`. Vivado inferred sequential elements for all seven, then stripped them:

```
WARNING: [Synth 8-6014] Unused sequential element rd_Y_ptr_next_reg was removed.
```

The RTL and the synthesized netlist therefore did not describe the same storage. Three of them (`rd_Y_ptr_next`, `rd_H_ptr_next`, `rd_grad_weight_ptr_next`) were observably X in behavioral simulation; RTL X-optimism (`if (X)` silently takes the else branch) hid this, while gate-level mux trees propagated it.

**Fix:** all seven are now `automatic` variables declared inside the `always_ff` block, which is what they always meant:

```systemverilog
always_ff @(posedge clk or posedge rst) begin
    automatic logic [15:0]        wr_ptr_next             = '0;
    automatic logic [15:0]        rd_input_ptr_next       = '0;
    automatic logic signed [15:0] rd_weight_ptr_next      = '0;
    ...
```

### Issue #5 — dead `grad_descent_in_reg` ✅ Fixed

`src/gradient_descent.sv` declared `logic grad_descent_in_reg;` which was never assigned or read — permanently X. Removed.

### Verification of #1–#5

| Check | Before | After |
|---|---|---|
| `synth_design` on unmodified `src/` | fails (`Synth 8-6735`) | **0 errors** |
| `xvlog`/`xelab` of RTL + SVA + TB | fails (`VRFC 10-1103`) | **builds** |
| Netlist survives without `DONT_TOUCH` | 0 cells | **22,265 LUTs** |
| `rd_weight_skip_size_reg` set/reset warning | present | **gone** |
| `Unused sequential element` warnings | 7 | **0** |
| Total synthesis warnings | 29 | **21** |
| X signals in behavioral sim | 5 | **0** |
| Behavioral output data | `0040`/`0000`, `ffd6`/`0030`, `ffd6`/`ffeb`, `ff8c`/`ffeb` | **identical** |
| Behavioral valid cycles | 14 | **14** |

No functional regression.

---

## 4. Bitstream

Generated from a full non-OOC implementation:

```
write_bitstream completed successfully
/tmp/tpu_verify/tpu.bit — 9,730,752 bytes
```

| Resource | Used | Available | Util % |
|---|---|---|---|
| Slice LUTs | 22,265 | 133,800 | **16.64%** |
| Occupied Slices | 6,218 | 33,450 | 18.59% |
| Slice Registers | 3,293 | 269,200 | 1.22% |
| DSP48E1 | 13 | 740 | 1.76% |
| Block RAM Tile | 0 | 365 | **0%** |
| Bonded IOB | 182 | 285 | 63.86% |
| BUFGCTRL | 1 | 32 | 3.13% |

**WNS +0.163 ns at 16 ns — "All user specified timing constraints are met."** Hold clean (WHS +0.140 ns, 0 failing endpoints of 5,557).

### Bitstream caveats — read before using this file

1. **Pins are auto-placed.** No board XDC exists, so Vivado assigned all 182 I/O arbitrarily. This bitstream is **not loadable on any specific board**.
2. **Two DRCs were downgraded** from error to warning to permit the write:
   ```tcl
   set_property SEVERITY {Warning} [get_drc_checks NSTD-1]   ;# unspecified I/O standard
   set_property SEVERITY {Warning} [get_drc_checks UCIO-1]   ;# unconstrained logical port
   ```
   These exist precisely to catch what was done here. Acceptable for a fit-and-timing proof; **not for hardware**.
3. **The raw port interface is not a real host interface.** 182 parallel port bits is a simulation construct; a board build needs JTAG/UART/AXI (see `mnist_demo/` and `xor_demo/` for the DE1-SoC approach).

For an actual board: add a pin XDC, wrap `tpu` in a host interface, re-run, and confirm both DRCs pass on their own.

---

## 5. Resource utilization and where the area goes

Area is dominated by one module. Hierarchical breakdown (rev 1 measurement; proportions unchanged by the fixes):

| Instance | Module | Total LUTs | FFs | DSPs |
|---|---|---|---|---|
| `tpu` | (top) | **22,335** | 3,295 | 13 |
| `ub_inst` | `unified_buffer` | **21,638** | 2,859 | 3 |
| — `(ub_inst)` core | | **14,496** | 2,803 | 1 |
| — `gradient_descent_gen[0]` | `gradient_descent` | 2,577 | 39 | 1 |
| `vpu_inst` | `vpu` | 458 | 172 | 6 |
| — `loss_parent_inst` | | 152 | 34 | 2 |
| — `bias_parent_inst` | | 128 | 34 | 0 |
| — `leaky_relu_parent_inst` | | 126 | 34 | 2 |
| — `leaky_relu_derivative_parent_inst` | | 54 | 34 | 2 |
| `systolic_inst` | `systolic` (4 PEs) | 239 | 264 | 4 |
| — `pe11` / `pe12` / `pe21` / `pe22` | `pe` | 71 / 60 / 58 / 49 | 82/66/65/49 | 1 each |

### Issue #9 — the unified buffer is 97% of the design ⚠️ Open

The entire compute — full systolic array plus the whole VPU — is 697 LUTs, about 3%.

**Cause:** `ub_memory[0:127]` (`src/unified_buffer.sv:59`) is a 128×16 flip-flop array with an async-reset `for` loop and many concurrent read pointers (input, weight, bias, Y, H, gradient-bias, gradient-weight). It cannot infer as BRAM, so it becomes registers plus a very wide LUT mux tree: **~14.5k LUTs to store 2 KB**.

**Optimization:** convert to a synchronous, BRAM-inferable memory — drop the reset loop, register the read path, and reduce or time-multiplex the concurrent read ports. Expect roughly an order-of-magnitude LUT reduction and ~1 of 365 BRAMs. This is what frees room to scale past 2×2; the DSP budget (740) already permits far more than the 13 in use.

> Note: LUT count rose 21,268 → 22,265 (+4.7%) after the Issue #4 fix, because scoping the temporaries changed what synthesis could optimize away. Immaterial to the fit conclusion.

---

## 6. Timing

| Clock constraint | Stage | WNS | Result |
|---|---|---|---|
| 10 ns (100 MHz) | post-synth | **−6.081 ns** | **Fails** — TNS −690.224 ns, 192/5545 failing endpoints |
| 16 ns (62.5 MHz) | post-route, OOC | **+0.090 ns** | Passes |
| 16 ns (62.5 MHz) | post-route, bitstream flow | **+0.163 ns** | Passes |

Hold passes throughout. **Practical Fmax ≈ 63 MHz** — the margin means there is essentially nothing left at 16 ns.

### Issue #10 — the VPU critical path is combinational ⚠️ Open

```
Source:      ub_inst/ub_rd_Y_data_out_reg[0][0]/C          (FDCE)
Destination: vpu_inst/loss_parent_inst/first_column/gradient_out_reg[15]/D
Data Path Delay: 16.026 ns  (logic 9.134 ns / 57%, route 6.892 ns / 43%)
Logic Levels:    19  (CARRY4=9, DSP48E1=1, LUT6=3, LUT4=4, LUT5=1, LUT1=1)
```

The path runs **UB read → loss subtract → leaky-ReLU multiply (DSP) → gradient register** with no pipeline register between. The README describes the VPU modules as "pipelined," but the loss → activation path is combinational in practice. **A single pipeline register between `loss_parent` and `leaky_relu_parent` is the cheapest route past 63 MHz.** The nine chained CARRY4s make the Q8.8 `fxp_addsub` chain a secondary target.

---

## 7. Behavioral RTL simulation

**Setup:** `sva/tb_tpu.sv` (XOR-training sequence from `test/test_tpu.py`) with all ten `sva/*_assertions.sv` bound via `sva/bind_all_assertions.sv`, in xsim.

**Result:** completes at 1085 ns (~108 cycles), `$finish` reached normally, **zero X signals** (was 5 before the fixes), 14 valid output cycles with plausible Q8.8 values:

| valid # | `vpu_data_out_1` | `vpu_data_out_2` |
|---|---|---|
| 1 | `0040` (+0.250) | `0000` |
| 2 | `ffd6` (−0.164) | `0030` (+0.188) |
| 3 | `ffd6` (−0.164) | `ffeb` (−0.082) |
| 4 | `ff8c` (−0.453) | `ffeb` (−0.082) |

### Issue #7 — 207 assertion failures are an xsim bug ➖ Not a design bug

All 207 come from one module:

| Assertion | Count | Message |
|---|---|---|
| GD-A3 | 107 | `grad_descent_done_out != registered(valid_in)` |
| GD-A4 | 98 | `value_updated_out changed while valid_in=0` |
| GD-A2 | 1 | `rst did not clear grad_descent_done_out` |
| GD-A1 | 1 | `rst did not clear value_updated_out` |

Instrumenting the exact comparison shows the assertion firing on a **true** comparison:

```
Error: A3b t=15000 rst=0 done=0 past_v_reg=0 past_rst=0
Error: A4b t=15000 rst=0 upd=0000 past_upd_reg=0000
```

`done (0) == $past(valid_in) (0)` — and it fails anyway. Probing confirmed the signals are clean 0, never X.

**Reduced to a minimal reproducer:** an assertion using `$past`, bound into a module, gives **0 failures** when the module is instantiated normally and **11 failures** when the identical module is instantiated inside a `generate ... for` block. `unified_buffer` instantiates `gradient_descent` inside `gradient_descent_gen[i]` — exactly the trigger.

Consistent with `docs/fv_results.md` reporting 129/129 assertions passing under QuestaSim, which does not have the bug. **Action:** filter GD-A1..A4 as known tool noise in any xsim-based CI. The count is unchanged by the rev 2 fixes, as expected.

### Issue #8 — neither testbench checks correctness ⚠️ Open

`sva/tb_tpu.sv` sequences on `dut.vpu_valid_out_1` and `$display`s diagnostics, but **never compares outputs against expected values**. The expected XOR results exist only as comments in `test/test_tpu.py`. So "behavioral simulation passes" means *runs without assertion violations*, **not** *computes the right answer*.

The numerical assertions live in the cocotb suite, which could not be run: `iverilog` and `cocotb` are not installed, and cocotb does not support Vivado's xsim. **This is the highest-value testing improvement available** — porting the `test_tpu.py` value checks into `tb_tpu.sv` would make the SV testbench genuinely self-checking, and would turn Issue #6 below from "X appears" into "the answer is wrong, here."

---

## 8. Post-route gate-level timing simulation

**Setup:** OOC place-and-route, `write_verilog -mode timesim` (13.9 MB netlist) + `write_sdf` (41.1 MB SDF), xsim with real device delays, 16 ns clock.

Three corrections were needed to get a valid run, all worth recording:

1. **`$sdf_annotate` alone does not annotate.** Setup/hold limits print as `(0:0:0)` and the resulting "violations" are zero-delay races. Explicit `xelab -sdfmax tb_tpu_timing/dut=<file>.sdf` is required; success is confirmed by `INFO: [XSIM 43-3452] SDF backannotation was successful`.
2. **`glbl.GSR` starves the stimulus.** The global set/reset holds every FF in reset for the first 100 ns while the TB releases `rst` at 12 ns. Fixed with `initial force glbl.GSR = 1'b0;` — safe because every FF has an async CLR.
3. **Netlist ports are flattened.** `ub_wr_host_data_in[0:1]` becomes escaped scalar ports `\ub_wr_host_data_in[0]`, so the TB instantiation must be rewired and the deep-hierarchy `[COL2-DIAG]` monitor removed.

> Use the **OOC** netlist for functional gate-level simulation. The bitstream-flow netlist has IBUFs and a BUFG, so the internal clock lands ~5 ns after the TB's edge while the TB drives inputs at zero delay; it mis-samples for reasons unrelated to the RTL (measured: 0 valid cycles). This is a testbench/constraint artifact, not a design fault.

### Issue #6 — residual X-propagation ⚠️ Open (improved)

| Metric | Before fixes | After fixes |
|---|---|---|
| Simulation completes | ❌ hung to 50 µs watchdog | ✅ **completes at 1736 ns** |
| Valid output cycles | 4 | ✅ **14** (matches behavioral) |
| Column 1, cycles 1–4 | X from cycle 4 | ✅ **matches behavioral exactly** |
| Column 2 | X from cycle 2 | ⚠️ **still X from cycle 2** |
| Column 1, cycle 5+ | — | ⚠️ **X** |

Substantial progress — the design now runs the full sequence at gate level and column 1 tracks behavioral for the first four cycles — but **X is not eliminated**.

**Localized to:** `vpu_inst/bias_parent_inst/column_2/add_inst`, fed by `ub_rd_bias_data_out[1]` — the UB's bias read for channel 1. The X appears on carry-chain nets at bit positions 3, 7, 11, 14, 15.

**Root cause not identified.** The obvious hypothesis — that `ub_memory[rd_bias_ptr + i]` indexes past the 128-entry array with its 16-bit pointer — was **tested and disproved**: in behavioral simulation `rd_bias_ptr` never exceeds 126 and `ub_rd_bias_data_out_1` is never X. This is a third, distinct RTL-vs-gate divergence in the bias path, separate from Issues #3 and #4.

**Caveats on the violation counts:** 126 before → 344 after. Not a meaningful regression signal — Vivado writes FF setup/hold as `(0:0:0)` in SDF and relies on static timing for signoff, `-pulse_r 0` disables glitch filtering, and the post-fix run executes the real sequence instead of stalling early. §6 is the timing verdict, not this number.

---

## 9. Recommended next steps

Ordered by value:

1. **Close Issue #6** — root-cause the residual X in `bias_parent/column_2`. This is the gate for board bring-up.
2. **Close Issue #8** — port the expected-value assertions from `test/test_tpu.py` into `sva/tb_tpu.sv`. Without this, neither simulation proves numerical correctness, and Issue #6 is harder to diagnose than it needs to be.
3. **Build a real board target** — pin XDC + host interface (JTAG/UART/AXI), then confirm `NSTD-1` and `UCIO-1` pass without downgrade.
4. **Close Issue #9** — rework the UB for BRAM inference. Highest-leverage change for scaling; frees ~60% of design area.
5. **Close Issue #10** — pipeline the loss → activation path to move past 63 MHz.
6. **Filter GD-A1..A4** (Issue #7) in any xsim-based CI as known tool noise.

---

## 10. Reproducing this work

All artifacts live in ephemeral scratch directories and are not checked in. To regenerate from a clean tree:

**Synthesis, implementation and bitstream** (`bit.tcl`) — runs against `src/` directly, no patching needed as of rev 2:
```tcl
foreach f [lsort [glob /path/to/tiny-tpu/src/*.sv]] { read_verilog -sv $f }
synth_design -top tpu -part xc7a200tfbg484-1
create_clock -name clk -period 16.000 [get_ports clk]
opt_design; place_design; phys_opt_design; route_design
report_utilization -file util.rpt
report_timing_summary -file timing.rpt
set_property SEVERITY {Warning} [get_drc_checks NSTD-1]
set_property SEVERITY {Warning} [get_drc_checks UCIO-1]
write_bitstream -force tpu.bit
```

**OOC netlist for gate-level simulation** (`ooc.tcl`):
```tcl
foreach f [lsort [glob /path/to/tiny-tpu/src/*.sv]] { read_verilog -sv $f }
synth_design -top tpu -part xc7a200tfbg484-1 -mode out_of_context
create_clock -name clk -period 16.000 [get_ports clk]
opt_design; place_design; phys_opt_design; route_design
write_verilog -mode timesim -sdf_anno true -force ooc_ts.v
write_sdf -force ooc_ts.sdf
```

**Behavioral simulation:**
```bash
xvlog -sv src/*.sv sva/*_assertions.sv sva/bind_all_assertions.sv sva/tb_tpu.sv
xelab -debug typical tb_tpu bind_wrapper -s tbsim
xsim tbsim -R
```

**Gate-level timing simulation** (needs the TB adaptations in §8; ~12 min to elaborate):
```bash
xvlog -sv tb_tpu_timing.sv && xvlog ooc_ts.v
xvlog $XILINX_VIVADO/data/verilog/src/glbl.v
xelab -relax -maxdelay -transport_int_delays -pulse_r 0 -pulse_int_r 0 \
      -sdfmax tb_tpu_timing/dut=$PWD/ooc_ts.sdf \
      -L simprims_ver -L unisims_ver tb_tpu_timing glbl -s oocsim
xsim oocsim -R
```
