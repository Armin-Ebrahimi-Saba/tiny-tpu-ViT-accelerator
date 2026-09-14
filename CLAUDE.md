# Goal

Run the smallest Depth Anything V2 (ViT-S/14) on the board. Model weights
live in DDR3. Write in the student-related files; do not change third-party
libraries if it can be avoided. Read `docs/` first, and `report.md` for how
every piece was arrived at.

# Hardware

Xilinx Artix-7 XC7A200T (Nexys Video), `xc7a200tsbg484-1`, 50 MHz.

# Layout

- `src/v2/` — the accelerator RTL. The Python emulator in `sw/` is the
  specification; every module is checked bit-exactly against it by
  `test/v2/run.sh` (Verilator). `sw/tiling.py` sizes the buffers.
- `rvlab/` — git submodule: the SoC (CV32E40P, TL-UL, DDR3, PyDesignFlow).
  `rvlab/src/rtl/tinytpu` is a symlink back to `src/v2`; the peripheral,
  its register map and the driver are `rvlab/src/rtl/student/tinytpu*.sv`,
  `rvlab/src/design/reggen/tinytpu.hjson`, `rvlab/src/sw/project/main.c`.
- `docs/LESSONS.md` — portable rules from a sibling project's bring-up on the
  same SoC. Read before debugging anything on the bus. `docs/DEBUGGING.md` is
  the full account behind them, including the wrong theories.

# Run

`flow` is `.venv/bin/flow`, run from `rvlab/`, as `flow <block> <task>`.
With no arguments it lists every target and its status.

    # 1. module testbenches, seconds (from the repo root)
    test/v2/run.sh

    # 2. whole SoC, no DDR3 -- ~12 minutes; the end-to-end check
    flow systb_project sim_rtl_xsim_batch
    grep hostio build/systb_project/sim_rtl_xsim_batch/xsim.log

    # 3. software, after any *.hjson change: BOTH, in this order
    flow reggen generate
    flow libsys build -R && flow sw_project build -R

    # 4. hardware
    flow rvlab_fpga_top syn
    flow rvlab_fpga_top pnr
    flow rvlab_fpga_top bitstream        # builds syn+pnr if missing
    flow rvlab_fpga_top program

A finished target is not rebuilt when its sources change. Force it with
`flow rvlab_fpga_top <task> --clean` for each of bitstream, pnr, syn, then
run bitstream; or `-R` to rebuild dependencies. A run that reports numbers
identical to the last one to the byte did not run -- check the timestamps.

DDR3 simulation is very slow, do not do it. `srcs_noddr` leaves
`WITH_EXT_DRAM` undefined, so the batch simulation has no DDR3 at all;
anything on that path is checked by synthesis and on the board.

After running bitstream, pnr and syn check the following files for warnings
and fix them without changing third party libraries if possible:

- rvlab/build/rvlab_fpga_top/bitstream/rvlab_fpga_top.io_report.txt
- rvlab/build/rvlab_fpga_top/syn/rvlab_fpga_top.*.txt
- rvlab/build/rvlab_fpga_top/pnr/rvlab_fpga_top.*.txt

The warnings that remain are listed with reasons in `report.md` §6.13; the
ones rooted in OpenTitan's `prim_subreg` async reset are third-party.

# Traps

**Adding a register to an `.hjson` shifts every register after it.** Append
new registers at the end. If one must go in the middle, rebuild `libsys`
*and* `sw_project`; software built against a stale map writes control bits
into the wrong address and fails in confusing ways. reggen's generated
`*_MASK` is the field width, not a mask in place.

**`pkill -f pattern` matches its own command line** and kills the shell
running it. Use `pgrep -f "[p]attern"`. Killing the simulator child (`xsimk`)
leaves the `flow` parent holding its memory; killing the parent leaves an
`xsim --gui` window alive with a dead kernel. `pkill -f "[x]sim"` (no
trailing `k`) gets both.

**The vector unit's in-place ops depend on the reader leading the packer by
a word.** Softmax over a slot in place is safe; a transpose in place is not.

# Debug

**Define a hardware register when the debugger can't answer.** A CPU wedged
on a bus request that will never be answered cannot retire the instruction,
so it cannot enter debug mode: the debugger reports "running" and every
register read comes back empty, and there is no PC to fetch. JTAG system-bus
access still works. `rvlab/src/rtl/student/student_tl_watch.sv` latches the
oldest unanswered request on the DDR3 port -- address, opcode, source,
outstanding count, saturating stall counter -- into `ddr_ctrl.wdog_addr` /
`wdog_stat`. It drives nothing, so it cannot disturb what it watches. Add
counters of issued and acknowledged requests to any new bus master before
they are needed.

Attaching the debugger resets the core: every PC read after an attach is of a
freshly restarted program, not a stalled one. Target-side printing is a hang
generator -- the hostio ring drains at tens of bytes per second. When a run
may hang, check progress every 30-60 s rather than waiting on it.
