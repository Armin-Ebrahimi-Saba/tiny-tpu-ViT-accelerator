#!/usr/bin/env bash
# Verilator testbench for rvlab's tinytpu DMA (copy, im2col gather, write-back),
# against a TL-UL memory model. Seconds; the SoC simulation has no DDR3 and the
# board is the only other place these modes run.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
rv="$here/../../rvlab/src/rtl"
build="$here/build"
mkdir -p "$build"
verilator --binary -j 0 --quiet -Wall -Wno-fatal \
    -Wno-BLKSEQ -Wno-INITIALDLY -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM -Wno-DECLFILENAME \
    --timing -I"$rv/inc" --Mdir "$build/tb_tinytpu_dma" -o tb_tinytpu_dma \
    --top-module tb_tinytpu_dma \
    "$rv/inc/prim_assert.sv" "$rv/rvlab_fpga/pkg/top_pkg.sv" "$rv/tlul/pkg/tlul_pkg.sv" \
    "$rv/rv_dm/tlul_adapter_host.sv" "$rv/student/tinytpu_wdma.sv" \
    "$here/tb_tinytpu_dma.sv"
"$build/tb_tinytpu_dma/tb_tinytpu_dma" | tee "$build/tb_tinytpu_dma.log"
grep -q "PASS" "$build/tb_tinytpu_dma.log"
