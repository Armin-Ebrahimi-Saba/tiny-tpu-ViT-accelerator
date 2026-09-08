# ABOUTME: Out-of-context synthesis of one 16x16 int8 GEMM tile on the xc7a200t, for area/Fmax sizing.
# ABOUTME: Reports utilization and the critical path so tpu-v2 can be budgeted before a full top-level exists.
set root [file normalize [file dirname [info script]]/../../..]
read_verilog -sv [list \
    $root/src/v2/pe_int8.sv \
    $root/src/v2/systolic_int8.sv \
    $root/src/v2/requant_unit.sv \
    $root/src/v2/gemm_tile_int8.sv ]
synth_design -top gemm_tile_int8 -part xc7a200tfbg484-1 -mode out_of_context \
    -generic ROWS=16 -generic COLS=16
create_clock -period 20.000 -name clk [get_ports clk]
report_utilization -file $root/test/v2/synth/utilization.rpt
report_timing_summary -delay_type max -max_paths 5 -file $root/test/v2/synth/timing.rpt
