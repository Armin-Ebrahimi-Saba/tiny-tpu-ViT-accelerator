`timescale 1ns/1ps

// ABOUTME: One tpu-v2 GEMM tile: int8 systolic array followed by per-output-channel requantization.
// ABOUTME: End-to-end equivalent of qlinear() in sw/kernels_int.py for a single ROWS x COLS tile.
//
//     out[m][n] = sat(round_half_up((bias[n] + sum_k x[m][k]*w[k][n]) * mult[n] / 2**shift[n]))
//
// Each column owns its own multiplier and shift, which is what makes per-channel
// requantization free here: the requant constants are indexed by output column,
// so they live in a COLS-entry register file rather than in the datapath.
//
// Latency from a column's psum to its int8 output is 2 cycles (the requant
// pipeline); the array's own latency is described in systolic_int8.sv.

module gemm_tile_int8 #(
    parameter int ROWS     = 16,
    parameter int COLS     = 16,
    parameter int ACC_BITS = 32,
    parameter int OUT_BITS = 8,
    parameter int OUT_MIN  = -127,
    parameter int OUT_MAX  =  127
)(
    input  logic clk,
    input  logic rst,

    // Activations, west edge
    input  logic signed [7:0]          data_in   [ROWS],
    input  logic                       valid_in  [ROWS],
    input  logic                       switch_in,

    // Weights and bias, north edge
    input  logic signed [7:0]          weight_in [COLS],
    input  logic                       accept_w  [COLS],
    input  logic signed [ACC_BITS-1:0] bias_in   [COLS],

    // Per-output-channel requantization constants
    input  logic signed [31:0]         mult_in   [COLS],
    input  logic [5:0]                 shift_in  [COLS],

    input  logic [$clog2(COLS+1)-1:0]  active_cols,

    // Requantized activations, south edge
    output logic signed [OUT_BITS-1:0] out       [COLS],
    output logic                       out_valid [COLS]
);

    logic signed [ACC_BITS-1:0] psum   [COLS];
    logic                       psum_v [COLS];

    systolic_int8 #(
        .ROWS(ROWS), .COLS(COLS), .ACC_BITS(ACC_BITS)
    ) u_array (
        .clk(clk), .rst(rst),
        .data_in(data_in), .valid_in(valid_in), .switch_in(switch_in),
        .weight_in(weight_in), .accept_w(accept_w), .bias_in(bias_in),
        .active_cols(active_cols),
        .psum_out(psum), .valid_out(psum_v)
    );

    for (genvar c = 0; c < COLS; c++) begin : g_requant
        requant_unit #(
            .ACC_BITS(ACC_BITS), .OUT_BITS(OUT_BITS),
            .OUT_MIN(OUT_MIN), .OUT_MAX(OUT_MAX)
        ) u_rq (
            .clk(clk), .rst(rst),
            .valid_in(psum_v[c]),
            .acc_in(psum[c]),
            .mult_in(mult_in[c]),
            .shift_in(shift_in[c]),
            .valid_out(out_valid[c]),
            .out(out[c])
        );
    end

endmodule
