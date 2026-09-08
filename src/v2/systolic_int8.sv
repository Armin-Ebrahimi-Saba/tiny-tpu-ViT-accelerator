`timescale 1ns/1ps

// ABOUTME: Parameterized ROWS x COLS weight-stationary int8 systolic array for tpu-v2.
// ABOUTME: Computes y[m][n] = bias[n] + sum_k x[m][k]*w[k][n] in exact int32, no rounding anywhere.
//
// The array is pure structure: every timing decision (weight load order, the
// staggered activation feed, when to raise switch) belongs to the control unit.
// That keeps this module trivially comparable against sw/kernels_int.py.
//
// Feeding contract:
//   - weights: hold accept_w[c] for ROWS cycles while driving weight_in[c] in
//     reverse k order (w[ROWS-1][c] first, w[0][c] last). accept_w is broadcast
//     to the whole column at once; the shadow registers are the shift chain.
//   - switch: pulse switch_in once; it walks the diagonal
//   - activations: present x[m][k] on data_in[k] with valid_in[k] high, at cycle
//     t0 + k + m (row k lags row 0 by k cycles)
//   - bias: hold bias_in[n] for the duration of the pass; it enters the top of
//     column n and is carried down with the partial sum

module systolic_int8 #(
    parameter int ROWS     = 16,
    parameter int COLS     = 16,
    parameter int ACC_BITS = 32
)(
    input  logic clk,
    input  logic rst,

    // West edge: one activation lane per reduction index k
    input  logic signed [7:0]          data_in   [ROWS],
    input  logic                       valid_in  [ROWS],
    input  logic                       switch_in,

    // North edge: one weight lane and one bias per output column n
    input  logic signed [7:0]          weight_in [COLS],
    input  logic                       accept_w  [COLS],
    input  logic signed [ACC_BITS-1:0] bias_in   [COLS],

    // Column enable: columns >= active_cols are held in reset so an N-wide
    // array can run a narrower tile without emitting garbage.
    input  logic [$clog2(COLS+1)-1:0]  active_cols,

    // South edge
    output logic signed [ACC_BITS-1:0] psum_out  [COLS],
    output logic                       valid_out [COLS]
);

    // [row][col] meshes. Index r is the reduction axis k, index c the output n.
    logic signed [7:0]          w_h    [ROWS+1][COLS];
    logic signed [ACC_BITS-1:0] psum_h [ROWS+1][COLS];

    logic signed [7:0]          d_h    [ROWS][COLS+1];
    logic                       v_h    [ROWS][COLS+1];
    logic                       sw_h   [ROWS][COLS+1];

    logic col_enable [COLS];

    for (genvar c = 0; c < COLS; c++) begin : g_north
        assign w_h[0][c]     = weight_in[c];
        assign psum_h[0][c]  = bias_in[c];
        assign col_enable[c] = ($clog2(COLS+1)'(c) < active_cols);
    end

    for (genvar r = 0; r < ROWS; r++) begin : g_west
        assign d_h[r][0]  = data_in[r];
        assign v_h[r][0]  = valid_in[r];
        // The switch pulse enters at the top-left and walks the diagonal: each
        // PE passes it east, and row r takes it from row r-1's column 0 output.
        assign sw_h[r][0] = (r == 0) ? switch_in : sw_h[r-1][1];
    end

    for (genvar r = 0; r < ROWS; r++) begin : g_row
        for (genvar c = 0; c < COLS; c++) begin : g_col
            pe_int8 #(.ACC_BITS(ACC_BITS)) u_pe (
                .clk         (clk),
                .rst         (rst),
                .enable      (col_enable[c]),

                .psum_in     (psum_h[r][c]),
                .weight_in   (w_h[r][c]),
                .accept_w_in (accept_w[c]),

                .data_in     (d_h[r][c]),
                .valid_in    (v_h[r][c]),
                .switch_in   (sw_h[r][c]),

                .psum_out    (psum_h[r+1][c]),
                .weight_out  (w_h[r+1][c]),

                .data_out    (d_h[r][c+1]),
                .valid_out   (v_h[r][c+1]),
                .switch_out  (sw_h[r][c+1])
            );
        end
    end

    for (genvar c = 0; c < COLS; c++) begin : g_south
        assign psum_out[c] = psum_h[ROWS][c];
        // Valid rides east with the data, so column c is ready on the cycle the
        // bottom row's PE in that column reports valid.
        assign valid_out[c] = v_h[ROWS-1][c+1];
    end

endmodule
