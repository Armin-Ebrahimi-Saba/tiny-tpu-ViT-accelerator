`timescale 1ns/1ps

// ABOUTME: Residual add of two tensors on different scales, bit-exact against qadd() in sw/kernels_int.py.
// ABOUTME: Both operands are rescaled keeping 12 guard bits, summed, then rounded exactly once.
//
// The guard bits are the whole point. Rescaling each operand to int8 before
// adding would inject up to half an LSB per add, and Depth Anything V2 Small has
// 24 residual adds in series -- the errors accumulate down the skip chain rather
// than cancelling. Keeping GUARD_BITS of sub-LSB precision through the sum and
// rounding once at the end costs a wider adder and nothing else.
//
// `apply_keep(x, GUARD)` shifts by `shift - GUARD`, which is a *signed* amount:
// a real multiplier above 1 gives a small shift, and the reference then shifts
// left instead of right. In practice residual scale ratios are close to 1 and
// the multiplier is normalized to [2**30, 2**31), so `shift` lands near 30 and
// the left-shift branch never fires -- but the reference has it, so it is here.
//
// OUT_MIN/OUT_MAX widen the result where the residual stream is carried at int16
// rather than int8, matching the reference's optional arguments.

module qadd_unit #(
    parameter int IN_BITS    = 8,
    parameter int OUT_BITS   = 8,
    parameter int GUARD_BITS = 12,   // _ADD_GUARD_BITS
    parameter int WIDE_BITS  = 64,
    parameter int OUT_MIN    = -127,
    parameter int OUT_MAX    =  127
)(
    input  logic clk,
    input  logic rst,

    input  logic                        valid_in,
    input  logic signed [IN_BITS-1:0]   a_in,
    input  logic signed [IN_BITS-1:0]   b_in,
    input  logic signed [31:0]          mult_a,
    input  logic signed [31:0]          mult_b,
    input  logic [5:0]                  shift_a,
    input  logic [5:0]                  shift_b,

    output logic                        valid_out,
    output logic signed [OUT_BITS-1:0]  out
);

    // Stage 1: the two multiplies.
    logic                       v1;
    logic signed [WIDE_BITS-1:0] prod_a, prod_b;
    logic [5:0]                 sha_q, shb_q;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            v1 <= 1'b0;
        end else begin
            v1     <= valid_in;
            prod_a <= WIDE_BITS'(a_in) * WIDE_BITS'(mult_a);
            prod_b <= WIDE_BITS'(b_in) * WIDE_BITS'(mult_b);
            sha_q  <= shift_a;
            shb_q  <= shift_b;
        end
    end

    // Stage 2: apply_keep on each operand independently.
    function automatic logic signed [WIDE_BITS-1:0] keep_shift(
        input logic signed [WIDE_BITS-1:0] prod,
        input logic [5:0]                  shift
    );
        logic signed [WIDE_BITS-1:0] half;
        int sh;
        sh = int'(shift) - GUARD_BITS;
        if (sh > 0) begin
            half = (sh == 0) ? '0 : (WIDE_BITS'(1) <<< (sh - 1));
            return (prod + half) >>> sh;
        end else begin
            return prod <<< (-sh);
        end
    endfunction

    logic                        v2;
    logic signed [WIDE_BITS-1:0] wide_a, wide_b;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            v2 <= 1'b0;
        end else begin
            v2     <= v1;
            wide_a <= keep_shift(prod_a, sha_q);
            wide_b <= keep_shift(prod_b, shb_q);
        end
    end

    // Stage 3: one sum, one rounding, one saturation.
    wire signed [WIDE_BITS-1:0] acc     = wide_a + wide_b;
    wire signed [WIDE_BITS-1:0] half    = WIDE_BITS'(1) <<< (GUARD_BITS - 1);
    wire signed [WIDE_BITS-1:0] rounded = (acc + half) >>> GUARD_BITS;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_out <= 1'b0;
            out       <= '0;
        end else begin
            valid_out <= v2;
            if (rounded > WIDE_BITS'(OUT_MAX))      out <= OUT_BITS'(OUT_MAX);
            else if (rounded < WIDE_BITS'(OUT_MIN)) out <= OUT_BITS'(OUT_MIN);
            else                                    out <= OUT_BITS'(rounded);
        end
    end

endmodule
