`timescale 1ns/1ps

// ABOUTME: int32 accumulator -> int8 (or int16) requantizer: sat(round_half_up(acc * mult / 2**shift)).
// ABOUTME: Bit-exact mirror of Requant.apply in sw/numerics.py; multiplier is normalized to [2**30, 2**31).
//
// Two pipeline stages so the 32x32 multiply gets a full cycle of its own:
//   s1: prod = acc * mult                        (one DSP48 cascade)
//   s2: out  = saturate((prod + 2**(shift-1)) >>> shift)
//
// The +2**(shift-1) before an arithmetic right shift is round-half-up (ties go
// toward +inf, not away from zero). sw/numerics.py does exactly this, and the
// difference matters on negative ties, so do not "fix" it to round-half-away.
//
// shift == 0 is a pass-through with no rounding term. The offline normalizer
// never emits it (a multiplier in [2**30, 2**31) with shift 0 would mean a real
// multiplier >= 2**30), but defining the case keeps the unit total.

module requant_unit #(
    parameter int ACC_BITS      = 32,
    parameter int OUT_BITS      = 8,
    parameter int INTER_BITS    = 64,   // see machines/tpu_v2.json: intermediate_bits
    parameter int OUT_MIN       = -127,
    parameter int OUT_MAX       =  127
)(
    input  logic clk,
    input  logic rst,

    input  logic                       valid_in,
    input  logic signed [ACC_BITS-1:0] acc_in,
    input  logic signed [31:0]         mult_in,    // normalized to [2**30, 2**31)
    input  logic [5:0]                 shift_in,   // 0..62

    output logic                       valid_out,
    output logic signed [OUT_BITS-1:0] out
);

    // ---- stage 1: multiply -------------------------------------------------
    logic signed [INTER_BITS-1:0] prod_q;
    logic [5:0]                   shift_q;
    logic                         valid_q;

    wire signed [INTER_BITS-1:0] prod = INTER_BITS'(acc_in) * INTER_BITS'(mult_in);

    always_ff @(posedge clk) begin
        if (rst) begin
            prod_q  <= '0;
            shift_q <= '0;
            valid_q <= 1'b0;
        end else begin
            prod_q  <= prod;
            shift_q <= shift_in;
            valid_q <= valid_in;
        end
    end

    // ---- stage 2: round, shift, saturate -----------------------------------
    // Widen by one bit so prod + half cannot wrap: |prod| < 2**62 and
    // half <= 2**61, so the sum needs 64 bits of magnitude plus sign.
    wire signed [INTER_BITS:0] half    = (shift_q == 6'd0)
                                       ? '0
                                       : ((INTER_BITS+1)'(1) <<< (shift_q - 6'd1));
    wire signed [INTER_BITS:0] rounded = (INTER_BITS+1)'(prod_q) + half;
    wire signed [INTER_BITS:0] shifted = rounded >>> shift_q;

    always_ff @(posedge clk) begin
        if (rst) begin
            out       <= '0;
            valid_out <= 1'b0;
        end else begin
            valid_out <= valid_q;
            if (shifted > (INTER_BITS+1)'(OUT_MAX))      out <= OUT_BITS'(OUT_MAX);
            else if (shifted < (INTER_BITS+1)'(OUT_MIN)) out <= OUT_BITS'(OUT_MIN);
            else                                         out <= OUT_BITS'(shifted);
        end
    end

endmodule
