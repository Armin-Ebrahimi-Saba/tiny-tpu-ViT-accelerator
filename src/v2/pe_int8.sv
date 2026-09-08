`timescale 1ns/1ps

// ABOUTME: Weight-stationary int8 x int8 -> int32 processing element for the tpu-v2 inference array.
// ABOUTME: Mirrors sw/kernels_int.py exact_int_matmul exactly: no rounding, no saturation, plain integer MAC.
//
// Dataflow (classic weight-stationary TPU):
//   - the shadow registers of a column form a shift chain: each PE hands its own
//     shadow value south, so ROWS load beats spread w[0..ROWS-1] down the column
//   - `switch_in` promotes shadow to active, propagating down the diagonal so
//     every PE switches on the cycle its activation arrives
//   - activations flow west -> east, one row per reduction index k
//   - partial sums flow north -> south, accumulating y[m][n] = sum_k x[m][k]*w[k][n]
//
// `accept_w_in` is a column-wide broadcast, NOT a propagating pulse. Every PE in
// the column must shift on the same cycle or the chain stalls; if instead each PE
// mirrored its own input, all of them would end the load holding the last weight
// driven rather than one weight each.
//
// PE(r, c) holds w[r][c]. Row r must be presented its activation r cycles after
// row 0, which is the standard staggered feed the scheduler in sw/lower.py assumes.

module pe_int8 #(
    parameter int ACC_BITS = 32
)(
    input  logic clk,
    input  logic rst,

    // North edge
    input  logic signed [ACC_BITS-1:0] psum_in,
    input  logic signed [7:0]          weight_in,
    input  logic                       accept_w_in,

    // West edge
    input  logic signed [7:0]          data_in,
    input  logic                       valid_in,
    input  logic                       switch_in,
    input  logic                       enable,

    // South edge
    output logic signed [ACC_BITS-1:0] psum_out,
    output logic signed [7:0]          weight_out,

    // East edge
    output logic signed [7:0]          data_out,
    output logic                       valid_out,
    output logic                       switch_out
);

    logic signed [7:0] weight_active;
    logic signed [7:0] weight_shadow;

    // int8 x int8 fits in 16 bits (|-127 * -127| = 16129), so the product is
    // always exact and the sign-extended add into ACC_BITS cannot lose bits.
    // A 16x16 column accumulates at most 16 * 16129 = 258064, far inside int32;
    // the int32 width exists to carry the int32 bias injected at row 0.
    wire signed [15:0]          product = data_in * weight_active;
    wire signed [ACC_BITS-1:0]  mac     = psum_in + ACC_BITS'(product);

    // The shift chain's output is the register itself, so the PE below receives
    // this PE's previous weight on the next beat.
    assign weight_out = weight_shadow;

    always_ff @(posedge clk or posedge rst) begin
        if (rst || !enable) begin
            data_out      <= 8'sd0;
            weight_active <= 8'sd0;
            weight_shadow <= 8'sd0;
            valid_out     <= 1'b0;
            switch_out    <= 1'b0;
            psum_out      <= '0;
        end else begin
            valid_out  <= valid_in;
            switch_out <= switch_in;

            if (switch_in) begin
                weight_active <= weight_shadow;
            end

            if (accept_w_in) begin
                weight_shadow <= weight_in;
            end

            if (valid_in) begin
                data_out <= data_in;
                psum_out <= mac;
            end else begin
                data_out <= 8'sd0;
                psum_out <= '0;
            end
        end
    end

endmodule
