`timescale 1ns/1ps

// ABOUTME: Fully pipelined floor divider, one result per cycle, for the softmax and LayerNorm datapaths.
// ABOUTME: Same bit-exact semantics as divider.sv -- Python's `//`, rounding toward -infinity.
//
// Why this exists alongside the sequential divider.sv: both references divide
// once per *element*, not once per row.
//
//   qsoftmax    probs = (e << 15) // total     11.3 M divisions per ViT-S block
//   qlayernorm  norm  = (d << 14) // r          1.1 M divisions per block
//
// At 32 cycles each that is ~7.2 s per transformer block at 50 MHz, against
// ~300 ms for all the GEMMs in the same block -- the divider would become 95% of
// the runtime. Unrolling the restoring loop into one stage per quotient bit
// turns it into 1 result/cycle for roughly NUM_BITS x (DEN_BITS+1) LUTs, which
// on this part is a rounding error next to the systolic array.
//
// The obvious cheaper alternative -- reciprocate once per row and multiply --
// is rejected on purpose: it does not reproduce a per-element floor division,
// and sw/kernels_int.py is the specification, not an approximation of it.

module divider_pipe #(
    parameter int NUM_BITS = 32,   // signed numerator width
    parameter int DEN_BITS = 32    // unsigned divisor width
)(
    input  logic clk,
    input  logic rst,

    input  logic                       valid_in,
    input  logic signed [NUM_BITS-1:0] num_in,
    input  logic        [DEN_BITS-1:0] den_in,

    output logic                       valid_out,
    output logic signed [NUM_BITS-1:0] quot_out,
    output logic                       div_zero
);

    // Stage 0 holds the registered inputs; stage i has i quotient bits resolved.
    logic [NUM_BITS-1:0] s_num  [NUM_BITS+1];
    logic [NUM_BITS-1:0] s_quot [NUM_BITS+1];
    logic [DEN_BITS:0]   s_rem  [NUM_BITS+1];
    logic [DEN_BITS-1:0] s_den  [NUM_BITS+1];
    logic                s_neg  [NUM_BITS+1];
    logic                s_dz   [NUM_BITS+1];
    logic                s_val  [NUM_BITS+1];

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            s_val[0] <= 1'b0;
        end else begin
            s_val[0]  <= valid_in;
            s_num[0]  <= num_in[NUM_BITS-1] ? NUM_BITS'(-num_in) : NUM_BITS'(num_in);
            s_neg[0]  <= num_in[NUM_BITS-1];
            s_den[0]  <= den_in;
            s_dz[0]   <= (den_in == '0);
            s_quot[0] <= '0;
            s_rem[0]  <= '0;
        end
    end

    for (genvar i = 0; i < NUM_BITS; i++) begin : g_step
        // Shift the next numerator bit in at the bottom of the remainder and
        // subtract the divisor if it fits. A zero divisor "fits" every time, so
        // it is masked out here and the quotient is forced to 0 at the end.
        wire [DEN_BITS:0] rem_sh = {s_rem[i][DEN_BITS-1:0], s_num[i][NUM_BITS-1]};
        wire              fits   = !s_dz[i] && (rem_sh >= {1'b0, s_den[i]});

        always_ff @(posedge clk or posedge rst) begin
            if (rst) begin
                s_val[i+1] <= 1'b0;
            end else begin
                s_val[i+1]  <= s_val[i];
                s_num[i+1]  <= {s_num[i][NUM_BITS-2:0], 1'b0};
                s_rem[i+1]  <= fits ? (rem_sh - {1'b0, s_den[i]}) : rem_sh;
                s_quot[i+1] <= {s_quot[i][NUM_BITS-2:0], fits};
                s_den[i+1]  <= s_den[i];
                s_neg[i+1]  <= s_neg[i];
                s_dz[i+1]   <= s_dz[i];
            end
        end
    end

    // Floor correction. Truncation is already correct for a non-negative
    // numerator and for an exact division; only an inexact negative has to round
    // away from zero, which a non-zero remainder detects.
    localparam int L = NUM_BITS;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_out <= 1'b0;
            quot_out  <= '0;
            div_zero  <= 1'b0;
        end else begin
            valid_out <= s_val[L];
            div_zero  <= s_dz[L];
            if (s_dz[L]) begin
                quot_out <= '0;
            end else if (s_neg[L]) begin
                quot_out <= (s_rem[L] != '0) ? -NUM_BITS'(s_quot[L] + NUM_BITS'(1))
                                             : -NUM_BITS'(s_quot[L]);
            end else begin
                quot_out <= NUM_BITS'(s_quot[L]);
            end
        end
    end

endmodule
