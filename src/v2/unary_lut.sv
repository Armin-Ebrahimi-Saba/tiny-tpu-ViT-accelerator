`timescale 1ns/1ps

// ABOUTME: 256-entry int8->int8 lookup table for GELU and any other scalar activation.
// ABOUTME: Bit-exact mirror of apply_unary_lut() in sw/kernels_int.py.
//
// The table is indexed by ``x_q + 128``, which for a two's-complement byte is
// just the sign bit inverted -- no adder, and the ROM is addressed directly by
// the raw activation. The contents are built offline by build_unary_lut() and
// loaded over the write port, so one instance serves GELU, leaky-ReLU, or any
// other elementwise function without respinning the bitstream.
//
// On a 7-series part a 256x8 table infers as distributed RAM (a handful of
// LUTRAMs), not a BRAM, which keeps the block RAM free for the unified buffer.

module unary_lut #(
    parameter int ENTRIES   = 256,
    parameter int DATA_BITS = 8
)(
    input  logic clk,
    input  logic rst,

    // Table load port
    input  logic                         wr_en,
    input  logic [$clog2(ENTRIES)-1:0]   wr_addr,
    input  logic signed [DATA_BITS-1:0]  wr_data,

    // Lookup
    input  logic                         valid_in,
    input  logic signed [7:0]            x_in,
    output logic                         valid_out,
    output logic signed [DATA_BITS-1:0]  y_out
);

    logic signed [DATA_BITS-1:0] table_q [ENTRIES];

    // x_q + 128 on a signed byte is the sign bit flipped.
    wire [7:0] index = {~x_in[7], x_in[6:0]};

    always_ff @(posedge clk) begin
        if (wr_en) begin
            table_q[wr_addr] <= wr_data;
        end
    end

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            y_out     <= '0;
            valid_out <= 1'b0;
        end else begin
            valid_out <= valid_in;
            y_out     <= table_q[index];
        end
    end

endmodule
