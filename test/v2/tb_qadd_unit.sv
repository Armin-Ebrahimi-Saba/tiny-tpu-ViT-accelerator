`timescale 1ns/1ps

// ABOUTME: Self-checking testbench proving qadd_unit matches qadd() in sw/kernels_int.py.
// ABOUTME: Drives back to back so the three pipeline stages must keep operands paired.

module tb_qadd_unit;

    `include "params.svh"

    localparam int W = 8 + 8 + 32 + 8 + 32 + 8 + 8;
    localparam int LATENCY = 4;   // one TB stimulus register plus the DUT's three

    logic clk = 1'b0;
    logic rst = 1'b1;
    always #5 clk = ~clk;

    logic [W-1:0] vec [N_QADD];

    logic               valid_in;
    logic signed [7:0]  a_in, b_in;
    logic signed [31:0] mult_a, mult_b;
    logic [5:0]         shift_a, shift_b;
    logic               valid_out;
    logic signed [7:0]  out;

    qadd_unit #(.IN_BITS(8), .OUT_BITS(8)) dut (
        .clk(clk), .rst(rst),
        .valid_in(valid_in), .a_in(a_in), .b_in(b_in),
        .mult_a(mult_a), .mult_b(mult_b), .shift_a(shift_a), .shift_b(shift_b),
        .valid_out(valid_out), .out(out)
    );

    int drive_i  = 0;
    int checked  = 0;
    int failures = 0;

    initial begin
        $readmemh("vec/qadd.hex", vec);
        valid_in = 1'b0;
        repeat (4) @(posedge clk);
        rst <= 1'b0;
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            valid_in <= 1'b0;
        end else if (drive_i < N_QADD) begin
            valid_in <= 1'b1;
            a_in     <= $signed(vec[drive_i][W-1     -: 8]);
            b_in     <= $signed(vec[drive_i][W-1-8   -: 8]);
            mult_a   <= $signed(vec[drive_i][W-1-16  -: 32]);
            shift_a  <= 6'(vec[drive_i][W-1-48  -: 8]);
            mult_b   <= $signed(vec[drive_i][W-1-56  -: 32]);
            shift_b  <= 6'(vec[drive_i][W-1-88  -: 8]);
            drive_i  <= drive_i + 1;
        end else begin
            valid_in <= 1'b0;
        end
    end

    always_ff @(posedge clk) begin
        if (!rst && valid_out) begin
            if (out !== $signed(vec[checked][0 +: 8])) begin
                $display("FAIL qadd[%0d]: got %0d, expected %0d",
                         checked, out, $signed(vec[checked][0 +: 8]));
                failures <= failures + 1;
            end
            checked <= checked + 1;
        end
    end

    initial begin
        wait (checked == N_QADD);
        @(posedge clk);
        if (failures != 0) $fatal(1, "qadd_unit: %0d/%0d mismatched", failures, checked);
        $display("PASS qadd_unit: %0d/%0d residual adds bit-exact vs qadd()", checked, N_QADD);
        $finish;
    end

    initial begin
        #1_000_000;
        $fatal(1, "qadd_unit: timeout after %0d/%0d checks", checked, N_QADD);
    end

endmodule
