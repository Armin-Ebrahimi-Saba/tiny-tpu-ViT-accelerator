`timescale 1ns/1ps

// ABOUTME: Self-checking testbench proving divider matches Python's floor division.
// ABOUTME: Vectors include exact and inexact negatives, which is where truncation would diverge.

module tb_divider;

    `include "params.svh"

    localparam int NB = DIV_NUM_BITS;
    localparam int DB = DIV_DEN_BITS;
    localparam int W  = NB + DB + NB;

    logic clk = 1'b0;
    logic rst = 1'b1;
    always #5 clk = ~clk;

    logic [W-1:0] vec [N_DIV];

    logic                 start;
    logic signed [NB-1:0] num_in;
    logic        [DB-1:0] den_in;
    logic                 ready;
    logic                 valid_out;
    logic signed [NB-1:0] quot_out;
    logic                 div_zero;

    divider #(.NUM_BITS(NB), .DEN_BITS(DB)) dut (
        .clk(clk), .rst(rst),
        .start(start), .num_in(num_in), .den_in(den_in), .ready(ready),
        .valid_out(valid_out), .quot_out(quot_out), .div_zero(div_zero)
    );

    int failures = 0;
    int checked  = 0;

    initial begin
        automatic logic signed [NB-1:0] expect_q;

        $readmemh("vec/div.hex", vec);

        start = 1'b0;
        repeat (4) @(posedge clk);
        rst <= 1'b0;
        @(posedge clk);

        for (int i = 0; i < N_DIV; i++) begin
            num_in   = $signed(vec[i][W-1 -: NB]);
            den_in   = vec[i][NB +: DB];
            expect_q = $signed(vec[i][0 +: NB]);

            wait (ready);
            @(posedge clk);
            start <= 1'b1;
            @(posedge clk);
            start <= 1'b0;

            wait (valid_out);
            if (quot_out !== expect_q || div_zero !== 1'b0) begin
                $display("FAIL div[%0d]: %0d / %0d got %0d (dz=%0b), expected %0d",
                         i, num_in, den_in, quot_out, div_zero, expect_q);
                failures++;
            end
            checked++;
            @(posedge clk);
        end

        // Division by zero must report rather than hang.
        wait (ready);
        @(posedge clk);
        num_in <= 64'sd42;
        den_in <= '0;
        start  <= 1'b1;
        @(posedge clk);
        start <= 1'b0;
        wait (valid_out);
        if (div_zero !== 1'b1 || quot_out !== '0) begin
            $display("FAIL div-by-zero: got q=%0d dz=%0b, expected q=0 dz=1", quot_out, div_zero);
            failures++;
        end
        @(posedge clk);

        if (failures != 0) $fatal(1, "divider: %0d/%0d mismatched", failures, checked);
        $display("PASS divider: %0d/%0d floor divisions bit-exact vs Python //", checked, N_DIV);
        $finish;
    end

    initial begin
        #5_000_000;
        $fatal(1, "divider: timeout after %0d/%0d checks", checked, N_DIV);
    end

endmodule
