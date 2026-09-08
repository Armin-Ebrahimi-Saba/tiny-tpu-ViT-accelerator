`timescale 1ns/1ps

// ABOUTME: Self-checking testbench proving isqrt matches _isqrt() in sw/kernels_int.py.
// ABOUTME: Vectors cluster on k**2 - 1 / k**2 / k**2 + 1, which is where floor(sqrt) steps.

module tb_isqrt;

    `include "params.svh"

    localparam int XB = ISQRT_BITS;
    localparam int RB = 32;

    logic clk = 1'b0;
    logic rst = 1'b1;
    always #5 clk = ~clk;

    logic [XB+RB-1:0] vec [N_ISQRT];

    logic            start;
    logic [XB-1:0]   x_in;
    logic            ready;
    logic            valid_out;
    logic [XB/2-1:0] root_out;

    isqrt #(.BITS(XB)) dut (
        .clk(clk), .rst(rst),
        .start(start), .x_in(x_in), .ready(ready),
        .valid_out(valid_out), .root_out(root_out)
    );

    int checked  = 0;
    int failures = 0;

    initial begin
        automatic logic [RB-1:0] expect_r;

        $readmemh("vec/isqrt.hex", vec);
        start = 1'b0;
        repeat (4) @(posedge clk);
        rst <= 1'b0;
        @(posedge clk);

        for (int i = 0; i < N_ISQRT; i++) begin
            x_in     = vec[i][XB+RB-1 -: XB];
            expect_r = vec[i][0 +: RB];

            wait (ready);
            @(posedge clk);
            start <= 1'b1;
            @(posedge clk);
            start <= 1'b0;

            wait (valid_out);
            if (root_out !== (XB/2)'(expect_r)) begin
                $display("FAIL isqrt[%0d]: sqrt(%0d) got %0d, expected %0d",
                         i, x_in, root_out, expect_r);
                failures++;
            end
            checked++;
            @(posedge clk);
        end

        if (failures != 0) $fatal(1, "isqrt: %0d/%0d mismatched", failures, checked);
        $display("PASS isqrt: %0d/%0d exact floor sqrt vs sw/kernels_int.py", checked, N_ISQRT);
        $finish;
    end

    initial begin
        #5_000_000;
        $fatal(1, "isqrt: timeout after %0d/%0d checks", checked, N_ISQRT);
    end

endmodule
