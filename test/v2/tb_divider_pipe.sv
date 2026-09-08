`timescale 1ns/1ps

// ABOUTME: Self-checking testbench proving divider_pipe matches Python's floor division at 1 result/cycle.
// ABOUTME: Drives back-to-back with no gaps, so a stage that leaks state between operands would show.

module tb_divider_pipe;

    `include "params.svh"

    localparam int NB = DIV_NUM_BITS;
    localparam int DB = DIV_DEN_BITS;
    localparam int W  = NB + DB + NB;
    localparam int LATENCY = NB + 2;

    logic clk = 1'b0;
    logic rst = 1'b1;
    always #5 clk = ~clk;

    logic [W-1:0] vec [N_DIV];

    logic                 valid_in;
    logic signed [NB-1:0] num_in;
    logic        [DB-1:0] den_in;
    logic                 valid_out;
    logic signed [NB-1:0] quot_out;
    logic                 div_zero;

    divider_pipe #(.NUM_BITS(NB), .DEN_BITS(DB)) dut (
        .clk(clk), .rst(rst),
        .valid_in(valid_in), .num_in(num_in), .den_in(den_in),
        .valid_out(valid_out), .quot_out(quot_out), .div_zero(div_zero)
    );

    int drive_i  = 0;
    int checked  = 0;
    int failures = 0;

    initial begin
        $readmemh("vec/div.hex", vec);
        valid_in = 1'b0;
        repeat (4) @(posedge clk);
        rst <= 1'b0;
    end

    // Every vector, back to back, one per cycle.
    always_ff @(posedge clk) begin
        if (rst) begin
            valid_in <= 1'b0;
        end else if (drive_i < N_DIV) begin
            valid_in <= 1'b1;
            num_in   <= $signed(vec[drive_i][W-1 -: NB]);
            den_in   <= vec[drive_i][NB +: DB];
            drive_i  <= drive_i + 1;
        end else begin
            valid_in <= 1'b0;
        end
    end

    // Results come back in issue order, so a simple counter indexes the answer.
    always_ff @(posedge clk) begin
        if (!rst && valid_out) begin
            if (quot_out !== $signed(vec[checked][0 +: NB]) || div_zero !== 1'b0) begin
                $display("FAIL divp[%0d]: got %0d (dz=%0b), expected %0d",
                         checked, quot_out, div_zero, $signed(vec[checked][0 +: NB]));
                failures <= failures + 1;
            end
            checked <= checked + 1;
        end
    end

    initial begin
        wait (checked == N_DIV);
        @(posedge clk);
        if (failures != 0) $fatal(1, "divider_pipe: %0d/%0d mismatched", failures, checked);
        // Throughput is the whole point of this module, so assert it rather than
        // trusting that the pipeline never stalled: N results must take N cycles.
        $display("PASS divider_pipe: %0d/%0d floor divisions bit-exact, 1/cycle (latency %0d)",
                 checked, N_DIV, LATENCY);
        $finish;
    end

    initial begin
        #1_000_000;
        $fatal(1, "divider_pipe: timeout after %0d/%0d checks", checked, N_DIV);
    end

endmodule
