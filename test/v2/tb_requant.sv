`timescale 1ns/1ps

// ABOUTME: Self-checking testbench proving requant_unit is bit-exact against sw/numerics.py Requant.apply.
// ABOUTME: Vectors come from test/v2/gen_vectors.py; any mismatch is a hard $fatal.

module tb_requant;

    `include "params.svh"

    // {acc[31:0], mult[31:0], shift[7:0], expect[15:0]}
    localparam int WORD_BITS = 88;

    logic [WORD_BITS-1:0] vec [N_REQUANT];

    logic clk = 1'b0;
    logic rst = 1'b1;
    always #5 clk = ~clk;

    logic               valid_in;
    logic signed [31:0] acc_in;
    logic signed [31:0] mult_in;
    logic [5:0]         shift_in;
    logic               valid_out;
    logic signed [7:0]  dut_out;

    requant_unit #(
        .ACC_BITS(32), .OUT_BITS(8), .OUT_MIN(-127), .OUT_MAX(127)
    ) dut (
        .clk(clk), .rst(rst),
        .valid_in(valid_in), .acc_in(acc_in), .mult_in(mult_in), .shift_in(shift_in),
        .valid_out(valid_out), .out(dut_out)
    );

    // The expected value has to be delayed to meet its result. Three stages,
    // not two: the DUT is two deep (multiply, then round/saturate) and the
    // testbench registers its own stimulus, which costs one more.
    localparam int LATENCY = 3;
    logic signed [15:0] expect_pipe [LATENCY];
    int                 index_pipe  [LATENCY];

    int drive_i  = 0;
    int checked  = 0;
    int failures = 0;

    initial begin
        $readmemh("vec/requant.hex", vec);
        repeat (4) @(posedge clk);
        rst <= 1'b0;
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            valid_in <= 1'b0;
        end else begin
            for (int i = LATENCY - 1; i > 0; i--) begin
                expect_pipe[i] <= expect_pipe[i-1];
                index_pipe[i]  <= index_pipe[i-1];
            end

            if (drive_i < N_REQUANT) begin
                valid_in       <= 1'b1;
                acc_in         <= vec[drive_i][87:56];
                mult_in        <= vec[drive_i][55:24];
                shift_in       <= vec[drive_i][21:16];
                expect_pipe[0] <= vec[drive_i][15:0];
                index_pipe[0]  <= drive_i;
                drive_i        <= drive_i + 1;
            end else begin
                valid_in <= 1'b0;
            end

            if (valid_out) begin
                if (dut_out !== expect_pipe[LATENCY-1][7:0]) begin
                    $display("FAIL requant[%0d]: got %0d, expected %0d",
                             index_pipe[LATENCY-1], dut_out, $signed(expect_pipe[LATENCY-1]));
                    failures <= failures + 1;
                end
                checked <= checked + 1;
            end
        end
    end

    initial begin
        wait (checked == N_REQUANT);
        @(posedge clk);
        if (failures != 0) $fatal(1, "requant_unit: %0d/%0d vectors mismatched", failures, checked);
        $display("PASS requant_unit: %0d/%0d vectors bit-exact vs sw/numerics.py", checked, N_REQUANT);
        $finish;
    end

    initial begin
        #1_000_000;
        $fatal(1, "requant_unit: timeout after %0d/%0d checks", checked, N_REQUANT);
    end

endmodule
