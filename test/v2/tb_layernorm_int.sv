`timescale 1ns/1ps

// ABOUTME: Self-checking testbench proving layernorm_int matches qlayernorm() in sw/kernels_int.py.
// ABOUTME: Includes a constant row, the only input that reaches the max(r, 1) clamp.

module tb_layernorm_int;

    `include "params.svh"

    logic clk = 1'b0;
    logic rst = 1'b1;
    always #5 clk = ~clk;

    logic [7:0]  x_mem [LN_ROWS][LN_LEN];
    logic [7:0]  y_mem [LN_ROWS][LN_LEN];
    logic [7:0]  g_mem [LN_LEN];
    logic [31:0] b_mem [LN_LEN];

    logic                              par_wr_en;
    logic [$clog2(LN_LEN)-1:0]         par_wr_addr;
    logic signed [7:0]                 par_wr_gamma;
    logic signed [31:0]                par_wr_beta;
    logic                              in_valid;
    logic signed [7:0]                 in_data;
    logic                              in_ready;
    logic                              out_valid;
    logic signed [7:0]                 out_data;

    layernorm_int #(.MAX_LEN(LN_LEN), .IN_BITS(8)) dut (
        .clk(clk), .rst(rst),
        .par_wr_en(par_wr_en), .par_wr_addr(par_wr_addr),
        .par_wr_gamma(par_wr_gamma), .par_wr_beta(par_wr_beta),
        .cfg_len($clog2(LN_LEN+1)'(LN_LEN)),
        .cfg_eps_q(64'(LN_EPS_Q)),
        .cfg_mult(32'(LN_MULT)), .cfg_shift(6'(LN_SHIFT)),
        .in_valid(in_valid), .in_data(in_data), .in_ready(in_ready),
        .out_valid(out_valid), .out_data(out_data)
    );

    int checked  = 0;
    int failures = 0;

    always_ff @(posedge clk) begin
        if (!rst && out_valid) begin
            if (out_data !== $signed(y_mem[checked / LN_LEN][checked % LN_LEN])) begin
                $display("FAIL layernorm[%0d][%0d]: got %0d, expected %0d",
                         checked / LN_LEN, checked % LN_LEN, out_data,
                         $signed(y_mem[checked / LN_LEN][checked % LN_LEN]));
                failures <= failures + 1;
            end
            checked <= checked + 1;
        end
    end

    initial begin
        $readmemh("vec/ln_x.hex", x_mem);
        $readmemh("vec/ln_y.hex", y_mem);
        $readmemh("vec/ln_gamma.hex", g_mem);
        $readmemh("vec/ln_beta.hex", b_mem);

        par_wr_en = 1'b0;
        in_valid  = 1'b0;
        repeat (4) @(posedge clk);
        rst <= 1'b0;
        @(posedge clk);

        for (int i = 0; i < LN_LEN; i++) begin
            par_wr_en    <= 1'b1;
            par_wr_addr  <= $clog2(LN_LEN)'(i);
            par_wr_gamma <= $signed(g_mem[i]);
            par_wr_beta  <= $signed(b_mem[i]);
            @(posedge clk);
        end
        par_wr_en <= 1'b0;
        @(posedge clk);

        for (int row = 0; row < LN_ROWS; row++) begin
            for (int col = 0; col < LN_LEN; col++) begin
                in_valid <= 1'b1;
                in_data  <= $signed(x_mem[row][col]);
                @(posedge clk);
                while (!in_ready) @(posedge clk);
            end
            in_valid <= 1'b0;
            wait (checked == (row + 1) * LN_LEN);
            @(posedge clk);
        end
    end

    initial begin
        wait (checked == LN_ROWS * LN_LEN);
        @(posedge clk);
        if (failures != 0) $fatal(1, "layernorm_int: %0d/%0d mismatched", failures, checked);
        $display("PASS layernorm_int: %0d/%0d outputs (%0d rows of %0d) bit-exact vs qlayernorm()",
                 checked, LN_ROWS * LN_LEN, LN_ROWS, LN_LEN);
        $finish;
    end

    initial begin
        #10_000_000;
        $fatal(1, "layernorm_int: timeout after %0d/%0d checks", checked, LN_ROWS * LN_LEN);
    end

endmodule
