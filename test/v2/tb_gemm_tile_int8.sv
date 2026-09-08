`timescale 1ns/1ps

// ABOUTME: Self-checking testbench proving gemm_tile_int8 reproduces qlinear() from sw/kernels_int.py.
// ABOUTME: Covers the whole path: staggered feed, int32 accumulation, per-channel requant to int8.

module tb_gemm_tile_int8;

    `include "params.svh"

    localparam int ROWS     = GEMM_K;
    localparam int COLS     = GEMM_N;
    localparam int ACC_BITS = 32;

    logic clk = 1'b0;
    logic rst = 1'b1;
    always #5 clk = ~clk;

    logic [7:0]  x_mem     [GEMM_M * GEMM_K];
    logic [7:0]  w_mem     [GEMM_K * GEMM_N];
    logic [31:0] bias_mem  [GEMM_N];
    logic [31:0] mult_mem  [GEMM_N];
    logic [7:0]  shift_mem [GEMM_N];
    logic [7:0]  out_mem   [GEMM_M * GEMM_N];

    logic signed [7:0]          data_in   [ROWS];
    logic                       valid_in  [ROWS];
    logic                       switch_in;
    logic signed [7:0]          weight_in [COLS];
    logic                       accept_w  [COLS];
    logic signed [ACC_BITS-1:0] bias_in   [COLS];
    logic signed [31:0]         mult_in   [COLS];
    logic [5:0]                 shift_in  [COLS];
    logic signed [7:0]          out       [COLS];
    logic                       out_valid [COLS];

    gemm_tile_int8 #(
        .ROWS(ROWS), .COLS(COLS), .ACC_BITS(ACC_BITS),
        .OUT_BITS(8), .OUT_MIN(-127), .OUT_MAX(127)
    ) dut (
        .clk(clk), .rst(rst),
        .data_in(data_in), .valid_in(valid_in), .switch_in(switch_in),
        .weight_in(weight_in), .accept_w(accept_w), .bias_in(bias_in),
        .mult_in(mult_in), .shift_in(shift_in),
        .active_cols($clog2(COLS+1)'(COLS)),
        .out(out), .out_valid(out_valid)
    );

    // See tb_systolic_int8.sv for the derivation of this schedule.
    localparam int SWITCH_CYC = ROWS + 1;
    localparam int DATA_START = SWITCH_CYC + 1;

    int cyc = 0;
    int checked [COLS];
    int failures = 0;
    int total_checked = 0;

    initial begin
        $readmemh("vec/gemm_x.hex",     x_mem);
        $readmemh("vec/gemm_w.hex",     w_mem);
        $readmemh("vec/gemm_bias.hex",  bias_mem);
        $readmemh("vec/gemm_mult.hex",  mult_mem);
        $readmemh("vec/gemm_shift.hex", shift_mem);
        $readmemh("vec/gemm_out.hex",   out_mem);
        for (int c = 0; c < COLS; c++) checked[c] = 0;
        repeat (4) @(posedge clk);
        rst <= 1'b0;
    end

    always_comb begin
        for (int c = 0; c < COLS; c++) begin
            bias_in[c]  = $signed(bias_mem[c]);
            mult_in[c]  = $signed(mult_mem[c]);
            shift_in[c] = shift_mem[c][5:0];
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            cyc       <= 0;
            switch_in <= 1'b0;
            for (int c = 0; c < COLS; c++) begin
                accept_w[c]  <= 1'b0;
                weight_in[c] <= 8'sd0;
            end
            for (int r = 0; r < ROWS; r++) begin
                data_in[r]  <= 8'sd0;
                valid_in[r] <= 1'b0;
            end
        end else begin
            cyc <= cyc + 1;

            if (cyc < ROWS) begin
                for (int c = 0; c < COLS; c++) begin
                    accept_w[c]  <= 1'b1;
                    weight_in[c] <= $signed(w_mem[(ROWS - 1 - cyc) * GEMM_N + c]);
                end
            end else begin
                for (int c = 0; c < COLS; c++) accept_w[c] <= 1'b0;
            end

            switch_in <= (cyc == SWITCH_CYC - 1);

            for (int r = 0; r < ROWS; r++) begin
                automatic int m = (cyc + 1) - DATA_START - r;
                if (m >= 0 && m < GEMM_M) begin
                    data_in[r]  <= $signed(x_mem[m * GEMM_K + r]);
                    valid_in[r] <= 1'b1;
                end else begin
                    data_in[r]  <= 8'sd0;
                    valid_in[r] <= 1'b0;
                end
            end
        end
    end

    always_ff @(posedge clk) begin
        automatic int inc = 0;
        automatic int bad = 0;
        if (!rst) begin
            for (int c = 0; c < COLS; c++) begin
                if (out_valid[c]) begin
                    if (checked[c] >= GEMM_M) begin
                        $display("FAIL col %0d: extra valid beyond %0d results", c, GEMM_M);
                        bad++;
                    end else begin
                        if (out[c] !== $signed(out_mem[checked[c] * GEMM_N + c])) begin
                            $display("FAIL qlinear[m=%0d][n=%0d]: got %0d, expected %0d",
                                     checked[c], c, out[c], $signed(out_mem[checked[c] * GEMM_N + c]));
                            bad++;
                        end
                        checked[c] <= checked[c] + 1;
                        inc++;
                    end
                end
            end
            total_checked <= total_checked + inc;
            failures      <= failures + bad;
        end
    end

    initial begin
        wait (total_checked == GEMM_M * GEMM_N);
        @(posedge clk);
        if (failures != 0)
            $fatal(1, "gemm_tile_int8: %0d mismatches over %0d results", failures, total_checked);
        $display("PASS gemm_tile_int8 %0dx%0d: %0d/%0d int8 outputs bit-exact vs qlinear()",
                 ROWS, COLS, total_checked, GEMM_M * GEMM_N);
        $finish;
    end

    initial begin
        #1_000_000;
        $fatal(1, "gemm_tile_int8: timeout after %0d/%0d results", total_checked, GEMM_M * GEMM_N);
    end

endmodule
