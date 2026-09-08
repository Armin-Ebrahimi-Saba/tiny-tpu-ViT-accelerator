`timescale 1ns/1ps

// ABOUTME: Self-checking testbench proving systolic_int8 reproduces sw/kernels_int.py exact_int_matmul exactly.
// ABOUTME: Drives the full weight-load / switch / staggered-feed sequence a real control unit must issue.

module tb_systolic_int8;

    `include "params.svh"

    localparam int ROWS     = GEMM_K;
    localparam int COLS     = GEMM_N;
    localparam int ACC_BITS = 32;

    logic clk = 1'b0;
    logic rst = 1'b1;
    always #5 clk = ~clk;

    // $readmemh fills sequentially across whitespace, so 2-D vectors arrive flat.
    logic [7:0]  x_mem    [GEMM_M * GEMM_K];
    logic [7:0]  w_mem    [GEMM_K * GEMM_N];
    logic [31:0] bias_mem [GEMM_N];
    logic [31:0] acc_mem  [GEMM_M * GEMM_N];

    logic signed [7:0]          data_in   [ROWS];
    logic                       valid_in  [ROWS];
    logic                       switch_in;
    logic signed [7:0]          weight_in [COLS];
    logic                       accept_w  [COLS];
    logic signed [ACC_BITS-1:0] bias_in   [COLS];
    logic signed [ACC_BITS-1:0] psum_out  [COLS];
    logic                       valid_out [COLS];

    systolic_int8 #(.ROWS(ROWS), .COLS(COLS), .ACC_BITS(ACC_BITS)) dut (
        .clk(clk), .rst(rst),
        .data_in(data_in), .valid_in(valid_in), .switch_in(switch_in),
        .weight_in(weight_in), .accept_w(accept_w), .bias_in(bias_in),
        .active_cols($clog2(COLS+1)'(COLS)),
        .psum_out(psum_out), .valid_out(valid_out)
    );

    // Schedule. Weights shift down one PE per cycle, so PE(r,c) ends up holding
    // whatever was driven at cycle ROWS-1-r; the load therefore runs in reverse
    // k order. The switch pulse then walks the diagonal, reaching PE(r,c) at
    // SWITCH_CYC+r+c, which is why row r's activations start r cycles late.
    //
    // Switch must trail the *last* weight beat by one cycle, not land on it:
    // PE(r,c) is still writing w[r] into its shadow register on cycle ROWS+r, so
    // a switch seen on that same cycle promotes the previous k's weight instead.
    // The pulse reaches PE(r,c) at SWITCH_CYC+r+c, which is one cycle later --
    // exactly right, and uniformly so for every r.
    localparam int SWITCH_CYC = ROWS + 1;
    localparam int DATA_START = SWITCH_CYC + 1;

    int cyc = 0;
    int checked [COLS];
    int failures = 0;
    int total_checked = 0;

    initial begin
        $readmemh("vec/gemm_x.hex",    x_mem);
        $readmemh("vec/gemm_w.hex",    w_mem);
        $readmemh("vec/gemm_bias.hex", bias_mem);
        $readmemh("vec/gemm_acc.hex",  acc_mem);
        for (int c = 0; c < COLS; c++) checked[c] = 0;
        repeat (4) @(posedge clk);
        rst <= 1'b0;
    end

    always_comb begin
        for (int c = 0; c < COLS; c++) bias_in[c] = $signed(bias_mem[c]);
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

            // --- weight load, reverse k order ---
            if (cyc < ROWS) begin
                for (int c = 0; c < COLS; c++) begin
                    accept_w[c]  <= 1'b1;
                    weight_in[c] <= $signed(w_mem[(ROWS - 1 - cyc) * GEMM_N + c]);
                end
            end else begin
                for (int c = 0; c < COLS; c++) accept_w[c] <= 1'b0;
            end

            switch_in <= (cyc == SWITCH_CYC - 1);

            // --- staggered activation feed: x[m][r] at cycle DATA_START+r+m ---
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

    // Column c emits its GEMM_M results in m order; scoreboard them as they land.
    // Several columns retire on the same cycle (result m of column c appears at
    // a time that depends on m+c), so the counters have to be summed locally and
    // written back once -- a `x <= x + 1` inside the loop would collapse them all
    // into a single increment.
    always_ff @(posedge clk) begin
        automatic int inc = 0;
        automatic int bad = 0;
        if (!rst) begin
            for (int c = 0; c < COLS; c++) begin
                if (valid_out[c]) begin
                    if (checked[c] >= GEMM_M) begin
                        $display("FAIL col %0d: extra valid beyond %0d results", c, GEMM_M);
                        bad++;
                    end else begin
                        if (psum_out[c] !== $signed(acc_mem[checked[c] * GEMM_N + c])) begin
                            $display("FAIL gemm[m=%0d][n=%0d]: got %0d, expected %0d",
                                     checked[c], c, psum_out[c], $signed(acc_mem[checked[c] * GEMM_N + c]));
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
            $fatal(1, "systolic_int8: %0d mismatches over %0d results", failures, total_checked);
        $display("PASS systolic_int8 %0dx%0d: %0d/%0d int32 accumulations bit-exact vs sw/kernels_int.py",
                 ROWS, COLS, total_checked, GEMM_M * GEMM_N);
        $finish;
    end

    initial begin
        #1_000_000;
        $fatal(1, "systolic_int8: timeout after %0d/%0d results", total_checked, GEMM_M * GEMM_N);
    end

endmodule
