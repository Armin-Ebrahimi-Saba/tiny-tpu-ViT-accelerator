`timescale 1ns/1ps

// ABOUTME: Self-checking testbench proving unary_lut matches apply_unary_lut() in sw/kernels_int.py.
// ABOUTME: Uses a real GELU table, and pins the int8 endpoints where an index-offset bug would show.

module tb_unary_lut;

    `include "params.svh"

    logic clk = 1'b0;
    logic rst = 1'b1;
    always #5 clk = ~clk;

    logic [7:0] lut_mem [256];
    logic [7:0] x_mem   [N_GELU];
    logic [7:0] y_mem   [N_GELU];

    logic              wr_en;
    logic [7:0]        wr_addr;
    logic signed [7:0] wr_data;
    logic              valid_in;
    logic signed [7:0] x_in;
    logic              valid_out;
    logic signed [7:0] y_out;

    unary_lut #(.ENTRIES(256), .DATA_BITS(8)) dut (
        .clk(clk), .rst(rst),
        .wr_en(wr_en), .wr_addr(wr_addr), .wr_data(wr_data),
        .valid_in(valid_in), .x_in(x_in),
        .valid_out(valid_out), .y_out(y_out)
    );

    // One TB stimulus register plus the DUT's own output register.
    localparam int LATENCY = 2;
    logic signed [7:0] expect_pipe [LATENCY];
    int                index_pipe  [LATENCY];

    int  drive_i  = 0;
    int  checked  = 0;
    int  failures = 0;
    bit  loaded   = 1'b0;

    initial begin
        $readmemh("vec/gelu_lut.hex", lut_mem);
        $readmemh("vec/gelu_x.hex",   x_mem);
        $readmemh("vec/gelu_y.hex",   y_mem);

        wr_en    = 1'b0;
        valid_in = 1'b0;
        repeat (4) @(posedge clk);
        rst <= 1'b0;
        @(posedge clk);

        // Load the table one entry per cycle.
        for (int i = 0; i < 256; i++) begin
            wr_en   <= 1'b1;
            wr_addr <= i[7:0];
            wr_data <= $signed(lut_mem[i]);
            @(posedge clk);
        end
        wr_en  <= 1'b0;
        loaded <= 1'b1;
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            valid_in <= 1'b0;
        end else begin
            for (int i = LATENCY - 1; i > 0; i--) begin
                expect_pipe[i] <= expect_pipe[i-1];
                index_pipe[i]  <= index_pipe[i-1];
            end

            if (loaded && drive_i < N_GELU) begin
                valid_in       <= 1'b1;
                x_in           <= $signed(x_mem[drive_i]);
                expect_pipe[0] <= $signed(y_mem[drive_i]);
                index_pipe[0]  <= drive_i;
                drive_i        <= drive_i + 1;
            end else begin
                valid_in <= 1'b0;
            end

            if (valid_out) begin
                if (y_out !== expect_pipe[LATENCY-1]) begin
                    $display("FAIL gelu[%0d]: x=%0d got %0d, expected %0d",
                             index_pipe[LATENCY-1], x_in, y_out, expect_pipe[LATENCY-1]);
                    failures <= failures + 1;
                end
                checked <= checked + 1;
            end
        end
    end

    initial begin
        wait (checked == N_GELU);
        @(posedge clk);
        if (failures != 0) $fatal(1, "unary_lut: %0d/%0d mismatched", failures, checked);
        $display("PASS unary_lut: %0d/%0d GELU lookups bit-exact vs sw/kernels_int.py",
                 checked, N_GELU);
        $finish;
    end

    initial begin
        #1_000_000;
        $fatal(1, "unary_lut: timeout after %0d/%0d checks", checked, N_GELU);
    end

endmodule
