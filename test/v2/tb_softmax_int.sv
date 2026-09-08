`timescale 1ns/1ps

// ABOUTME: Self-checking testbench proving softmax_int matches qsoftmax() in sw/kernels_int.py.
// ABOUTME: Runs several rows back to back, including a flat row and a one-hot row.

module tb_softmax_int;

    `include "params.svh"

    logic clk = 1'b0;
    logic rst = 1'b1;
    always #5 clk = ~clk;

    logic [15:0] lut_lo_mem [256];
    logic [15:0] lut_hi_mem [256];
    logic [15:0] x_mem [SM_ROWS][SM_LEN];
    logic [7:0]  y_mem [SM_ROWS][SM_LEN];

    logic               lut_wr_en;
    logic               lut_wr_sel;
    logic [7:0]         lut_wr_addr;
    logic [15:0]        lut_wr_data;
    logic               in_valid;
    logic signed [15:0] in_data;
    logic               in_ready;
    logic               out_valid;
    logic signed [7:0]  out_data;

    softmax_int #(.MAX_LEN(SM_LEN), .IN_BITS(16)) dut (
        .clk(clk), .rst(rst),
        .lut_wr_en(lut_wr_en), .lut_wr_sel(lut_wr_sel),
        .lut_wr_addr(lut_wr_addr), .lut_wr_data(lut_wr_data),
        .cfg_len($clog2(SM_LEN+1)'(SM_LEN)),
        .cfg_mult(32'(SM_MULT)), .cfg_shift(6'(SM_SHIFT)),
        .in_valid(in_valid), .in_data(in_data), .in_ready(in_ready),
        .out_valid(out_valid), .out_data(out_data)
    );

    int row      = 0;
    int col      = 0;
    int checked  = 0;
    int failures = 0;

    // Results arrive in row-major order, so a single running index identifies
    // which reference element each one belongs to.
    always_ff @(posedge clk) begin
        if (!rst && out_valid) begin
            if (out_data !== $signed(y_mem[checked / SM_LEN][checked % SM_LEN])) begin
                $display("FAIL softmax[%0d][%0d]: got %0d, expected %0d",
                         checked / SM_LEN, checked % SM_LEN, out_data,
                         $signed(y_mem[checked / SM_LEN][checked % SM_LEN]));
                failures <= failures + 1;
            end
            checked <= checked + 1;
        end
    end

    initial begin
        $readmemh("vec/sm_lut_lo.hex", lut_lo_mem);
        $readmemh("vec/sm_lut_hi.hex", lut_hi_mem);
        $readmemh("vec/sm_x.hex", x_mem);
        $readmemh("vec/sm_y.hex", y_mem);

        lut_wr_en = 1'b0;
        in_valid  = 1'b0;
        repeat (4) @(posedge clk);
        rst <= 1'b0;
        @(posedge clk);

        for (int i = 0; i < 256; i++) begin
            lut_wr_en   <= 1'b1;
            lut_wr_sel  <= 1'b0;
            lut_wr_addr <= i[7:0];
            lut_wr_data <= lut_lo_mem[i];
            @(posedge clk);
            lut_wr_sel  <= 1'b1;
            lut_wr_data <= lut_hi_mem[i];
            @(posedge clk);
        end
        lut_wr_en <= 1'b0;
        @(posedge clk);

        for (row = 0; row < SM_ROWS; row++) begin
            for (col = 0; col < SM_LEN; col++) begin
                in_valid <= 1'b1;
                in_data  <= $signed(x_mem[row][col]);
                @(posedge clk);
                while (!in_ready) @(posedge clk);
            end
            in_valid <= 1'b0;
            // The row must fully retire before the next one is offered.
            wait (checked == (row + 1) * SM_LEN);
            @(posedge clk);
        end
    end

    initial begin
        wait (checked == SM_ROWS * SM_LEN);
        @(posedge clk);
        if (failures != 0) $fatal(1, "softmax_int: %0d/%0d mismatched", failures, checked);
        $display("PASS softmax_int: %0d/%0d outputs (%0d rows of %0d) bit-exact vs qsoftmax()",
                 checked, SM_ROWS * SM_LEN, SM_ROWS, SM_LEN);
        $finish;
    end

    initial begin
        #10_000_000;
        $fatal(1, "softmax_int: timeout after %0d/%0d checks", checked, SM_ROWS * SM_LEN);
    end

endmodule
