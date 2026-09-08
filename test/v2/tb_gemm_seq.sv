`timescale 1ns/1ps

// ABOUTME: Self-checking testbench proving gemm_seq runs a GEMM larger than the array in every dimension.
// ABOUTME: Checked against qlinear(); the shape needs 3 k tiles, 2 n tiles and 3 m blocks, so all three loops run.
// ABOUTME: Run twice -- once at base zero, once at three unrelated non-zero operand bases -- since the bases are what let a result stay where it lands.

module tb_gemm_seq;

    `include "params.svh"

    localparam int ROWS = 16;
    localparam int COLS = 16;
    localparam int MBLK = 16;               // < SEQ_M, so the M loop really blocks
    localparam int ADDR_BITS = 20;
    localparam int NP = SEQ_NT * COLS;      // padded output channels

    logic clk = 1'b0;
    logic rst = 1'b1;
    always #5 clk = ~clk;

    // Buffers, exactly as the sequencer addresses them.
    // Headroom above each tensor so the second run can be placed at an
    // arbitrary base. The bases are deliberately unequal and not multiples of
    // anything the sequencer counts in.
    localparam int A_BASE = 13;
    localparam int B_BASE = 37;
    localparam int D_BASE = 5;

    logic [8*ROWS-1:0] act_mem [A_BASE + SEQ_M * SEQ_KT];
    logic [8*COLS-1:0] wgt_mem [B_BASE + SEQ_KT * ROWS * SEQ_NT];
    logic [8*COLS-1:0] out_mem [D_BASE + SEQ_M * SEQ_NT];
    logic [7:0]        exp_mem [SEQ_M][SEQ_N];
    logic [31:0]       bias_mem [NP];
    logic [31:0]       mult_mem [NP];
    logic [7:0]        shift_mem [NP];

    logic                          cfg_wr_en;
    logic [$clog2(2048)-1:0]       cfg_wr_addr;
    logic signed [31:0]            cfg_wr_bias, cfg_wr_mult;
    logic [5:0]                    cfg_wr_shift;
    logic                          start, busy, done;
    logic [15:0]                   cfg_a_word, cfg_b_word, cfg_d_word;
    // The address bus is sized for real tensors; these test buffers are small,
    // so the upper address bits are legitimately unused here.
    /* verilator lint_off UNUSEDSIGNAL */
    logic [ADDR_BITS-1:0]          act_addr, wgt_addr, out_addr;
    /* verilator lint_on UNUSEDSIGNAL */
    logic [8*ROWS-1:0]             act_data;
    logic [8*COLS-1:0]             wgt_data, out_data;
    logic                          out_we;

    gemm_seq #(
        .ROWS(ROWS), .COLS(COLS), .MBLK(MBLK), .ADDR_BITS(ADDR_BITS)
    ) dut (
        .clk(clk), .rst(rst),
        .cfg_wr_en(cfg_wr_en), .cfg_wr_addr(cfg_wr_addr),
        .cfg_wr_bias(cfg_wr_bias), .cfg_wr_mult(cfg_wr_mult), .cfg_wr_shift(cfg_wr_shift),
        .start(start),
        .cfg_m($clog2(2049)'(SEQ_M)),
        .cfg_k_tiles($clog2(129)'(SEQ_KT)), .cfg_n_tiles($clog2(129)'(SEQ_NT)),
        .cfg_a_word(cfg_a_word), .cfg_b_word(cfg_b_word), .cfg_d_word(cfg_d_word),
        .busy(busy), .done(done),
        .act_addr(act_addr), .act_data(act_data),
        .wgt_addr(wgt_addr), .wgt_data(wgt_data),
        .out_we(out_we), .out_addr(out_addr), .out_data(out_data)
    );

    // Synchronous buffer reads, one cycle behind the address -- what a BRAM does.
    // The address bus is ADDR_BITS wide; each buffer uses only as much of it as
    // its own depth needs.
    // A one-word buffer still needs a one-bit address, so guard $clog2(1) == 0.
    localparam int AA = $clog2(A_BASE + SEQ_M * SEQ_KT + 1);
    localparam int WA = $clog2(B_BASE + SEQ_KT * ROWS * SEQ_NT + 1);
    localparam int OA = $clog2(D_BASE + SEQ_M * SEQ_NT + 1);

    always_ff @(posedge clk) begin
        act_data <= act_mem[AA'(act_addr)];
        wgt_data <= wgt_mem[WA'(wgt_addr)];
        if (out_we) out_mem[OA'(out_addr)] <= out_data;
    end

    // busy must bracket the run: high while the sequencer works, low at done.
    always_ff @(posedge clk) begin
        if (!rst && done && busy)
            $fatal(1, "gemm_seq: done asserted while still busy");
    end

    int failures = 0;

    // One full run at the given bases, checked against the emulator. The
    // operand memories are reloaded each time so the second run cannot pass on
    // data the first one left behind.
    task automatic run_at(input int ab, input int bb, input int db);
        for (int i = 0; i < A_BASE + SEQ_M * SEQ_KT; i++) act_mem[i] = '0;
        for (int i = 0; i < B_BASE + SEQ_KT * ROWS * SEQ_NT; i++) wgt_mem[i] = '0;
        for (int i = 0; i < D_BASE + SEQ_M * SEQ_NT; i++) out_mem[i] = '0;
        $readmemh("vec/seq_act.hex", act_mem, ab);
        $readmemh("vec/seq_wgt.hex", wgt_mem, bb);

        cfg_a_word <= 16'(ab);
        cfg_b_word <= 16'(bb);
        cfg_d_word <= 16'(db);
        @(posedge clk);

        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;

        wait (done);
        @(posedge clk);

        for (int m = 0; m < SEQ_M; m++) begin
            for (int n = 0; n < SEQ_N; n++) begin
                automatic logic signed [7:0] got =
                    $signed(out_mem[db + m * SEQ_NT + (n / COLS)][8*(n % COLS) +: 8]);
                if (got !== $signed(exp_mem[m][n])) begin
                    if (failures < 10)
                        $display("FAIL seq[%0d][%0d] @base %0d/%0d/%0d: got %0d, expected %0d",
                                 m, n, ab, bb, db, got, $signed(exp_mem[m][n]));
                    failures++;
                end
            end
        end
    endtask

    initial begin
        $readmemh("vec/seq_out.hex",   exp_mem);
        $readmemh("vec/seq_bias.hex",  bias_mem);
        $readmemh("vec/seq_mult.hex",  mult_mem);
        $readmemh("vec/seq_shift.hex", shift_mem);

        cfg_wr_en = 1'b0;
        start     = 1'b0;
        cfg_a_word = '0;
        cfg_b_word = '0;
        cfg_d_word = '0;
        repeat (4) @(posedge clk);
        rst <= 1'b0;
        @(posedge clk);

        for (int i = 0; i < NP; i++) begin
            cfg_wr_en    <= 1'b1;
            cfg_wr_addr  <= $clog2(2048)'(i);
            cfg_wr_bias  <= $signed(bias_mem[i]);
            cfg_wr_mult  <= $signed(mult_mem[i]);
            cfg_wr_shift <= 6'(shift_mem[i]);
            @(posedge clk);
        end
        cfg_wr_en <= 1'b0;
        @(posedge clk);

        run_at(0, 0, 0);
        run_at(A_BASE, B_BASE, D_BASE);

        if (failures != 0)
            $fatal(1, "gemm_seq: %0d/%0d mismatched", failures, 2 * SEQ_M * SEQ_N);
        $display("PASS gemm_seq %0dx%0dx%0d (%0d k tiles, %0d n tiles, MBLK=%0d): %0d/%0d bit-exact vs qlinear(), at base 0 and base %0d/%0d/%0d",
                 SEQ_M, SEQ_K, SEQ_N, SEQ_KT, SEQ_NT, MBLK,
                 2 * SEQ_M * SEQ_N, 2 * SEQ_M * SEQ_N, A_BASE, B_BASE, D_BASE);
        $finish;
    end

    initial begin
        #20_000_000;
        $fatal(1, "gemm_seq: timeout (done never asserted)");
    end

endmodule
