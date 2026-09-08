`timescale 1ns/1ps

// ABOUTME: Self-checking testbench for vpu_seq: every vector op streamed out of a buffer and back.
// ABOUTME: Checked against qsoftmax/qlayernorm/qadd/apply_unary_lut/numpy .T, with distinct base words per op.

module tb_vpu_seq;

    `include "params.svh"

    localparam int LANES  = 16;
    localparam int WORDS  = 128;
    localparam int AB     = 20;
    // Smaller than the SoC instance on purpose: the kernels' row buffers scale
    // with these, and the sequencer's behaviour does not.
    localparam int SMAX   = 256;
    localparam int LMAX   = 128;

    // Base words. Every op gets its own source and destination so a sequencer
    // that ignored the base would still pass on op 0 and fail everywhere else.
    localparam int UN_A = 0,  UN_D = 64;
    localparam int QA_A = 16, QA_B = 0,  QA_D = 80;
    localparam int SM_A = 32, SM_D = 96;
    localparam int LN_A = 48, LN_D = 104;
    localparam int TR_A = 56, TR_D = 112;
    localparam int T2_A = 72, T2_D = 88;

    logic clk = 1'b0;
    logic rst = 1'b1;
    always #5 clk = ~clk;

    logic [8*LANES-1:0] a_mem [WORDS];
    logic [8*LANES-1:0] b_mem [WORDS];
    logic [8*LANES-1:0] d_mem [WORDS];

    logic                 sm_lut_wr_en, sm_lut_wr_sel;
    logic [7:0]           sm_lut_wr_addr;
    logic [15:0]          sm_lut_wr_data;
    logic                 un_lut_wr_en;
    logic [7:0]           un_lut_wr_addr;
    logic signed [7:0]    un_lut_wr_data;
    logic                 ln_par_wr_en;
    logic [$clog2(LMAX)-1:0] ln_par_wr_addr;
    logic signed [7:0]    ln_par_wr_gamma;
    logic signed [31:0]   ln_par_wr_beta;

    logic                 start, busy, done;
    logic [3:0]           op;
    logic [15:0]          cfg_len, cfg_rows, cfg_a_word, cfg_b_word, cfg_d_word;
    logic signed [31:0]   cfg_mult, cfg_mult_b;
    logic [5:0]           cfg_shift, cfg_shift_b;
    logic [63:0]          cfg_eps;

    /* verilator lint_off UNUSEDSIGNAL */
    logic [AB-1:0]        a_addr, b_addr, o_addr;
    /* verilator lint_on UNUSEDSIGNAL */
    logic [8*LANES-1:0]   a_data, b_data, o_data;
    logic [LANES-1:0]     o_wmask;
    logic                 o_we;

    vpu_seq #(
        .LANES(LANES), .SM_LEN(SMAX), .LN_LEN(LMAX), .ADDR_BITS(AB)
    ) dut (
        .clk(clk), .rst(rst),
        .sm_lut_wr_en(sm_lut_wr_en), .sm_lut_wr_sel(sm_lut_wr_sel),
        .sm_lut_wr_addr(sm_lut_wr_addr), .sm_lut_wr_data(sm_lut_wr_data),
        .un_lut_wr_en(un_lut_wr_en), .un_lut_wr_addr(un_lut_wr_addr),
        .un_lut_wr_data(un_lut_wr_data),
        .ln_par_wr_en(ln_par_wr_en), .ln_par_wr_addr(ln_par_wr_addr),
        .ln_par_wr_gamma(ln_par_wr_gamma), .ln_par_wr_beta(ln_par_wr_beta),
        .start(start), .op(op),
        .cfg_len(cfg_len), .cfg_rows(cfg_rows),
        .cfg_mult(cfg_mult), .cfg_shift(cfg_shift),
        .cfg_mult_b(cfg_mult_b), .cfg_shift_b(cfg_shift_b),
        .cfg_eps(cfg_eps),
        .cfg_a_word(cfg_a_word), .cfg_b_word(cfg_b_word), .cfg_d_word(cfg_d_word),
        .busy(busy), .done(done),
        .a_addr(a_addr), .a_data(a_data),
        .b_addr(b_addr), .b_data(b_data),
        .o_we(o_we), .o_addr(o_addr), .o_wmask(o_wmask), .o_data(o_data)
    );

    // Synchronous buffer reads and a byte-masked write, exactly like tinytpu_buf.
    localparam int WA = $clog2(WORDS);
    always_ff @(posedge clk) begin
        a_data <= a_mem[WA'(a_addr)];
        b_data <= b_mem[WA'(b_addr)];
        if (o_we)
            for (int l = 0; l < LANES; l++)
                if (o_wmask[l]) d_mem[WA'(o_addr)][8*l +: 8] <= o_data[8*l +: 8];
    end

    always_ff @(posedge clk) begin
        if (!rst && done && busy)
            $fatal(1, "vpu_seq: done asserted while still busy");
    end

    int failures = 0;

    // Expected streams, one byte per line. A single buffer reloaded per op,
    // rather than one array per op passed by reference: Verilator 5.032
    // mis-generates `const ref` unpacked-array task arguments.
    localparam int EXP_MAX = 4096;
    logic [7:0] exp_buf [EXP_MAX];

    logic [15:0] lut_lo [256];
    logic [15:0] lut_hi [256];
    logic [7:0]  un_tab [256];
    logic [7:0]  ln_gam [VPU_LN_LEN];
    logic [31:0] ln_bet [VPU_LN_LEN];

    task automatic run_op(input logic [3:0] o,
                          input logic [15:0] len, input logic [15:0] rows,
                          input logic [15:0] a_base, input logic [15:0] b_base,
                          input logic [15:0] d_base,
                          input logic signed [31:0] mult, input logic [5:0] sh,
                          input logic signed [31:0] mult_b, input logic [5:0] sh_b,
                          input logic [63:0] eps);
        op          = o;
        cfg_len     = len;
        cfg_rows    = rows;
        cfg_a_word  = a_base;
        cfg_b_word  = b_base;
        cfg_d_word  = d_base;
        cfg_mult    = mult;
        cfg_shift   = sh;
        cfg_mult_b  = mult_b;
        cfg_shift_b = sh_b;
        cfg_eps     = eps;
        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;
        wait (done);
        @(negedge clk);
    endtask

    // The destination is pre-filled with 0xAA. Anything past the last element of
    // a run must still read 0xAA: that is the only check on the tail-word byte
    // mask, and a mask stuck at all-ones passes every element comparison.
    function automatic void check(input string name, input int d_base, input int count);
        int words;
        logic [7:0] got;
        words = (count + LANES - 1) / LANES;
        for (int i = 0; i < count; i++) begin
            got = d_mem[d_base + i / LANES][8*(i % LANES) +: 8];
            if (got !== exp_buf[i]) begin
                if (failures < 8)
                    $display("%s[%0d]: got %02x expected %02x", name, i, got, exp_buf[i]);
                failures++;
            end
        end
        for (int i = count; i < words * LANES; i++) begin
            got = d_mem[d_base + i / LANES][8*(i % LANES) +: 8];
            if (got !== 8'haa) begin
                $display("%s: tail byte %0d was overwritten with %02x", name, i, got);
                failures++;
            end
        end
    endfunction

    initial begin
        start = 1'b0; op = '0;
        sm_lut_wr_en = 1'b0; un_lut_wr_en = 1'b0; ln_par_wr_en = 1'b0;
        sm_lut_wr_sel = 1'b0; sm_lut_wr_addr = '0; sm_lut_wr_data = '0;
        un_lut_wr_addr = '0; un_lut_wr_data = '0;
        ln_par_wr_addr = '0; ln_par_wr_gamma = '0; ln_par_wr_beta = '0;
        cfg_len = '0; cfg_rows = '0; cfg_mult = '0; cfg_shift = '0;
        cfg_mult_b = '0; cfg_shift_b = '0; cfg_eps = '0;
        cfg_a_word = '0; cfg_b_word = '0; cfg_d_word = '0;

        for (int i = 0; i < WORDS; i++) begin
            a_mem[i] = '0;
            b_mem[i] = '0;
            d_mem[i] = {LANES{8'haa}};
        end

        $readmemh("vec/vpu_un_x.hex",  a_mem, UN_A);
        $readmemh("vec/vpu_qa_a.hex",  a_mem, QA_A);
        $readmemh("vec/vpu_sm_x.hex",  a_mem, SM_A);
        $readmemh("vec/vpu_ln_x.hex",  a_mem, LN_A);
        $readmemh("vec/vpu_tr_x.hex",  a_mem, TR_A);
        $readmemh("vec/vpu_tr2_x.hex", a_mem, T2_A);
        $readmemh("vec/vpu_qa_b.hex",  b_mem, QA_B);

        $readmemh("vec/vpu_un_lut.hex", un_tab);
        $readmemh("vec/sm_lut_lo.hex", lut_lo);
        $readmemh("vec/sm_lut_hi.hex", lut_hi);
        $readmemh("vec/vpu_ln_gamma.hex", ln_gam);
        $readmemh("vec/vpu_ln_beta.hex", ln_bet);

        repeat (4) @(negedge clk);
        rst = 1'b0;

        // Tables first: every kernel's state is loaded through the sequencer's
        // pass-through ports, which is exactly how the register map will do it.
        for (int i = 0; i < 256; i++) begin
            un_lut_wr_en = 1'b1; un_lut_wr_addr = 8'(i); un_lut_wr_data = un_tab[i];
            sm_lut_wr_en = 1'b1; sm_lut_wr_sel = 1'b0;
            sm_lut_wr_addr = 8'(i); sm_lut_wr_data = lut_lo[i];
            @(negedge clk);
            sm_lut_wr_sel = 1'b1; sm_lut_wr_data = lut_hi[i];
            un_lut_wr_en = 1'b0;
            @(negedge clk);
        end
        sm_lut_wr_en = 1'b0;

        for (int i = 0; i < VPU_LN_LEN; i++) begin
            ln_par_wr_en = 1'b1;
            ln_par_wr_addr = $clog2(LMAX)'(i);
            ln_par_wr_gamma = ln_gam[i];
            ln_par_wr_beta = ln_bet[i];
            @(negedge clk);
        end
        ln_par_wr_en = 1'b0;

        $readmemh("vec/vpu_un_y.hex", exp_buf);
        run_op(4'd4, 16'(VPU_UN_N), 16'd1, 16'(UN_A), 16'd0, 16'(UN_D), '0, '0, '0, '0, '0);
        check("unary", UN_D, VPU_UN_N);

        $readmemh("vec/vpu_qa_y.hex", exp_buf);
        run_op(4'd3, 16'(VPU_QA_N), 16'd1, 16'(QA_A), 16'(QA_B), 16'(QA_D),
               VPU_QA_MULT_A, 6'(VPU_QA_SHIFT_A),
               VPU_QA_MULT_B, 6'(VPU_QA_SHIFT_B), '0);
        check("qadd", QA_D, VPU_QA_N);

        $readmemh("vec/vpu_sm_y.hex", exp_buf);
        run_op(4'd1, 16'(VPU_SM_LEN), 16'(VPU_SM_ROWS), 16'(SM_A), 16'd0, 16'(SM_D),
               VPU_SM_MULT, 6'(VPU_SM_SHIFT), '0, '0, '0);
        check("softmax", SM_D, VPU_SM_ROWS * VPU_SM_LEN);

        $readmemh("vec/vpu_ln_y.hex", exp_buf);
        run_op(4'd2, 16'(VPU_LN_LEN), 16'(VPU_LN_ROWS), 16'(LN_A), 16'd0, 16'(LN_D),
               VPU_LN_MULT, 6'(VPU_LN_SHIFT), '0, '0, 64'(VPU_LN_EPS_Q));
        check("layernorm", LN_D, VPU_LN_ROWS * VPU_LN_LEN);

        // cfg_rows/cfg_len describe the *source*: the result is COLS x ROWS.
        // Both dimensions are odd multiples of nothing in particular, so the
        // read index generator has to be right rather than merely consistent.
        $readmemh("vec/vpu_tr_y.hex", exp_buf);
        run_op(4'd5, 16'(VPU_TR_COLS), 16'(VPU_TR_ROWS), 16'(TR_A), 16'd0, 16'(TR_D),
               '0, '0, '0, '0, '0);
        check("transpose", TR_D, VPU_TR_ROWS * VPU_TR_COLS);

        $readmemh("vec/vpu_tr2_y.hex", exp_buf);
        run_op(4'd5, 16'(VPU_TR2_N), 16'(VPU_TR2_N), 16'(T2_A), 16'd0, 16'(T2_D),
               '0, '0, '0, '0, '0);
        check("transpose sq", T2_D, VPU_TR2_N * VPU_TR2_N);

        if (failures == 0)
            $display("tb_vpu_seq: PASS (%0d unary, %0d qadd, %0dx%0d softmax, %0dx%0d layernorm, %0dx%0d + %0dx%0d transpose)",
                     VPU_UN_N, VPU_QA_N, VPU_SM_ROWS, VPU_SM_LEN,
                     VPU_LN_ROWS, VPU_LN_LEN, VPU_TR_ROWS, VPU_TR_COLS,
                     VPU_TR2_N, VPU_TR2_N);
        else
            $fatal(1, "tb_vpu_seq: %0d mismatches", failures);
        $finish;
    end

endmodule
