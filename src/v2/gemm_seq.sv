`timescale 1ns/1ps

// ABOUTME: Block sequencer that runs a full M x K x N GEMM on the fixed ROWS x COLS array.
// ABOUTME: Replaces the hand-written schedule that previously existed only inside test/v2/tb_systolic_int8.sv.
//
// The array computes one ROWS x COLS tile. A transformer block needs GEMMs up
// to 1370 x 1536 x 1536, so the tile has to be swept in three nested loops:
//
//   for n_tile:                      output channels, COLS at a time
//     for m_block:                   activation rows, MBLK at a time
//       for k_tile:                  reduction depth, ROWS at a time
//         load weights, switch, stream MBLK activations, accumulate int32
//       requantize the accumulators and write MBLK output words
//
// **Why M is blocked.** The reduction over k spans several tiles, so the int32
// partial sums have to survive between them. Holding them for every row at once
// would need M x COLS x 4 bytes -- 87 KB at M=1370 -- which does not fit on
// chip. Blocking M to MBLK rows bounds that to MBLK x COLS x 4 (2 KB at
// MBLK=32) and costs nothing but re-loading the same weight tile once per
// block, which is amortized over MBLK rows of work.
//
// **Why the loop order is n outermost.** Weights are the expensive operand to
// fetch from DRAM and the array is weight-stationary. With n outermost and k
// innermost, each weight tile is loaded once per m_block; with m outermost it
// would be loaded once per (m_block, n_tile) pair. The activations re-read
// instead, and those are already on chip.
//
// **Bias enters the array, not the accumulator.** bias_in rides down the column
// with the partial sum, so it is driven only on the first k tile and zeroed
// afterwards -- otherwise it would be added once per tile.
//
// Padding is the host's job: K and N are padded up to whole tiles with zero
// weights, which contribute nothing to the sum. That is what a compiler does
// anyway, and it keeps every tile in the sweep a full ROWS x COLS tile.

module gemm_seq #(
    parameter int ROWS      = 16,
    parameter int COLS      = 16,
    parameter int MBLK      = 32,    // activation rows held in the accumulators
    parameter int ACC_BITS  = 32,
    parameter int ADDR_BITS = 20,
    parameter int MAX_M     = 2048,
    parameter int MAX_N     = 2048
)(
    input  logic clk,
    input  logic rst,

    // Per-output-channel constants, written before a GEMM starts.
    input  logic                          cfg_wr_en,
    input  logic [$clog2(MAX_N)-1:0]      cfg_wr_addr,
    input  logic signed [ACC_BITS-1:0]    cfg_wr_bias,
    input  logic signed [31:0]            cfg_wr_mult,
    input  logic [5:0]                    cfg_wr_shift,

    // GEMM shape, in rows and whole tiles.
    input  logic                          start,
    input  logic [$clog2(MAX_M+1)-1:0]    cfg_m,
    input  logic [$clog2(MAX_N/COLS+1)-1:0] cfg_k_tiles,
    input  logic [$clog2(MAX_N/COLS+1)-1:0] cfg_n_tiles,

    // Where the three operands sit, as word indices into their buffers. The
    // sequencer's own counters stay relative and the base is added on the way
    // out, so nothing about the tiling depends on placement. This is what lets
    // a result be left where it lands instead of being copied to its slot.
    input  logic [15:0]                   cfg_a_word,
    input  logic [15:0]                   cfg_b_word,
    input  logic [15:0]                   cfg_d_word,

    output logic                          busy,
    output logic                          done,

    // Activation buffer: one ROWS-byte word per (row, k tile).
    output logic [ADDR_BITS-1:0]          act_addr,
    input  logic [8*ROWS-1:0]             act_data,

    // Weight buffer: one COLS-byte word per (k, n tile).
    output logic [ADDR_BITS-1:0]          wgt_addr,
    input  logic [8*COLS-1:0]             wgt_data,

    // Output buffer: one COLS-byte word per (row, n tile).
    output logic                          out_we,
    output logic [ADDR_BITS-1:0]          out_addr,
    output logic [8*COLS-1:0]             out_data
);

    localparam int MIDX_BITS = $clog2(MBLK + 1);
    // Counters run up to MBLK/COLS inclusive, so they are one bit wider than
    // the memories they address. Indexing with the wider value truncates.
    localparam int AB        = (MBLK <= 1) ? 1 : $clog2(MBLK);
    localparam int CB        = (COLS <= 1) ? 1 : $clog2(COLS);
    localparam int TIDX_BITS = $clog2(MAX_N/COLS + 1);
    localparam int GM_BITS   = $clog2(MAX_M + 1);
    localparam int WCNT_BITS = $clog2(ROWS + 2);

    typedef enum logic [2:0] {
        S_IDLE, S_NCFG, S_WLOAD, S_SWITCH, S_STREAM, S_RETIRE, S_REQ
    } state_e;
    state_e state;

    // ------------------------------------------------------ per-channel config
    // MAX_N deep and written once per layer, read once per n tile: this is
    // block RAM, and saying so is worth ~3k LUTs of distributed RAM on Artix-7.
    (* ram_style = "block" *) logic signed [ACC_BITS-1:0] bias_mem  [MAX_N];
    (* ram_style = "block" *) logic signed [31:0]         mult_mem  [MAX_N];
    (* ram_style = "block" *) logic [5:0]                 shift_mem [MAX_N];

    always_ff @(posedge clk) begin
        if (cfg_wr_en) begin
            bias_mem[cfg_wr_addr]  <= cfg_wr_bias;
            mult_mem[cfg_wr_addr]  <= cfg_wr_mult;
            shift_mem[cfg_wr_addr] <= cfg_wr_shift;
        end
    end

    // Copied into COLS registers once per n tile so the datapath never indexes
    // a MAX_N-deep memory.
    logic signed [ACC_BITS-1:0] bias_r  [COLS];
    logic signed [31:0]         mult_r  [COLS];
    logic [5:0]                 shift_r [COLS];

    // ------------------------------------------------------------ loop counters
    logic [TIDX_BITS-1:0] n_tile, k_tile;
    logic [GM_BITS-1:0]   m_base;
    logic [MIDX_BITS-1:0] mrows;          // rows in the current m block
    logic [MIDX_BITS-1:0] mm, req_i, wr_cnt;
    logic [WCNT_BITS-1:0] wcnt;
    logic [$clog2(COLS+1)-1:0] ncfg_i;
    wire  [$clog2(MAX_N)-1:0]  ncfg_addr =
        $clog2(MAX_N)'(COLS * int'(n_tile) + int'(ncfg_i));
    logic                 first_kt, last_kt;

    logic [ADDR_BITS-1:0] act_addr_q, wgt_addr_q, out_addr_q;
    logic [15:0]          a_base_q, b_base_q, d_base_q;
    assign act_addr = act_addr_q + ADDR_BITS'(a_base_q);
    assign wgt_addr = wgt_addr_q + ADDR_BITS'(b_base_q);
    assign out_addr = out_addr_q + ADDR_BITS'(d_base_q);

    logic [ADDR_BITS-1:0] k_words, n_words;
    assign k_words = ADDR_BITS'(cfg_k_tiles);
    assign n_words = ADDR_BITS'(cfg_n_tiles);

    // ------------------------------------------------------------- array feeds
    logic signed [7:0]          a_data  [ROWS];
    logic                       a_valid [ROWS];
    logic                       switch_in;
    logic signed [7:0]          w_in    [COLS];
    logic                       accept_w[COLS];
    logic signed [ACC_BITS-1:0] bias_in [COLS];
    logic signed [ACC_BITS-1:0] psum    [COLS];
    logic                       psum_v  [COLS];

    logic wv;   // weight word from wgt_data is valid this cycle
    logic av;   // activation word from act_data is valid this cycle

    for (genvar c = 0; c < COLS; c++) begin : g_wfeed
        assign w_in[c]     = $signed(wgt_data[8*c +: 8]);
        assign accept_w[c] = wv;
        assign bias_in[c]  = first_kt ? bias_r[c] : '0;
    end

    // Row k of the activation word must arrive k cycles after row 0, which is
    // what lines each activation up with the partial sum descending its column.
    // A triangular delay chain does that: lane k reads out of stage k-1.
    logic signed [7:0] sk_d [ROWS][ROWS];
    logic              sk_v [ROWS][ROWS];

    always_ff @(posedge clk) begin
        if (rst) begin
            for (int k = 0; k < ROWS; k++)
                for (int j = 0; j < ROWS; j++) sk_v[k][j] <= 1'b0;
        end else begin
            for (int k = 0; k < ROWS; k++) begin
                sk_d[k][0] <= $signed(act_data[8*k +: 8]);
                sk_v[k][0] <= av;
                for (int j = 1; j < ROWS; j++) begin
                    sk_d[k][j] <= sk_d[k][j-1];
                    sk_v[k][j] <= sk_v[k][j-1];
                end
            end
        end
    end

    for (genvar k = 0; k < ROWS; k++) begin : g_skew
        assign a_data[k]  = (k == 0) ? $signed(act_data[8*k +: 8]) : sk_d[k][k-1];
        assign a_valid[k] = (k == 0) ? av                          : sk_v[k][k-1];
    end

    systolic_int8 #(.ROWS(ROWS), .COLS(COLS), .ACC_BITS(ACC_BITS)) u_array (
        .clk(clk), .rst(rst),
        .data_in(a_data), .valid_in(a_valid), .switch_in(switch_in),
        .weight_in(w_in), .accept_w(accept_w), .bias_in(bias_in),
        .active_cols($clog2(COLS+1)'(COLS)),
        .psum_out(psum), .valid_out(psum_v)
    );

    // ----------------------------------------------------------- accumulators
    // One MBLK-deep int32 memory per column. Column c retires its results in m
    // order, so a per-column counter is the write address and no arbitration is
    // needed between columns.
    logic signed [ACC_BITS-1:0] acc_mem [COLS][MBLK];
    logic signed [ACC_BITS-1:0] acc_rd  [COLS];
    logic [MIDX_BITS-1:0]       mc      [COLS];
    logic [MIDX_BITS-1:0]       ac_addr [COLS];
    logic signed [ACC_BITS-1:0] ac_psum [COLS];
    logic                       ac_v    [COLS];
    logic [AB-1:0]              acc_raddr [COLS];

    wire all_retired = (mc[COLS-1] == mrows);

    for (genvar c = 0; c < COLS; c++) begin : g_accrd
        assign acc_raddr[c] = AB'((state == S_REQ) ? req_i : mc[c]);
    end

    always_ff @(posedge clk) begin
        for (int c = 0; c < COLS; c++) acc_rd[c] <= acc_mem[c][acc_raddr[c]];
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            for (int c = 0; c < COLS; c++) begin
                mc[c]   <= '0;
                ac_v[c] <= 1'b0;
            end
        end else begin
            for (int c = 0; c < COLS; c++) begin
                ac_v[c] <= psum_v[c];
                if (psum_v[c]) begin
                    ac_addr[c] <= mc[c];
                    ac_psum[c] <= psum[c];
                    mc[c]      <= mc[c] + MIDX_BITS'(1);
                end
                // acc_rd[c] was issued at address mc[c] on the same cycle the
                // psum arrived, so it lands here alongside ac_psum.
                if (ac_v[c])
                    acc_mem[c][AB'(ac_addr[c])] <= first_kt ? ac_psum[c]
                                                       : acc_rd[c] + ac_psum[c];
            end
            if (state == S_WLOAD && wcnt == '0)
                for (int c = 0; c < COLS; c++) mc[c] <= '0;
        end
    end

    // ------------------------------------------------------------- requantize
    logic                       rq_v;
    logic signed [7:0]          rq_out   [COLS];
    logic                       rq_valid [COLS];

    for (genvar c = 0; c < COLS; c++) begin : g_requant
        requant_unit #(.ACC_BITS(ACC_BITS), .OUT_BITS(8)) u_rq (
            .clk(clk), .rst(rst),
            .valid_in(rq_v), .acc_in(acc_rd[c]),
            .mult_in(mult_r[c]), .shift_in(shift_r[c]),
            .valid_out(rq_valid[c]), .out(rq_out[c])
        );
        assign out_data[8*c +: 8] = rq_out[c];
    end

    assign out_we = rq_valid[0];

    always_ff @(posedge clk) begin
        if (rst) begin
            state      <= S_IDLE;
            busy       <= 1'b0;
            done       <= 1'b0;
            wv         <= 1'b0;
            av         <= 1'b0;
            switch_in  <= 1'b0;
            rq_v       <= 1'b0;
        end else begin
            wv        <= 1'b0;
            av        <= 1'b0;
            switch_in <= 1'b0;
            rq_v      <= 1'b0;
            done      <= 1'b0;

            if (rq_valid[0]) begin
                out_addr_q <= out_addr_q + n_words;
                wr_cnt     <= wr_cnt + MIDX_BITS'(1);
            end

            case (state)
                S_IDLE: begin
                    if (start) begin
                        busy    <= 1'b1;
                        a_base_q <= cfg_a_word;
                        b_base_q <= cfg_b_word;
                        d_base_q <= cfg_d_word;
                        n_tile  <= '0;
                        m_base  <= '0;
                        ncfg_i  <= '0;
                        state   <= S_NCFG;
                    end
                end

                // Copy this n tile's COLS constants out of the deep memories.
                S_NCFG: begin
                    bias_r[CB'(ncfg_i)]  <= bias_mem[ncfg_addr];
                    mult_r[CB'(ncfg_i)]  <= mult_mem[ncfg_addr];
                    shift_r[CB'(ncfg_i)] <= shift_mem[ncfg_addr];
                    if (ncfg_i == $clog2(COLS+1)'(COLS - 1)) begin
                        ncfg_i   <= '0;
                        k_tile   <= '0;
                        first_kt <= 1'b1;
                        last_kt  <= (cfg_k_tiles == TIDX_BITS'(1));
                        mrows    <= (GM_BITS'(m_base) + GM_BITS'(MBLK) <= cfg_m)
                                    ? MIDX_BITS'(MBLK) : MIDX_BITS'(cfg_m - m_base);
                        wcnt     <= '0;
                        state    <= S_WLOAD;
                    end else begin
                        ncfg_i <= ncfg_i + $clog2(COLS+1)'(1);
                    end
                end

                // Weights stream in reverse k order, one beat per cycle. The
                // read is issued a cycle before accept_w so the data is on the
                // bus when the column takes it.
                S_WLOAD: begin
                    if (wcnt == '0) begin
                        wgt_addr_q <= ADDR_BITS'((ROWS*k_tile + ROWS - 1) * n_words
                                                 + ADDR_BITS'(n_tile));
                    end else begin
                        wgt_addr_q <= wgt_addr_q - n_words;
                        wv         <= 1'b1;
                    end
                    if (wcnt == WCNT_BITS'(ROWS)) begin
                        wcnt  <= '0;
                        state <= S_SWITCH;
                    end else begin
                        wcnt <= wcnt + WCNT_BITS'(1);
                    end
                end

                // One cycle after the last weight beat; the first activation
                // read is issued here so lane 0 has it on the next cycle.
                // One cycle after the last weight beat. The activation buffer
                // is read synchronously, so row 0's address goes on the bus
                // here and its data -- and therefore `av` -- lands a cycle
                // later, exactly when the contract wants x[0][0] on lane 0.
                S_SWITCH: begin
                    switch_in  <= 1'b1;
                    act_addr_q <= ADDR_BITS'(GM_BITS'(m_base) * k_words) + ADDR_BITS'(k_tile);
                    mm         <= '0;
                    state      <= S_STREAM;
                end

                S_STREAM: begin
                    if (mm < mrows) begin
                        // `av` trails its address by one cycle, so it is raised
                        // here for the word already on the bus.
                        av <= 1'b1;
                        if (mm + MIDX_BITS'(1) < mrows)
                            act_addr_q <= act_addr_q + k_words;
                        mm <= mm + MIDX_BITS'(1);
                    end else begin
                        state <= S_RETIRE;
                    end
                end

                // Every column must land its last partial sum before the tile's
                // weights are overwritten.
                S_RETIRE: begin
                    if (all_retired) begin
                        if (last_kt) begin
                            req_i      <= '0;
                            wr_cnt     <= '0;
                            out_addr_q <= ADDR_BITS'(GM_BITS'(m_base) * n_words)
                                          + ADDR_BITS'(n_tile);
                            state      <= S_REQ;
                        end else begin
                            k_tile   <= k_tile + TIDX_BITS'(1);
                            first_kt <= 1'b0;
                            last_kt  <= (k_tile + TIDX_BITS'(2) == cfg_k_tiles);
                            wcnt     <= '0;
                            state    <= S_WLOAD;
                        end
                    end
                end

                // Drain: all COLS accumulators share an address here, so the
                // requantized lanes emerge aligned and pack straight into a word.
                S_REQ: begin
                    if (req_i < mrows) begin
                        rq_v   <= 1'b1;
                        req_i  <= req_i + MIDX_BITS'(1);
                    end else if (wr_cnt == mrows) begin
                        if (GM_BITS'(m_base) + GM_BITS'(MBLK) >= cfg_m) begin
                            if (n_tile + TIDX_BITS'(1) == cfg_n_tiles) begin
                                busy  <= 1'b0;
                                done  <= 1'b1;
                                state <= S_IDLE;
                            end else begin
                                n_tile <= n_tile + TIDX_BITS'(1);
                                m_base <= '0;
                                state  <= S_NCFG;
                            end
                        end else begin
                            m_base <= m_base + GM_BITS'(MBLK);
                            state  <= S_NCFG;
                        end
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
