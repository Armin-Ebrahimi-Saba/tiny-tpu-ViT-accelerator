`timescale 1ns/1ps

// ABOUTME: Integer softmax over one row, bit-exact against qsoftmax() in sw/kernels_int.py.
// ABOUTME: Three passes over the row -- max, exp+sum, divide+requant -- because each needs the previous one's reduction.
//
// The structure follows the reference exactly:
//
//   m     = max(x)
//   d     = clamp(m - x, 0, ENTRIES**2 - 1)
//   e     = (lut_lo[d % ENTRIES] * lut_hi[d / ENTRIES]) >> 15
//   total = sum(e)
//   probs = (e << 15) // total
//   out   = requant(probs)
//
// Two things about that are worth stating because they are what make the
// hardware cheap:
//
// The exp table is split in two. A single 256-entry ROM is enough for int8
// logits but not for the int16 attention logits, whose useful range of `d`
// spans tens of thousands. Factoring exp(-s*(256*hi + lo)) into
// exp(-s*256*hi) * exp(-s*lo) turns one impossible 65536-entry ROM into two
// 256-entry ones and a multiply. Both tables carry the same uniform scale
// factor, which cancels in the normalization below, so it costs no accuracy.
//
// Subtracting the max first means the exponent argument is always <= 0, so the
// table is one-sided and `d` is unsigned. That is also what keeps `total` from
// overflowing: every term is <= 2**15.
//
// Three passes need the row buffered, and the row cannot be streamed straight
// through, because `total` is not known until the last element has been seen.
// The buffers are the reason MAX_LEN is a parameter: at 1370 tokens they are
// ~2.7 KB and ~5.5 KB, i.e. a couple of BRAMs.

module softmax_int #(
    parameter int MAX_LEN   = 1370,  // longest row (ViT-S/14 at 518x518 => 1370 tokens)
    parameter int IN_BITS   = 16,    // int16 logits; int8 sign-extends into this
    parameter int ENTRIES   = 256,
    parameter int LUT_BITS  = 16
)(
    input  logic clk,
    input  logic rst,

    // Exp table load. Both halves share an address and data bus; `lut_sel`
    // picks lo (0) or hi (1).
    input  logic                        lut_wr_en,
    input  logic                        lut_wr_sel,
    input  logic [$clog2(ENTRIES)-1:0]  lut_wr_addr,
    input  logic [LUT_BITS-1:0]         lut_wr_data,

    // Per-row configuration, sampled when a row starts.
    input  logic [$clog2(MAX_LEN+1)-1:0] cfg_len,
    input  logic [31:0]                  cfg_mult,
    input  logic [5:0]                   cfg_shift,

    input  logic                       in_valid,
    input  logic signed [IN_BITS-1:0]  in_data,
    output logic                       in_ready,

    output logic                       out_valid,
    output logic signed [7:0]          out_data
);

    localparam int IDX_BITS   = $clog2(MAX_LEN + 1);
    // Counters need one more bit than the memories do, because they count *up
    // to* MAX_LEN. Addressing with the wider value would be a width truncation.
    localparam int MEM_BITS   = (MAX_LEN <= 1) ? 1 : $clog2(MAX_LEN);
    localparam int ADDR_BITS  = $clog2(ENTRIES);
    localparam int D_BITS     = 2 * ADDR_BITS;                 // clamp target: ENTRIES**2 - 1
    localparam int E_BITS     = LUT_BITS;                      // (lo*hi)>>15 stays under 2**15
    localparam int TOTAL_BITS = E_BITS + IDX_BITS;             // MAX_LEN terms, each < 2**E_BITS
    localparam int FRAC_BITS  = 15;                            // _SOFTMAX_FRAC_BITS
    localparam int LUT_SHIFT  = 15;                            // _EXP_LUT_SHIFT
    localparam int NUM_BITS   = 32;

    typedef enum logic [1:0] { S_IDLE, S_LOAD, S_EXP, S_DIV } state_e;
    state_e state;

    logic signed [IN_BITS-1:0] x_mem [MAX_LEN];
    logic        [E_BITS-1:0]  e_mem [MAX_LEN];
    logic        [LUT_BITS-1:0] lut_lo [ENTRIES];
    logic        [LUT_BITS-1:0] lut_hi [ENTRIES];

    always_ff @(posedge clk) begin
        if (lut_wr_en) begin
            if (lut_wr_sel) lut_hi[lut_wr_addr] <= lut_wr_data;
            else            lut_lo[lut_wr_addr] <= lut_wr_data;
        end
    end

    logic signed [IN_BITS-1:0]  max_q;
    logic [IDX_BITS-1:0]        len_q;
    logic [IDX_BITS-1:0]        cnt;       // load / issue counter
    logic [IDX_BITS-1:0]        out_cnt;
    logic [TOTAL_BITS-1:0]      total_q;
    logic [31:0]                mult_q;
    logic [5:0]                 shift_q;

    assign in_ready = (state == S_IDLE) || (state == S_LOAD);

    // ---------------------------------------------------------------- exp pass
    // Stage a: address issued.  Stage b: x_mem output valid.
    // Stage c: d registered, so the LUTs are addressed.  Stage d: both LUT
    // outputs valid -> multiply, store, accumulate.
    logic                      va, vb, vc, vd;
    logic [MEM_BITS-1:0]       ia, ib, ic, id;  // memory addresses, narrower than cnt
    logic signed [IN_BITS-1:0] x_rd;
    logic [D_BITS-1:0]         d_q;
    logic [LUT_BITS-1:0]       lo_rd, hi_rd;

    // m - x is non-negative by construction, but it can exceed the table's
    // reach, so it saturates rather than wrapping.
    wire signed [IN_BITS:0] diff      = (IN_BITS+1)'(max_q) - (IN_BITS+1)'(x_rd);
    wire                    diff_over = (diff > (IN_BITS+1)'((1 << D_BITS) - 1));
    wire [D_BITS-1:0]       d_next    = diff_over ? {D_BITS{1'b1}} : D_BITS'(diff);

    wire [2*LUT_BITS-1:0] e_prod = lo_rd * hi_rd;
    wire [E_BITS-1:0]     e_next = E_BITS'(e_prod >> LUT_SHIFT);

    // ---------------------------------------------------------------- div pass
    logic                 dva, dvb;
    logic [E_BITS-1:0]    e_rd;
    logic                 q_valid;
    logic signed [NUM_BITS-1:0] q_out;
    logic                 q_dz;

    divider_pipe #(.NUM_BITS(NUM_BITS), .DEN_BITS(TOTAL_BITS)) u_div (
        .clk(clk), .rst(rst),
        .valid_in(dvb), .num_in(NUM_BITS'({{(NUM_BITS-E_BITS){1'b0}}, e_rd} << FRAC_BITS)),
        .den_in(total_q),
        .valid_out(q_valid), .quot_out(q_out), .div_zero(q_dz)
    );

    requant_unit #(.ACC_BITS(NUM_BITS), .OUT_BITS(8)) u_rq (
        .clk(clk), .rst(rst),
        .valid_in(q_valid), .acc_in(q_out), .mult_in(mult_q), .shift_in(shift_q),
        .valid_out(out_valid), .out(out_data)
    );

    always_ff @(posedge clk) begin
        if (rst) begin
            state   <= S_IDLE;
            cnt     <= '0;
            out_cnt <= '0;
            total_q <= '0;
            max_q   <= '0;
            len_q   <= '0;
            {va, vb, vc, vd}  <= '0;
            {dva, dvb}        <= '0;
        end else begin
            va <= 1'b0;
            dva <= 1'b0;

            // Read-address pipeline registers; the memories are read
            // synchronously so each result trails its address by one cycle.
            vb <= va; ib <= ia;
            vc <= vb; ic <= ib;
            vd <= vc; id <= ic;
            dvb <= dva;

            if (vb) d_q <= d_next;

            if (vd) begin
                e_mem[id] <= e_next;
                total_q   <= total_q + TOTAL_BITS'(e_next);
            end

            case (state)
                S_IDLE: begin
                    total_q <= '0;
                    out_cnt <= '0;
                    if (in_valid) begin
                        x_mem[0] <= in_data;
                        max_q    <= in_data;
                        len_q    <= cfg_len;
                        mult_q   <= cfg_mult;
                        shift_q  <= cfg_shift;
                        cnt      <= IDX_BITS'(1);
                        state    <= (cfg_len == IDX_BITS'(1)) ? S_EXP : S_LOAD;
                    end
                end

                S_LOAD: begin
                    if (in_valid) begin
                        x_mem[MEM_BITS'(cnt)] <= in_data;
                        if (in_data > max_q) max_q <= in_data;
                        cnt <= cnt + IDX_BITS'(1);
                        if (cnt + IDX_BITS'(1) == len_q) begin
                            cnt   <= '0;
                            state <= S_EXP;
                        end
                    end
                end

                S_EXP: begin
                    if (cnt < len_q) begin
                        va  <= 1'b1;
                        ia  <= MEM_BITS'(cnt);
                        cnt <= cnt + IDX_BITS'(1);
                    end else if (!va && !vb && !vc && !vd) begin
                        // Last accumulation has retired, so total_q is final.
                        cnt   <= '0;
                        state <= S_DIV;
                    end
                end

                S_DIV: begin
                    if (cnt < len_q) begin
                        dva <= 1'b1;
                        ia  <= MEM_BITS'(cnt);
                        cnt <= cnt + IDX_BITS'(1);
                    end
                    if (out_valid) begin
                        out_cnt <= out_cnt + IDX_BITS'(1);
                        if (out_cnt + IDX_BITS'(1) == len_q) begin
                            cnt   <= '0;
                            state <= S_IDLE;
                        end
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // Synchronous reads shared by both passes.
    always_ff @(posedge clk) begin
        x_rd  <= x_mem[ia];
        e_rd  <= e_mem[ia];
        lo_rd <= lut_lo[d_q[ADDR_BITS-1:0]];
        hi_rd <= lut_hi[d_q[D_BITS-1:ADDR_BITS]];
    end

    // A zero denominator means the exp table underflowed the whole row; the
    // reference raises there, so flag it loudly in simulation rather than
    // silently emitting zeros.
    // synthesis translate_off
    always_ff @(posedge clk) begin
        if (!rst && q_valid && q_dz)
            $fatal(1, "softmax_int: exp LUT underflowed -- total == 0");
    end
    // synthesis translate_on

endmodule
