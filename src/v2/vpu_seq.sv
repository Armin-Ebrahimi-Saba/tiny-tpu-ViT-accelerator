// ABOUTME: Streams rows out of a 128-bit buffer through one of the vector ops and back.
// ABOUTME: Turns softmax / layernorm / residual-add / activation-LUT / transpose into one start-and-poll op.

// Vector sequencer.
//
// `gemm_seq.sv` made the systolic array driveable from a register write; this
// does the same for the vector kernels. Each of them is a one-element-per-cycle
// stream processor, but the buffers hand out 128 bits at a time, so what sits
// between them is a serializer on the read side and a packer on the write side.
//
// Three fixed ports, exactly like the GEMM: operand A, operand B (residual add
// only), and a destination. There is no region select. Software places data
// where an op expects it, and the weight DMA is what moves it -- including
// buffer-to-buffer, since the fast aperture is itself a device on the main
// crossbar, so a DMA whose source is 0x2xxxxxxx copies one region into another
// with the CPU out of the loop. That is how a transformer block passes results
// from one op to the next.
//
// Rows are packed flat: `rows * len` int8 elements starting at a word-aligned
// offset. Only the final 128-bit word of the destination can be partial, which
// is why the write port carries a byte mask.
//
// Element rate is one per two cycles: the read is registered, and a new one is
// only issued once the outstanding datum is known to be consumed. Overlapping
// them needs the read pipeline to know the kernel's back-pressure a cycle early,
// which softmax's three-phase `in_ready` does not offer. Same trade as the
// weight DMA: correctness now, throughput after the sequencing is proven.
//
// Softmax's input port is int16 -- it exists for logits wider than a GEMM
// output -- but everything here is int8, sign-extended on the way in, because
// int8 is what a requantized GEMM produces and what the next op consumes.
//
// The transpose is the odd one out: it has no kernel at all, only a read
// address pattern. It is here because attention needs K-transpose and the GEMM
// reads its stationary operand row-major, so without it a transformer block
// cannot be expressed at all -- and doing it on the CPU is 8 bits per bus round
// trip, which is the one place that cost is unarguable.

module vpu_seq #(
    parameter int LANES     = 16,          // int8 lanes per buffer word
    parameter int SM_LEN    = 1370,        // longest softmax row (1370 ViT-S tokens)
    parameter int LN_LEN    = 1024,        // longest layernorm row (384 for ViT-S)
    parameter int ADDR_BITS = 20,
    parameter int CNT_BITS  = 24
)(
    input  logic clk,
    input  logic rst,

    // ---- table and parameter loads, passed through to the kernels ----
    input  logic                       sm_lut_wr_en,
    input  logic                       sm_lut_wr_sel,   // 0 = lo half, 1 = hi half
    input  logic [7:0]                 sm_lut_wr_addr,
    input  logic [15:0]                sm_lut_wr_data,

    input  logic                       un_lut_wr_en,
    input  logic [7:0]                 un_lut_wr_addr,
    input  logic signed [7:0]          un_lut_wr_data,

    input  logic                       ln_par_wr_en,
    input  logic [$clog2(LN_LEN)-1:0]  ln_par_wr_addr,
    input  logic signed [7:0]          ln_par_wr_gamma,
    input  logic signed [31:0]         ln_par_wr_beta,

    // ---- command ----
    input  logic                       start,
    input  logic [3:0]                 op,
    input  logic [15:0]                cfg_len,      // elements per row
    input  logic [15:0]                cfg_rows,
    input  logic signed [31:0]         cfg_mult,
    input  logic [5:0]                 cfg_shift,
    input  logic signed [31:0]         cfg_mult_b,   // residual add, operand B
    input  logic [5:0]                 cfg_shift_b,
    input  logic [63:0]                cfg_eps,      // layernorm epsilon, already scaled
    input  logic [15:0]                cfg_a_word,
    input  logic [15:0]                cfg_b_word,
    input  logic [15:0]                cfg_d_word,

    output logic                       busy,
    output logic                       done,

    // ---- buffer ports ----
    output logic [ADDR_BITS-1:0]       a_addr,
    input  logic [8*LANES-1:0]         a_data,
    output logic [ADDR_BITS-1:0]       b_addr,
    input  logic [8*LANES-1:0]         b_data,

    output logic                       o_we,
    output logic [ADDR_BITS-1:0]       o_addr,
    output logic [LANES-1:0]           o_wmask,
    output logic [8*LANES-1:0]         o_data
);

    localparam int LB = $clog2(LANES);

    localparam logic [3:0] OP_SOFTMAX   = 4'd1;
    localparam logic [3:0] OP_LAYERNORM = 4'd2;
    localparam logic [3:0] OP_QADD      = 4'd3;
    localparam logic [3:0] OP_UNARY     = 4'd4;
    localparam logic [3:0] OP_TRANSPOSE = 4'd5;

    localparam int SM_CB = $clog2(SM_LEN + 1);
    localparam int LN_CB = $clog2(LN_LEN + 1);

    typedef enum logic [1:0] { S_IDLE, S_RUN, S_DRAIN, S_DONE } state_e;
    state_e state;

    logic [3:0]           op_q;
    logic [CNT_BITS-1:0]  total_q;      // rows * len
    logic [15:0]          a_base_q, b_base_q, d_base_q;
    logic [15:0]          rows_q, len_q;
    logic [CNT_BITS-1:0]  fi;           // elements fed
    logic [CNT_BITS-1:0]  oi;           // elements collected

    // ------------------------------------------------------------ read side
    logic                 pend, have;
    logic [LB-1:0]        pend_lane;
    logic signed [7:0]    hold_a, hold_b;

    wire feeding   = (state == S_RUN) && (fi < total_q);
    logic unit_ready;
    wire consume   = have && unit_ready;
    wire can_issue = feeding && !pend && (!have || consume);

    // Every op but the transpose reads its operands in the order it writes
    // them, so the read index is the feed counter. The transpose is the whole
    // reason `ridx` exists as a separate register: it walks a column of the
    // source while the write side walks a row of the destination, which is one
    // stride of `len_q` per element and a step back to the next source column
    // at the end of each output row. Nothing else in the datapath changes --
    // the transpose is an addressing pattern, not a kernel.
    logic [CNT_BITS-1:0] ridx, rbase;
    logic [15:0]         tcol;
    wire  sel_tr = (op_q == OP_TRANSPOSE);
    assign a_addr = ADDR_BITS'(a_base_q) + ADDR_BITS'(ridx >> LB);
    assign b_addr = ADDR_BITS'(b_base_q) + ADDR_BITS'(ridx >> LB);

    always_ff @(posedge clk) begin
        if (rst) begin
            ridx  <= '0;
            rbase <= '0;
            tcol  <= '0;
        end else if (state == S_IDLE) begin
            ridx  <= '0;
            rbase <= '0;
            tcol  <= '0;
        end else if (can_issue) begin
            if (!sel_tr) begin
                ridx <= ridx + CNT_BITS'(1);
            end else if (tcol == rows_q - 16'd1) begin
                tcol  <= '0;
                rbase <= rbase + CNT_BITS'(1);
                ridx  <= rbase + CNT_BITS'(1);
            end else begin
                tcol <= tcol + 16'd1;
                ridx <= ridx + CNT_BITS'(len_q);
            end
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            pend <= 1'b0;
            have <= 1'b0;
        end else begin
            pend      <= can_issue;
            pend_lane <= ridx[LB-1:0];
            if (pend) begin
                hold_a <= a_data[8*pend_lane +: 8];
                hold_b <= b_data[8*pend_lane +: 8];
                have   <= 1'b1;
            end else if (consume) begin
                have <= 1'b0;
            end
        end
    end

    // ---------------------------------------------------------- the kernels
    wire sel_sm = (op_q == OP_SOFTMAX);
    wire sel_ln = (op_q == OP_LAYERNORM);
    wire sel_qa = (op_q == OP_QADD);
    wire sel_un = (op_q == OP_UNARY);

    logic              sm_ready, sm_ovalid, ln_ready, ln_ovalid;
    logic signed [7:0] sm_out, ln_out;
    logic              qa_ovalid, un_ovalid;
    logic signed [7:0] qa_out, un_out;
    logic              tr_ovalid;
    logic signed [7:0] tr_out;

    // The softmax and layernorm kernels stall; the add and the lookup are fixed
    // latency and always accept, so their ready is a constant.
    always_comb begin
        unique case (op_q)
            OP_SOFTMAX:   unit_ready = sm_ready;
            OP_LAYERNORM: unit_ready = ln_ready;
            default:      unit_ready = 1'b1;
        endcase
    end

    softmax_int #(
        .MAX_LEN(SM_LEN), .IN_BITS(16)
    ) u_softmax (
        .clk, .rst,
        .lut_wr_en(sm_lut_wr_en), .lut_wr_sel(sm_lut_wr_sel),
        .lut_wr_addr(sm_lut_wr_addr), .lut_wr_data(sm_lut_wr_data),
        .cfg_len(SM_CB'(cfg_len)), .cfg_mult(cfg_mult), .cfg_shift(cfg_shift),
        .in_valid(have && sel_sm), .in_data(16'(signed'(hold_a))),
        .in_ready(sm_ready),
        .out_valid(sm_ovalid), .out_data(sm_out)
    );

    layernorm_int #(
        .MAX_LEN(LN_LEN), .IN_BITS(8)
    ) u_layernorm (
        .clk, .rst,
        .par_wr_en(ln_par_wr_en), .par_wr_addr(ln_par_wr_addr),
        .par_wr_gamma(ln_par_wr_gamma), .par_wr_beta(ln_par_wr_beta),
        .cfg_len(LN_CB'(cfg_len)), .cfg_eps_q(cfg_eps),
        .cfg_mult(cfg_mult), .cfg_shift(cfg_shift),
        .in_valid(have && sel_ln), .in_data(hold_a), .in_ready(ln_ready),
        .out_valid(ln_ovalid), .out_data(ln_out)
    );

    qadd_unit u_qadd (
        .clk, .rst,
        .valid_in(have && sel_qa), .a_in(hold_a), .b_in(hold_b),
        .mult_a(cfg_mult), .mult_b(cfg_mult_b),
        .shift_a(cfg_shift), .shift_b(cfg_shift_b),
        .valid_out(qa_ovalid), .out(qa_out)
    );

    unary_lut u_unary (
        .clk, .rst,
        .wr_en(un_lut_wr_en), .wr_addr(un_lut_wr_addr), .wr_data(un_lut_wr_data),
        .valid_in(have && sel_un), .x_in(hold_a),
        .valid_out(un_ovalid), .y_out(un_out)
    );

    // The transpose has no kernel behind it, but it still has to look like one
    // to the write side: a single register gives it the same one-cycle latency
    // the LUT and the adder have, so the packer needs no special case.
    always_ff @(posedge clk) begin
        if (rst) begin
            tr_ovalid <= 1'b0;
        end else begin
            tr_ovalid <= have && sel_tr;
            tr_out    <= hold_a;
        end
    end

    logic              out_valid;
    logic signed [7:0] out_byte;
    always_comb begin
        unique case (op_q)
            OP_SOFTMAX:   begin out_valid = sm_ovalid; out_byte = sm_out; end
            OP_LAYERNORM: begin out_valid = ln_ovalid; out_byte = ln_out; end
            OP_QADD:      begin out_valid = qa_ovalid; out_byte = qa_out; end
            OP_UNARY:     begin out_valid = un_ovalid; out_byte = un_out; end
            OP_TRANSPOSE: begin out_valid = tr_ovalid; out_byte = tr_out; end
            default:      begin out_valid = 1'b0;      out_byte = '0;     end
        endcase
    end

    // ----------------------------------------------------------- write side
    logic [8*LANES-1:0] stage;
    logic [LANES-1:0]   smask;
    wire [LB-1:0]       olane = oi[LB-1:0];

    logic [8*LANES-1:0] stage_next;
    logic [LANES-1:0]   smask_next;
    always_comb begin
        stage_next = stage;
        smask_next = smask;
        if (out_valid) begin
            stage_next[8*olane +: 8] = out_byte;
            smask_next[olane]        = 1'b1;
        end
    end

    // The last element of a run rarely lands on lane 15, so the tail word is
    // committed short and the mask keeps it from touching its neighbours.
    wire commit = out_valid && ((olane == LB'(LANES-1)) ||
                                (oi == total_q - CNT_BITS'(1)));

    always_ff @(posedge clk) begin
        if (rst) begin
            stage <= '0;
            smask <= '0;
            o_we  <= 1'b0;
        end else begin
            o_we <= commit;
            if (commit) begin
                o_addr  <= ADDR_BITS'(d_base_q) + ADDR_BITS'(oi >> LB);
                o_data  <= stage_next;
                o_wmask <= smask_next;
                stage   <= '0;
                smask   <= '0;
            end else begin
                stage <= stage_next;
                smask <= smask_next;
            end
        end
    end

    // ----------------------------------------------------------------- FSM
    always_ff @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            fi    <= '0;
            oi    <= '0;
            done  <= 1'b0;
            busy  <= 1'b0;
        end else begin
            done <= 1'b0;

            if (can_issue) fi <= fi + CNT_BITS'(1);
            if (out_valid) oi <= oi + CNT_BITS'(1);

            unique case (state)
                S_IDLE: begin
                    if (start && op != 4'd0 && cfg_len != '0 && cfg_rows != '0) begin
                        op_q     <= op;
                        total_q  <= CNT_BITS'(cfg_rows) * CNT_BITS'(cfg_len);
                        rows_q   <= cfg_rows;
                        len_q    <= cfg_len;
                        a_base_q <= cfg_a_word;
                        b_base_q <= cfg_b_word;
                        d_base_q <= cfg_d_word;
                        fi       <= '0;
                        oi       <= '0;
                        busy     <= 1'b1;
                        state    <= S_RUN;
                    end
                end

                S_RUN: begin
                    // The commit registered above still has to reach the buffer,
                    // so `done` waits a cycle rather than racing the last write.
                    if (out_valid && (oi == total_q - CNT_BITS'(1)))
                        state <= S_DRAIN;
                end

                S_DRAIN: state <= S_DONE;

                S_DONE: begin
                    done  <= 1'b1;
                    busy  <= 1'b0;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
