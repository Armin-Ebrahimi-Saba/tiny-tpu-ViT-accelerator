`timescale 1ns/1ps

// ABOUTME: Integer LayerNorm over one row, bit-exact against qlayernorm() in sw/kernels_int.py.
// ABOUTME: Three passes -- sum, centered-square-sum, then normalize -- plus one isqrt per row.
//
// Following the reference exactly:
//
//   total  = sum(x)
//   d      = N*x - total                 == N*(x - mean), exact, no division
//   denom  = sum(d*d) + eps_q
//   r      = max(isqrt(denom), 1)
//   norm   = (d << 14) // r
//   out    = clip(requant_keep(norm * gamma) + beta, -127, 127)
//
// Two properties of that formulation let the whole normalization run in the raw
// quantized domain with no scale at all:
//
// The input scale cancels inside (x - mean) / sqrt(var), so only the affine tail
// needs `requant`. The host folds the sqrt(N) / 2**14 constant into that
// multiplier (see quantize.py), which is why no factor of N appears here.
//
// The mean is never actually computed. Multiplying through by N keeps `d` an
// exact integer; a rounded mean would inject an error into every element before
// the variance is even taken.
//
// `eps_q` arrives pre-converted: eps lives in the real domain, and the host
// scales it once by eps / scale_in**2 * N**3 to reach the squared quantized
// domain of `denom`.
//
// Cost is ~3N cycles per row plus one 32-step isqrt. The square root is per row
// -- 2740 per transformer block against ~1.05 M divisions -- which is why it
// stays sequential while the divide is fully pipelined.

module layernorm_int #(
    parameter int MAX_LEN   = 1370,
    parameter int IN_BITS   = 8,
    parameter int FRAC_BITS = 14,     // _LN_FRAC_BITS
    parameter int SUM2_BITS = 64
)(
    input  logic clk,
    input  logic rst,

    // Per-channel affine parameters, indexed by position in the row.
    input  logic                          par_wr_en,
    input  logic [$clog2(MAX_LEN)-1:0]    par_wr_addr,
    input  logic signed [7:0]             par_wr_gamma,
    input  logic signed [31:0]            par_wr_beta,

    // Per-row configuration, sampled when a row starts.
    input  logic [$clog2(MAX_LEN+1)-1:0]  cfg_len,
    input  logic [SUM2_BITS-1:0]          cfg_eps_q,
    input  logic signed [31:0]            cfg_mult,
    input  logic [5:0]                    cfg_shift,

    input  logic                          in_valid,
    input  logic signed [IN_BITS-1:0]     in_data,
    output logic                          in_ready,

    output logic                          out_valid,
    output logic signed [7:0]             out_data
);

    localparam int IDX_BITS  = $clog2(MAX_LEN + 1);
    localparam int MEM_BITS  = (MAX_LEN <= 1) ? 1 : $clog2(MAX_LEN);
    localparam int TOT_BITS  = IN_BITS + IDX_BITS;             // sum of MAX_LEN int8 values
    localparam int D_BITS    = IN_BITS + IDX_BITS + 1;         // |N*x - total| <= 2*N*127
    localparam int NUM_BITS  = D_BITS + FRAC_BITS;             // d << FRAC_BITS
    localparam int ACC_BITS  = NUM_BITS + 8;                   // norm * gamma
    localparam int ROOT_BITS = SUM2_BITS / 2;
    localparam int DIV_LAT   = NUM_BITS + 2;                   // divider_pipe latency
    localparam int RQ_LAT    = 2;                              // requant_unit latency

    typedef enum logic [2:0] { S_IDLE, S_LOAD, S_CENTER, S_SQRT, S_NORM } state_e;
    state_e state;

    logic signed [IN_BITS-1:0] x_mem     [MAX_LEN];
    logic signed [D_BITS-1:0]  d_mem     [MAX_LEN];
    logic signed [7:0]         gamma_mem [MAX_LEN];
    logic signed [31:0]        beta_mem  [MAX_LEN];

    always_ff @(posedge clk) begin
        if (par_wr_en) begin
            gamma_mem[par_wr_addr] <= par_wr_gamma;
            beta_mem[par_wr_addr]  <= par_wr_beta;
        end
    end

    logic [IDX_BITS-1:0]        len_q, cnt, out_cnt;
    logic signed [TOT_BITS-1:0] total_q;
    logic [SUM2_BITS-1:0]       sum_d2;
    logic [SUM2_BITS-1:0]       eps_q;
    logic signed [31:0]         mult_q;
    logic [5:0]                 shift_q;
    logic [ROOT_BITS-1:0]       r_q;

    assign in_ready = (state == S_IDLE) || (state == S_LOAD);

    // ------------------------------------------------------------ read stages
    // Every memory is read synchronously, so a result trails its address by one
    // cycle. va/vb track the centering pass, dva/dvb the normalize pass.
    logic                      va, vb, dva, dvb;
    logic [MEM_BITS-1:0]       ia, ib;
    logic signed [IN_BITS-1:0] x_rd;
    logic signed [D_BITS-1:0]  d_rd;
    logic signed [7:0]         gamma_rd;
    logic signed [31:0]        beta_rd;

    // d = N*x - total, exact. N is at most MAX_LEN, so this is a small multiply.
    wire signed [D_BITS-1:0] len_s  = D_BITS'($signed({1'b0, len_q}));
    wire signed [D_BITS-1:0] d_next = (len_s * D_BITS'(x_rd)) - D_BITS'(total_q);
    wire [2*D_BITS-1:0]      d_sq   = (2*D_BITS)'(d_next * d_next);

    // ------------------------------------------------------------------ isqrt
    logic                 sq_start, sq_ready, sq_valid;
    logic [ROOT_BITS-1:0] sq_root;

    isqrt #(.BITS(SUM2_BITS)) u_isqrt (
        .clk(clk), .rst(rst),
        .start(sq_start), .x_in(sum_d2 + eps_q), .ready(sq_ready),
        .valid_out(sq_valid), .root_out(sq_root)
    );

    // ------------------------------------------------------------- divide pipe
    logic                       q_valid;
    logic signed [NUM_BITS-1:0] q_out;
    logic                       q_dz;

    divider_pipe #(.NUM_BITS(NUM_BITS), .DEN_BITS(ROOT_BITS)) u_div (
        .clk(clk), .rst(rst),
        .valid_in(dvb),
        .num_in(NUM_BITS'(d_rd) <<< FRAC_BITS),
        .den_in(r_q),
        .valid_out(q_valid), .quot_out(q_out), .div_zero(q_dz)
    );

    // gamma and beta travel alongside their element rather than being re-read at
    // the far end. The divider's latency is fixed, so a delay line is exact and
    // needs no reasoning about which cycle a synchronous read would land on.
    logic signed [7:0]  gpipe [DIV_LAT];
    logic signed [31:0] bpipe [DIV_LAT+RQ_LAT];

    // requant_unit here is given a full int32 output range, which turns it into
    // apply_keep(acc, 0): rescale and round, but do not saturate. Saturation
    // happens once at the very end, after beta is added -- exactly as in the
    // reference, where clipping an intermediate would change the result.
    logic signed [31:0] rq_out;
    logic               rq_valid;

    requant_unit #(
        .ACC_BITS(ACC_BITS), .OUT_BITS(32), .INTER_BITS(ACC_BITS + 32),
        .OUT_MIN(-(2**31 - 1)), .OUT_MAX(2**31 - 1)
    ) u_rq (
        .clk(clk), .rst(rst),
        .valid_in(q_valid),
        .acc_in(ACC_BITS'(q_out) * ACC_BITS'(gpipe[DIV_LAT-1])),
        .mult_in(mult_q), .shift_in(shift_q),
        .valid_out(rq_valid), .out(rq_out)
    );

    wire signed [32:0] biased = 33'(rq_out) + 33'(bpipe[DIV_LAT+RQ_LAT-1]);

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            out_valid <= 1'b0;
            out_data  <= '0;
        end else begin
            out_valid <= rq_valid;
            if (biased > 33'sd127)       out_data <= 8'sd127;
            else if (biased < -33'sd127) out_data <= -8'sd127;
            else                         out_data <= 8'(biased);
        end
    end

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state    <= S_IDLE;
            cnt      <= '0;
            out_cnt  <= '0;
            total_q  <= '0;
            sum_d2   <= '0;
            r_q      <= '0;
            sq_start <= 1'b0;
            {va, vb, dva, dvb} <= '0;
        end else begin
            va       <= 1'b0;
            dva      <= 1'b0;
            sq_start <= 1'b0;

            vb  <= va;  ib <= ia;
            dvb <= dva;

            for (int i = DIV_LAT - 1; i > 0; i--) gpipe[i] <= gpipe[i-1];
            for (int i = DIV_LAT + RQ_LAT - 1; i > 0; i--) bpipe[i] <= bpipe[i-1];
            gpipe[0] <= gamma_rd;
            bpipe[0] <= beta_rd;

            if (vb) begin
                d_mem[ib] <= d_next;
                sum_d2    <= sum_d2 + SUM2_BITS'(d_sq);
            end

            case (state)
                S_IDLE: begin
                    total_q <= '0;
                    sum_d2  <= '0;
                    out_cnt <= '0;
                    if (in_valid) begin
                        x_mem[0] <= in_data;
                        total_q  <= TOT_BITS'(in_data);
                        len_q    <= cfg_len;
                        eps_q    <= cfg_eps_q;
                        mult_q   <= cfg_mult;
                        shift_q  <= cfg_shift;
                        cnt      <= IDX_BITS'(1);
                        state    <= (cfg_len == IDX_BITS'(1)) ? S_CENTER : S_LOAD;
                    end
                end

                S_LOAD: begin
                    if (in_valid) begin
                        x_mem[MEM_BITS'(cnt)] <= in_data;
                        total_q <= total_q + TOT_BITS'(in_data);
                        cnt     <= cnt + IDX_BITS'(1);
                        if (cnt + IDX_BITS'(1) == len_q) begin
                            cnt   <= '0;
                            state <= S_CENTER;
                        end
                    end
                end

                S_CENTER: begin
                    if (cnt < len_q) begin
                        va  <= 1'b1;
                        ia  <= MEM_BITS'(cnt);
                        cnt <= cnt + IDX_BITS'(1);
                    end else if (!va && !vb) begin
                        // Every square has retired, so the denominator is final.
                        sq_start <= 1'b1;
                        state    <= S_SQRT;
                    end
                end

                S_SQRT: begin
                    if (sq_valid) begin
                        // The reference clamps to 1; a zero root only happens on
                        // an exactly constant row with eps_q == 0.
                        r_q   <= (sq_root == '0) ? ROOT_BITS'(1) : sq_root;
                        cnt   <= '0;
                        state <= S_NORM;
                    end
                end

                S_NORM: begin
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

    always_ff @(posedge clk) begin
        x_rd     <= x_mem[ia];
        d_rd     <= d_mem[ia];
        gamma_rd <= gamma_mem[ia];
        beta_rd  <= beta_mem[ia];
    end

    // synthesis translate_off
    // The FSM only starts a root between rows, so it can never collide with one
    // already running. Assert it rather than assume it.
    always_ff @(posedge clk) begin
        if (!rst && sq_start && !sq_ready)
            $fatal(1, "layernorm_int: isqrt started while busy");
    end

    always_ff @(posedge clk) begin
        if (!rst && q_valid && q_dz)
            $fatal(1, "layernorm_int: divide by zero -- r was not clamped");
    end
    // synthesis translate_on

endmodule
