`timescale 1ns/1ps

// ABOUTME: Shared sequential integer divider with floor semantics, used by softmax and LayerNorm.
// ABOUTME: Bit-exact mirror of Python's `//` operator, which rounds toward -infinity.
//
// Both consumers need exactly one division per element:
//
//   softmax    probs = (e << 15) // total        numerator >= 0
//   layernorm  norm  = (d << 14) // r            numerator signed
//
// The signed case is the reason this is not a plain restoring divider. Python's
// `//` floors, so -7 // 2 == -4, whereas the natural hardware result (and C, and
// SystemVerilog's own `/`) truncates toward zero and gives -3. Since sw/ is the
// specification, the correction below is mandatory, not cosmetic: LayerNorm's
// numerator `d = N*x - mean*N` is negative for every below-average channel, i.e.
// about half of all elements, so getting this wrong would be a systematic bias
// rather than a rare edge case.
//
// The core is unsigned restoring division: one quotient bit per cycle, a compare
// and a conditional subtract. That costs NUM_BITS cycles, which is affordable
// here because divisions are per-row (softmax) or per-token (LayerNorm), not per
// MAC -- the systolic array does the work that has to be fast.
//
// Division by zero returns quotient 0 and asserts `div_zero`; callers are
// expected to have clamped the divisor (both references do: softmax raises on a
// zero denominator, LayerNorm applies max(r, 1)).

module divider #(
    parameter int NUM_BITS = 64,   // signed numerator width
    parameter int DEN_BITS = 32    // unsigned divisor width
)(
    input  logic clk,
    input  logic rst,

    // Accepts a new operand pair whenever `ready` is high.
    input  logic                          start,
    input  logic signed [NUM_BITS-1:0]    num_in,
    input  logic        [DEN_BITS-1:0]    den_in,
    output logic                          ready,

    output logic                          valid_out,
    output logic signed [NUM_BITS-1:0]    quot_out,
    output logic                          div_zero
);

    localparam int STEP_BITS = $clog2(NUM_BITS + 1);

    typedef enum logic [1:0] { S_IDLE, S_RUN, S_FIX } state_e;
    state_e state;

    logic [NUM_BITS-1:0]   num_abs;   // |numerator|, shifted left one bit per step
    logic [NUM_BITS-1:0]   quot_mag;  // unsigned magnitude of the quotient
    logic [DEN_BITS:0]     rem;       // one bit wider than the divisor for the trial shift
    logic [DEN_BITS-1:0]   den_q;
    logic                  num_neg;
    logic [STEP_BITS-1:0]  step;

    // Trial subtraction for this cycle: shift the next numerator bit in from the
    // top and see whether the divisor fits.
    wire [DEN_BITS:0] rem_shifted = {rem[DEN_BITS-1:0], num_abs[NUM_BITS-1]};
    wire              fits        = (rem_shifted >= {1'b0, den_q});
    wire [DEN_BITS:0] rem_next    = fits ? (rem_shifted - {1'b0, den_q}) : rem_shifted;

    assign ready = (state == S_IDLE);

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state     <= S_IDLE;
            valid_out <= 1'b0;
            quot_out  <= '0;
            div_zero  <= 1'b0;
            num_abs   <= '0;
            quot_mag  <= '0;
            rem       <= '0;
            den_q     <= '0;
            num_neg   <= 1'b0;
            step      <= '0;
        end else begin
            valid_out <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        // -(-2**(NUM_BITS-1)) is not representable, but neither
                        // consumer reaches it: softmax is non-negative and
                        // LayerNorm's numerator is bounded far below that.
                        num_abs  <= num_in[NUM_BITS-1] ? NUM_BITS'(-num_in) : NUM_BITS'(num_in);
                        num_neg  <= num_in[NUM_BITS-1];
                        den_q    <= den_in;
                        quot_mag <= '0;
                        rem      <= '0;
                        step     <= '0;
                        if (den_in == '0) begin
                            // Report rather than hang; quotient is defined as 0.
                            quot_out  <= '0;
                            div_zero  <= 1'b1;
                            valid_out <= 1'b1;
                        end else begin
                            div_zero <= 1'b0;
                            state    <= S_RUN;
                        end
                    end
                end

                S_RUN: begin
                    rem      <= rem_next;
                    num_abs  <= {num_abs[NUM_BITS-2:0], 1'b0};
                    quot_mag <= {quot_mag[NUM_BITS-2:0], fits};
                    if (step == STEP_BITS'(NUM_BITS - 1)) begin
                        state <= S_FIX;
                    end else begin
                        step <= step + STEP_BITS'(1);
                    end
                end

                S_FIX: begin
                    // Floor correction: truncation already gives the right answer
                    // for a non-negative numerator and for an exact division. It
                    // is only the inexact negative case that must round away from
                    // zero, which `rem != 0` detects.
                    if (num_neg) begin
                        quot_out <= (rem != '0) ? -NUM_BITS'(quot_mag + NUM_BITS'(1))
                                                : -NUM_BITS'(quot_mag);
                    end else begin
                        quot_out <= NUM_BITS'(quot_mag);
                    end
                    valid_out <= 1'b1;
                    state     <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
