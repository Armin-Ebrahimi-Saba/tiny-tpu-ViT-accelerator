`timescale 1ns/1ps

// ABOUTME: Exact integer square root, floor(sqrt(x)), for the LayerNorm variance denominator.
// ABOUTME: Bit-exact with _isqrt() in sw/kernels_int.py without reproducing its float64 seed.
//
// The reference seeds Newton from a float64 sqrt and then refines in integer
// arithmetic. RTL has no float64, and simply swapping in a cheap seed does not
// work: Newton's convergence depends on the seed, and from a power-of-two seed
// a 62-bit input can need ~30 iterations to settle.
//
// What makes the substitution legal is the *tail* of the reference, not its
// seed. After the iterations it does:
//
//     r = r - 1  if  r*r > x
//     r = r + 1  if  (r+1)*(r+1) <= x
//
// Those two comparisons pin the exact floor(sqrt(x)) for any r already within
// one of it. In other words the reference's contract is "exact floor sqrt", and
// the float seed is an implementation detail of reaching it. So this module
// computes floor(sqrt(x)) directly, by the classic restoring digit-by-digit
// method: two bits of radicand per step, one bit of root, BITS/2 steps, no seed
// and no convergence question at all.
//
// LayerNorm needs one of these per token -- 2740 per transformer block, against
// ~1.05 M divisions -- so unlike the divider this one stays sequential.

module isqrt #(
    parameter int BITS = 64           // must be even; x is unsigned
)(
    input  logic clk,
    input  logic rst,

    input  logic                  start,
    input  logic [BITS-1:0]       x_in,
    output logic                  ready,

    output logic                  valid_out,
    output logic [BITS/2-1:0]     root_out
);

    localparam int STEPS     = BITS / 2;
    localparam int STEP_BITS = $clog2(STEPS + 1);
    localparam int REM_BITS  = BITS/2 + 3;   // rem*4 plus the two shifted-in bits

    logic [BITS-1:0]      x_q;       // shifted left two bits per step
    logic [BITS/2:0]      rem;       // stays below 2*root+1 by construction
    logic [BITS/2-1:0]    root;
    logic [STEP_BITS-1:0] step;
    logic                 busy;

    assign ready = !busy;

    wire [REM_BITS-1:0] rem_sh = {rem, x_q[BITS-1:BITS-2]};
    wire [REM_BITS-1:0] trial  = {{(REM_BITS-BITS/2-2){1'b0}}, root, 2'b01};
    wire                fits   = (rem_sh >= trial);

    always_ff @(posedge clk) begin
        if (rst) begin
            busy      <= 1'b0;
            valid_out <= 1'b0;
            root_out  <= '0;
            x_q       <= '0;
            rem       <= '0;
            root      <= '0;
            step      <= '0;
        end else begin
            valid_out <= 1'b0;

            if (!busy) begin
                if (start) begin
                    x_q  <= x_in;
                    rem  <= '0;
                    root <= '0;
                    step <= '0;
                    busy <= 1'b1;
                end
            end else begin
                rem  <= (BITS/2+1)'(fits ? (rem_sh - trial) : rem_sh);
                root <= {root[BITS/2-2:0], fits};
                x_q  <= {x_q[BITS-3:0], 2'b00};
                if (step == STEP_BITS'(STEPS - 1)) begin
                    busy      <= 1'b0;
                    valid_out <= 1'b1;
                    root_out  <= {root[BITS/2-2:0], fits};
                end else begin
                    step <= step + STEP_BITS'(1);
                end
            end
        end
    end

endmodule
