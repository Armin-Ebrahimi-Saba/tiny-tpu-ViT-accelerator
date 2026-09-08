#!/usr/bin/env bash
# ABOUTME: Regenerates tpu-v2 test vectors from the sw/ reference and runs every Verilator testbench.
# ABOUTME: Exits non-zero on the first failure so it can be dropped straight into CI.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
build="$here/build"

RTL=(
    "$root/src/v2/pe_int8.sv"
    "$root/src/v2/systolic_int8.sv"
    "$root/src/v2/requant_unit.sv"
    "$root/src/v2/gemm_tile_int8.sv"
    "$root/src/v2/unary_lut.sv"
    "$root/src/v2/divider.sv"
    "$root/src/v2/divider_pipe.sv"
    "$root/src/v2/softmax_int.sv"
    "$root/src/v2/isqrt.sv"
    "$root/src/v2/layernorm_int.sv"
    "$root/src/v2/qadd_unit.sv"
    "$root/src/v2/gemm_seq.sv"
    "$root/src/v2/vpu_seq.sv"
)

TBS=(tb_requant tb_systolic_int8 tb_gemm_tile_int8 tb_unary_lut tb_divider tb_divider_pipe tb_softmax_int tb_isqrt tb_layernorm_int tb_qadd_unit tb_gemm_seq tb_vpu_seq)

build_tb() {
    local top="$1"
    mkdir -p "$build/$top"
    # -Wall keeps the RTL honest; the suppressions below are all testbench-only
    # idioms (a blocking clock generator, a shared reset, unused shared params).
    verilator --binary -j 0 --quiet \
        -Wall -Wno-fatal \
        -Wno-BLKSEQ -Wno-INITIALDLY -Wno-SYNCASYNCNET -Wno-UNUSEDPARAM \
        -Wno-UNOPTFLAT \
        --timing \
        -I"$here/vec" \
        --Mdir "$build/$top" \
        -o "$top" \
        --top-module "$top" \
        "${RTL[@]}" "$here/$top.sv"
}

run_all() {
    for top in "${TBS[@]}"; do
        build_tb "$top"
        ( cd "$here" && "$build/$top/$top" )
    done
}

# A single geometry can hide a misalignment that happens to cancel, so sweep a
# few shapes -- including non-square and K=1/N=1 degenerate tiles -- with a
# different seed each time.
# The sequencer sweep rides along: each entry also picks a GEMM larger than the
# array, so the k, n and m-block loops are exercised together with the tile.
SHAPES=(
    "-m 8  -k 16 -n 16 --seq-m 40 --seq-k 45 --seq-n 30"
    "-m 1  -k 16 -n 16 --seq-m 33 --seq-k 48 --seq-n 48"
    "-m 12 -k 4  -n 7  --seq-m 1  --seq-k 16 --seq-n 16"
    "-m 5  -k 7  -n 3  --seq-m 17 --seq-k 1  --seq-n 1"
    "-m 3  -k 1  -n 5  --seq-m 64 --seq-k 64 --seq-n 16"
    "-m 4  -k 9  -n 1  --seq-m 16 --seq-k 32 --seq-n 32"
    "-m 16 -k 16 -n 16 --seq-m 48 --seq-k 16 --seq-n 64"
    # The two deepest tiles a real ViT-S block asks for, at the m and n tiling
    # sw/tiling.py picks for a 48 KB weight buffer: fc2 (K=1536) and attention's
    # P.V (K=1370). K is the dimension the weight buffer is measured in, so
    # these are the shapes that decide whether the buffers are sized right.
    "-m 4 -k 8 -n 12 --seq-m 21 --seq-k 1536 --seq-n 32"
    "-m 6 -k 3 -n 9  --seq-m 5  --seq-k 1370 --seq-n 32"
)

if [ "$#" -gt 0 ]; then
    python3 "$here/gen_vectors.py" "$@"
    run_all
else
    seed=1
    for shape in "${SHAPES[@]}"; do
        echo "=== shape: $shape (seed $seed) ==="
        # shellcheck disable=SC2086
        python3 "$here/gen_vectors.py" $shape --seed "$seed"
        run_all
        seed=$((seed + 1))
    done
fi

echo
echo "all tpu-v2 testbenches passed"
