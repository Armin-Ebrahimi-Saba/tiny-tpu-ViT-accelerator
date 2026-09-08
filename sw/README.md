# `sw/` — Depth Anything V2 toolchain for tiny-tpu

Software side of running [Depth Anything V2](https://depth-anything-v2.github.io/) (ViT-S/14
encoder + DPT depth head, 24.8 M parameters) on the tiny-tpu datapath.

This is the **compiler and bit-accurate emulator**, not the RTL. It defines exactly what the
hardware must compute, and it proves that definition produces correct depth maps before a
single line of SystemVerilog is written.

## The honest framing

The RTL in `src/` today is a 2×2 Q8.8 training array with bias, leaky-ReLU, MSE loss and
gradient descent. It cannot run DA-V2: no convolution, no softmax, no LayerNorm, no GELU,
no bilinear upsampling, and Q8.8's ±128 range clips ViT attention logits.

So this toolchain targets **`tpu-v2`** — an extended datapath described in
[`machines/tpu_v2.json`](machines/tpu_v2.json). That file is the contract:

> Everything in `supported_ops`, `datapath`, `requant`, and `vector_unit` is something the
> RTL must implement. The emulator here is the bit-exact reference it must match.

Nothing silently degrades. Point the compiler at a machine that lacks an op and it raises
`UnsupportedOpError` naming the op — a compiler that quietly approximates produces a model
that looks like it works and is wrong.

## What is validated

| Claim | Evidence |
|---|---|
| The fp32 graph **is** Depth Anything V2 | 131 dB SQNR, cosine 1.00000000 vs HuggingFace's independent implementation at 518×518 (`test_matches_official_implementation`) |
| Every fp32 kernel is correct | 27 tests against `torch.nn.functional` — conv2d, conv_transpose2d, linear, layer_norm, softmax, gelu, bilinear, bicubic (`test_kernels_float.py`) |
| Every integer kernel is correct | SQNR floor of 25 dB vs the fp32 kernel, per op (`test_kernels_int.py`) |
| Requantization is exactly specified | Compared against the integer expression in Python bignums, not against floats (`test_numerics.py`) |
| Tiling is arithmetically sound | Tiled 16×16 GEMM is bit-identical to the whole GEMM (`test_tiled_gemm_equals_whole_gemm`) |
| Weights fit | 24.2 MiB int8, 4.7 % of the 512 MB DDR3 |

105 tests, ~2 s. `python -m pytest sw/tests -q` (add `--run-slow` for the full-resolution
equivalence check, which downloads the HF model).

## Cost on the target

`python -m sw info --img-size 518`:

```
tpu-v2: 16x16 weight_stationary array, int8 x int8 -> int32, 50 MHz (12.80 GMAC/s peak)
  MACs             57.63 G
  cycles           0.272 G
  compute time     5.4 s/frame
  array efficiency 82.8 % of peak
  DRAM traffic     27.2 MiB/frame
  bound by         compute
```

The two `head/conv2` and `head/conv1` convolutions alone are 4.1 GMAC — they run at 518×518
and 296×296 with 32 channels, and cost more than any three transformer blocks. If frame time
ever matters, that is where to look first, not at attention.

## Accuracy of int8 quantization

Measured at 518×518 against the fp32 reference. Calibrated on three photographs
(`astronaut`, `coffee`, `rocket` from `skimage.data`) and evaluated on a **held-out**
fourth (`chelsea`), so there is a genuine calibration/evaluation gap.

| Configuration | SQNR dB | cosine | Pearson r | AbsRel | δ<1.25 |
|---|---:|---:|---:|---:|---:|
| no clipping (`--calibration amax`) | 3.78 | 0.874 | 0.167 | — | 5.1 % |
| `--per-channel`, no clipping | 4.18 | 0.884 | 0.174 | — | 4.1 % |
| `--percentile 99.99` | 5.85 | 0.887 | 0.202 | — | 69.9 % |
| **`--percentile 99.9`** | **15.90** | 0.987 | 0.937 | 42.2 % | **81.4 %** |
| `--percentile 99.95` | 15.77 | **0.993** | **0.971** | 54.6 % | 78.0 % |
| `--percentile 99.9 --per-channel` | 15.50 | 0.986 | 0.937 | — | 75.9 % |
| `--percentile 99.9 --residual-int16` | 15.57 | 0.986 | 0.931 | 39.1 % | 81.9 % |
| `--percentile 99.8` | 13.89 | 0.986 | 0.929 | 43.1 % | 65.4 % |
| `--percentile 99.5` | 6.67 | 0.886 | −0.058 | — | 31.2 % |
| `--percentile 99.9 --bias-correction` | 15.24 | 0.985 | 0.925 | 40.7 % | 80.9 % |
| `--calibration mse --norm 1.0` | 13.97 | 0.985 | 0.922 | 39.7 % | 69.7 % |
| `--calibration mse --norm 2.0` | 6.21 | 0.925 | 0.558 | — | 13.3 % |
| `--calibration mse --norm 0.5` | 4.77 | 0.883 | −0.215 | — | 7.3 % |

`--percentile 99.9` is the default, because it is the best measured setting.

**Clipping is the whole game.** Going from no clipping to the 99.9th percentile is worth
**+12 dB** — more than every other knob combined. Nothing else moves the result by more
than about 1 dB.

**The optimum is sharp.** 99.9 and 99.95 are both good; 99.99 and 99.5 both collapse to
around 6 dB. If you change the model, the resolution, or the calibration set, re-sweep it —
do not assume 99.9 transfers.

Four hypotheses were tested and **disproved**. They are recorded so they are not retried, and
each one is still implemented and available behind a flag — the measurement is what is
valuable, not the code:

- *Per-channel weight scales will help.* They do not (−0.4 dB). Weight resolution is not what
  is being lost. `--per-channel`.
- *The residual stream needs int16.* It does not (−0.3 dB). The per-tensor harness shows the
  residual stream degrading monotonically with depth — 16.3 dB at block 0 down to 1.7 dB at
  block 7, with ~0 % saturation — and yet widening it changes nothing end to end. The
  LayerNorm that consumes it evidently does not care how coarse it is.
  `--residual-int16`.
- *An MSE-optimal threshold will beat a hand-picked percentile.* It does not (−1.9 dB at its
  best exponent), and it is violently sensitive to that exponent: p=1.0 gives 14.0 dB, p=0.7
  gives 3.6 dB. See `mse_optimal_threshold` for why. `--calibration mse`.
- *Bias correction will remove the residual gain/offset error.* Neutral (−0.7 dB SQNR,
  −1.5 pp AbsRel). `--bias-correction`.

One genuine bug was found this way and is worth knowing about if you extend `quantize.py`:
correcting every layer's bias at once from the *observed* output shift of the quantized
network counts each layer's inherited error again at every later layer, and it destroys the
model outright — 15.9 dB to 0.05 dB. `correct_biases` evaluates `(W − Ŵ)·E[x]` analytically
instead, which isolates each layer.

### Where the remaining error is

From `python -m sw evaluate --top 30`, mean SQNR by op type at the best setting:

```
softmax   1.4 dB      matmul       7.4 dB      conv2d       9.0 dB
gelu      3.1 dB      batch_matmul 7.8 dB      interpolate 11.5 dB
mul       4.6 dB      add          6.8 dB
layernorm 4.7 dB
```

Attention probabilities are the worst tensors in the network by a wide margin, despite the
int16 logits and the two-stage exponential. That is where to look next.

The end-to-end result — Pearson 0.94, δ<1.25 of 81 % — is a usable depth map with visible
degradation, not a broken one. Whether that is good enough is an application decision. If it
is not, the next steps are quantization-aware training or an advanced PTQ method (AdaRound,
BRECQ, SmoothQuant). None of them change the hardware contract; they change how
`quantize.py` picks scales, which is why that stage is isolated behind `QuantParams`.

Caveat: four photographs is a small evaluation. Re-run against a real corpus before treating
any single number as final.

## Files

Dependencies run one way: `machine` → `numerics` → `ir` → kernels → `execute` → `quantize`
→ `lower`/`harness` → `cli`. Nothing below imports anything above it, and only
`frontend/dav2.py` imports torch.

### `machine.py` + `machines/tpu_v2.json`

**What the hardware is allowed to do.** `MachineSpec.load("tpu-v2")` reads the JSON;
`spec.require(op_type)` raises `UnsupportedOpError` for anything undeclared, and
`spec.check_attr(op, key, value)` enforces per-op limits (kernel size, stride, dilation).
`summary()` prints the one-line target description.

Edit the **JSON**, not the Python, to change the target: array geometry, clock, bit widths,
requant form, unified-buffer and DRAM sizes, which vector units exist, and the
`supported_ops` whitelist. This file is the RTL requirements document — everything in it is
something the hardware must implement.

### `numerics.py`

**The int8 quantization contract**, and the one file an RTL engineer should read first.

- `scale_from_amax(amax, n_levels)` — symmetric scale, no zero point.
- `quantize_tensor` / `dequantize_tensor` / `quantize_bias` — real ↔ integer conversion.
- `amax_per_channel` / `quantize_per_channel` — per-output-channel weight scales.
- `Requant` — the accumulator rescaler. `from_real_multiplier(M)` normalizes M into an int32
  multiplier plus a shift via `frexp`; `apply(acc)` is the integer-only expression the RTL
  must reproduce exactly. Accepts a *vector* of multipliers for per-channel requant, with
  `broadcast_to(ndim, axis)` to place them on the right axis.
- `apply_keep(acc, keep_bits)` — rescale while retaining sub-LSB precision, for summing
  several terms before a single rounding.
- `sqnr_db`, `cosine_similarity` — the quality metrics every report uses.

### `ir.py`

**The graph representation.** `Op` (type, name, inputs, outputs, attrs), `Graph` (ops,
inputs, outputs, initializers), and `GraphBuilder` for constructing one. `Graph.validate()`
enforces single assignment and topological order. `Graph.summary()` prints op counts and
parameter totals. Deliberately minimal — no shape inference, no type system; shapes come
from actually running the graph.

### `kernels_float.py`

**The fp32 golden reference,** pure numpy. `conv2d`, `conv_transpose2d`, `linear`,
`batch_matmul`, `layernorm`, `softmax`, `gelu`, `relu`, `interpolate_bilinear`,
`interpolate_bicubic`, plus `im2col` (the exact layout the array consumes). Every one is
validated against `torch.nn.functional` in `tests/test_kernels_float.py`. Touch this only if
a kernel is wrong; the tests will tell you.

### `kernels_int.py`

**What the RTL must compute, bit for bit.** No float arithmetic touches a datapath value.

- `qlinear`, `qconv2d`, `qconv_transpose2d`, `qbatch_matmul` — int32 accumulation with an
  explicit overflow check, then requant.
- `exact_int_matmul` — integer matmul routed through float64 BLAS where that is *provably*
  exact (K·127² < 2⁵³), because numpy has no integer BLAS and an int64 einsum over a ViT is
  unusably slow.
- `qadd` — residual add with guard bits, optionally int16 out.
- `qlayernorm` — integer LayerNorm; the input scale cancels, so it runs in the raw quantized
  domain with an integer `rsqrt` (`_isqrt`, float seed + integer Newton).
- `qsoftmax` + `build_exp_lut` — two cascaded 256-entry exponential ROMs, int16 logits in.
- `build_unary_lut` / `apply_unary_lut` — GELU and any other pointwise function.
- `qinterpolate_bilinear` — fixed-point resampling, scale-preserving.

### `execute.py`

**The two interpreters.** `run_float(graph, inputs, observer=, keep=)` and
`run_quant(graph, qparams, inputs, machine=, observer=, keep=)` walk the same IR, so any
divergence is attributable to a named tensor. The `observer` callback fires on every produced
tensor — fp32 values from the float run, *raw integers* from the quantized run, so callers
can measure saturation. `_Values` frees intermediates once their last consumer has run,
which matters: attention probabilities alone are ~45 MB per block at 518×518.
`RangeObserver` is the calibration collector.

### `quantize.py`

**fp32 graph + calibration data → every run-time integer constant.** The stage to modify
when you want better accuracy; nothing here changes the hardware contract.

- `calibrate(graph, batches, method=, percentile=, norm=)` — choose each tensor's clipping
  threshold: `"amax"` (no clipping), `"percentile"`, or `"mse"` (two passes, histogram, then
  minimize `E|x−quant(x)|^p`).
- `mse_optimal_threshold(hist, amax, p=)` — the threshold search itself.
- `correct_biases(graph, qp, batches, machine=)` — fold weight-rounding's mean shift into
  the biases, analytically per layer.
- `unify_scale_groups` / `resolve_scales` — union-find forcing one scale across
  reshape/transpose/slice/concat/interpolate groups, since those preserve values.
- `wide_tensors(graph, residual=)` — which tensors are carried at int16.
- `build_quant_params(...)` — derives requant multipliers, quantized biases, GELU and
  exponential LUTs per op.
- `quantize_graph(...)` — the one-call path: calibrate, resolve, build.

Output is a `QuantParams`: `scales`, `const_q` (quantized weights), `op_params`, `rails`,
`weight_scales`.

### `lower.py`

**Does it fit, and how long does it take.** `lower_graph(graph, machine, input_shapes)`
returns a `Schedule`: every op tiled to the array, with MACs, cycles, DRAM traffic and
unified-buffer blocking. `Schedule.text()` prints the report, `hottest(n)` ranks by cycles.
`infer_shapes` propagates shapes by running the graph on zeros — cheaper than reimplementing
shape rules, and it cannot disagree with the executor because it *is* the executor.
`run_tiled_gemm` executes a GEMM the way the array does, so the tiling model can be *proved*
equivalent rather than assumed. `require_supported` gates a graph against a machine.

### `harness.py`

**Where did the accuracy go.** `compare(graph, qp, inputs, machine=, per_tensor=)` runs both
executors on one input and returns a `Report` with per-tensor SQNR, cosine, scale and
saturation. `worst(n)` and `most_saturated(n)` are the diagnostics that found every real
problem in this project. `depth_metrics()` gives AbsRel / RMSE / δ<1.25 / Pearson r.
`per_tensor=False` compares outputs only and uses far less memory.

### `imageio.py`

**Data in, pictures out.** `load_image(path, size)` applies DA-V2 preprocessing (bicubic
resize, ImageNet normalization, NCHW). `find_images(dir)` enumerates a calibration corpus.
`calibration_batches(...)` yields batches, falling back to `synthetic_image` — band-limited
1/f noise, which at least has natural spectral statistics, unlike white noise.
`save_depth_png` writes a normalized visualization.

### `frontend/dav2.py`

**Checkpoint → IR graph.** `load_state_dict(path)` reads the `.pth` into plain numpy (the
only place torch is needed). `build_dav2_graph(state_dict, DAV2Config(img_size=...))`
constructs the ViT-S/14 encoder and DPT head. `DAV2Config` holds the geometry.

Structure is derived from the checkpoint's own tensor names and shapes, not copied from the
reference repo. Two compile-time transforms live here: the fused qkv projection is split into
three matmuls, and `head_dim**-0.5` is folded into the q weights. Positional embeddings are
resampled bicubically off the native 37×37 grid, so any multiple of 14 works.

### `cli.py` + `__main__.py`

**`python -m sw <command>`.** `info` (target, graph, tiled schedule), `evaluate` (per-tensor
fp32-vs-int8 report), `run` (one image through the emulator to a PNG). Quantization flags:
`--per-channel`, `--percentile`, `--residual-int16`, `--calib-dir`, `--calib-count`.

### `tests/`

- `test_kernels_float.py` — every fp32 kernel against `torch.nn.functional`.
- `test_kernels_int.py` — every integer kernel against its fp32 counterpart at an SQNR floor.
- `test_numerics.py` — the requant contract, in Python bignums rather than floats.
- `test_compiler.py` — machine gating, IR invariants, scale unification, tiling equivalence,
  per-channel quantization.
- `test_dav2.py` — graph structure, and (with `--run-slow`) equivalence with HuggingFace's
  independent DA-V2 implementation.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install numpy pillow pytest

# What does the target look like, and what does the model cost on it?
.venv/bin/python -m sw info --img-size 518

# How much accuracy does int8 cost? Use real photographs for calibration.
# Defaults are already the measured-best quantization settings.
.venv/bin/python -m sw evaluate --img-size 518 --calib-dir /path/to/photos --top 20

# Run one image through the integer emulator and write a depth map.
.venv/bin/python -m sw run --image photo.jpg --calib-dir /path/to/photos --out depth.png
```

Without `--calib-dir` the tools calibrate on synthetic 1/f noise and say so on stderr.
Those numbers are indicative only — activation ranges from noise are not the ranges real
photographs produce.

Use `--img-size 70` for fast iteration: the positional embedding is resampled bicubically
exactly as DINOv2 does it, so the model stays valid at any multiple of 14.

## How the numerics work

Symmetric per-tensor int8, int32 accumulate. `real = scale * q`, no zero point. For a GEMM
with input scale `sx` and weight scale `sw` producing output scale `sy`:

- Bias is quantized at `sx * sw` and added into the int32 accumulator.
- The accumulator is rescaled by `M = sx * sw / sy`, expressed as an int32 multiplier
  normalized to `[2³⁰, 2³¹)` plus a shift: `y = sat_int8(round_half_up(acc * mult / 2^shift))`.
  One 32×32→64 multiply and an arithmetic shift — a single DSP slice.

Three things are less obvious and matter:

**LayerNorm needs no input scale.** `(x - mean) / sqrt(var)` is scale-invariant, so the
whole normalization runs in the raw quantized domain using an integer `rsqrt`; only the
`gamma`/`beta` tail carries a scale. This is why `qlayernorm` takes `scale_in` only for the
epsilon term.

**Residual adds keep guard bits.** DA-V2 has 24 residual adds in series. Rounding each
operand to int8 before adding injects half an LSB every time, so `qadd` rescales both
operands retaining 12 fractional bits and rounds once.

**Scale-preserving ops must share a scale.** `reshape`, `transpose`, `slice`, `concat` and
`interpolate` permute or blend values without changing them, so a union-find pass forces one
scale across each such group. Missing this silently rescales the `cls_token` where it is
concatenated with the patch tokens.

**The fused qkv projection is split.** The checkpoint stores one `[3E, E]` projection, and
emitting it fused would force q, k and v onto a single scale — a slice preserves values, so
it preserves scale, and v's much narrower range would be crushed by q's. `frontend/dav2.py`
splits the weight into three matmuls, which is exact and gives each its own scale. The
`head_dim**-0.5` factor is folded into the q weights at the same time.

**Two tensor classes are carried at int16, not int8.** Both were found by the per-tensor
harness, not predicted:

- *Attention logits*, always. At int8 the exponential's argument is granular enough that
  every probability in a row is distorted; the measured SQNR was **negative**. Softmax
  therefore consumes int16 and indexes `exp()` through two cascaded 256-entry ROMs —
  `exp(-s·(256·hi + lo)) = exp(-s·256·hi) · exp(-s·lo)` — two ROMs and one multiply.
- *The transformer residual stream*, under `--residual-int16`. A ViT accumulates a few very
  large channels down the residual path; at int8 they set the scale and ordinary values get
  one or two levels.

No wide tensor ever reaches the systolic array: `32767 × 127 × 576` overflows the int32
accumulator, so `wide_tensors()` only widens tensors whose consumers are all vector ops.

## Known limitations

- **PTQ accuracy, as measured above.** The stack compiles, tiles, costs and emulates DA-V2
  correctly; what it cannot currently do is preserve the model's accuracy at int8. QAT or
  advanced PTQ is the fix, and it lives entirely on the software side.
- **Calibration is percentile-of-batch, not a true global percentile.** Exact would need a
  histogram pass. Documented in `RangeObserver`.
- **No instruction encoding yet.** This stack ends at a costed, verified tile schedule. The
  ISA encoder, assembler, DRAM layout planner and `.bin` emitter are the next layer, and
  none of them can be validated until the RTL exists.
- **Batch size 1, square inputs.** The reference preserves aspect ratio and rounds each side
  to a multiple of 14; this compiles for one fixed square resolution.
- **Per-channel requant costs hardware.** It needs a (multiplier, shift) RAM addressed by
  output column — 5 bytes per column, so 80 bytes for a 16-wide array. The machine
  description now declares `requant.per_channel: true` on the strength of the measurement
  above; if the RTL cannot afford it, set it back to `false` and the compiler will refuse
  `--per-channel` rather than silently emitting something the hardware cannot run.
