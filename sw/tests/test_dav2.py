# ABOUTME: Validates the DA-V2 frontend: graph structure, and equivalence with the official implementation.
# ABOUTME: The equivalence test is what makes the fp32 path a golden reference rather than a guess.

from __future__ import annotations

import numpy as np
import pytest

from sw.execute import run_float, run_quant
from sw.frontend import DAV2Config, build_dav2_graph, load_state_dict
from sw.imageio import calibration_batches
from sw.lower import lower_graph, require_supported
from sw.machine import MachineSpec
from sw.numerics import cosine_similarity, sqnr_db
from sw.quantize import quantize_graph

CHECKPOINT_PARAMS = 24_785_089


@pytest.fixture(scope="module")
def state_dict() -> dict[str, np.ndarray]:
    try:
        return load_state_dict()
    except FileNotFoundError as exc:
        pytest.skip(f"DA-V2 checkpoint unavailable: {exc}")


@pytest.fixture(scope="module")
def machine() -> MachineSpec:
    return MachineSpec.load("tpu-v2")


@pytest.fixture(scope="module")
def small_graph(state_dict):
    return build_dav2_graph(state_dict, DAV2Config(img_size=70))


def test_checkpoint_has_the_expected_parameter_count(state_dict) -> None:
    assert sum(v.size for v in state_dict.values()) == CHECKPOINT_PARAMS


def test_graph_structure(small_graph) -> None:
    counts = small_graph.op_types()
    assert counts["softmax"] == 12          # one per transformer block
    assert counts["gelu"] == 12
    assert counts["layernorm"] == 12 * 2 + 4  # two per block plus the four taps
    assert counts["batch_matmul"] == 24     # qk and av per block
    assert counts["conv_transpose2d"] == 2  # DPT resize levels 0 and 1
    assert small_graph.outputs == ["depth"]


def test_qkv_is_split_so_q_k_and_v_get_independent_scales(small_graph) -> None:
    """A fused qkv matmul plus slices would force one scale across all three."""
    from sw.quantize import unify_scale_groups

    groups = unify_scale_groups(small_graph)
    q = groups["blk0/q_heads"]
    k = groups["blk0/k_heads"]
    v = groups["blk0/v_heads"]
    assert len({q, k, v}) == 3


def test_attention_logits_are_carried_at_int16(small_graph, machine) -> None:
    from sw.quantize import WIDE_LEVELS, wide_tensors

    wide = wide_tensors(small_graph)
    assert len(wide) == 12                       # one per block
    assert all("logits" in name for name in wide)

    qp = quantize_graph(small_graph, calibration_batches(70, count=1), machine=machine)
    for name in wide:
        assert qp.rails[name] == WIDE_LEVELS
    producer = {t: op for op in small_graph.ops for t in op.outputs}
    for name in wide:
        rq = qp.op_params[producer[name].name].requant
        assert (rq.out_min, rq.out_max) == (-WIDE_LEVELS, WIDE_LEVELS)


def test_softmax_exp_lut_is_two_staged(small_graph, machine) -> None:
    qp = quantize_graph(small_graph, calibration_batches(70, count=1), machine=machine)
    softmax_ops = [op for op in small_graph.ops if op.type == "softmax"]
    lo, hi = qp.op_params[softmax_ops[0].name].exp_lut
    assert lo.shape == hi.shape == (256,)
    assert lo[0] == hi[0]           # exp(0) in both stages
    assert lo[1] > hi[1]            # the high stage decays 256x faster


def test_every_op_is_supported_by_the_machine(small_graph, machine) -> None:
    require_supported(small_graph, machine)


def test_graph_is_single_assignment(small_graph) -> None:
    small_graph.validate()


def test_odd_image_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="multiple of patch"):
        DAV2Config(img_size=100).patch_grid


def test_pos_embed_is_resampled_off_the_native_grid(state_dict) -> None:
    native = build_dav2_graph(state_dict, DAV2Config(img_size=518))
    small = build_dav2_graph(state_dict, DAV2Config(img_size=70))
    assert native.initializers["pos_embed"].shape[1] == 37 * 37 + 1
    assert small.initializers["pos_embed"].shape[1] == 5 * 5 + 1


def test_schedule_is_costed_and_compute_bound(small_graph, machine) -> None:
    sched = lower_graph(small_graph, machine, {"image": (1, 3, 70, 70)})
    assert sched.total_macs > 0
    assert sched.seconds > 0
    assert 0.3 < sched.utilization <= 1.0


def test_int8_emulation_tracks_fp32(small_graph, machine) -> None:
    qp = quantize_graph(small_graph, calibration_batches(70, count=3), machine=machine)
    sample = next(iter(calibration_batches(70, count=1, seed=42)))
    ref = run_float(small_graph, sample)["depth"]
    got = run_quant(small_graph, qp, sample, machine=machine)["depth"]
    assert got.shape == ref.shape
    assert sqnr_db(ref, got) > 12.0
    assert cosine_similarity(ref, got) > 0.99


def test_weights_fit_in_dram(small_graph, machine) -> None:
    qp = quantize_graph(small_graph, calibration_batches(70, count=1), machine=machine)
    assert qp.weight_bytes() < machine.dram["bytes"]
    assert qp.weight_bytes() > 20_000_000  # int8, so roughly one byte per parameter


@pytest.mark.slow
def test_matches_official_implementation(state_dict) -> None:
    """Compare the fp32 path against HuggingFace's independent DA-V2 at native resolution."""
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")

    model = transformers.AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf")
    model.eval()
    x = (np.random.default_rng(0).standard_normal((1, 3, 518, 518)) * 0.5).astype(np.float32)
    with torch.no_grad():
        reference = model(pixel_values=torch.from_numpy(x)).predicted_depth.numpy()[0]

    graph = build_dav2_graph(state_dict, DAV2Config(img_size=518))
    mine = run_float(graph, {"image": x})["depth"][0, 0]

    assert sqnr_db(reference, mine) > 100.0
    assert cosine_similarity(reference, mine) > 0.999999
