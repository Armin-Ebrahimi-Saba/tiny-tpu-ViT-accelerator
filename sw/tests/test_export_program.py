# ABOUTME: A fixed-scale encoder program must not depend on the picture it was traced with,
# ABOUTME: and its emulator must agree with the image-specific lowering wherever the scales agree.

import numpy as np
import pytest

import sw.export_tpu as ex
from sw.export_tpu import Blob, Calibration, build_encoder, encoder_im2col, encoder_input

SIZE = 28                        # 2x2 patches, 5 tokens: the whole encoder in seconds


@pytest.fixture(scope="module")
def state_dict():
    from sw.frontend import load_state_dict
    try:
        return load_state_dict()
    except FileNotFoundError as exc:
        pytest.skip(f"DA-V2 checkpoint unavailable: {exc}")


def image(seed):
    return np.random.default_rng(seed).standard_normal((1, 3, SIZE, SIZE)).astype(np.float32)


def test_im2col_rows_are_patches_with_an_empty_cls_row():
    img = image(0)
    a = encoder_im2col(img)
    assert a.shape == (5, 592)
    assert not a[0].any() and not a[:, 588:].any()
    assert np.array_equal(a[2, :588], img[0, :, 0:14, 14:28].reshape(-1))


def recorded(state_dict, seeds):
    saved, ex.CALIB = ex.CALIB, Calibration()
    try:
        for s in seeds:
            build_encoder(Blob(), state_dict, image(s), check=False)
        return ex.CALIB.scales()
    finally:
        ex.CALIB = saved


def program(state_dict, scales, seed, tmp_path):
    saved, ex.CALIB = ex.CALIB, Calibration(fixed=scales)
    try:
        blob = Blob()
        side = build_encoder(blob, state_dict, image(seed), check=False, input_in_arena=True)
        path = tmp_path / f"p{seed}.bin"
        blob.write(path)
        return path.read_bytes(), side
    finally:
        ex.CALIB = saved


def test_program_is_the_same_whatever_picture_traced_it(state_dict, tmp_path):
    scales = recorded(state_dict, [1, 2])
    a, side_a = program(state_dict, scales, 3, tmp_path)
    b, _ = program(state_dict, scales, 4, tmp_path)
    assert a == b
    # ...and the input it wants is exactly encoder_input() at the recorded scale.
    assert side_a["input"]["scale"] == scales["embed.in"]
    assert np.array_equal(side_a["input"]["emu"], encoder_input(image(3), scales["embed.in"]))


def test_fixed_scales_from_one_image_reproduce_that_image_exactly(state_dict):
    """Calibrating on a single picture and running that picture is the
    image-specific lowering: every tap byte must match."""
    scales = recorded(state_dict, [5])
    dyn = build_encoder(Blob(), state_dict, image(5), check=False)
    fixed = ex.run_encoder_emulator(state_dict, image(5), scales)
    for t in dyn["taps"]:
        assert np.array_equal(t["emu"], fixed[t["name"]])
