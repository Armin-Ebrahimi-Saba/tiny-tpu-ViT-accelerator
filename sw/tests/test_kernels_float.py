# ABOUTME: Validates every numpy fp32 reference kernel against torch's independent implementation.
# ABOUTME: Without this the "golden" model would only be self-consistent, not correct.

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from sw import kernels_float as K  # noqa: E402

RNG = np.random.default_rng(0)


def t(x: np.ndarray) -> "torch.Tensor":
    return torch.from_numpy(np.ascontiguousarray(x.astype(np.float32)))


def assert_close(mine: np.ndarray, ref: "torch.Tensor", atol: float = 1e-4, rtol: float = 1e-4) -> None:
    np.testing.assert_allclose(mine, ref.detach().numpy(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("stride,pad,k", [(1, 0, 1), (1, 1, 3), (2, 1, 3), (4, 0, 4)])
def test_conv2d(stride: int, pad: int, k: int) -> None:
    x = RNG.standard_normal((2, 5, 11, 13)).astype(np.float32)
    w = RNG.standard_normal((7, 5, k, k)).astype(np.float32)
    b = RNG.standard_normal(7).astype(np.float32)
    assert_close(K.conv2d(x, w, b, stride, pad), F.conv2d(t(x), t(w), t(b), stride, pad))


def test_conv2d_no_bias() -> None:
    x = RNG.standard_normal((1, 3, 8, 8)).astype(np.float32)
    w = RNG.standard_normal((4, 3, 3, 3)).astype(np.float32)
    assert_close(K.conv2d(x, w, None, 1, 1), F.conv2d(t(x), t(w), None, 1, 1))


@pytest.mark.parametrize("stride,pad,k", [(4, 0, 4), (2, 0, 2), (1, 0, 3), (2, 1, 3)])
def test_conv_transpose2d(stride: int, pad: int, k: int) -> None:
    x = RNG.standard_normal((2, 6, 5, 7)).astype(np.float32)
    w = RNG.standard_normal((6, 4, k, k)).astype(np.float32)
    b = RNG.standard_normal(4).astype(np.float32)
    assert_close(
        K.conv_transpose2d(x, w, b, stride, pad),
        F.conv_transpose2d(t(x), t(w), t(b), stride, pad),
    )


def test_linear() -> None:
    x = RNG.standard_normal((2, 9, 16)).astype(np.float32)
    w = RNG.standard_normal((32, 16)).astype(np.float32)
    b = RNG.standard_normal(32).astype(np.float32)
    assert_close(K.linear(x, w, b), F.linear(t(x), t(w), t(b)))


def test_layernorm() -> None:
    x = RNG.standard_normal((2, 9, 16)).astype(np.float32)
    g = RNG.standard_normal(16).astype(np.float32)
    b = RNG.standard_normal(16).astype(np.float32)
    assert_close(K.layernorm(x, g, b, 1e-6), F.layer_norm(t(x), (16,), t(g), t(b), 1e-6))


def test_softmax() -> None:
    x = (RNG.standard_normal((2, 6, 40, 40)) * 8).astype(np.float32)
    assert_close(K.softmax(x, -1), F.softmax(t(x), dim=-1), atol=1e-6)


def test_gelu_matches_exact_erf_gelu() -> None:
    x = np.linspace(-8, 8, 20001).astype(np.float32)
    assert_close(K.gelu(x), F.gelu(t(x), approximate="none"), atol=1e-6, rtol=1e-5)


def test_relu() -> None:
    x = RNG.standard_normal(1000).astype(np.float32)
    assert_close(K.relu(x), F.relu(t(x)))


@pytest.mark.parametrize("align", [True, False])
@pytest.mark.parametrize("size", [(10, 10), (37, 41), (7, 3)])
def test_interpolate_bilinear(align: bool, size: tuple[int, int]) -> None:
    x = RNG.standard_normal((2, 3, 5, 6)).astype(np.float32)
    assert_close(
        K.interpolate_bilinear(x, size, align),
        F.interpolate(t(x), size=size, mode="bilinear", align_corners=align),
        atol=1e-5,
    )


@pytest.mark.parametrize("align", [False, True])
@pytest.mark.parametrize("size", [(9, 9), (37, 37), (3, 5)])
def test_interpolate_bicubic(align: bool, size: tuple[int, int]) -> None:
    x = RNG.standard_normal((1, 4, 7, 7)).astype(np.float32)
    assert_close(
        K.interpolate_bicubic(x, size, align),
        F.interpolate(t(x), size=size, mode="bicubic", align_corners=align),
        atol=1e-4,
    )


def test_im2col_reconstructs_conv() -> None:
    """The im2col layout is what the array actually consumes, so it must be exact."""
    x = RNG.standard_normal((1, 3, 6, 6)).astype(np.float32)
    w = RNG.standard_normal((5, 3, 3, 3)).astype(np.float32)
    cols = K.im2col(x, 3, 3, 1, 1)
    manual = (w.reshape(5, -1) @ cols[0]).reshape(1, 5, 6, 6)
    assert_close(manual, F.conv2d(t(x), t(w), None, 1, 1))
