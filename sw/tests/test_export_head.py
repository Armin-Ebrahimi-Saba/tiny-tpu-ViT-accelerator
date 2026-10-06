# ABOUTME: The head lowering must compute the reference graph's head: its float path is checked
# ABOUTME: against run_float, and the int pieces (im2col, tap split, upsampling) against plain numpy.

import numpy as np
import pytest

from sw.export_head import T, Head, build_head, conv_weight, im2col
from sw.export_tpu import BUFFERS, Blob, qscale, quant


@pytest.fixture(scope="module")
def state_dict():
    from sw.frontend import load_state_dict
    try:
        return load_state_dict()
    except FileNotFoundError as exc:
        pytest.skip(f"DA-V2 checkpoint unavailable: {exc}")


def test_im2col_times_weight_is_the_convolution():
    """Against a direct loop: (ky, kx, c) order, zero padding, stride."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((7, 6, 16))
    w = rng.standard_normal((5, 16, 3, 3))
    a, ho, wo = im2col(x, 3, 2, 1)
    y = (a @ conv_weight(w)).reshape(ho, wo, 5)
    xp = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    for oy in range(ho):
        for ox in range(wo):
            patch = xp[2 * oy:2 * oy + 3, 2 * ox:2 * ox + 3]          # [3][3][C]
            assert np.allclose(y[oy, ox], np.einsum("yxc,oc yx->o".replace(" ", ""), patch, w))
    # tap ranges are column slices of the whole
    a2, _, _ = im2col(x, 3, 2, 1, 4, 9)
    assert np.array_equal(a2, a[:, 4 * 16:])


def test_head_float_path_is_the_reference_head(state_dict):
    from sw.execute import run_float
    from sw.frontend.dav2 import DAV2Config, build_dav2_graph
    size = 56
    g = build_dav2_graph(state_dict, DAV2Config(img_size=size))
    img = np.random.default_rng(1).standard_normal((1, 3, size, size)).astype(np.float32)
    names = [f"tap{j}/norm" for j in range(4)]
    vals = run_float(g, {"image": img}, keep=names)
    blob = Blob()
    taps = []
    for n in names:
        f = vals[n][0].astype(np.float64)
        taps.append({"q": quant(f, qscale(f)), "ref": blob.scratch(f.size), "s": qscale(f), "f": f})
    d = build_head(blob, state_dict, taps, size // 14, size, check=False)
    ref = vals[g.outputs[0]].squeeze()
    assert np.abs(d["f"][..., 0] - ref).max() < 1e-4 * max(1.0, np.abs(ref).max())


def test_a_conv_too_deep_for_the_weight_buffer_is_split_and_summed(state_dict):
    """resize_layers.3 is a 3x3 over 384 channels: K = 3456 > the weight buffer."""
    assert 9 * 384 > BUFFERS.wgt_words
    blob = Blob()
    rng = np.random.default_rng(2)
    f = rng.standard_normal((3, 3, 384))
    x = T(quant(f, qscale(f)), blob.scratch(f.size), qscale(f), f)
    y = Head(blob, state_dict, check=False).conv("t", x, 3, stride=2, pad=1,
                                                 prefix="resize_layers.3")
    assert y.shape == (2, 2, 384)
    gemms = [d for d in blob.descs if d.op == 0]
    assert len({(d.g_k >> 20) & 0x3F for d in gemms}) == 2          # two tap groups
    deq = y.q.astype(float) * y.s
    assert np.corrcoef(deq.ravel(), y.f.ravel())[0, 1] > 0.99
