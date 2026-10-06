# ABOUTME: Checks that feeding the fp32 graph its own taps through `overrides` changes nothing,
# ABOUTME: which is what makes a depth map finished from board taps comparable to the reference.

import numpy as np

from sw.execute import run_float
from sw.ir import GraphBuilder


def test_override_replaces_an_intermediate_and_skips_its_producer():
    b = GraphBuilder("t")
    x = b.input("x")
    h = b.emit("relu", [x], "h")
    y = b.emit("relu", [h], "y")
    b.output(y)
    g = b.build()
    inp = {"x": np.array([-1.0, 2.0], dtype=np.float32)}
    assert np.array_equal(run_float(g, inp)[y], [0.0, 2.0])
    out = run_float(g, inp, overrides={h: np.array([5.0, -3.0])})[y]
    assert np.array_equal(out, [5.0, 0.0])
