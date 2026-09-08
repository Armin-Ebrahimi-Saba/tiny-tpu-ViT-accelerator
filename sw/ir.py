# ABOUTME: Minimal single-assignment graph IR that both the float reference and the integer emulator execute.
# ABOUTME: One graph, two backends, so any fp32-vs-int8 difference is attributable to a specific op.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


@dataclass
class Op:
    """A single node. ``inputs``/``outputs`` are tensor names, not values."""

    type: str
    name: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)

    def attr(self, key: str, default: Any = ...) -> Any:
        if key in self.attrs:
            return self.attrs[key]
        if default is ...:
            raise KeyError(f"op {self.name!r} ({self.type}) has no attribute {key!r}")
        return default

    def __repr__(self) -> str:
        return f"Op({self.type}, {self.name}, in={self.inputs}, out={self.outputs})"


@dataclass
class Graph:
    name: str
    ops: list[Op] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    # Constant tensors (weights, biases, positional embeddings) by name.
    initializers: dict[str, np.ndarray] = field(default_factory=dict)

    def producer(self, tensor: str) -> Op | None:
        for op in self.ops:
            if tensor in op.outputs:
                return op
        return None

    def consumers(self, tensor: str) -> list[Op]:
        return [op for op in self.ops if tensor in op.inputs]

    def op_types(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for op in self.ops:
            counts[op.type] = counts.get(op.type, 0) + 1
        return dict(sorted(counts.items()))

    def parameter_count(self) -> int:
        return int(sum(v.size for v in self.initializers.values()))

    def validate(self) -> None:
        """Check single assignment and that every read is dominated by a write."""
        defined: set[str] = set(self.inputs) | set(self.initializers)
        seen_ops: set[str] = set()
        for op in self.ops:
            if op.name in seen_ops:
                raise ValueError(f"duplicate op name {op.name!r}")
            seen_ops.add(op.name)
            for t in op.inputs:
                if t not in defined:
                    raise ValueError(
                        f"op {op.name!r} ({op.type}) reads undefined tensor {t!r}; "
                        "ops must be listed in topological order"
                    )
            for t in op.outputs:
                if t in defined:
                    raise ValueError(f"tensor {t!r} assigned more than once")
                defined.add(t)
        for t in self.outputs:
            if t not in defined:
                raise ValueError(f"graph output {t!r} is never produced")

    def summary(self) -> str:
        params = self.parameter_count()
        types = ", ".join(f"{k}x{v}" for k, v in self.op_types().items())
        return (
            f"graph {self.name!r}: {len(self.ops)} ops, {params:,} constant params "
            f"({params / 1e6:.2f} M)\n  {types}"
        )


class GraphBuilder:
    """Convenience wrapper that names tensors and appends ops in order."""

    def __init__(self, name: str) -> None:
        self.graph = Graph(name=name)
        self._counter: dict[str, int] = {}

    # -- naming ----------------------------------------------------------

    def _fresh(self, stem: str) -> str:
        n = self._counter.get(stem, 0)
        self._counter[stem] = n + 1
        return stem if n == 0 else f"{stem}_{n}"

    # -- declarations ----------------------------------------------------

    def input(self, name: str) -> str:
        self.graph.inputs.append(name)
        return name

    def constant(self, name: str, value: np.ndarray) -> str:
        name = self._fresh(name)
        self.graph.initializers[name] = np.asarray(value, dtype=np.float32)
        return name

    def output(self, tensor: str) -> str:
        self.graph.outputs.append(tensor)
        return tensor

    # -- op emission -----------------------------------------------------

    def emit(self, op_type: str, inputs: Iterable[str], out_stem: str, **attrs: Any) -> str:
        inputs = list(inputs)
        name = self._fresh(f"{out_stem}/{op_type}")
        out = self._fresh(out_stem)
        self.graph.ops.append(Op(type=op_type, name=name, inputs=inputs, outputs=[out], attrs=attrs))
        return out

    def build(self) -> Graph:
        self.graph.validate()
        return self.graph
