# ABOUTME: Loads and validates the machine description that the compiler and emulator target.
# ABOUTME: The description is the contract between this toolchain and the RTL that must implement it.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MACHINES_DIR = Path(__file__).parent / "machines"


class UnsupportedOpError(NotImplementedError):
    """Raised when a graph contains an op the target machine does not declare.

    This is deliberately fatal. A compiler that silently approximates an
    unsupported op produces a model that looks like it works and is wrong.
    """


@dataclass(frozen=True)
class MachineSpec:
    name: str
    description: str
    clock_mhz: float
    array: dict[str, Any]
    datapath: dict[str, Any]
    requant: dict[str, Any]
    unified_buffer: dict[str, Any]
    dram: dict[str, Any]
    vector_unit: dict[str, Any]
    supported_ops: dict[str, dict[str, Any]] = field(default_factory=dict)

    # --- geometry -------------------------------------------------------

    @property
    def rows(self) -> int:
        return int(self.array["rows"])

    @property
    def cols(self) -> int:
        return int(self.array["cols"])

    @property
    def macs_per_cycle(self) -> int:
        return self.rows * self.cols

    @property
    def peak_macs_per_second(self) -> float:
        return self.macs_per_cycle * self.clock_mhz * 1e6

    # --- datapath limits ------------------------------------------------

    @property
    def act_min(self) -> int:
        return int(self.datapath["act_min"])

    @property
    def act_max(self) -> int:
        return int(self.datapath["act_max"])

    @property
    def accum_min(self) -> int:
        return -(1 << (int(self.datapath["accum_bits"]) - 1))

    @property
    def accum_max(self) -> int:
        return (1 << (int(self.datapath["accum_bits"]) - 1)) - 1

    # --- op support -----------------------------------------------------

    def supports(self, op_type: str) -> bool:
        return op_type in self.supported_ops

    def require(self, op_type: str) -> dict[str, Any]:
        """Return the op's constraints, or raise if the machine cannot run it."""
        if op_type not in self.supported_ops:
            known = ", ".join(sorted(self.supported_ops))
            raise UnsupportedOpError(
                f"machine {self.name!r} does not implement op {op_type!r}; "
                f"declared ops are: {known}"
            )
        return self.supported_ops[op_type]

    def check_attr(self, op_type: str, key: str, value: Any) -> None:
        """Validate one op attribute against the declared constraint.

        A constraint that is a list is a whitelist of allowed values; a
        constraint named ``max_*`` is an inclusive upper bound. Anything the
        machine file does not mention is unconstrained.
        """
        constraints = self.require(op_type)
        if key in constraints:
            allowed = constraints[key]
            if isinstance(allowed, list) and value not in allowed:
                raise UnsupportedOpError(
                    f"{op_type}.{key}={value!r} unsupported on {self.name}; allowed: {allowed}"
                )
            if isinstance(allowed, bool) and not allowed and value:
                raise UnsupportedOpError(f"{op_type}.{key} not supported on {self.name}")
        bound_key = f"max_{key}"
        if bound_key in constraints and value > constraints[bound_key]:
            raise UnsupportedOpError(
                f"{op_type}.{key}={value} exceeds {self.name} limit {constraints[bound_key]}"
            )

    # --- loading --------------------------------------------------------

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MachineSpec:
        missing = {
            "name", "description", "clock_mhz", "array", "datapath", "requant",
            "unified_buffer", "dram", "vector_unit", "supported_ops",
        } - set(raw)
        if missing:
            raise ValueError(f"machine description missing keys: {sorted(missing)}")
        spec = cls(
            name=raw["name"],
            description=raw["description"],
            clock_mhz=float(raw["clock_mhz"]),
            array=raw["array"],
            datapath=raw["datapath"],
            requant=raw["requant"],
            unified_buffer=raw["unified_buffer"],
            dram=raw["dram"],
            vector_unit=raw["vector_unit"],
            supported_ops=raw["supported_ops"],
        )
        spec.validate()
        return spec

    @classmethod
    def load(cls, name_or_path: str = "tpu-v2") -> MachineSpec:
        path = Path(name_or_path)
        if not path.exists():
            path = MACHINES_DIR / f"{name_or_path.replace('-', '_')}.json"
        if not path.exists():
            available = sorted(p.stem for p in MACHINES_DIR.glob("*.json"))
            raise FileNotFoundError(f"no machine {name_or_path!r}; available: {available}")
        return cls.from_dict(json.loads(path.read_text()))

    def validate(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("array dimensions must be positive")
        if int(self.datapath["accum_bits"]) < 2 * int(self.datapath["act_bits"]):
            raise ValueError("accumulator narrower than a single product")
        lo, hi = self.requant["multiplier_range"]
        if not 0 < lo < hi < (1 << 31):
            raise ValueError("requant multiplier_range must lie inside int32")
        lo_shift = int(self.requant["shift_min"])
        hi_shift = int(self.requant["shift_max"])
        if not 0 <= lo_shift < hi_shift:
            raise ValueError("requant shift range must satisfy 0 <= shift_min < shift_max")

    def summary(self) -> str:
        gmacs = self.peak_macs_per_second / 1e9
        return (
            f"{self.name}: {self.rows}x{self.cols} {self.array['dataflow']} array, "
            f"int{self.datapath['act_bits']} x int{self.datapath['weight_bits']} "
            f"-> int{self.datapath['accum_bits']}, {self.clock_mhz:g} MHz "
            f"({gmacs:.2f} GMAC/s peak), {self.unified_buffer['bytes'] // 1024} KB UB, "
            f"{self.dram['bytes'] // (1 << 20)} MB DRAM"
        )
