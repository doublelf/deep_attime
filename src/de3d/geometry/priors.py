"""Class-level 3D priors (size, height, grounded flag, subject weight)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = ["ClassPrior", "PriorTable"]


@dataclass(frozen=True)
class ClassPrior:
    """Prior geometry of one object category."""

    dims: tuple[float, float, float]  # width, height, length (metres)
    height: float
    grounded: bool
    weight: float

    @property
    def width(self) -> float:
        return self.dims[0]

    @property
    def object_height(self) -> float:
        return self.dims[1]

    @property
    def length(self) -> float:
        return self.dims[2]


@dataclass
class PriorTable:
    """Look-up of :class:`ClassPrior` objects by category name."""

    defaults: ClassPrior
    classes: dict[str, ClassPrior]

    @classmethod
    def from_yaml(cls, path: str | Path) -> PriorTable:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle) or {}

        defaults = cls._build(raw.get("default") or {})
        classes = {
            str(name): cls._build(spec or {})
            for name, spec in (raw.get("classes") or {}).items()
        }
        return cls(defaults=defaults, classes=classes)

    @staticmethod
    def _build(spec: dict[str, Any]) -> ClassPrior:
        dims = [float(v) for v in (spec.get("dims") or [0.6, 0.6, 0.6])]
        while len(dims) < 3:
            dims.append(dims[-1])
        return ClassPrior(
            dims=(dims[0], dims[1], dims[2]),
            height=float(spec.get("height", dims[1])),
            grounded=bool(spec.get("grounded", False)),
            weight=float(spec.get("weight", 0.1)),
        )

    def get(self, class_name: str) -> ClassPrior:
        """Prior of ``class_name`` (falls back to the default entry)."""
        return self.classes.get(class_name, self.defaults)

    @property
    def class_weights(self) -> dict[str, float]:
        """Subject-scoring weight per category."""
        return {name: prior.weight for name, prior in self.classes.items()}
