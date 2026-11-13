"""Score every detection and pick the frame's main subject.

The score blends four cheap cues; see ``config/subject.yaml`` for the weights.

* ``area``      -- square root of the bbox area relative to the image.
* ``center``    -- closeness of the bbox centre to the image centre.
* ``confidence``-- the detector's own confidence.
* ``class``     -- a prior weight per category (a person is more likely to be
  the subject of the shot than a distant traffic cone).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..detect.base import Detection

__all__ = ["SubjectScorer", "ScoredDetection"]


@dataclass
class ScoredDetection:
    """A detection together with its subject score."""

    detection: Detection
    score: float

    @property
    def box(self) -> np.ndarray:
        return self.detection.box


@dataclass
class SubjectScorer:
    """Turn a list of detections into a ranked list of subject candidates."""

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "area": 1.0,
            "center": 0.6,
            "confidence": 0.8,
            "class": 0.4,
        }
    )
    min_area_ratio: float = 0.003
    class_weights: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        class_weights: dict[str, float] | None = None,
        preset: str | None = None,
    ) -> SubjectScorer:
        weights = dict(config.get("weights") or {})
        if preset:
            presets = config.get("presets") or {}
            if preset not in presets:
                raise ValueError(
                    f"unknown subject preset {preset!r}; available: {sorted(presets)}"
                )
            weights.update(presets[preset])
        return cls(
            weights=weights,
            min_area_ratio=float(config.get("min_area_ratio", 0.003)),
            class_weights=dict(class_weights or {}),
        )

    # -- scoring --------------------------------------------------------
    def score(
        self,
        detection: Detection,
        image_width: int,
        image_height: int,
    ) -> float:
        """Subject score of a single detection (higher is better)."""
        image_area = float(image_width * image_height)
        area = np.sqrt(detection.area / image_area) if image_area > 0 else 0.0

        diag = float(np.hypot(image_width, image_height))
        cx, cy = detection.center
        offset = float(np.hypot(cx - image_width / 2.0, cy - image_height / 2.0))
        center = 1.0 - min(offset / diag, 1.0) if diag > 0 else 0.0

        confidence = float(detection.score)
        klass = float(self.class_weights.get(detection.class_name, 0.0))

        w = self.weights
        return (
            w.get("area", 0.0) * area
            + w.get("center", 0.0) * center
            + w.get("confidence", 0.0) * confidence
            + w.get("class", 0.0) * klass
        )

    def filter_detections(
        self,
        detections: Iterable[Detection],
        image_width: int,
        image_height: int,
    ) -> list[Detection]:
        """Drop detections that are too small to carry a reliable depth."""
        min_area = self.min_area_ratio * float(image_width * image_height)
        return [d for d in detections if d.area >= min_area]

    def rank(
        self,
        detections: Iterable[Detection],
        image_width: int,
        image_height: int,
    ) -> list[ScoredDetection]:
        """Score and sort the detections, best first."""
        scored = [
            ScoredDetection(detection=d, score=self.score(d, image_width, image_height))
            for d in detections
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored

    def select(
        self,
        detections: Iterable[Detection],
        image_width: int,
        image_height: int,
        max_subjects: int = 1,
    ) -> list[ScoredDetection]:
        """Return the best ``max_subjects`` candidates."""
        if max_subjects <= 0:
            return []
        return self.rank(detections, image_width, image_height)[:max_subjects]
