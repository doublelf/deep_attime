"""Detection container shared by the detector and the subject selector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Detection", "box_iou"]


@dataclass
class Detection:
    """A single 2D detection in *original* image coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    class_name: str

    @property
    def width(self) -> float:
        return float(self.x2 - self.x1)

    @property
    def height(self) -> float:
        return float(self.y2 - self.y1)

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)

    @property
    def center(self) -> tuple[float, float]:
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))

    @property
    def box(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], np.float64)

    @classmethod
    def from_box(
        cls,
        box: np.ndarray,
        score: float,
        class_id: int,
        class_name: str,
    ) -> Detection:
        return cls(
            x1=float(box[0]),
            y1=float(box[1]),
            x2=float(box[2]),
            y2=float(box[3]),
            score=float(score),
            class_id=int(class_id),
            class_name=class_name,
        )


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(ix2 - ix1, 0.0)
    ih = max(iy2 - iy1, 0.0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0
