"""2D detector wrapper around ultralytics YOLO."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import Detection

__all__ = ["Yolo2DDetector"]


class Yolo2DDetector:
    """Thin wrapper that returns :class:`Detection` objects in image pixels.

    Parameters
    ----------
    weights:
        Path to the ultralytics checkpoint (``.pt``).
    device:
        Torch device; ``"cpu"`` by default.
    conf, iou:
        Detection and NMS thresholds.
    imgsz:
        Inference size; ``640`` keeps a good balance on CPU.
    """

    def __init__(
        self,
        weights: str | Path,
        device: str = "cpu",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        classes: list[int] | None = None,
    ) -> None:
        from ultralytics import YOLO  # imported lazily: heavy dependency

        self._model = YOLO(str(weights))
        self._device = device
        self._conf = float(conf)
        self._iou = float(iou)
        self._imgsz = int(imgsz)
        self._classes = classes

    @property
    def names(self) -> dict[int, str]:
        return dict(self._model.names)

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        """Run detection on a BGR frame."""
        results = self._model.predict(
            source=image_bgr,
            device=self._device,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            classes=self._classes,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        names = self.names
        detections: list[Detection] = []
        for xyxy, conf, cls in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy(),
            strict=True,
        ):
            class_id = int(cls)
            detections.append(
                Detection.from_box(
                    xyxy,
                    score=float(conf),
                    class_id=class_id,
                    class_name=str(names.get(class_id, class_id)),
                )
            )
        return detections

    def describe(self) -> dict[str, Any]:
        return {
            "weights": str(self._model.ckpt_path),
            "device": self._device,
            "conf": self._conf,
            "iou": self._iou,
            "imgsz": self._imgsz,
        }
