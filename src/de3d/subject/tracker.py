"""Temporal lock on the frame's main subject.

Once a subject has been acquired it is kept across frames so that the 3D box
does not jump to a different object while the camera pans.  The lock is
released automatically when the target disappears for too long.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..detect.base import Detection, box_iou

__all__ = ["SubjectTracker", "TrackerState"]


@dataclass
class TrackerState:
    """Result of a tracker update for one frame."""

    detection: Detection | None = None
    box: np.ndarray | None = None
    locked: bool = False
    matched: bool = False
    lost_frames: int = 0


@dataclass
class SubjectTracker:
    """Keeps a single subject locked across frames."""

    iou_threshold: float = 0.3
    center_distance_ratio: float = 0.25
    ema: float = 0.6
    max_lost_frames: int = 10

    locked_box: np.ndarray | None = field(default=None, init=False)
    locked_detection: Detection | None = field(default=None, init=False)
    lost_frames: int = field(default=0, init=False)

    # -- lock handling --------------------------------------------------
    @property
    def is_locked(self) -> bool:
        return self.locked_box is not None

    def acquire(self, detection: Detection) -> None:
        """Start tracking ``detection``."""
        self.locked_box = detection.box.copy()
        self.locked_detection = detection
        self.lost_frames = 0

    def release(self) -> None:
        """Drop the current lock."""
        self.locked_box = None
        self.locked_detection = None
        self.lost_frames = 0

    # -- matching --------------------------------------------------------
    def _match(
        self,
        detections: list[Detection],
        image_width: int,
        image_height: int,
    ) -> Detection | None:
        if self.locked_box is None or not detections:
            return None
        locked_cx = 0.5 * (self.locked_box[0] + self.locked_box[2])
        locked_cy = 0.5 * (self.locked_box[1] + self.locked_box[3])
        max_center_distance = self.center_distance_ratio * float(
            np.hypot(image_width, image_height)
        )

        best: Detection | None = None
        best_iou = 0.0
        for det in detections:
            iou = box_iou(self.locked_box, det.box)
            cx, cy = det.center
            distance = float(np.hypot(cx - locked_cx, cy - locked_cy))
            if distance > max_center_distance:
                continue
            if iou >= self.iou_threshold and iou > best_iou:
                best = det
                best_iou = iou
        return best

    # -- main entry point ------------------------------------------------
    def update(
        self,
        detections: list[Detection],
        image_width: int,
        image_height: int,
    ) -> TrackerState:
        """Advance the tracker by one frame."""
        if self.locked_box is None:
            return TrackerState()

        match = self._match(detections, image_width, image_height)
        if match is not None:
            # Smooth the locked box so the 3D box does not jitter.
            alpha = float(np.clip(self.ema, 0.0, 1.0))
            smoothed = alpha * match.box + (1.0 - alpha) * self.locked_box
            self.locked_box = smoothed
            self.locked_detection = match
            self.lost_frames = 0
            return TrackerState(
                detection=match,
                box=self.locked_box.copy(),
                locked=True,
                matched=True,
                lost_frames=0,
            )

        self.lost_frames += 1
        if self.lost_frames > self.max_lost_frames:
            self.release()
            return TrackerState()
        return TrackerState(
            detection=self.locked_detection,
            box=None if self.locked_box is None else self.locked_box.copy(),
            locked=True,
            matched=False,
            lost_frames=self.lost_frames,
        )
