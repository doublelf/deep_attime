"""Subject scoring and temporal locking."""

from __future__ import annotations

import numpy as np

from de3d.detect.base import Detection
from de3d.subject.scorer import SubjectScorer
from de3d.subject.tracker import SubjectTracker


def _det(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    score: float = 0.9,
    name: str = "person",
    class_id: int = 0,
) -> Detection:
    return Detection(x1, y1, x2, y2, score, class_id, name)


def test_scorer_prefers_large_central_confident_object() -> None:
    scorer = SubjectScorer(weights={"area": 1.0, "center": 0.6, "confidence": 0.8, "class": 0.4})
    big = _det(400, 300, 900, 900)
    small = _det(10, 10, 80, 80)
    off_center = _det(0, 0, 200, 200)
    best = scorer.select([small, off_center, big], 1000, 1000, 1)[0]
    assert best.detection is big


def test_scorer_filters_tiny_detections() -> None:
    scorer = SubjectScorer(min_area_ratio=0.01)
    tiny = _det(0, 0, 50, 50)
    assert scorer.filter_detections([tiny], 1000, 1000) == []


def test_scorer_uses_class_weight() -> None:
    scorer = SubjectScorer(
        weights={"area": 1.0, "center": 0.0, "confidence": 0.0, "class": 1.0},
        class_weights={"person": 1.0, "traffic cone": 0.1},
    )
    person = _det(100, 100, 200, 200, name="person")
    cone = _det(100, 100, 200, 200, name="traffic cone")
    assert scorer.score(person, 1000, 1000) > scorer.score(cone, 1000, 1000)


def test_tracker_locks_and_smooths() -> None:
    tracker = SubjectTracker(iou_threshold=0.3, ema=0.5, max_lost_frames=2)
    tracker.acquire(_det(100, 100, 200, 300))
    state = tracker.update([_det(102, 102, 202, 302)], 1000, 1000)
    assert state.matched and state.locked
    assert state.box is not None
    # EMA with alpha 0.5 halves the shift of the 2 px move.
    assert state.box[0] == 101.0


def test_tracker_releases_after_max_lost_frames() -> None:
    tracker = SubjectTracker(iou_threshold=0.3, max_lost_frames=2)
    tracker.acquire(_det(100, 100, 200, 300))
    assert tracker.update([], 1000, 1000).locked
    assert tracker.update([], 1000, 1000).locked
    state = tracker.update([], 1000, 1000)
    assert not state.locked
    assert not tracker.is_locked


def test_tracker_ignores_distant_detections() -> None:
    tracker = SubjectTracker(iou_threshold=0.3, center_distance_ratio=0.05, max_lost_frames=5)
    tracker.acquire(_det(100, 100, 200, 300))
    state = tracker.update([_det(700, 700, 800, 900)], 1000, 1000)
    assert not state.matched
    assert state.locked


def test_tracker_box_iou_helper() -> None:
    from de3d.detect.base import box_iou

    a = np.array([0.0, 0.0, 10.0, 10.0])
    assert box_iou(a, a) == 1.0
    assert box_iou(a, np.array([20.0, 20.0, 30.0, 30.0])) == 0.0
