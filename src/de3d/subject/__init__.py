"""Subject selection: pick the frame's main object and keep it locked."""

from .scorer import ScoredDetection, SubjectScorer
from .tracker import SubjectTracker, TrackerState

__all__ = [
    "SubjectScorer",
    "ScoredDetection",
    "SubjectTracker",
    "TrackerState",
]
