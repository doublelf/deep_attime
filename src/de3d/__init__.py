"""Monocular 3D cuboid extraction.

Extracts one metric 3D cuboid (eight corners, camera frame) per frame from a
monocular RGB video and draws it on the frame.

Typical use::

    python -m de3d.cli --input video.mp4 --camera config/cameras/sample_1920x1080.yaml
"""

from __future__ import annotations

__version__ = "0.1.0"

from .camera import CameraModel, EquidistantFisheyeCamera, PinholeCamera
from .detect.base import Detection
from .geometry.box3d import Box3D
from .geometry.priors import ClassPrior, PriorTable
from .pipeline import FrameResult, Pipeline, PipelineConfig
from .subject.scorer import SubjectScorer
from .subject.tracker import SubjectTracker

__all__ = [
    "__version__",
    "CameraModel",
    "PinholeCamera",
    "EquidistantFisheyeCamera",
    "Detection",
    "Box3D",
    "ClassPrior",
    "PriorTable",
    "Pipeline",
    "PipelineConfig",
    "FrameResult",
    "SubjectScorer",
    "SubjectTracker",
]
