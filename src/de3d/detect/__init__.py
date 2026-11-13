"""2D detection."""

from .base import Detection, box_iou
from .onnx_yolo import OnnxYoloDetector
from .yolo2d import Yolo2DDetector

__all__ = ["Detection", "box_iou", "Yolo2DDetector", "OnnxYoloDetector"]
