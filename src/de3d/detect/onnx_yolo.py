"""2D detector running on ONNX Runtime.

Uses the very same ONNX file that is meant for the RKNN conversion, so the
PC runs exactly the graph that will later be executed on the NPU.  It is also
noticeably faster than the PyTorch path on CPU because the pre- and
post-processing are done in numpy instead of PyTorch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import Detection

__all__ = ["OnnxYoloDetector", "letterbox"]

# Ultralytics uses a single padding value for the letterbox.
_PAD_VALUE = 114


def letterbox(
    image: np.ndarray,
    size: int,
    pad_value: int = _PAD_VALUE,
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize keeping the aspect ratio and pad to a square.

    Returns the padded image, the scale factor and the (x, y) padding.
    """
    import cv2

    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_w = int(round(width * scale))
    new_h = int(round(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (size - new_w) / 2.0
    pad_y = (size - new_h) / 2.0
    top = int(round(pad_y - 0.1))
    left = int(round(pad_x - 0.1))
    canvas = np.full((size, size, 3), pad_value, np.uint8)
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, scale, (pad_x, pad_y)


def _non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    """Plain NMS over ``(x1, y1, x2, y2)`` boxes."""
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[index, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[index, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[index, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[index, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_a = (boxes[index, 2] - boxes[index, 0]) * (boxes[index, 3] - boxes[index, 1])
        area_b = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(area_a + area_b - inter, 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


class OnnxYoloDetector:
    """YOLO (ultralytics export format) on ONNX Runtime.

    The exported graph outputs ``(1, 4 + num_classes, num_anchors)`` with the
    boxes in ``(cx, cy, w, h)`` format and sigmoid class scores.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        threads: int = 4,
        class_names: dict[int, str] | None = None,
        classes: list[int] | None = None,
    ) -> None:
        import onnxruntime as ort

        self._path = Path(onnx_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(threads)
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(self._path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._input = self._session.get_inputs()[0]
        self._output = self._session.get_outputs()[0]
        self._conf = float(conf)
        self._iou = float(iou)
        self._imgsz = int(imgsz)
        self._names = class_names or COCO_NAMES
        self._num_classes = len(self._names)
        self._classes = set(classes) if classes else None

    @property
    def names(self) -> dict[int, str]:
        return self._names

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        """Run detection on a BGR frame."""
        import cv2

        height, width = image_bgr.shape[:2]
        canvas, scale, (pad_x, pad_y) = letterbox(image_bgr, self._imgsz)
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = blob.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...].astype(np.float32)

        output = self._session.run([self._output.name], {self._input.name: blob})[0]
        # The exported graph outputs (1, 4 + classes, anchors); transpose it to
        # one row per anchor.
        prediction = np.asarray(output, np.float32)[0].transpose(1, 0)

        class_scores = prediction[:, 4:]
        best_class = class_scores.argmax(axis=1)
        best_score = class_scores[np.arange(class_scores.shape[0]), best_class]
        keep_mask = best_score >= self._conf
        if self._classes is not None:
            keep_mask &= np.isin(best_class, list(self._classes))
        if not np.any(keep_mask):
            return []

        selected = prediction[keep_mask]
        best_class = best_class[keep_mask]
        best_score = best_score[keep_mask]

        cx, cy, bw, bh = selected[:, 0], selected[:, 1], selected[:, 2], selected[:, 3]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)

        keep = _non_max_suppression(boxes, best_score, self._iou)
        detections: list[Detection] = []
        for index in keep:
            x1, y1, x2, y2 = boxes[index]
            # Undo the letterbox and map back to the original image.
            x1 = (x1 - pad_x) / scale
            x2 = (x2 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            y2 = (y2 - pad_y) / scale
            x1 = float(np.clip(x1, 0, width - 1))
            x2 = float(np.clip(x2, 0, width - 1))
            y1 = float(np.clip(y1, 0, height - 1))
            y2 = float(np.clip(y2, 0, height - 1))
            class_id = int(best_class[index])
            detections.append(
                Detection.from_box(
                    np.array([x1, y1, x2, y2]),
                    score=float(best_score[index]),
                    class_id=class_id,
                    class_name=str(self._names.get(class_id, class_id)),
                )
            )
        return detections

    def describe(self) -> dict[str, Any]:
        return {
            "weights": str(self._path),
            "backend": "onnxruntime",
            "conf": self._conf,
            "iou": self._iou,
            "imgsz": self._imgsz,
        }


COCO_NAMES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}
