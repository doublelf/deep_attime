"""Compare the ONNX Runtime detector with the ultralytics (PyTorch) one.

Usage::

    python scripts/compare_detectors.py --image <frame.png>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from de3d.detect.onnx_yolo import OnnxYoloDetector  # noqa: E402
from de3d.detect.yolo2d import Yolo2DDetector  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=str(REPO_ROOT / "outputs" / "test3" / "frame12.png"))
    parser.add_argument("--weights", default=str(REPO_ROOT / "models" / "yolo11n.pt"))
    parser.add_argument("--onnx", default=str(REPO_ROOT / "models" / "onnx" / "yolo11n.onnx"))
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args(argv)

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"[error] cannot read {args.image}", file=sys.stderr)
        return 1

    torch_detector = Yolo2DDetector(args.weights, device="cpu", conf=args.conf)
    start = time.perf_counter()
    torch_boxes = torch_detector.detect(frame)
    torch_time = time.perf_counter() - start

    onnx_detector = OnnxYoloDetector(args.onnx, conf=args.conf)
    start = time.perf_counter()
    onnx_boxes = onnx_detector.detect(frame)
    onnx_time = time.perf_counter() - start

    print(f"torch : {len(torch_boxes):3d} detections in {torch_time*1000:7.1f} ms")
    for det in sorted(torch_boxes, key=lambda d: -d.score)[:8]:
        print(
            f"        {det.class_name:14s} {det.score:.3f} "
            f"[{det.x1:7.1f} {det.y1:7.1f} {det.x2:7.1f} {det.y2:7.1f}]"
        )
    print(f"onnx  : {len(onnx_boxes):3d} detections in {onnx_time*1000:7.1f} ms")
    for det in sorted(onnx_boxes, key=lambda d: -d.score)[:8]:
        print(
            f"        {det.class_name:14s} {det.score:.3f} "
            f"[{det.x1:7.1f} {det.y1:7.1f} {det.x2:7.1f} {det.y2:7.1f}]"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
