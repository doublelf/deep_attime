"""Probe the detector's per-frame latency on individual video frames.

Usage::

    python scripts/probe_detector.py --frames 10,31,35,45
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        default=r"D:\Desktop\chat\opencode\project\poseball_RKNN\ceshi\data\inputs\sample-a.mp4",
    )
    parser.add_argument("--frames", default="10,31,35,45")
    parser.add_argument("--onnx", default=str(REPO_ROOT / "models" / "onnx" / "yolo11n.onnx"))
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args(argv)

    detector = OnnxYoloDetector(args.onnx, threads=4)
    for frame_index in [int(v) for v in args.frames.split(",")]:
        capture = cv2.VideoCapture(args.video)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            print(f"frame {frame_index}: cannot read")
            continue
        detector.detect(frame)
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            detector.detect(frame)
            samples.append((time.perf_counter() - start) * 1000)
        samples.sort()
        print(
            f"frame {frame_index:4d}: min {samples[0]:6.0f} ms  median {samples[len(samples)//2]:6.0f} ms"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
