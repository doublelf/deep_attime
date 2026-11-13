"""Benchmark the Depth-Anything ONNX model on CPU.

The exported graph interpolates its position embeddings, so the input size is
free; this script finds the fastest useful configuration.

Usage::

    python scripts/benchmark_depth.py --image <path>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_image(path: str | None) -> np.ndarray:
    if path and Path(path).exists():
        image = cv2.imread(str(path))
        if image is not None:
            return image
    return (np.random.default_rng(0).random((1080, 1920, 3)) * 255).astype(np.uint8)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(REPO_ROOT / "models" / "depth_anything_v2_small.onnx"))
    parser.add_argument("--image", default=None)
    parser.add_argument("--threads", default="4,8,12,16")
    parser.add_argument("--long-sides", default="518,392,308,252")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)

    import onnxruntime as ort

    image = _read_image(args.image)
    height, width = image.shape[:2]
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)

    print(f"image {width}x{height}")
    print(f"{'threads':>8} {'long':>6} {'tokens':>8} {'ms/frame':>10} {'fps':>7}")
    for threads in [int(v) for v in args.threads.split(",")]:
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            args.model, sess_options=options, providers=["CPUExecutionProvider"]
        )
        name = session.get_inputs()[0].name
        for long_side in [int(v) for v in args.long_sides.split(",")]:
            scale = long_side / max(width, height)
            target_w = max(int(round(width * scale / 14)) * 14, 14)
            target_h = max(int(round(height * scale / 14)) * 14, 14)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            blob = resized.astype(np.float32) / 255.0
            blob = (blob - mean) / std
            blob = np.transpose(blob, (2, 0, 1))[None, ...].astype(np.float32)

            session.run(None, {name: blob})  # warm-up
            samples = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                session.run(None, {name: blob})
                samples.append(time.perf_counter() - start)
            elapsed = min(samples)
            tokens = (target_h // 14) * (target_w // 14)
            print(
                f"{threads:>8} {long_side:>6} {tokens:>8} {elapsed*1000:>10.0f} "
                f"{1.0/elapsed:>7.2f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
