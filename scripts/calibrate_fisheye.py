"""Calibrate an equidistant fisheye camera and write a camera profile YAML.

The script follows the OpenCV fisheye model used by the ``ppo`` rig:
``cv2.fisheye.calibrate`` for the intrinsics and the 4-term equidistant
polynomial for the distortion.

Usage::

    python scripts/calibrate_fisheye.py --images calib/ --out config/cameras/my_cam.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

TEMPLATE = """# Equidistant fisheye camera calibrated with cv2.fisheye.calibrate.
# Images: {count} view(s) from {source}

model: equidistant_fisheye
width: {width}
height: {height}
fx: {fx:.4f}
fy: {fy:.4f}
cx: {cx:.4f}
cy: {cy:.4f}
distortion: [{k1:.6g}, {k2:.6g}, {k3:.6g}, {k4:.6g}]
rms: {rms:.4f}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="directory with calibration views")
    parser.add_argument("--out", default="config/cameras/fisheye.yaml")
    parser.add_argument("--cols", type=int, default=9, help="chessboard columns (inner corners)")
    parser.add_argument("--rows", type=int, default=6, help="chessboard rows (inner corners)")
    parser.add_argument("--square", type=float, default=0.025, help="square size in metres")
    args = parser.parse_args(argv)

    image_dir = Path(args.images)
    files = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )
    if not files:
        print(f"[error] no images in {image_dir}", file=sys.stderr)
        return 1

    objp = np.zeros((args.rows * args.cols, 3), np.float64)
    objp[:, :2] = np.mgrid[0 : args.cols, 0 : args.rows].T.reshape(-1, 2)
    objp *= args.square

    object_points = []
    image_points = []
    size = None
    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows), None)
        if not found:
            print(f"[warn] no chessboard in {path.name}")
            continue
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001,
            ),
        )
        object_points.append(objp)
        image_points.append(corners)
        print(f"[ok  ] {path.name}")

    if len(object_points) < 3:
        print("[error] need at least 3 usable views", file=sys.stderr)
        return 1

    K = np.zeros((3, 3), np.float64)
    D = np.zeros((4, 1), np.float64)
    rms = cv2.fisheye.calibrate(
        object_points, image_points, size, K, D, K, flags=(cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC)
    )
    K = np.asarray(K).reshape(3, 3)
    D = np.asarray(D).reshape(4)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        TEMPLATE.format(
            count=len(object_points),
            source=image_dir,
            width=size[0],
            height=size[1],
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            k1=D[0],
            k2=D[1],
            k3=D[2],
            k4=D[3],
            rms=rms,
        ),
        encoding="utf-8",
    )
    print(f"[ok  ] wrote {out} (rms={rms:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
