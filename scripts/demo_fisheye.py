"""End-to-end check of the fisheye branch.

A frame of a pinhole video is re-rendered through the fisheye camera model, so
the image is consistent with ``config/cameras/ppo_fisheye_480.yaml``.  The
pipeline then runs on that fisheye image and the resulting cuboid has to land
on the same object as in the original frame.

Usage::

    python scripts/demo_fisheye.py --input <video.mp4> --frame 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from de3d.camera import load_camera_profile  # noqa: E402
from de3d.camera.fisheye import EquidistantFisheyeCamera  # noqa: E402
from de3d.camera.pinhole import PinholeCamera  # noqa: E402


def render_fisheye(frame: np.ndarray, fisheye: EquidistantFisheyeCamera) -> np.ndarray:
    """Re-render a pinhole frame through the fisheye camera model."""
    height, width = frame.shape[:2]
    pinhole = PinholeCamera(
        fx=1371.1, fy=1371.1, cx=width / 2.0, cy=height / 2.0, width=width, height=height
    )
    ys, xs = np.mgrid[0 : fisheye.height, 0 : fisheye.width]
    pixels = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    rays = fisheye.unproject(pixels)
    source = pinhole.project(rays).reshape(fisheye.height, fisheye.width, 2)
    return cv2.remap(
        frame,
        source[..., 0].astype(np.float32),
        source[..., 1].astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--frame", type=int, default=12)
    parser.add_argument("--camera", default=str(REPO_ROOT / "config" / "cameras" / "ppo_fisheye_480.yaml"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "fisheye"))
    args = parser.parse_args(argv)

    camera, _ = load_camera_profile(args.camera)
    assert isinstance(camera, EquidistantFisheyeCamera)

    capture = cv2.VideoCapture(args.input)
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        print("[error] cannot read the frame", file=sys.stderr)
        return 1

    fisheye_frame = render_fisheye(frame, camera)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "fisheye_input.png"), fisheye_frame)

    from de3d.cli import main as cli_main

    argv2 = [
        "--input",
        str(out_dir / "fisheye_input.png"),
        "--camera",
        args.camera,
        "--out-dir",
        str(out_dir),
        "--json",
        str(out_dir / "subject.json"),
        "--no-video",
    ]
    return cli_main(argv2)


if __name__ == "__main__":
    sys.exit(main())
