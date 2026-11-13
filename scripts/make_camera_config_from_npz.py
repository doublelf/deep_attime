"""Convert the ``ppo`` fisheye rig configuration into a camera profile YAML.

The ``ppo`` project stores its rig in an ``.npz`` archive with the keys
``K`` (3x3), ``D`` (4x1), ``yaws`` and ``size``.  This script turns that into
the YAML profile consumed by ``de3d``.

Usage::

    python scripts/make_camera_config_from_npz.py \\
        --npz ../ppo/vendor/camera_config.npz \\
        --out config/cameras/ppo_fisheye_480.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

TEMPLATE = """# Equidistant fisheye surround rig converted from
#   {source}
#
# !! The source values are placeholders unless the rig was really calibrated
#    with cv2.fisheye.calibrate -- check the generator script that produced it.
#
# Lens model: equidistant (theta_d = theta * (1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8)),
# identical to ppo/panorama/geometry/fisheye.py.

default_yaw: 0.0

cameras:
{cameras}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz",
        default=str(Path(__file__).resolve().parents[2] / "ppo" / "vendor" / "camera_config.npz"),
    )
    parser.add_argument("--out", default="config/cameras/ppo_fisheye_480.yaml")
    parser.add_argument("--prefix", default="cam")
    args = parser.parse_args(argv)

    source = Path(args.npz)
    if not source.exists():
        print(f"[error] {source} not found", file=sys.stderr)
        return 1
    data = np.load(source, allow_pickle=True)
    K = np.asarray(data["K"], np.float64).reshape(3, 3)
    D = np.asarray(data["D"], np.float64).reshape(-1)
    yaws = np.asarray(data["yaws"], np.float64).reshape(-1)
    size = int(np.asarray(data["size"]).reshape(-1)[0])

    blocks = []
    for index, yaw in enumerate(yaws):
        blocks.append(
            f"""  - name: {args.prefix}{index}_{int(round(float(yaw))) % 360:03d}
    yaw: {float(yaw):.1f}
    pitch: 0.0
    model: equidistant_fisheye
    width: {size}
    height: {size}
    fx: {K[0, 0]:.4f}
    fy: {K[1, 1]:.4f}
    cx: {K[0, 2]:.4f}
    cy: {K[1, 2]:.4f}
    distortion: [{D[0]:.6g}, {D[1]:.6g}, {D[2]:.6g}, {D[3]:.6g}]
"""
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        TEMPLATE.format(source=source, cameras="\n".join(blocks)), encoding="utf-8"
    )
    print(f"[ok  ] wrote {out} ({len(yaws)} cameras, size {size})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
