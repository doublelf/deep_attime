"""Camera model factory and configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .base import CameraModel
from .fisheye import EquidistantFisheyeCamera
from .pinhole import PinholeCamera

__all__ = [
    "CameraModel",
    "PinholeCamera",
    "EquidistantFisheyeCamera",
    "build_camera",
    "load_camera_profile",
]


def build_camera(spec: dict[str, Any]) -> CameraModel:
    """Create a camera model from a plain dictionary.

    Expected keys: ``model`` (``pinhole`` or ``equidistant_fisheye``),
    ``width``, ``height`` and the intrinsics.
    """
    model = str(spec.get("model", "pinhole")).lower()
    width = int(spec["width"])
    height = int(spec["height"])
    if model in {"pinhole", "rectilinear"}:
        return PinholeCamera(
            fx=float(spec["fx"]),
            fy=float(spec.get("fy", spec["fx"])),
            cx=float(spec["cx"]),
            cy=float(spec["cy"]),
            width=width,
            height=height,
        )
    if model in {"equidistant_fisheye", "fisheye", "equidistant"}:
        dist = spec.get("distortion") or [0.0, 0.0, 0.0, 0.0]
        return EquidistantFisheyeCamera(
            fx=float(spec["fx"]),
            fy=float(spec.get("fy", spec["fx"])),
            cx=float(spec["cx"]),
            cy=float(spec["cy"]),
            width=width,
            height=height,
            k1=float(dist[0]),
            k2=float(dist[1]) if len(dist) > 1 else 0.0,
            k3=float(dist[2]) if len(dist) > 2 else 0.0,
            k4=float(dist[3]) if len(dist) > 3 else 0.0,
        )
    raise ValueError(f"unknown camera model: {model!r}")


def load_camera_profile(path: str | Path) -> tuple[CameraModel, dict[str, Any]]:
    """Load a camera YAML profile.

    The profile may describe a single camera (``model``/``width``/``height``
    at the top level) or a rig (``cameras: [...]``).  For a rig the camera
    with ``yaw == 0`` is returned, together with the full profile so that the
    caller can select another member of the rig.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        profile = yaml.safe_load(handle) or {}

    if "cameras" in profile:
        members = list(profile["cameras"])
        if not members:
            raise ValueError(f"{path}: 'cameras' is empty")
        default_yaw = float(profile.get("default_yaw", 0.0))
        chosen = min(members, key=lambda c: abs(float(c.get("yaw", 0.0)) - default_yaw))
        return build_camera(chosen), profile
    return build_camera(profile), profile
