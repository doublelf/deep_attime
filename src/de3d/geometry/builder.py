"""Turn a 2D detection plus a depth map into a metric 3D cuboid.

The pipeline is:

1. pick the anchor pixel of the object -- the bottom-centre for objects that
   stand on the ground, the box centre for suspended ones;
2. estimate the metric depth of that anchor (ground plane, depth model or a
   class height prior, see :mod:`de3d.depth.metric`);
3. place the cuboid centre and size it from the class prior, optionally
   refined with the apparent size of the object.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..camera.base import CameraModel, pixels_per_meter_at_depth
from ..depth.depth_anything import DepthMap
from ..depth.metric import MetricScale
from ..detect.base import Detection
from .box3d import Box3D
from .priors import ClassPrior

__all__ = ["DepthContext", "build_box3d", "estimate_yaw"]

_GROUND_BAND = (0.05, 0.25)  # fractions of the box height above the bottom
_CENTRE_BAND = 0.40  # half-width of the central region used for suspended objects

DepthProvider = Callable[[], "tuple[DepthMap | None, MetricScale | None]"]


@dataclass
class DepthContext:
    """Everything needed to lift a 2D detection into 3D.

    ``depth_provider`` is called lazily: the depth model is only executed when
    the metric depth of the depth model is actually required (for suspended
    objects, or when the ground plane cannot be used).
    """

    camera: CameraModel
    camera_height: float
    pitch_deg: float = 0.0
    depth_provider: DepthProvider | None = None

    def region_depth(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Median metric depth of the depth model inside a pixel region."""
        if self.depth_provider is None:
            return float("nan")
        depth_map, scale = self.depth_provider()
        if depth_map is None or scale is None or not scale.valid:
            return float("nan")
        raw = depth_map.region_median(x1, y1, x2, y2)
        if not np.isfinite(raw):
            return float("nan")
        return float(scale.apply(np.array([raw], np.float64))[0])


def estimate_yaw(measured_width: float, prior_width: float, prior_length: float) -> float:
    """Yaw that best explains the observed horizontal extent.

    The horizontal extent of a box of width ``w`` and length ``l`` rotated by
    ``yaw`` is approximately ``w*|cos| + l*|sin|``.  The smaller of the two
    possible solutions is returned (objects are more often seen close to
    front-on than side-on).
    """
    if not np.isfinite(measured_width) or measured_width <= 0:
        return 0.0
    width = max(prior_width, 1e-3)
    length = max(prior_length, 1e-3)
    target = float(measured_width)
    peak = float(np.hypot(width, length))
    if target >= peak:
        return float(np.arctan2(length, width))
    lo, hi = 0.0, float(np.arctan2(length, width))
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        value = width * np.cos(mid) + length * np.sin(mid)
        if value < target:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def build_box3d(
    detection: Detection,
    prior: ClassPrior,
    context: DepthContext,
    depth_mode: str = "auto",
    yaw_mode: str = "zero",
    aspect_refine: float = 0.5,
) -> Box3D | None:
    """Build the metric cuboid of ``detection``.

    Parameters
    ----------
    depth_mode:
        ``auto``     ground plane for grounded objects, depth model otherwise;
        ``ground``   ground plane only;
        ``model``    depth model only;
        ``prior``    class height prior only.
    yaw_mode:
        ``zero``   axis aligned box;
        ``ratio``  yaw estimated from the apparent width.
    aspect_refine:
        Weight of the measured size (0 = pure class prior, 1 = pure measurement).
    """
    x1, y1, x2, y2 = detection.box
    box_w = max(x2 - x1, 1.0)
    box_h = max(y2 - y1, 1.0)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    grounded = prior.grounded
    source = "unknown"
    depth = float("nan")

    if grounded:
        anchor = np.array([[cx, y2]], np.float64)
        band_top = y2 - _GROUND_BAND[1] * box_h
        band_bottom = y2 - _GROUND_BAND[0] * box_h
        sample_region = (x1, band_top, x2, band_bottom)
    else:
        anchor = np.array([[cx, cy]], np.float64)
        half_w = _CENTRE_BAND * 0.5 * box_w
        half_h = _CENTRE_BAND * 0.5 * box_h
        sample_region = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)

    # -- depth ----------------------------------------------------------
    if depth_mode in {"auto", "ground"} and grounded:
        ray = context.camera.unproject(anchor)[0]
        depth = float(
            context.camera.ground_depth(
                ray[None, :], context.camera_height, context.pitch_deg
            )[0]
        )
        source = "ground"
    if (not np.isfinite(depth)) and depth_mode in {"auto", "model"}:
        depth = context.region_depth(*sample_region)
        source = "model"
    if (not np.isfinite(depth)) and depth_mode == "auto":
        from ..depth.metric import depth_from_prior

        depth = depth_from_prior(
            context.camera,
            detection.box,
            prior.height,
            context.camera_height,
            context.pitch_deg,
        )
        source = "prior"
    if depth_mode == "prior":
        from ..depth.metric import depth_from_prior

        depth = depth_from_prior(
            context.camera,
            detection.box,
            prior.height,
            context.camera_height,
            context.pitch_deg,
        )
        source = "prior"
    if not np.isfinite(depth) or depth <= 0:
        return None

    # -- size -----------------------------------------------------------
    width, height, length = prior.dims
    if aspect_refine > 0:
        ppm = pixels_per_meter_at_depth(context.camera, cx, cy, depth)
        if np.isfinite(ppm) and ppm > 1e-6:
            measured_w = box_w / ppm
            measured_h = box_h / ppm
            blend = float(np.clip(aspect_refine, 0.0, 1.0))
            width = (1 - blend) * width + blend * measured_w
            height = (1 - blend) * height + blend * measured_h
            if prior.width > 1e-6:
                length = length * (width / prior.width)
            width = float(np.clip(width, 0.5 * prior.width, 1.8 * prior.width))
            height = float(np.clip(height, 0.5 * prior.height, 1.8 * prior.height))
            length = float(np.clip(length, 0.5 * prior.length, 1.8 * prior.length))

    # -- 3D centre ------------------------------------------------------
    ray = context.camera.unproject(anchor)[0]
    point = ray * (depth / ray[2])
    if grounded:
        # The anchor is the contact point with the ground; lift by half the
        # (final) object height to reach the centre of the cuboid.
        centre = point + np.array([0.0, -height / 2.0, 0.0])
    else:
        centre = point

    # -- yaw ------------------------------------------------------------
    yaw = 0.0
    if yaw_mode == "ratio" and aspect_refine > 0:
        ppm = pixels_per_meter_at_depth(context.camera, cx, cy, depth)
        if np.isfinite(ppm) and ppm > 1e-6:
            yaw = estimate_yaw(box_w / ppm, prior.width, prior.length)

    return Box3D(
        category=detection.class_name,
        score=float(detection.score),
        center=centre,
        dims=np.array([width, height, length], np.float64),
        yaw=yaw,
        depth=float(depth),
        depth_source=source,
        bbox2d=(float(x1), float(y1), float(x2), float(y2)),
    )
