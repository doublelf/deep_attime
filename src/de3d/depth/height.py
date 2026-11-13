"""Estimate the camera height from the scene.

The ground-plane depth of an object that stands on the ground is

``Z = h * Z_ground(h = 1)``

so it scales with the camera height ``h``.  The same object also gives an
independent estimate of its distance through its known physical height:

``Z = f_local * H_prior / h_px``

Equating the two yields the camera height

``h = Z_prior / Z_ground(h = 1)``

Averaging that over many frames gives a stable camera height, and with it a
stable metric depth -- much less noisy than the per-frame size estimate it is
derived from.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..camera.base import CameraModel, local_focal_pixels

__all__ = ["CameraHeightEstimator"]


class CameraHeightEstimator:
    """Robust camera-height estimate from grounded subjects."""

    def __init__(self, camera: CameraModel, window: int = 45) -> None:
        self._camera = camera
        self._samples: deque[float] = deque(maxlen=int(window))

    @property
    def samples(self) -> list[float]:
        return list(self._samples)

    @property
    def height(self) -> float | None:
        """Median of the collected estimates (metres), or ``None``."""
        if not self._samples:
            return None
        return float(np.median(self._samples))

    def reset(self) -> None:
        self._samples.clear()

    def update(
        self,
        bbox: np.ndarray,
        height_prior: float,
        camera_height_hint: float = 1.0,
    ) -> float | None:
        """Add one observation and return the current estimate.

        Parameters
        ----------
        bbox:
            ``(x1, y1, x2, y2)`` box of an object that stands on the ground.
        height_prior:
            Real-world height of the object, in metres.
        camera_height_hint:
            Value used for the reference ground depth; only the ratio matters.
        """
        x1, y1, x2, y2 = [float(v) for v in bbox]
        pixel_height = y2 - y1
        if pixel_height < 20 or height_prior <= 0:
            return None

        # Contact point: the middle of the bottom edge.
        foot_u = 0.5 * (x1 + x2)
        foot_v = y2
        ray = self._camera.unproject(np.array([[foot_u, foot_v]], np.float64))[0]
        ground = self._camera.ground_depth(
            ray[None, :], camera_height_hint, self._camera_pitch
        )[0]
        if not np.isfinite(ground) or ground <= 0:
            return None

        focal = local_focal_pixels(self._camera, foot_u, 0.5 * (y1 + y2))
        if focal <= 1e-6:
            return None
        prior_depth = focal * height_prior / pixel_height
        if not np.isfinite(prior_depth) or prior_depth <= 0:
            return None

        estimate = prior_depth / ground * camera_height_hint
        if not np.isfinite(estimate) or estimate <= 0.05 or estimate > 20:
            return None
        self._samples.append(float(estimate))
        return self.height

    # The pitch is not known while estimating; a level camera is assumed.
    _camera_pitch = 0.0
