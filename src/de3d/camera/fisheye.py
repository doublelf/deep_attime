"""Fisheye camera model of the ``ppo`` surround rig.

The lens model is identical to the one used by the ``ppo`` panoramic rig
(``ppo/panorama/geometry/fisheye.py``), so the 3D boxes drawn on a raw
fisheye frame stay consistent with the panorama warping.

Forward model (camera frame, ``+x`` right, ``+y`` down, ``+z`` forward)::

    gx = X / Z,  gy = Y / Z
    theta   = atan2(hypot(gx, gy), 1)          # angle from the optical axis
    theta_d = theta * (1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8)
    scale   = theta_d / theta
    u = cx + scale * f * gx
    v = cy + scale * f * gy

Note that the pixel radius follows ``r = f * (theta_d / theta) * tan(theta)``:
the radial polynomial stretches the image towards the rim while the tangent
law of the pinhole projection is kept.

The inverse maps a pixel back to a ray.  Writing ``r = hypot(u - cx, v - cy)``
the radial constraint is ``r / f = (1 + k1 t^2 + ...) * tan(theta)``, which is
monotonically increasing on ``[0, pi/2)`` and is solved with a vectorised
bisection.
"""

from __future__ import annotations

import numpy as np

from .base import CameraModel

__all__ = ["EquidistantFisheyeCamera"]

_THETA_MAX = 1.55  # radians; tan() is still finite here
_BISECTION_STEPS = 60


class EquidistantFisheyeCamera(CameraModel):
    """Equidistant fisheye camera with a 4-term radial distortion.

    Parameters
    ----------
    fx, fy, cx, cy:
        Intrinsics of the *equidistant* pinhole-equivalent model.
    k1, k2, k3, k4:
        Radial distortion coefficients of the equidistant polynomial.
    width, height:
        Image size the intrinsics were calibrated for.
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        k1: float = 0.0,
        k2: float = 0.0,
        k3: float = 0.0,
        k4: float = 0.0,
    ) -> None:
        super().__init__(width, height)
        self._fx = float(fx)
        self._fy = float(fy)
        self._cx = float(cx)
        self._cy = float(cy)
        self.k = np.array([k1, k2, k3, k4], np.float64)

    # -- properties -----------------------------------------------------
    @property
    def fx(self) -> float:
        return self._fx

    @property
    def fy(self) -> float:
        return self._fy

    @property
    def cx(self) -> float:
        return self._cx

    @property
    def cy(self) -> float:
        return self._cy

    # -- lens maths -----------------------------------------------------
    def _distortion_ratio(self, theta: np.ndarray) -> np.ndarray:
        """``theta_d / theta`` of the equidistant polynomial."""
        theta2 = theta * theta
        ratio = (
            1.0
            + self.k[0] * theta2
            + self.k[1] * theta2**2
            + self.k[2] * theta2**3
            + self.k[3] * theta2**4
        )
        return np.where(np.abs(theta) < 1e-12, 1.0, ratio)

    def _radial_forward(self, theta: np.ndarray) -> np.ndarray:
        """``r / f`` produced by a ray at angle ``theta`` from the axis."""
        return self._distortion_ratio(theta) * np.tan(theta)

    def _theta_from_radius(self, radius: np.ndarray) -> np.ndarray:
        """Invert :meth:`_radial_forward` with a vectorised bisection."""
        target = np.asarray(radius, np.float64) / self._fx
        lo = np.zeros_like(target)
        hi = np.full_like(target, _THETA_MAX)
        for _ in range(_BISECTION_STEPS):
            mid = 0.5 * (lo + hi)
            too_small = self._radial_forward(mid) < target
            lo = np.where(too_small, mid, lo)
            hi = np.where(too_small, hi, mid)
        return 0.5 * (lo + hi)

    # -- geometry -------------------------------------------------------
    def unproject(self, pixels: np.ndarray) -> np.ndarray:
        pixels = np.asarray(pixels, np.float64).reshape(-1, 2)
        dx = pixels[:, 0] - self._cx
        dy = pixels[:, 1] - self._cy
        radius = np.hypot(dx, dy)
        theta = self._theta_from_radius(radius)
        scale = self._distortion_ratio(theta)
        # dx = scale * f * gx  ->  gx = dx / (scale * f)
        gx = dx / (scale * self._fx)
        gy = dy / (scale * self._fy)
        rays = np.stack([gx, gy, np.ones_like(gx)], axis=1)
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    def project(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, np.float64).reshape(-1, 3)
        z = np.where(np.abs(points[:, 2]) < 1e-9, 1e-9, points[:, 2])
        gx = points[:, 0] / z
        gy = points[:, 1] / z
        theta = np.arctan2(np.hypot(gx, gy), 1.0)
        scale = self._distortion_ratio(theta)
        u = self._cx + scale * self._fx * gx
        v = self._cy + scale * self._fy * gy
        return np.stack([u, v], axis=1)

    def _rescale(
        self, sx: float, sy: float, width: int, height: int
    ) -> EquidistantFisheyeCamera:
        return EquidistantFisheyeCamera(
            fx=self._fx * sx,
            fy=self._fy * sy,
            cx=self._cx * sx,
            cy=self._cy * sy,
            width=width,
            height=height,
            k1=self.k[0],
            k2=self.k[1],
            k3=self.k[2],
            k4=self.k[3],
        )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"EquidistantFisheyeCamera(fx={self._fx:.1f}, fy={self._fy:.1f}, "
            f"cx={self._cx:.1f}, cy={self._cy:.1f}, k={self.k.tolist()}, "
            f"size={self.width}x{self.height})"
        )
