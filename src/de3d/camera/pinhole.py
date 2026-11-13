"""Rectilinear pinhole camera model."""

from __future__ import annotations

import numpy as np

from .base import CameraModel

__all__ = ["PinholeCamera"]


class PinholeCamera(CameraModel):
    """Standard rectified pinhole camera.

    Parameters
    ----------
    fx, fy:
        Focal lengths in pixels.
    cx, cy:
        Principal point in pixels.
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
    ) -> None:
        super().__init__(width, height)
        self._fx = float(fx)
        self._fy = float(fy)
        self._cx = float(cx)
        self._cy = float(cy)

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

    # -- geometry -------------------------------------------------------
    def unproject(self, pixels: np.ndarray) -> np.ndarray:
        pixels = np.asarray(pixels, np.float64).reshape(-1, 2)
        rays = np.empty((pixels.shape[0], 3), np.float64)
        rays[:, 0] = (pixels[:, 0] - self._cx) / self._fx
        rays[:, 1] = (pixels[:, 1] - self._cy) / self._fy
        rays[:, 2] = 1.0
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    def project(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, np.float64).reshape(-1, 3)
        z = np.where(np.abs(points[:, 2]) < 1e-9, 1e-9, points[:, 2])
        u = self._fx * points[:, 0] / z + self._cx
        v = self._fy * points[:, 1] / z + self._cy
        return np.stack([u, v], axis=1)

    def _rescale(self, sx: float, sy: float, width: int, height: int) -> PinholeCamera:
        return PinholeCamera(
            fx=self._fx * sx,
            fy=self._fy * sy,
            cx=self._cx * sx,
            cy=self._cy * sy,
            width=width,
            height=height,
        )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"PinholeCamera(fx={self._fx:.1f}, fy={self._fy:.1f}, "
            f"cx={self._cx:.1f}, cy={self._cy:.1f}, size={self.width}x{self.height})"
        )
