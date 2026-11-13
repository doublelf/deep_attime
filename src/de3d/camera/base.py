"""Camera models shared by the whole pipeline.

Every model works in the OpenCV camera frame: ``+x`` right, ``+y`` down,
``+z`` towards the scene.  The same convention is used by the Omni3D
benchmark, so 3D boxes stay compatible with other monocular detectors.

Two models are provided:

* :class:`PinholeCamera` -- the usual rectified pinhole camera.
* :class:`EquidistantFisheyeCamera` -- the equidistant lens model used by the
  ``ppo`` panoramic rig (see ``ppo/panorama/geometry/fisheye.py``).  The
  distortion polynomial is identical, so boxes drawn on a raw fisheye frame
  match the panorama warping.

The unified interface is intentionally small so that the 3D box maths never
has to know which lens was used:

``unproject(pixels)``
    image pixels -> unit ray directions in the camera frame.
``project(points)``
    camera-frame points -> image pixels.
``ground_depth(rays, camera_height, pitch_deg)``
    metric depth of the ground point along each ray.

All models share the same ground-plane model: the ground is a horizontal
plane ``camera_height`` metres below the optical centre and the optical axis
is pitched down by ``pitch_deg``.  With world axes ``X`` right, ``Y``
forward and ``Z`` up, a camera pitched down by ``p`` maps a camera-frame
point ``q`` to the world height ``-qy*cos(p) - qz*sin(p)``.  The ground
therefore satisfies ``qy*cos(p) + qz*sin(p) = h`` and a ray ``(dx, dy, dz)``
hits it at the depth

``Z = h / ((dy / dz) * cos(p) + sin(p))``

which reduces to the familiar ``h * fy / (v - cy)`` for a level pinhole
camera.
"""

from __future__ import annotations

import abc

import numpy as np

__all__ = [
    "CameraModel",
    "pixels_per_meter_at_ground",
    "pixels_per_meter_at_depth",
    "local_focal_pixels",
    "drop_to_ground",
]


class CameraModel(abc.ABC):
    """Base class for camera intrinsics."""

    def __init__(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)

    # -- geometry ------------------------------------------------------
    @abc.abstractmethod
    def unproject(self, pixels: np.ndarray) -> np.ndarray:
        """Map image pixels to unit ray directions.

        Parameters
        ----------
        pixels:
            ``(N, 2)`` array of ``(u, v)`` pixel coordinates.

        Returns
        -------
        ``(N, 3)`` array of unit vectors in the camera frame.
        """

    @abc.abstractmethod
    def project(self, points: np.ndarray) -> np.ndarray:
        """Map camera-frame points to image pixels.

        Parameters
        ----------
        points:
            ``(N, 3)`` array of ``(X, Y, Z)`` points.

        Returns
        -------
        ``(N, 2)`` array of ``(u, v)`` pixel coordinates.
        """

    def ground_depth(
        self,
        rays: np.ndarray,
        camera_height: float,
        pitch_deg: float = 0.0,
    ) -> np.ndarray:
        """Metric depth of the ground intersection along each ray.

        Parameters
        ----------
        rays:
            ``(N, 3)`` unit ray directions in the camera frame.
        camera_height:
            Height of the optical centre above the ground, in metres.
        pitch_deg:
            Downward pitch of the optical axis, in degrees.

        Returns
        -------
        ``(N,)`` array of depths (NaN where the ray does not hit the ground).
        """
        rays = np.asarray(rays, np.float64).reshape(-1, 3)
        pitch = np.radians(float(pitch_deg))
        with np.errstate(divide="ignore", invalid="ignore"):
            gy = rays[:, 1] / rays[:, 2]
        # World axes: X right, Y forward (horizontal), Z up.  A camera pitched
        # down by p has forward = (0, cos p, -sin p) and down = (0, -sin p,
        # -cos p), so the world height of a camera-frame point q is
        #     -qy * cos(p) - qz * sin(p)
        # and the ground (h below the optical centre) satisfies
        #     qy * cos(p) + qz * sin(p) = h.
        denom = gy * np.cos(pitch) + np.sin(pitch)
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = float(camera_height) / denom
        valid = np.isfinite(depth) & (depth > 0)
        return np.where(valid, depth, np.nan)

    # -- helpers --------------------------------------------------------
    def scaled(self, width: int, height: int) -> CameraModel:
        """Return an equivalent model for a resized image.

        The field of view is preserved, so the focal length and the principal
        point are scaled by the corresponding factor.
        """
        if width == self.width and height == self.height:
            return self
        sx = width / float(self.width)
        sy = height / float(self.height)
        return self._rescale(sx, sy, int(width), int(height))

    @abc.abstractmethod
    def _rescale(self, sx: float, sy: float, width: int, height: int) -> CameraModel:
        """Return a copy with scaled intrinsics."""

    @property
    def diagonal(self) -> float:
        """Diagonal of the image in pixels (used for normalised distances)."""
        return float(np.hypot(self.width, self.height))

    def pixels_per_meter(self, depth: float) -> float:
        """Vertical pixels covered by one metre at a given depth."""
        return self.fy / max(depth, 1e-6)

    @property
    @abc.abstractmethod
    def fx(self) -> float:
        """Focal length in pixels along x."""

    @property
    @abc.abstractmethod
    def fy(self) -> float:
        """Focal length in pixels along y."""

    @property
    @abc.abstractmethod
    def cx(self) -> float:
        """Principal point x."""

    @property
    @abc.abstractmethod
    def cy(self) -> float:
        """Principal point y."""


def pixels_per_meter_at_ground(
    camera: CameraModel,
    u: float,
    v: float,
    camera_height: float,
    pitch_deg: float = 0.0,
) -> float:
    """Vertical pixels per metre at the ground point below pixel ``(u, v)``."""
    ray = camera.unproject(np.array([[u, v]], np.float64))[0]
    depth = camera.ground_depth(ray[None, :], camera_height, pitch_deg)[0]
    if not np.isfinite(depth) or depth <= 0:
        return float("nan")
    return local_focal_pixels(camera, u, v) / depth


def drop_to_ground(
    camera: CameraModel,
    points: np.ndarray,
    camera_height: float,
    pitch_deg: float = 0.0,
) -> np.ndarray:
    """Drop camera-frame points vertically onto the ground plane.

    Used to draw the "shadow" of a cuboid, i.e. its footprint on the ground.
    """
    points = np.asarray(points, np.float64).reshape(-1, 3)
    pitch = np.radians(float(pitch_deg))
    cos_p = float(np.cos(pitch))
    sin_p = float(np.sin(pitch))
    # Height above the ground of every point, in the camera frame.
    above = float(camera_height) - (points[:, 1] * cos_p + points[:, 2] * sin_p)
    # Moving a point along +y (downwards in the image) lowers it by
    # s * cos(p) metres in the world.
    step = above / max(cos_p, 1e-6)
    out = points.copy()
    out[:, 1] = points[:, 1] + np.maximum(step, 0.0)
    return out


def pixels_per_meter_at_depth(
    camera: CameraModel,
    u: float,
    v: float,
    depth: float,
) -> float:
    """Vertical pixels per metre for a point at ``depth`` along a pixel ray."""
    if not np.isfinite(depth) or depth <= 0:
        return float("nan")
    return local_focal_pixels(camera, u, v) / depth


def local_focal_pixels(camera: CameraModel, u: float, v: float) -> float:
    """Pixels per radian in the vertical direction at pixel ``(u, v)``.

    For a pinhole camera this is simply ``fy``; for the fisheye model the
    scale shrinks towards the edge of the frame, which the numeric derivative
    accounts for.
    """
    ray = camera.unproject(np.array([[u, v]], np.float64))[0]
    delta = 1e-4

    def rotate_x(vector: np.ndarray, angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array(
            [
                vector[0],
                vector[1] * c - vector[2] * s,
                vector[1] * s + vector[2] * c,
            ]
        )

    upper = camera.project(rotate_x(ray, delta)[None, :])[0]
    lower = camera.project(rotate_x(ray, -delta)[None, :])[0]
    return float(np.hypot(*(upper - lower)) / (2.0 * delta))
