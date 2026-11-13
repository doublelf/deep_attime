"""Turn the depth model's relative disparity into metric depth.

Depth-Anything V2 predicts an *affine invariant* disparity ``d_raw`` (larger
values are closer).  The true metric disparity therefore satisfies

``1 / Z = a * d_raw + b``

for two unknowns ``a`` and ``b``.

The scene provides the missing metric anchor: a horizontal ground plane at a
known camera height.  For every pixel the ground intersection distance is
known from the camera geometry alone, so the bottom rows of the frame give a
set of ``(d_raw, 1 / Z_ground)`` observations that pin down ``a`` and ``b``.

Three estimators are provided:

``fit_ground_scale``
    the affine fit described above.
``ground_depth``
    metric depth straight from the ground plane (used for objects that
    visibly rest on the ground).
``depth_from_prior``
    fallback that derives the depth from a known object height.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..camera.base import CameraModel, local_focal_pixels
from .depth_anything import DepthMap

__all__ = [
    "MetricScale",
    "fit_ground_scale",
    "ground_depth_map",
    "depth_from_prior",
]

@dataclass(frozen=True)
class MetricScale:
    """Affine map from raw disparity to metric disparity."""

    a: float
    b: float
    residual: float = 0.0
    samples: int = 0

    def apply(self, disparity: np.ndarray) -> np.ndarray:
        """Convert raw disparity into metric depth (metres)."""
        true_disparity = self.a * np.asarray(disparity, np.float64) + self.b
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = 1.0 / true_disparity
        depth = np.where(np.isfinite(depth), depth, np.nan)
        return np.clip(depth, 0.05, 500.0)

    @property
    def valid(self) -> bool:
        return self.a > 0.0 and self.samples >= 50


def _sample_grid(height: int, width: int, max_points: int = 240) -> tuple[
    np.ndarray, np.ndarray
]:
    """Regularly spaced pixel coordinates used for the fits."""
    step = max(1, int(np.sqrt(height * width / max(max_points, 1))))
    ys = np.arange(0, height, step)
    xs = np.arange(0, width, step)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    return grid_y.ravel(), grid_x.ravel()


def ground_depth_map(
    camera: CameraModel,
    camera_height: float,
    pitch_deg: float,
    shape: tuple[int, int],
    step: int = 1,
) -> np.ndarray:
    """Metric depth of the ground for every pixel of an image.

    Returns a ``(H, W)`` array; pixels whose ray does not hit the ground hold
    ``NaN``.
    """
    height, width = shape
    ys = np.arange(0, height, step)
    xs = np.arange(0, width, step)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    pixels = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float64)
    rays = camera.unproject(pixels)
    depth = camera.ground_depth(rays, camera_height, pitch_deg)
    full = np.full((height, width), np.nan, np.float64)
    full[::step, ::step] = depth.reshape(len(ys), len(xs))
    return full


def fit_ground_scale(
    depth_map: DepthMap,
    camera: CameraModel,
    camera_height: float,
    pitch_deg: float = 0.0,
    bottom_fraction: float = 0.35,
    max_points: int = 240,
    outlier_sigma: float = 3.0,
) -> MetricScale | None:
    """Fit the affine disparity scale using the visible ground.

    Parameters
    ----------
    depth_map:
        Raw disparity of the frame.
    camera:
        Camera model used to compute the ground intersection.
    camera_height:
        Height of the optical centre above the ground, in metres.
    pitch_deg:
        Downward pitch of the optical axis, in degrees.
    bottom_fraction:
        Fraction of the image height (measured from the bottom) that is
        assumed to show ground.
    """
    height, width = camera.height, camera.width
    step = max(1, int(np.sqrt(height * width / max(max_points, 1))))
    ys = np.arange(0, height, step)
    xs = np.arange(0, width, step)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")

    row_start = int(height * (1.0 - bottom_fraction))
    keep = grid_y >= row_start
    if not np.any(keep):
        return None
    grid_y = grid_y[keep]
    grid_x = grid_x[keep]

    pixels = np.stack([grid_x, grid_y], axis=1).astype(np.float64)
    rays = camera.unproject(pixels)
    ground_depth = camera.ground_depth(rays, camera_height, pitch_deg)

    raw = depth_map.sample(grid_x.astype(np.float64), grid_y.astype(np.float64))
    valid = (
        np.isfinite(raw)
        & np.isfinite(ground_depth)
        & (ground_depth > 0.05)
        & (raw > 0)
    )
    if int(np.count_nonzero(valid)) < 50:
        return None

    x = raw[valid]
    y = 1.0 / ground_depth[valid]

    a, b, resid = _robust_line_fit(x, y, outlier_sigma)
    if a <= 0:
        return None
    return MetricScale(a=float(a), b=float(b), residual=float(resid), samples=int(x.size))


def _robust_line_fit(
    x: np.ndarray,
    y: np.ndarray,
    outlier_sigma: float,
    iterations: int = 2,
) -> tuple[float, float, float]:
    """Least squares fit ``y = a * x + b`` with MAD based outlier rejection."""
    mask = np.ones_like(x, dtype=bool)
    a, b = 0.0, 0.0
    residual = 0.0
    for _ in range(max(iterations, 1)):
        xs, ys = x[mask], y[mask]
        if xs.size < 10:
            break
        design = np.stack([xs, np.ones_like(xs)], axis=1)
        solution, *_ = np.linalg.lstsq(design, ys, rcond=None)
        a, b = float(solution[0]), float(solution[1])
        residuals = y - (a * x + b)
        scale = 1.4826 * float(np.median(np.abs(residuals - np.median(residuals))))
        if scale <= 1e-12:
            break
        new_mask = np.abs(residuals) <= outlier_sigma * scale
        if new_mask.sum() < 20 or np.array_equal(new_mask, mask):
            mask = new_mask
            break
        mask = new_mask
    residual = float(np.sqrt(np.mean((y[mask] - (a * x[mask] + b)) ** 2))) if mask.any() else 0.0
    return a, b, residual


def depth_from_prior(
    camera: CameraModel,
    bbox: np.ndarray,
    height_prior: float,
    camera_height: float,
    pitch_deg: float = 0.0,
) -> float:
    """Estimate the depth of an object from a known physical height.

    The apparent pixel height of the object is turned into a distance with the
    local focal length of the camera: ``Z = f * H / h_px``.
    """
    x1, y1, x2, y2 = [float(v) for v in bbox]
    pixel_height = max(y2 - y1, 1.0)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    focal = local_focal_pixels(camera, cx, cy)
    if focal <= 1e-6:
        return float("nan")
    return float(focal * height_prior / pixel_height)
