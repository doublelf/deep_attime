"""Camera model tests: projection round-trips and the ground plane."""

from __future__ import annotations

import numpy as np
import pytest

from de3d.camera.base import drop_to_ground, pixels_per_meter_at_depth
from de3d.camera.fisheye import EquidistantFisheyeCamera
from de3d.camera.pinhole import PinholeCamera


def _pinhole() -> PinholeCamera:
    return PinholeCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)


def _fisheye() -> EquidistantFisheyeCamera:
    return EquidistantFisheyeCamera(
        fx=220.0,
        fy=220.0,
        cx=240.0,
        cy=240.0,
        width=480,
        height=480,
        k1=0.02,
        k2=-0.005,
        k3=0.0005,
        k4=0.0,
    )


@pytest.mark.parametrize("camera", [_pinhole(), _fisheye()])
def test_project_unproject_round_trip(camera) -> None:
    rng = np.random.default_rng(0)
    pixels = np.stack(
        [
            rng.uniform(0, camera.width, 200),
            rng.uniform(0, camera.height, 200),
        ],
        axis=1,
    )
    rays = camera.unproject(pixels)
    # Points along the rays, one metre away.
    points = rays * 2.0
    again = camera.project(points)
    assert np.allclose(again, pixels, atol=1e-6)


@pytest.mark.parametrize("camera", [_pinhole(), _fisheye()])
def test_ray_is_normalised(camera) -> None:
    pixels = np.array([[10.0, 20.0], [400.0, 300.0], [camera.cx, camera.cy]])
    rays = camera.unproject(pixels)
    assert np.allclose(np.linalg.norm(rays, axis=1), 1.0)


@pytest.mark.parametrize("camera", [_pinhole(), _fisheye()])
def test_ground_depth_matches_manual_formula(camera) -> None:
    height = 1.5
    pixels = np.array([[camera.cx, camera.cy + 50.0], [camera.cx + 30.0, camera.cy + 120.0]])
    rays = camera.unproject(pixels)
    depth = camera.ground_depth(rays, height)
    assert np.all(np.isfinite(depth))
    assert np.all(depth > 0)
    # The ground point must really sit on the ground plane.
    points = rays * (depth[:, None] / rays[:, 2:3])
    assert np.allclose(points[:, 1], height, atol=1e-6)


@pytest.mark.parametrize("camera", [_pinhole(), _fisheye()])
def test_pixels_per_meter_matches_focal_ratio_for_pinhole(camera) -> None:
    depth = 8.0
    ppm = pixels_per_meter_at_depth(camera, camera.cx, camera.cy, depth)
    assert np.isfinite(ppm)
    if isinstance(camera, PinholeCamera):
        assert ppm == pytest.approx(camera.fy / depth, rel=1e-6)


@pytest.mark.parametrize("camera", [_pinhole(), _fisheye()])
def test_drop_to_ground_lands_on_ground(camera) -> None:
    height = 1.2
    pixels = np.array([[camera.cx, camera.cy + 80.0]])
    ray = camera.unproject(pixels)[0]
    depth = camera.ground_depth(ray[None, :], height)[0]
    point = ray * (depth / ray[2]) + np.array([0.0, -0.4, 0.0])
    dropped = drop_to_ground(camera, point[None, :], height)[0]
    assert dropped[1] == pytest.approx(height, abs=1e-6)


def test_scaled_preserves_field_of_view() -> None:
    camera = _pinhole()
    scaled = camera.scaled(1280, 960)
    assert scaled.fx == pytest.approx(camera.fx * 2.0)
    assert scaled.cx == pytest.approx(camera.cx * 2.0)
    pixels = np.array([[100.0, 120.0]])
    assert np.allclose(camera.unproject(pixels), scaled.unproject(pixels * 2.0))


def test_fisheye_matches_reference_implementation() -> None:
    """Compare against the projection used by ppo/panorama/geometry/fisheye.py."""
    camera = _fisheye()
    k = camera.k
    point = np.array([[0.3, -0.2, 2.0]])
    gx, gy = point[0, 0] / point[0, 2], point[0, 1] / point[0, 2]
    theta = np.arctan2(np.hypot(gx, gy), 1.0)
    theta_d = theta * (
        1 + k[0] * theta**2 + k[1] * theta**4 + k[2] * theta**6 + k[3] * theta**8
    )
    scale = theta_d / theta
    expected_u = camera.cx + scale * camera.fx * gx
    expected_v = camera.cy + scale * camera.fy * gy
    got = camera.project(point)[0]
    assert got[0] == pytest.approx(expected_u)
    assert got[1] == pytest.approx(expected_v)
