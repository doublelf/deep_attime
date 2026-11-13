"""Metric depth conversion and the 3D cuboid builder."""

from __future__ import annotations

import numpy as np
import pytest

from de3d.camera.pinhole import PinholeCamera
from de3d.depth.depth_anything import DepthMap
from de3d.depth.metric import MetricScale, depth_from_prior, fit_ground_scale
from de3d.detect.base import Detection
from de3d.geometry.box3d import Box3D
from de3d.geometry.builder import DepthContext, build_box3d, estimate_yaw
from de3d.geometry.priors import ClassPrior


def _camera() -> PinholeCamera:
    return PinholeCamera(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0, width=1280, height=720)


def _depth_map(disparity: np.ndarray, shape: tuple[int, int]) -> DepthMap:
    """Wrap a disparity array that is already at the original resolution."""
    return DepthMap(
        disparity=disparity.astype(np.float32),
        scale_x=disparity.shape[1] / float(shape[1]),
        scale_y=disparity.shape[0] / float(shape[0]),
    )


def _ground_disparity(camera, height=1.5, a=1.4, b=0.02, shape=(720, 1280)):
    """Build the disparity map of a flat ground plane seen by ``camera``."""
    ys = np.arange(shape[0])
    xs = np.arange(shape[1])
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    pixels = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float64)
    rays = camera.unproject(pixels)
    depth = camera.ground_depth(rays, height)
    with np.errstate(divide="ignore", invalid="ignore"):
        disparity = (1.0 / depth - b) / a
    disparity = disparity.reshape(shape)
    disparity[~np.isfinite(disparity)] = 0.0
    return disparity.astype(np.float64)


def test_fit_ground_scale_recovers_the_affine_map() -> None:
    camera = _camera()
    disparity = _ground_disparity(camera)
    scale = fit_ground_scale(_depth_map(disparity, disparity.shape), camera, 1.5)
    assert scale is not None
    assert scale.valid
    assert scale.a == pytest.approx(1.4, rel=1e-3)
    assert scale.b == pytest.approx(0.02, abs=5e-3)


def test_metric_depth_matches_the_ground_plane() -> None:
    camera = _camera()
    disparity = _ground_disparity(camera)
    depth_map = _depth_map(disparity, disparity.shape)
    scale = fit_ground_scale(depth_map, camera, 1.5)
    assert scale is not None
    metric = scale.apply(disparity)

    row = 600
    col = 640
    ray = camera.unproject(np.array([[col, row]], np.float64))
    expected = camera.ground_depth(ray, 1.5)[0]
    assert metric[row, col] == pytest.approx(expected, rel=0.02)


def test_fit_ground_scale_returns_none_without_ground() -> None:
    camera = _camera()
    disparity = np.zeros((720, 1280), np.float64)
    depth_map = _depth_map(disparity, disparity.shape)
    assert fit_ground_scale(depth_map, camera, 1.5) is None


def test_depth_from_prior() -> None:
    camera = _camera()
    # A 1.7 m tall person, 200 px high -> Z = f * 1.7 / 200.
    bbox = np.array([600.0, 400.0, 700.0, 600.0])
    depth = depth_from_prior(camera, bbox, 1.7, 1.5)
    assert depth == pytest.approx(1000.0 * 1.7 / 200.0, rel=0.02)


def test_estimate_yaw_is_zero_for_a_front_facing_box() -> None:
    assert estimate_yaw(1.8, 1.85, 4.6) == pytest.approx(0.0, abs=1e-6)


def test_estimate_yaw_finds_the_side_view() -> None:
    # A car seen from the side is 4.6 m wide on screen; the front view is
    # 1.85 m.  The estimator returns the smaller of the two solutions.
    yaw = estimate_yaw(4.6, 1.85, 4.6)
    assert yaw < np.arctan2(4.6, 1.85)
    assert 1.85 * np.cos(yaw) + 4.6 * np.sin(yaw) == pytest.approx(4.6, rel=1e-6)


def test_estimate_yaw_saturates_at_the_diagonal() -> None:
    yaw = estimate_yaw(10.0, 1.85, 4.6)
    assert yaw == pytest.approx(np.arctan2(4.6, 1.85), rel=1e-6)


def test_box3d_corners_are_consistent() -> None:
    box = Box3D(
        category="car",
        score=0.9,
        center=np.array([0.0, 0.0, 10.0]),
        dims=np.array([2.0, 1.5, 4.0]),
    )
    corners = box.corners
    assert corners.shape == (8, 3)
    extent = corners.max(axis=0) - corners.min(axis=0)
    assert extent == pytest.approx(np.array([2.0, 1.5, 4.0]), rel=1e-9)
    assert np.allclose(corners.mean(axis=0), box.center, atol=1e-9)


def test_builder_places_a_grounded_object() -> None:
    camera = _camera()
    prior = ClassPrior(dims=(0.6, 1.7, 0.4), height=1.7, grounded=True, weight=1.0)
    # Contact point 5 m in front of the camera, on the ground.
    contact_px = camera.project(np.array([[0.0, 1.5, 5.0]]))[0]
    detection = Detection(
        x1=contact_px[0] - 60,
        y1=contact_px[1] - 170,
        x2=contact_px[0] + 60,
        y2=contact_px[1],
        score=0.9,
        class_id=0,
        class_name="person",
    )
    context = DepthContext(camera=camera, camera_height=1.5)
    box = build_box3d(detection, prior, context, depth_mode="ground", aspect_refine=0.0)
    assert box is not None
    assert box.depth == pytest.approx(5.0, rel=0.02)
    assert box.depth_source == "ground"
    # The centre sits half the object height above the contact point.
    assert box.center[1] == pytest.approx(1.5 - 0.85, abs=1e-6)
    assert box.center[2] == pytest.approx(5.0, rel=0.02)
    # The projected corners fall inside the 2D box.
    projected = camera.project(box.corners)
    assert projected[:, 0].min() >= detection.x1 - 40
    assert projected[:, 0].max() <= detection.x2 + 40


def test_builder_uses_the_model_depth_for_suspended_objects() -> None:
    camera = _camera()
    prior = ClassPrior(dims=(0.22, 0.22, 0.22), height=0.22, grounded=False, weight=0.2)
    raw = np.full((720, 1280), 0.05, np.float64)
    depth_map = _depth_map(raw, raw.shape)
    scale = MetricScale(a=1.0, b=0.0, samples=100)
    detection = Detection(600, 300, 660, 360, 0.8, 32, "sports ball")
    context = DepthContext(
        camera=camera,
        camera_height=1.5,
        depth_provider=lambda: (depth_map, scale),
    )
    box = build_box3d(detection, prior, context, depth_mode="model", aspect_refine=0.0)
    assert box is not None
    assert box.depth == pytest.approx(20.0, rel=1e-6)
    assert box.depth_source == "model"


def test_builder_returns_none_without_depth() -> None:
    camera = _camera()
    prior = ClassPrior(dims=(0.6, 1.7, 0.4), height=1.7, grounded=True, weight=1.0)
    detection = Detection(600, 300, 660, 360, 0.8, 0, "person")
    context = DepthContext(camera=camera, camera_height=0.0, depth_provider=None)
    # An explicit "model" request must not silently fall back.
    assert build_box3d(detection, prior, context, depth_mode="model", aspect_refine=0.0) is None
    # "auto" falls back to the class height prior.
    box = build_box3d(detection, prior, context, depth_mode="auto", aspect_refine=0.0)
    assert box is not None
    assert box.depth_source == "prior"
