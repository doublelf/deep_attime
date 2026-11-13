"""Camera height estimation and the fisheye lens chain."""

from __future__ import annotations

import numpy as np
import pytest

from de3d.camera.fisheye import EquidistantFisheyeCamera
from de3d.camera.pinhole import PinholeCamera
from de3d.depth.height import CameraHeightEstimator


def _person_box(camera: PinholeCamera, depth: float, height: float, world_height: float):
    """Pixel box of a person of ``height`` standing ``depth`` metres away."""
    focal = camera.fy
    pixel_height = focal * height / depth
    foot_v = camera.cy + focal * world_height / depth
    top_v = foot_v - pixel_height
    return np.array(
        [
            camera.cx - pixel_height * 0.25,
            top_v,
            camera.cx + pixel_height * 0.25,
            foot_v,
        ]
    )


def test_height_estimator_recovers_the_camera_height() -> None:
    camera = PinholeCamera(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0, width=1280, height=720)
    truth = 1.15
    estimator = CameraHeightEstimator(camera, window=15)
    for depth in (3.0, 4.0, 5.5, 6.0):
        box = _person_box(camera, depth, 1.75, truth)
        estimator.update(box, 1.75)
    assert estimator.height == pytest.approx(truth, rel=0.02)


def test_height_estimator_ignores_tiny_objects() -> None:
    camera = PinholeCamera(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0, width=1280, height=720)
    estimator = CameraHeightEstimator(camera)
    assert estimator.update(np.array([0.0, 0.0, 10.0, 10.0]), 1.75) is None
    assert estimator.height is None


def test_fisheye_central_ray_is_the_optical_axis() -> None:
    camera = EquidistantFisheyeCamera(
        fx=220.0, fy=220.0, cx=240.0, cy=240.0, width=480, height=480
    )
    ray = camera.unproject(np.array([[240.0, 240.0]]))[0]
    assert np.allclose(ray, [0.0, 0.0, 1.0], atol=1e-9)


def test_fisheye_equidistant_geometry() -> None:
    """Without distortion the model reduces to a pinhole projection.

    The rig's model applies the radial polynomial to the angle but keeps the
    tangent law of the pinhole projection, so ``r / f = tan(theta)``.
    """
    camera = EquidistantFisheyeCamera(
        fx=220.0, fy=220.0, cx=240.0, cy=240.0, width=480, height=480
    )
    radius = 102.0
    pixel = np.array([[240.0, 240.0 + radius]])
    ray = camera.unproject(pixel)[0]
    angle = np.arctan2(np.hypot(ray[0], ray[1]), ray[2])
    assert angle == pytest.approx(np.arctan(radius / 220.0), rel=1e-6)


def test_fisheye_distortion_expands_the_radius() -> None:
    """The radial polynomial pushes the pixel further out than the pinhole."""
    plain = EquidistantFisheyeCamera(
        fx=220.0, fy=220.0, cx=240.0, cy=240.0, width=480, height=480
    )
    distorted = EquidistantFisheyeCamera(
        fx=220.0,
        fy=220.0,
        cx=240.0,
        cy=240.0,
        width=480,
        height=480,
        k1=0.02,
        k2=-0.005,
        k3=0.0005,
    )
    direction = np.array([[0.0, 0.3, 1.0]])
    assert distorted.project(direction)[0][1] > plain.project(direction)[0][1]


def test_fisheye_and_pinhole_agree_in_the_centre() -> None:
    """A fisheye and a pinhole with the same focal length share the centre."""
    pinhole = PinholeCamera(fx=220.0, fy=220.0, cx=240.0, cy=240.0, width=480, height=480)
    fisheye = EquidistantFisheyeCamera(
        fx=220.0, fy=220.0, cx=240.0, cy=240.0, width=480, height=480
    )
    pixels = np.array([[240.0, 240.0], [240.0, 260.0], [260.0, 240.0]])
    a = pinhole.unproject(pixels)
    b = fisheye.unproject(pixels)
    # The models differ away from the axis (tan vs the equidistant law) but
    # must stay close for small angles.
    assert np.allclose(a, b, atol=0.02)


def test_fisheye_box_projects_back_into_the_2d_box() -> None:
    from de3d.detect.base import Detection
    from de3d.geometry.box3d import Box3D

    camera = EquidistantFisheyeCamera(
        fx=220.0,
        fy=220.0,
        cx=240.0,
        cy=240.0,
        width=480,
        height=480,
        k1=0.02,
        k2=-0.005,
        k3=0.0005,
    )
    depth = 4.0
    height = 1.75
    ray = camera.unproject(np.array([[240.0, 300.0]]))[0]
    focal = camera.fy
    pixel_height = focal * height / depth
    foot = camera.project(np.array([[0.0, 1.0, depth]]))[0]
    detection = Detection(
        foot[0] - pixel_height * 0.2,
        foot[1] - pixel_height,
        foot[0] + pixel_height * 0.2,
        foot[1],
        0.9,
        0,
        "person",
    )
    box = Box3D(
        category="person",
        score=0.9,
        center=np.array([0.0, 1.0 - height / 2.0, depth]),
        dims=np.array([0.6, height, 0.4]),
    )
    projected = camera.project(box.corners)
    assert projected[:, 1].min() >= detection.y1 - 30
    assert projected[:, 1].max() <= detection.y2 + 30
    assert np.all(np.isfinite(projected))
    assert ray[2] > 0
