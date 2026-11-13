"""Rendering: rectangular faces, real edges, panel and the solid look."""

from __future__ import annotations

import numpy as np
import pytest

from de3d.camera.pinhole import PinholeCamera
from de3d.geometry.box3d import CUBE_EDGES, Box3D
from de3d.render.draw3d import (
    draw_box3d,
    draw_ground_shadow,
    draw_info_panel,
    info_panel_lines,
)


def _box(yaw: float = 0.0) -> Box3D:
    return Box3D(
        category="person",
        score=0.91,
        center=np.array([0.1, -0.2, 3.0]),
        dims=np.array([0.6, 1.7, 0.4]),
        yaw=yaw,
        depth=3.0,
        depth_source="ground",
    )


def _camera() -> PinholeCamera:
    return PinholeCamera(fx=800.0, fy=800.0, cx=640.0, cy=360.0, width=1280, height=720)


def test_there_are_twelve_real_edges() -> None:
    assert len(CUBE_EDGES) == 12
    assert len(set(CUBE_EDGES)) == 12


def test_no_edge_is_a_face_diagonal() -> None:
    """Every edge must connect two corners that share a face."""
    box = _box()
    corners = box.corners
    dimensions = {round(float(v), 6) for v in box.dims}
    for a, b in CUBE_EDGES:
        length_ab = float(np.linalg.norm(corners[a] - corners[b]))
        # A real edge has exactly one of the box dimensions; a diagonal would
        # be longer than the biggest dimension.
        assert round(length_ab, 6) in dimensions
        assert length_ab <= max(box.dims) + 1e-6


def test_every_face_is_a_rectangle() -> None:
    box = _box()
    for face in box.faces():
        corners = face.corners
        assert corners.shape == (4, 3)
        edges = [corners[(i + 1) % 4] - corners[i] for i in range(4)]
        lengths = [float(np.linalg.norm(edge)) for edge in edges]
        # Opposite sides are equal ...
        assert lengths[0] == pytest.approx(lengths[2], rel=1e-9)
        assert lengths[1] == pytest.approx(lengths[3], rel=1e-9)
        # ... and the adjacent sides are perpendicular.
        for i in range(4):
            dot = float(np.dot(edges[i], edges[(i + 1) % 4]))
            assert dot == pytest.approx(0.0, abs=1e-9)


def test_face_lengths_match_the_dimensions() -> None:
    box = _box()
    for face in box.faces():
        corners = face.corners
        lengths = {
            round(float(np.linalg.norm(corners[(i + 1) % 4] - corners[i])), 6)
            for i in range(4)
        }
        assert lengths <= {0.6, 1.7, 0.4}


def test_faces_are_rectangles_for_any_yaw() -> None:
    for yaw in (0.0, 0.3, 1.2, -2.5):
        box = _box(yaw)
        for face in box.faces():
            corners = face.corners
            edges = [corners[(i + 1) % 4] - corners[i] for i in range(4)]
            for i in range(4):
                dot = float(np.dot(edges[i], edges[(i + 1) % 4]))
                assert dot == pytest.approx(0.0, abs=1e-9)


def test_face_normals_point_outwards() -> None:
    box = _box()
    for face in box.faces():
        # The outward normal points away from the box centre.
        offset = face.center - box.center
        assert float(np.dot(face.normal, offset)) > 0


def test_visibility_matches_the_camera_position() -> None:
    """The camera sits at the origin of the camera frame (``+y`` points down)."""
    # A box standing on the ground in front of the camera: only the face that
    # points at the camera is visible.
    box = _box()
    visible = {face.name for face in box.faces() if face.visible}
    assert "near" in visible
    assert "far" not in visible

    # A box well below the camera: the top face becomes visible as well.
    lower = Box3D(
        category="car",
        score=0.8,
        center=np.array([0.0, 1.5, 4.0]),
        dims=np.array([1.8, 1.5, 4.5]),
    )
    visible = {face.name for face in lower.faces() if face.visible}
    assert "top" in visible
    assert "bottom" not in visible
    assert "near" in visible


def test_draw_box3d_keeps_the_frame_size() -> None:
    camera = _camera()
    frame = np.zeros((720, 1280, 3), np.uint8)
    out = draw_box3d(frame, _box(), camera, camera_height=1.0)
    assert out.shape == frame.shape
    assert out.any()  # something was drawn


def test_draw_box3d_does_not_fill_the_faces() -> None:
    """Only the wireframe and the corner markers are drawn, no face fill."""
    camera = _camera()
    frame = np.zeros((720, 1280, 3), np.uint8)
    out = draw_box3d(frame, _box(), camera, camera_height=1.0, draw_ground=False)
    # Every drawn pixel must be part of a thin stroke, never a filled area.
    coloured = np.any(out > 40, axis=2)
    assert coloured.sum() < 0.02 * coloured.size
    # A point in the middle of the front face stays untouched.
    centre = out[400, 900]
    assert not centre.any()


def test_vertices_are_marked_with_blue_circles() -> None:
    """Each corner carries a blue disc with a white rim."""
    camera = _camera()
    frame = np.zeros((720, 1280, 3), np.uint8)
    out = draw_box3d(frame, _box(), camera, camera_height=1.0, draw_ground=False)
    corners = camera.project(_box().corners)
    for u, v in corners:
        x, y = int(round(u)), int(round(v))
        patch = out[y - 3 : y + 4, x - 3 : x + 4]
        blue = (patch[:, :, 0] > 150) & (patch[:, :, 1] < 100) & (patch[:, :, 2] < 100)
        assert blue.any(), f"no blue core at corner ({x}, {y})"
        white = (patch[:, :, 0] > 200) & (patch[:, :, 1] > 200) & (patch[:, :, 2] > 200)
        assert white.any(), f"no white rim at corner ({x}, {y})"


def test_edge_thickness_parameter() -> None:
    """A thicker wireframe paints more pixels."""
    camera = _camera()
    box = _box()
    thin = draw_box3d(
        np.zeros((720, 1280, 3), np.uint8),
        box,
        camera,
        camera_height=1.0,
        draw_ground=False,
        thickness=1,
        vertex_radius=0,
    )
    thick = draw_box3d(
        np.zeros((720, 1280, 3), np.uint8),
        box,
        camera,
        camera_height=1.0,
        draw_ground=False,
        thickness=6,
        vertex_radius=0,
    )
    thin_pixels = int((np.any(thin > 40, axis=2)).sum())
    thick_pixels = int((np.any(thick > 40, axis=2)).sum())
    assert thick_pixels > thin_pixels * 1.5


def test_vertex_radius_parameter() -> None:
    """A larger radius paints a larger blue disc."""
    camera = _camera()
    box = _box()
    small = draw_box3d(
        np.zeros((720, 1280, 3), np.uint8),
        box,
        camera,
        camera_height=1.0,
        draw_ground=False,
        vertex_radius=3,
    )
    large = draw_box3d(
        np.zeros((720, 1280, 3), np.uint8),
        box,
        camera,
        camera_height=1.0,
        draw_ground=False,
        vertex_radius=10,
    )
    def blue(img: np.ndarray) -> int:
        return int(((img[:, :, 0] > 150) & (img[:, :, 1] < 100)).sum())

    assert blue(large) > blue(small) * 1.5


def test_ground_shadow_is_drawn_on_the_ground() -> None:
    camera = _camera()
    frame = np.zeros((720, 1280, 3), np.uint8)
    out = draw_ground_shadow(frame, _box(), camera, camera_height=1.0)
    assert out.any()


def test_info_panel_lines() -> None:
    lines = info_panel_lines(_box())
    assert len(lines) == 3
    assert "person" in lines[0]
    assert "conf 0.91" in lines[0]
    assert "dist" in lines[1]
    assert "size" in lines[2]
    assert info_panel_lines(None) == ["no subject"]


def test_draw_info_panel_keeps_the_frame_size() -> None:
    frame = np.zeros((720, 1280, 3), np.uint8)
    out = draw_info_panel(frame, info_panel_lines(_box()))
    assert out.shape == frame.shape
    # The panel must be in the top-left corner.
    assert out[:120, :400].any()
    assert not out[300:, 800:].any()
