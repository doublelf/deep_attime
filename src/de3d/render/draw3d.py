"""Draw the subject's 3D cuboid and the information panel.

The cuboid is drawn as a pure wireframe: the twelve real edges are stroked in
white and the eight corners are marked with a blue disc edged in white.  The
footprint of the box is additionally projected onto the ground.

Every face is a rectangle: the edges are derived from the face definitions in
:mod:`de3d.geometry.box3d`, so no diagonal can appear.
"""

from __future__ import annotations

import numpy as np

from ..camera.base import CameraModel, drop_to_ground
from ..geometry.box3d import CUBE_EDGES, CUBE_FACES, Box3D

__all__ = [
    "draw_box3d",
    "draw_ground_shadow",
    "draw_info_panel",
    "info_panel_lines",
]

# BGR colours
_EDGE_COLOR = (255, 255, 255)  # white wireframe
_VERTEX_EDGE_COLOR = (255, 255, 255)  # white rim of the vertex marker
_VERTEX_FILL_COLOR = (255, 0, 0)  # blue core of the vertex marker (BGR)
_GROUND_COLOR = (110, 190, 120)
_PANEL_COLOR = (25, 25, 28)
_PANEL_BORDER = (90, 200, 255)
_TEXT_COLOR = (245, 245, 245)
_ACCENT_COLOR = (90, 200, 255)  # first line of the panel

_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX


def info_panel_lines(box: Box3D | None) -> list[str]:
    """The three lines shown in the top-left panel."""
    if box is None:
        return ["no subject"]
    return [
        f"{box.category}   conf {box.score:.2f}",
        f"dist {box.depth:.2f} m",
        f"size {box.dims[0]:.2f} x {box.dims[1]:.2f} x {box.dims[2]:.2f} m",
    ]


def draw_info_panel(
    frame: np.ndarray,
    lines: list[str],
    origin: tuple[int, int] = (24, 34),
    line_height: int = 26,
    alpha: float = 0.55,
) -> np.ndarray:
    """Draw a translucent panel with the detection information."""
    import cv2

    if not lines:
        return frame
    scale = 0.62
    thickness = 1
    sizes = [cv2.getTextSize(text, _FONT, scale, thickness)[0] for text in lines]
    text_width = max(size[0] for size in sizes)
    box_width = text_width + 28
    box_height = line_height * len(lines) + 18
    x, y = origin
    top_left = (x - 14, y - 22)
    bottom_right = (x - 14 + box_width, y - 22 + box_height)

    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, _PANEL_COLOR, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, dst=frame)
    cv2.rectangle(frame, top_left, bottom_right, _PANEL_BORDER, 1, lineType=cv2.LINE_AA)

    for index, text in enumerate(lines):
        color = _ACCENT_COLOR if index == 0 else _TEXT_COLOR
        cv2.putText(
            frame,
            text,
            (x, y + index * line_height),
            _FONT,
            scale,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
    return frame


def draw_box3d(
    frame: np.ndarray,
    box: Box3D,
    camera: CameraModel,
    camera_height: float = 0.0,
    pitch_deg: float = 0.0,
    thickness: int = 3,
    vertex_radius: int = 5,
    draw_ground: bool = True,
) -> np.ndarray:
    """Draw the cuboid as a white wireframe with marked corners.

    Parameters
    ----------
    thickness:
        Stroke width of the wireframe, in pixels.
    vertex_radius:
        Radius of the corner markers, in pixels.  The blue core is drawn with
        ``radius - 2`` so the white rim stays visible.
    """
    import cv2

    corners_px = camera.project(box.corners)
    if not np.all(np.isfinite(corners_px)):
        return frame
    height, width = frame.shape[:2]
    if np.any(corners_px[:, 0] < -width) or np.any(corners_px[:, 0] > 2 * width):
        return frame
    if np.any(corners_px[:, 1] < -height) or np.any(corners_px[:, 1] > 2 * height):
        return frame

    # 1) the footprint on the ground
    if draw_ground:
        draw_ground_shadow(frame, box, camera, camera_height, pitch_deg)

    # 2) the twelve edges of the wireframe
    for a, b in CUBE_EDGES:
        p1 = (int(round(corners_px[a][0])), int(round(corners_px[a][1])))
        p2 = (int(round(corners_px[b][0])), int(round(corners_px[b][1])))
        cv2.line(frame, p1, p2, _EDGE_COLOR, thickness, lineType=cv2.LINE_AA)

    # 3) the eight corners: blue disc with a white rim
    core_radius = max(vertex_radius - 2, 1)
    for u, v in corners_px:
        centre = (int(round(u)), int(round(v)))
        cv2.circle(
            frame, centre, vertex_radius, _VERTEX_EDGE_COLOR, -1, lineType=cv2.LINE_AA
        )
        cv2.circle(
            frame, centre, core_radius, _VERTEX_FILL_COLOR, -1, lineType=cv2.LINE_AA
        )

    return frame


def draw_ground_shadow(
    frame: np.ndarray,
    box: Box3D,
    camera: CameraModel,
    camera_height: float = 0.0,
    pitch_deg: float = 0.0,
    alpha: float = 0.25,
) -> np.ndarray:
    """Draw the projection of the cuboid's bottom face onto the ground."""
    import cv2

    if camera_height <= 0:
        return frame
    # The bottom face, in ring order (see CUBE_FACES), so the polygon is a
    # proper rectangle instead of a crossed one.
    bottom = box.corners[list(CUBE_FACES[1])]
    on_ground = drop_to_ground(camera, bottom, camera_height, pitch_deg)
    pixels = camera.project(on_ground)
    if not np.all(np.isfinite(pixels)):
        return frame

    polygon = np.array([[int(round(u)), int(round(v))] for u, v in pixels], np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], _GROUND_COLOR)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, dst=frame)
    cv2.polylines(frame, [polygon], True, _GROUND_COLOR, 1, lineType=cv2.LINE_AA)
    return frame
