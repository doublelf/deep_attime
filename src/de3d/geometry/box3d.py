"""The 3D cuboid produced for the frame's subject."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["Box3D", "CUBE_EDGES", "CUBE_FACES", "FACE_NORMALS", "Face"]

# The eight corners of the cube, in the camera frame (``+x`` right, ``+y``
# down, ``+z`` towards the scene).  The loop order fixes the meaning of
# every index:
#
#   sy = -1 -> top,   +1 -> bottom
#   sz = -1 -> near,  +1 -> far
#   sx = -1 -> left,  +1 -> right
#
#   v0 = top-near-left      v1 = top-near-right
#   v2 = top-far-left       v3 = top-far-right
#   v4 = bottom-near-left   v5 = bottom-near-right
#   v6 = bottom-far-left    v7 = bottom-far-right
#
# The six faces are listed with their corners in counter-clockwise order, so
# every face is a proper rectangle and the edges follow from them.
CUBE_FACES: tuple[tuple[int, int, int, int], ...] = (
    (0, 1, 3, 2),  # top
    (4, 5, 7, 6),  # bottom
    (0, 1, 5, 4),  # near   (facing the camera)
    (2, 3, 7, 6),  # far
    (0, 2, 6, 4),  # left
    (1, 3, 7, 5),  # right
)

# Outward normals of the faces above, in the box's local frame.
FACE_NORMALS: tuple[tuple[float, float, float], ...] = (
    (0.0, -1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 0.0, 1.0),
    (-1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
)

FACE_NAMES: tuple[str, ...] = ("top", "bottom", "near", "far", "left", "right")

# The twelve real edges, derived from the faces so that no diagonal can slip
# in: every edge belongs to two faces and has one of the box dimensions.
CUBE_EDGES: tuple[tuple[int, int], ...] = tuple(
    sorted(
        {
            tuple(sorted((face[i], face[(i + 1) % 4])))
            for face in CUBE_FACES
            for i in range(4)
        }
    )
)


@dataclass
class Face:
    """One rectangular face of a cuboid."""

    index: int
    name: str
    indices: tuple[int, int, int, int]
    corners: np.ndarray  # (4, 3) camera frame, metres
    normal: np.ndarray  # (3,) outward normal, camera frame
    center: np.ndarray  # (3,) face centre, camera frame

    @property
    def visible(self) -> bool:
        """True when the face points towards the camera.

        The camera sits at the origin of the camera frame, so a face is
        visible when its outward normal points back at the origin.
        """
        return bool(np.dot(self.normal, self.center) < 0.0)


def _unit_corners() -> np.ndarray:
    """The eight cube corners in the camera frame (yaw = 0).

    ``x`` grows to the right, ``y`` grows downwards, ``z`` grows towards the
    scene, so the ground plane is at ``y = +h/2``.
    """
    half = 0.5
    corners = []
    for sy in (-1.0, 1.0):  # top / bottom
        for sz in (-1.0, 1.0):  # near / far
            for sx in (-1.0, 1.0):  # left / right
                corners.append((sx * half, sy * half, sz * half))
    return np.array(corners, np.float64)


UNIT_CORNERS = _unit_corners()


@dataclass
class Box3D:
    """A metric 3D cuboid in the camera frame."""

    category: str
    score: float
    center: np.ndarray  # (3,) camera frame, metres
    dims: np.ndarray  # (3,) width, height, length in metres
    yaw: float = 0.0
    depth: float = 0.0
    depth_source: str = "unknown"
    bbox2d: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    locked: bool = False
    lost_frames: int = 0
    corners: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, np.float64).reshape(3)
        self.dims = np.asarray(self.dims, np.float64).reshape(3)
        self.corners = self.compute_corners()

    def compute_corners(self) -> np.ndarray:
        """Return the eight corners of the cuboid in the camera frame."""
        width, height, length = self.dims
        local = UNIT_CORNERS * np.array([width, height, length], np.float64)
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], np.float64)
        return local @ rotation.T + self.center

    def faces(self) -> list[Face]:
        """Return the six rectangular faces with normals and visibility."""
        corners = self.corners
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], np.float64)
        faces: list[Face] = []
        for index, (indices, normal) in enumerate(zip(CUBE_FACES, FACE_NORMALS, strict=True)):
            face_corners = corners[list(indices)]
            world_normal = np.asarray(normal, np.float64) @ rotation.T
            faces.append(
                Face(
                    index=index,
                    name=FACE_NAMES[index],
                    indices=indices,
                    corners=face_corners,
                    normal=world_normal,
                    center=face_corners.mean(axis=0),
                )
            )
        return faces

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return {
            "category": self.category,
            "score": float(self.score),
            "bbox2d": [float(v) for v in self.bbox2d],
            "center_cam": [float(v) for v in self.center],
            "dims": [float(v) for v in self.dims],
            "yaw": float(self.yaw),
            "depth": float(self.depth),
            "depth_source": self.depth_source,
            "locked": bool(self.locked),
            "lost_frames": int(self.lost_frames),
            "corners_cam": [[float(c) for c in row] for row in self.corners],
        }
