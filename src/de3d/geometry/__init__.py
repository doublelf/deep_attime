"""Geometry helpers: metric cuboids and class priors."""

from .box3d import CUBE_EDGES, CUBE_FACES, FACE_NORMALS, Box3D, Face
from .builder import DepthContext, build_box3d, estimate_yaw
from .priors import ClassPrior, PriorTable

__all__ = [
    "Box3D",
    "Face",
    "CUBE_EDGES",
    "CUBE_FACES",
    "FACE_NORMALS",
    "DepthContext",
    "build_box3d",
    "estimate_yaw",
    "ClassPrior",
    "PriorTable",
]
