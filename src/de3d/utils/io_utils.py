"""Small helpers for video I/O and JSON output."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["VideoInfo", "open_video", "iter_frames", "write_json", "VideoWriter"]


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int
    path: str


def is_image(path: str | Path) -> bool:
    """True when ``path`` points at a still image."""
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def open_video(path: str | Path) -> tuple[Any, VideoInfo]:
    """Open a video and return ``(capture, info)``."""
    import cv2

    path = str(path)
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    info = VideoInfo(
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(capture.get(cv2.CAP_PROP_FPS)) or 25.0,
        frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        path=path,
    )
    return capture, info


def open_image(path: str | Path) -> tuple[np.ndarray, VideoInfo]:
    """Open a still image and return ``(frame, info)``."""
    import cv2

    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(f"cannot open image: {path}")
    height, width = frame.shape[:2]
    return frame, VideoInfo(
        width=width, height=height, fps=25.0, frames=1, path=str(path)
    )


def iter_frames(
    capture: Any,
    stride: int = 1,
    max_frames: int = 0,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, frame)`` pairs."""
    stride = max(int(stride), 1)
    index = 0
    emitted = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride == 0:
            yield index, frame
            emitted += 1
            if max_frames and emitted >= max_frames:
                break
        index += 1


class VideoWriter:
    """Thin wrapper around ``cv2.VideoWriter``."""

    def __init__(self, path: str | Path, width: int, height: int, fps: float) -> None:
        import cv2

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self.path), fourcc, float(fps), (width, height))

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def write_json(path: str | Path, data: Any) -> None:
    """Write ``data`` as UTF-8 JSON with a stable key order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
