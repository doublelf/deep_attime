"""ONNX Runtime wrapper around Depth-Anything V2 (small).

The checkpoint is the community ONNX export of ``depth-anything-v2-small``
(``onnx-community/depth-anything-v2-small``), fetched through a mirror when
``huggingface.co`` is not reachable.  It runs on CPU through ONNX Runtime,
which keeps the whole pipeline free of PyTorch at inference time and makes the
later RKNN conversion a one-liner.

The model predicts an *affine invariant disparity* (larger = closer).  The
metric conversion lives in :mod:`de3d.depth.metric`.

The exported graph interpolates its position embeddings, so the input size is
free.  The image is resized so that the long side matches ``size`` and both
sides are multiples of 14 (the patch size); the aspect ratio is preserved,
which keeps the token count as low as possible.

The disparity is returned at the model's native resolution together with the
scale factors needed to look it up in *original* image coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

__all__ = ["DepthAnything", "DepthMap", "LazyDepthAnything"]

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
_PATCH = 14


@dataclass
class DepthMap:
    """Raw disparity plus the mapping back to original image pixels."""

    disparity: np.ndarray  # (h, w) float32, native model resolution
    scale_x: float  # disparity_x = original_x * scale_x
    scale_y: float  # disparity_y = original_y * scale_y

    @property
    def shape(self) -> tuple[int, int]:
        return self.disparity.shape

    def sample(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Nearest-neighbour lookup of the disparity at original pixels."""
        height, width = self.disparity.shape
        xs = np.clip(np.rint(np.asarray(u, np.float64) * self.scale_x).astype(np.int64), 0, width - 1)
        ys = np.clip(np.rint(np.asarray(v, np.float64) * self.scale_y).astype(np.int64), 0, height - 1)
        return self.disparity[ys, xs]

    def region_median(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Median disparity inside an original-image pixel region."""
        height, width = self.disparity.shape
        ix1 = int(np.clip(np.floor(x1 * self.scale_x), 0, width - 1))
        ix2 = int(np.clip(np.ceil(x2 * self.scale_x), 0, width - 1))
        iy1 = int(np.clip(np.floor(y1 * self.scale_y), 0, height - 1))
        iy2 = int(np.clip(np.ceil(y2 * self.scale_y), 0, height - 1))
        if ix2 <= ix1 or iy2 <= iy1:
            return float("nan")
        patch = self.disparity[iy1 : iy2 + 1, ix1 : ix2 + 1]
        valid = patch[np.isfinite(patch)]
        if valid.size == 0:
            return float("nan")
        return float(np.median(valid))


class DepthAnything:
    """Depth-Anything V2 (small) running on ONNX Runtime.

    Parameters
    ----------
    onnx_path:
        Path to ``model.onnx``.
    size:
        Long side of the network input, in pixels.
    threads:
        ONNX Runtime intra-op thread count.  On hybrid CPUs (P-cores + E-cores)
        a small number is often *faster* than using every core.
    providers:
        Execution providers; CPU by default.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        size: int = 518,
        threads: int = 4,
        providers: list[str] | None = None,
    ) -> None:
        import onnxruntime as ort

        self._path = Path(onnx_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(threads)
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(self._path),
            sess_options=options,
            providers=providers or ["CPUExecutionProvider"],
        )
        self._input = self._session.get_inputs()[0]
        self._output = self._session.get_outputs()[0]
        self._size = int(size)

    @property
    def input_shape(self) -> tuple:
        return tuple(self._input.shape)

    def _target_size(self, width: int, height: int) -> tuple[int, int]:
        scale = self._size / float(max(width, height))
        target_w = max(int(round(width * scale / _PATCH)) * _PATCH, _PATCH)
        target_h = max(int(round(height * scale / _PATCH)) * _PATCH, _PATCH)
        return target_w, target_h

    def infer(self, image_bgr: np.ndarray) -> DepthMap:
        """Return the raw disparity map of a BGR frame."""
        height, width = image_bgr.shape[:2]
        target_w, target_h = self._target_size(width, height)

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        blob = resized.astype(np.float32) / 255.0
        blob = (blob - _IMAGENET_MEAN) / _IMAGENET_STD
        blob = np.transpose(blob, (2, 0, 1))[None, ...].astype(np.float32)

        output = self._session.run([self._output.name], {self._input.name: blob})[0]
        disparity = np.asarray(output, np.float32).reshape(-1)
        side = int(round(np.sqrt(disparity.size)))
        if side * side == disparity.size:
            disparity = disparity.reshape(side, side)
        else:
            disparity = disparity.reshape(output.shape[-2], output.shape[-1])

        disparity = self._refine(disparity, rgb)
        return DepthMap(
            disparity=disparity,
            scale_x=disparity.shape[1] / float(width),
            scale_y=disparity.shape[0] / float(height),
        )

    @staticmethod
    def _refine(disparity: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        """Guided-filter the disparity map so its edges follow the image."""
        guide = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        src = disparity.astype(np.float32)
        try:
            import cv2.ximgproc as ximgproc

            return ximgproc.guidedFilter(
                guide, src, 4, 1e-3, borderType=cv2.BORDER_REFLECT
            ).astype(np.float32)
        except Exception:  # pragma: no cover - builds without contrib
            return cv2.bilateralFilter(
                src, 5, 0.2, 5, borderType=cv2.BORDER_REFLECT
            ).astype(np.float32)


class LazyDepthAnything:
    """Creates the ONNX Runtime session on the first inference.

    Grounded objects never need the depth model (their depth comes from the
    ground plane), so deferring the session creation saves ~90 MB of resident
    memory and keeps the detector's inference speed untouched.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        size: int = 518,
        threads: int = 4,
        providers: list[str] | None = None,
    ) -> None:
        self._path = Path(onnx_path)
        self._size = int(size)
        self._threads = int(threads)
        self._providers = providers
        self._model: DepthAnything | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def infer(self, image_bgr: np.ndarray) -> DepthMap:
        if self._model is None:
            self._model = DepthAnything(
                self._path,
                size=self._size,
                threads=self._threads,
                providers=self._providers,
            )
        return self._model.infer(image_bgr)
