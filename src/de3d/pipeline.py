"""End-to-end pipeline: frame in, annotated frame + metric cuboid out."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .camera.base import CameraModel
from .depth.depth_anything import DepthMap
from .depth.height import CameraHeightEstimator
from .depth.metric import MetricScale, fit_ground_scale
from .detect.base import Detection
from .geometry.box3d import Box3D
from .geometry.builder import DepthContext, build_box3d
from .geometry.priors import PriorTable
from .render.draw3d import draw_box3d, draw_info_panel, info_panel_lines
from .subject.scorer import SubjectScorer
from .subject.tracker import SubjectTracker

__all__ = ["Pipeline", "PipelineConfig", "FrameResult"]


class _LazyDepth:
    """Runs the depth model on first use and caches the result."""

    def __init__(
        self,
        depth_model: Any,
        image: np.ndarray,
        camera: CameraModel,
        camera_height: float,
        pitch_deg: float,
        bottom_fraction: float,
    ) -> None:
        self._model = depth_model
        self._image = image
        self._camera = camera
        self._height = camera_height
        self._pitch = pitch_deg
        self._bottom_fraction = bottom_fraction
        self.map: DepthMap | None = None
        self.scale: MetricScale | None = None
        self.elapsed = 0.0
        self.fit_elapsed = 0.0

    def __call__(self) -> tuple[DepthMap | None, MetricScale | None]:
        if self.map is None:
            start = time.perf_counter()
            self.map = self._model.infer(self._image)
            self.elapsed = time.perf_counter() - start
            start = time.perf_counter()
            self.scale = fit_ground_scale(
                self.map,
                self._camera,
                self._height,
                self._pitch,
                bottom_fraction=self._bottom_fraction,
            )
            self.fit_elapsed = time.perf_counter() - start
        return self.map, self.scale


@dataclass
class PipelineConfig:
    """Static configuration of the pipeline."""

    camera: CameraModel
    camera_height: float
    priors: PriorTable
    scorer: SubjectScorer
    tracker: SubjectTracker
    pitch_deg: float = 0.0
    depth_mode: str = "auto"
    yaw_mode: str = "zero"
    aspect_refine: float = 0.5
    bottom_fraction: float = 0.35
    max_subjects: int = 1
    min_area_ratio: float = 0.003
    auto_height: bool = False
    box_thickness: int = 3
    vertex_radius: int = 5


@dataclass
class FrameResult:
    """Everything produced for a single frame."""

    index: int
    frame: np.ndarray
    box: Box3D | None = None
    detections: list[Detection] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    scale: MetricScale | None = None


class Pipeline:
    """Glue between the detector, the depth model and the 3D box builder."""

    def __init__(
        self,
        detector: Any,
        depth_model: Any,
        config: PipelineConfig,
    ) -> None:
        self._detector = detector
        self._depth = depth_model
        self._config = config
        self._height_estimator = CameraHeightEstimator(config.camera)

    # -- helpers --------------------------------------------------------
    @property
    def config(self) -> PipelineConfig:
        return self._config

    @property
    def camera_height_estimate(self) -> float | None:
        """Current camera-height estimate (``None`` when not auto calibrating)."""
        return self._height_estimator.height

    def _camera_for(self, width: int, height: int) -> CameraModel:
        return self._config.camera.scaled(width, height)

    # -- main entry point ------------------------------------------------
    def process(self, index: int, frame: np.ndarray) -> FrameResult:
        """Run the full pipeline on a single BGR frame."""
        cfg = self._config
        height, width = frame.shape[:2]
        camera = self._camera_for(width, height)
        timings: dict[str, float] = {}

        start = time.perf_counter()
        detections = self._detector.detect(frame)
        timings["detect"] = time.perf_counter() - start

        # -- subject selection + temporal lock ---------------------------
        start = time.perf_counter()
        filtered = cfg.scorer.filter_detections(detections, width, height)
        if not cfg.tracker.is_locked:
            candidates = cfg.scorer.select(filtered, width, height, 1)
            if candidates:
                cfg.tracker.acquire(candidates[0].detection)
        state = cfg.tracker.update(filtered, width, height)
        timings["subject"] = time.perf_counter() - start

        if state.box is None or state.detection is None:
            draw_info_panel(frame, info_panel_lines(None))
            return FrameResult(
                index=index,
                frame=frame,
                detections=detections,
                timings=timings,
            )

        # -- camera height ------------------------------------------------
        camera_height = cfg.camera_height
        if cfg.auto_height:
            prior = cfg.priors.get(state.detection.class_name)
            if prior.grounded:
                self._height_estimator.update(state.box, prior.height)
            estimate = self._height_estimator.height
            if estimate is not None:
                camera_height = estimate

        # -- depth -------------------------------------------------------
        # The depth model is only executed when its output is really needed:
        # objects that stand on the ground get their depth from the ground
        # plane, which is pure geometry and costs nothing.
        provider = _LazyDepth(
            depth_model=self._depth,
            image=frame,
            camera=camera,
            camera_height=camera_height,
            pitch_deg=cfg.pitch_deg,
            bottom_fraction=cfg.bottom_fraction,
        )

        # -- 3D box ------------------------------------------------------
        start = time.perf_counter()
        subject = Detection.from_box(
            state.box,
            score=state.detection.score,
            class_id=state.detection.class_id,
            class_name=state.detection.class_name,
        )
        prior = cfg.priors.get(subject.class_name)
        context = DepthContext(
            camera=camera,
            camera_height=camera_height,
            pitch_deg=cfg.pitch_deg,
            depth_provider=provider,
        )
        box = build_box3d(
            subject,
            prior,
            context,
            depth_mode=cfg.depth_mode,
            yaw_mode=cfg.yaw_mode,
            aspect_refine=cfg.aspect_refine,
        )
        timings["box3d"] = time.perf_counter() - start
        timings["depth"] = provider.elapsed
        timings["metric"] = provider.fit_elapsed

        if box is not None:
            box.locked = state.locked
            box.lost_frames = state.lost_frames

        # -- render ------------------------------------------------------
        start = time.perf_counter()
        if box is not None:
            draw_box3d(
                frame,
                box,
                camera,
                camera_height=camera_height,
                pitch_deg=cfg.pitch_deg,
                thickness=cfg.box_thickness,
                vertex_radius=cfg.vertex_radius,
            )
        draw_info_panel(frame, info_panel_lines(box))
        timings["render"] = time.perf_counter() - start

        return FrameResult(
            index=index,
            frame=frame,
            box=box,
            detections=detections,
            timings=timings,
            scale=provider.scale,
        )
