"""Command line entry point for the monocular 3D cuboid pipeline."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

from .camera import load_camera_profile
from .depth.depth_anything import LazyDepthAnything
from .detect.yolo2d import Yolo2DDetector
from .geometry.priors import PriorTable
from .pipeline import Pipeline, PipelineConfig
from .subject.scorer import SubjectScorer
from .subject.tracker import SubjectTracker
from .utils.io_utils import (
    VideoWriter,
    is_image,
    iter_frames,
    open_image,
    open_video,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = REPO_ROOT / "config"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract one metric 3D cuboid per frame from a monocular video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="video or image to process")
    parser.add_argument(
        "--camera",
        default=str(DEFAULT_CONFIG_DIR / "cameras" / "sample_1920x1080.yaml"),
        help="camera profile YAML",
    )
    parser.add_argument(
        "--priors",
        default=str(DEFAULT_CONFIG_DIR / "priors.yaml"),
        help="class prior YAML",
    )
    parser.add_argument(
        "--subject-config",
        default=str(DEFAULT_CONFIG_DIR / "subject.yaml"),
        help="subject scoring YAML",
    )
    parser.add_argument(
        "--subject-preset",
        default="generic",
        help="named weight preset from the subject config",
    )
    parser.add_argument(
        "--detector",
        default=str(REPO_ROOT / "models" / "yolo11n.pt"),
        help="ultralytics checkpoint",
    )
    parser.add_argument(
        "--detector-onnx",
        default=str(REPO_ROOT / "models" / "onnx" / "yolo11n.onnx"),
        help="ONNX Runtime detector (used when the file exists)",
    )
    parser.add_argument(
        "--depth-model",
        default=str(REPO_ROOT / "models" / "depth_anything_v2_small.onnx"),
        help="Depth-Anything V2 small ONNX file",
    )
    parser.add_argument(
        "--camera-height",
        default="auto",
        help="optical centre height above the ground in metres, or 'auto' to "
        "estimate it from grounded subjects",
    )
    parser.add_argument("--pitch", type=float, default=0.0, help="camera pitch (deg, down+)")
    parser.add_argument("--depth-mode", default="auto", choices=["auto", "ground", "model", "prior"])
    parser.add_argument("--yaw-mode", default="zero", choices=["zero", "ratio"])
    parser.add_argument("--aspect-refine", type=float, default=0.5)
    parser.add_argument("--bottom-fraction", type=float, default=0.35)
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=1,
        help="number of subjects per frame (the tracker currently follows one)",
    )
    parser.add_argument("--min-area-ratio", type=float, default=None)
    parser.add_argument("--classes", default=None, help="comma separated COCO class ids")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--depth-size", type=int, default=518, help="long side of the depth input")
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="ONNX Runtime intra-op threads (fewer is often faster on hybrid CPUs)",
    )
    parser.add_argument("--torch-threads", type=int, default=6)
    parser.add_argument("--no-track", action="store_true", help="disable the temporal lock")
    parser.add_argument("--max-lost-frames", type=int, default=10)
    parser.add_argument(
        "--box-thickness",
        type=int,
        default=3,
        help="wireframe stroke width in pixels",
    )
    parser.add_argument(
        "--vertex-radius",
        type=int,
        default=5,
        help="radius of the corner markers in pixels",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "outputs"))
    parser.add_argument("--json", default=None, help="path of the JSON report")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-video", action="store_true", help="do not write an annotated video")
    return parser


def _peak_rss_mb() -> float:
    """Peak resident set size of this process, in megabytes."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:  # pragma: no cover - psutil is optional
        return float("nan")


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    camera, profile = load_camera_profile(args.camera)
    priors = PriorTable.from_yaml(args.priors)
    subject_config = _load_yaml(args.subject_config)

    auto_height = str(args.camera_height).lower() == "auto"
    camera_height = 1.5 if auto_height else float(args.camera_height)

    scorer = SubjectScorer.from_config(
        subject_config,
        class_weights=priors.class_weights,
        preset=args.subject_preset,
    )
    if args.min_area_ratio is not None:
        scorer.min_area_ratio = args.min_area_ratio

    tracker = SubjectTracker(
        iou_threshold=float((subject_config.get("tracker") or {}).get("iou_threshold", 0.3)),
        center_distance_ratio=float(
            (subject_config.get("tracker") or {}).get("center_distance_ratio", 0.25)
        ),
        ema=float((subject_config.get("tracker") or {}).get("ema", 0.6)),
        max_lost_frames=args.max_lost_frames,
    )

    classes = None
    if args.classes:
        classes = [int(v) for v in args.classes.split(",") if v.strip()]
    if args.max_subjects != 1:
        print("[warn] the temporal tracker follows a single subject; using 1")

    detector = None
    onnx_detector = Path(args.detector_onnx)
    if onnx_detector.exists():
        # NOTE: do not import torch here.  PyTorch initialises an OpenMP
        # thread pool which fights with the ONNX Runtime spinning pool and
        # slows the detector down by ~6x (48 ms -> 300 ms per frame).
        from .detect.onnx_yolo import OnnxYoloDetector

        detector = OnnxYoloDetector(
            onnx_detector,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            threads=args.threads,
            classes=classes,
        )
    else:
        import torch

        torch.set_num_threads(max(1, args.torch_threads))
        detector = Yolo2DDetector(
            args.detector,
            device="cpu",
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            classes=classes,
        )
    depth_model = LazyDepthAnything(
        args.depth_model, size=args.depth_size, threads=args.threads
    )

    config = PipelineConfig(
        camera=camera,
        camera_height=camera_height,
        priors=priors,
        scorer=scorer,
        tracker=SubjectTracker() if args.no_track else tracker,
        pitch_deg=args.pitch,
        depth_mode=args.depth_mode,
        yaw_mode=args.yaw_mode,
        aspect_refine=args.aspect_refine,
        bottom_fraction=args.bottom_fraction,
        max_subjects=args.max_subjects,
        min_area_ratio=scorer.min_area_ratio,
        auto_height=auto_height,
        box_thickness=args.box_thickness,
        vertex_radius=args.vertex_radius,
    )
    pipeline = Pipeline(detector, depth_model, config)

    image_input = is_image(args.input)
    if image_input:
        frame, info = open_image(args.input)
        source = [(0, frame)]
        close_source = None
    else:
        capture, info = open_video(args.input)
        source = iter_frames(capture, args.stride, args.max_frames)
        close_source = capture

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = Path(args.json) if args.json else out_dir / "subject.json"
    writer = None
    if not image_input and not args.no_video and not args.no_render:
        writer = VideoWriter(
            out_dir / "annotated.mp4", info.width, info.height, info.fps
        )
    records: list[dict] = []
    started = time.perf_counter()
    frames = 0
    peak_rss = 0.0
    for index, frame in source:
        result = pipeline.process(index, frame)
        frames += 1
        peak_rss = max(peak_rss, _peak_rss_mb())
        if writer is not None:
            writer.write(result.frame)
        elif image_input:
            import cv2

            cv2.imwrite(str(out_dir / "annotated.png"), result.frame)
        record = {
            "frame": result.index,
            "timings_ms": {k: round(v * 1000.0, 2) for k, v in result.timings.items()},
            "detections": len(result.detections),
        }
        if result.box is not None:
            record["subject"] = result.box.to_dict()
        records.append(record)
        if frames % 25 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"[de3d] frame {index:5d}  {frames / max(elapsed, 1e-6):5.2f} fps",
                flush=True,
            )

    if writer is not None:
        writer.release()
    if close_source is not None:
        close_source.release()

    elapsed = time.perf_counter() - started
    payload = {
        "input": str(args.input),
        "camera": repr(camera),
        "camera_height": camera_height,
        "camera_height_mode": "auto" if auto_height else "fixed",
        "camera_height_estimate": (
            pipeline.camera_height_estimate if auto_height else None
        ),
        "pitch_deg": args.pitch,
        "depth_mode": args.depth_mode,
        "yaw_mode": args.yaw_mode,
        "frames_processed": frames,
        "fps": round(frames / max(elapsed, 1e-6), 3),
        "peak_rss_mb": round(peak_rss, 1),
        "frames": records,
    }
    write_json(json_path, payload)
    print(
        f"[de3d] {frames} frames in {elapsed:.1f}s "
        f"({frames / max(elapsed, 1e-6):.2f} fps, peak RSS {peak_rss:.0f} MB) -> {json_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
