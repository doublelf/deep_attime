"""Export the pipeline's models to ONNX so they can be converted to RKNN.

* YOLO: exported with the ultralytics exporter (static shape, NCHW, no
  dynamic axes -- that is what RKNN-Toolkit2 prefers).
* Depth-Anything: already an ONNX file; the script only verifies it and
  reports the input/output signature.

Usage::

    python scripts/export_onnx.py --out-dir models/onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def export_yolo(weights: Path, out_dir: Path, imgsz: int) -> Path | None:
    from ultralytics import YOLO

    print(f"[yolo] exporting {weights} (imgsz={imgsz})")
    model = YOLO(str(weights))
    target = model.export(format="onnx", imgsz=imgsz, simplify=True, opset=13)
    exported = Path(target)
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / exported.name
    if exported.resolve() != destination.resolve():
        destination.write_bytes(exported.read_bytes())
    print(f"[ok  ] {destination}")
    return destination

def simplify_onnx(path: Path) -> Path:
    """Run onnxsim over an ONNX file (optional, needs the ``onnxsim`` package)."""
    try:
        import onnx
        import onnxsim

        model = onnxsim.simplify(str(path))
        onnx.save(model, str(path))
        print(f"[ok  ] simplified {path.name}")
    except Exception as exc:  # noqa: BLE001 - simplification is optional
        print(f"[warn] simplification skipped: {type(exc).__name__}: {exc}")
    return path


def make_static_depth_model(source: Path, destination: Path, height: int, width: int) -> Path:
    """Pin the depth model's input to a fixed resolution.

    RKNN-Toolkit2 is happiest with static shapes, and the dynamic graph of the
    community export produces symbolic output dimensions.  ``height`` and
    ``width`` must be multiples of 14 (the ViT patch size).
    """
    import onnx

    if height % 14 or width % 14:
        raise ValueError("height and width must be multiples of 14")

    model = onnx.load(str(source))
    graph = model.graph
    dims = graph.input[0].type.tensor_type.shape.dim
    dims[0].dim_value = 1
    dims[1].dim_value = 3
    dims[2].dim_value = int(height)
    dims[3].dim_value = int(width)
    for dim in dims:
        dim.ClearField("dim_param")

    model = onnx.shape_inference.infer_shapes(model)
    out = model.graph.output[0]
    out_dims = out.type.tensor_type.shape.dim
    out_dims[0].dim_value = 1
    out_dims[1].dim_value = int(height)
    out_dims[2].dim_value = int(width)
    for dim in out_dims:
        dim.ClearField("dim_param")

    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(destination))
    print(f"[ok  ] {destination} (static {height}x{width})")
    return destination


def inspect_onnx(path: Path) -> None:
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)

    def shape_of(value_info) -> list[int | str]:
        dims = []
        for dim in value_info.type.tensor_type.shape.dim:
            dims.append(dim.dim_value if dim.HasField("dim_value") else dim.dim_param)
        return dims

    print(f"[onnx] {path.name}")
    for value_info in model.graph.input:
        print(f"       input  {value_info.name}: {shape_of(value_info)}")
    for value_info in model.graph.output:
        print(f"       output {value_info.name}: {shape_of(value_info)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(REPO_ROOT / "models" / "yolo11n.pt"))
    parser.add_argument("--depth-model", default=str(REPO_ROOT / "models" / "depth_anything_v2_small.onnx"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "models" / "onnx"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument(
        "--depth-size",
        default="518x294",
        help="static input of the exported depth model (HxW, multiples of 14); "
        "use 'dynamic' to keep the community export as is",
    )
    parser.add_argument("--no-simplify", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    if not args.skip_yolo:
        weights = Path(args.weights)
        if not weights.exists():
            print(f"[error] {weights} missing; run scripts/download_models.py", file=sys.stderr)
            return 1
        exported = export_yolo(weights, out_dir, args.imgsz)
        if exported is not None and not args.no_simplify:
            simplify_onnx(exported)

    depth = Path(args.depth_model)
    if not depth.exists():
        print(f"[warn] depth model {depth} missing", file=sys.stderr)
        return 1

    destination = out_dir / depth.name
    if args.depth_size.lower() != "dynamic":
        height, width = (int(v) for v in args.depth_size.lower().split("x"))
        make_static_depth_model(depth, destination, height, width)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        if depth.resolve() != destination.resolve():
            destination.write_bytes(depth.read_bytes())
        print(f"[ok  ] {destination} (dynamic)")
    if not args.no_simplify:
        simplify_onnx(destination)
    inspect_onnx(destination)
    return 0


if __name__ == "__main__":
    sys.exit(main())
