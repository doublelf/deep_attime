"""Download the model files the pipeline needs.

Everything is fetched with plain HTTP(S) because ``pip`` is not usable in this
environment and ``huggingface.co`` is not directly reachable; ``hf-mirror.com``
is used as the mirror.

Usage::

    python scripts/download_models.py
    python scripts/download_models.py --out-dir models --depth-size-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "models"

YOLO_URLS = [
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
]

DEPTH_URLS = [
    "https://hf-mirror.com/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx",
    "https://hf-mirror.com/depth-anything/Depth-Anything-V2-Small/resolve/main/onnx/model.onnx",
    "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx",
]


def download(urls: list[str], destination: Path, retries: int = 3) -> bool:
    """Download the first URL that works into ``destination``."""
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"[skip] {destination.name} already present ({destination.stat().st_size/1e6:.1f} MB)")
        return True

    for attempt in range(1, retries + 1):
        for url in urls:
            try:
                print(f"[get ] {url}")
                with requests.get(url, stream=True, timeout=60) as response:
                    response.raise_for_status()
                    total = int(response.headers.get("content-length", 0))
                    written = 0
                    tmp = destination.with_suffix(destination.suffix + ".part")
                    with tmp.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=1 << 20):
                            if not chunk:
                                continue
                            handle.write(chunk)
                            written += len(chunk)
                            if total:
                                print(
                                    f"       {written/1e6:6.1f} / {total/1e6:6.1f} MB",
                                    end="\r",
                                    flush=True,
                                )
                    tmp.replace(destination)
                    print(f"\n[ok  ] {destination.name}  {written/1e6:.1f} MB")
                    return True
            except Exception as exc:  # noqa: BLE001 - report and try the next url
                print(f"[warn] {url} -> {type(exc).__name__}: {exc}")
                time.sleep(1.0)
        if attempt < retries:
            time.sleep(2.0)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--yolo-name", default="yolo11n.pt")
    parser.add_argument("--depth-name", default="depth_anything_v2_small.onnx")
    parser.add_argument("--depth-only", action="store_true")
    parser.add_argument("--yolo-only", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    ok = True
    if not args.depth_only:
        ok &= download(YOLO_URLS, out_dir / args.yolo_name)
    if not args.yolo_only:
        ok &= download(DEPTH_URLS, out_dir / args.depth_name)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
