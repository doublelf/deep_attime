"""Print a compact summary of a pipeline JSON report."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    path = Path(argv[1] if len(argv) > 1 else "outputs/subject.json")
    limit = int(argv[2]) if len(argv) > 2 else 20
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"input     : {data.get('input')}")
    print(f"camera    : {data.get('camera')}")
    print(f"fps       : {data.get('fps')}   frames: {data.get('frames_processed')}")
    print(f"depth_mode: {data.get('depth_mode')}   yaw: {data.get('yaw_mode')}")
    print("-" * 96)
    header = f"{'frame':>6} {'det':>4} {'detect':>7} {'depth':>7} {'metric':>7} {'box3d':>7} {'render':>7}  subject"
    print(header)
    for record in data["frames"][:limit]:
        timings = record["timings_ms"]
        subject = record.get("subject")
        line = (
            f"{record['frame']:>6} {record['detections']:>4} "
            + " ".join(
                f"{timings.get(key, 0.0):>7.0f}"
                for key in ("detect", "depth", "metric", "box3d", "render")
            )
        )
        if subject:
            dims = ",".join(f"{v:.2f}" for v in subject["dims"])
            line += (
                f"  {subject['category']} d={subject['depth']:.2f}m"
                f" ({subject['depth_source']}) dims={dims}"
                f" locked={subject['locked']} yaw={subject['yaw']:.2f}"
            )
        else:
            line += "  -"
        print(line)

    depths = [
        record["subject"]["depth"]
        for record in data["frames"]
        if record.get("subject")
    ]
    if depths:
        print("-" * 96)
        print(
            f"subject frames: {len(depths)}/{data['frames_processed']}  "
            f"depth min={min(depths):.2f} max={max(depths):.2f} "
            f"mean={sum(depths)/len(depths):.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
