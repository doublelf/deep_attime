#!/usr/bin/env python
"""Launcher that puts ``src`` on ``sys.path`` and runs the CLI.

``pip`` is not usable in this environment, so the package is not installed;
this script provides the same entry point without touching the interpreter
setup::

    python run.py --input video.mp4 --camera config/cameras/sample_1920x1080.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from de3d.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
