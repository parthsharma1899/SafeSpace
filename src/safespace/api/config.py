"""Reads settings from the repo-root `.env` file into the environment.

Variables already set in the environment win. The file is gitignored; copy
`.env.example` to `.env` to start. Only simple `NAME=value` lines are read.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[3]


def load_env(path: Optional[Path] = None) -> None:
    path = path or ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name, value = name.strip(), value.strip().strip('"').strip("'")
        if name and name not in os.environ:
            os.environ[name] = value
