"""The interface every modality model obeys.

A wrapper turns one kind of input (a text string, an audio file, an image, a
window of wrist signals) into a stress score between 0 and 1. The rules, which
the fusion layer relies on:

- A wrapper never returns a string and never raises because of ordinary bad
  input (empty text, a silent clip, a photo with no face). It returns
  `ModalityScore(None, available=False, reason=...)`.
- A wrapper never substitutes a default file for missing input.
- `load()` reads the weights once and is safe to call again.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ModalityScore:
    score: Optional[float]   # stress score in [0, 1]; None when unavailable
    available: bool
    reason: str = ""         # why it is unavailable

    def __post_init__(self) -> None:
        if self.available:
            if self.score is None or not 0.0 <= float(self.score) <= 1.0:
                raise ValueError(f"an available score must be in [0, 1], got {self.score!r}")
        else:
            if self.score is not None:
                raise ValueError("an unavailable score must be None")


def artifacts_dir() -> Path:
    """Folder holding model weights: $SAFESPACE_ARTIFACTS, else <repo>/artifacts."""
    env = os.environ.get("SAFESPACE_ARTIFACTS")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "artifacts"
