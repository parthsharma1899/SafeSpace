"""Where the trained weights are hosted, and which files the app needs.

The weights live in a Hugging Face model repo, laid out exactly like `artifacts/`.
`scripts/model_manifest.json` (committed to git) lists every file with its size
and SHA-256, so a download is checked against what this commit expects.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "scripts" / "model_manifest.json"
DEFAULT_REPO = "parth18s/safespace-fusion-models"

# Files each model needs, as glob patterns relative to artifacts/.
MODEL_PATTERNS: dict[str, list[str]] = {
    "calibration": ["calibration.json"],
    "wrist": ["wrist/wrist_xgb.json"],
    "text": ["text/stress_classifier_epoch2/*"],
    "voice": ["voice/speech_model.keras"],
    "face": [
        "face/elm_hidden_f16.npz",
        "face/refit/*",
        "face/features/*",
        "face/resnet50_features/**/*",
        "face/vit_features/**/*",
    ],
}
IGNORED = {".DS_Store"}


def repo_id() -> str:
    return os.environ.get("SAFESPACE_MODELS_REPO", DEFAULT_REPO)


def local_files(artifacts: Path) -> dict[str, list[str]]:
    """The app's files present under `artifacts`, grouped by model."""
    out: dict[str, list[str]] = {}
    for model, patterns in MODEL_PATTERNS.items():
        found = set()
        for pattern in patterns:
            for p in artifacts.glob(pattern):
                if p.is_file() and p.name not in IGNORED:
                    found.add(p.relative_to(artifacts).as_posix())
        out[model] = sorted(found)
    return out


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def read_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(path.read_text())


def check(artifacts: Path, entries: list[dict]) -> list[str]:
    """Problems with the local copies of `entries`; empty when every file matches."""
    problems = []
    for e in entries:
        p = artifacts / e["path"]
        if not p.is_file():
            problems.append(f"missing: {e['path']}")
        elif p.stat().st_size != e["size"]:
            problems.append(f"wrong size: {e['path']}")
        elif sha256(p) != e["sha256"]:
            problems.append(f"checksum mismatch: {e['path']}")
    return problems
