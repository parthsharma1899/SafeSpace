#!/usr/bin/env python3
"""Write a compact copy of the face model's ELM hidden layer, and check it.

    python scripts/compact_face_elm.py [--check 400]

The original artifacts/face/elm_weights-001.npz stores W (153344 x 2000) in
float64: 2.45 GB. This writes artifacts/face/elm_hidden_f16.npz with W in
float16 (0.61 GB) and b, B unchanged. The face model uses the compact file when
it exists. Nothing is retrained and the original file is not changed.

With --check N (needs FER-2013), N test images per emotion are scored with both
files and the largest score difference and both AUCs are written to
results/face_compact.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safespace.models.base import artifacts_dir  # noqa: E402
from safespace.models.face import COMPACT_ELM, FaceModel  # noqa: E402
from safespace.sim.metrics import auc  # noqa: E402

STRESSED_FER = {0, 1, 2, 4}


def write_compact(face_dir: Path) -> Path:
    elm = np.load(face_dir / "elm_weights-001.npz")
    out = face_dir / COMPACT_ELM
    np.savez(out, W=elm["W"].astype(np.float16), b=elm["b"], B=elm["B"])
    return out


def score_all(model: FaceModel, crops) -> np.ndarray:
    model.load()
    return np.array([model._score_crop(c) for c in crops])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=int, default=0, help="FER-2013 test images per emotion to compare")
    args = ap.parse_args()

    face_dir = artifacts_dir() / "face"
    out = write_compact(face_dir)
    print(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB)")
    if not args.check:
        return

    import cv2
    import pandas as pd

    df = pd.read_csv(ROOT / "data" / "raw" / "fer2013" / "test.csv")
    df = pd.concat([g.sample(min(len(g), args.check), random_state=0) for _, g in df.groupby("emotion")])
    crops = [cv2.cvtColor(np.array(p.split(), dtype=np.uint8).reshape(48, 48), cv2.COLOR_GRAY2BGR) for p in df["pixels"]]
    y = df["emotion"].isin(STRESSED_FER).to_numpy()

    compact = score_all(FaceModel(face_dir), crops)
    tmp = face_dir / (COMPACT_ELM + ".off")
    out.rename(tmp)  # hide it so the model loads the original
    try:
        original = score_all(FaceModel(face_dir), crops)
    finally:
        tmp.rename(out)

    diff = float(np.max(np.abs(compact - original)))
    md = [
        "# Face model: compact ELM file",
        "",
        "Produced by `scripts/compact_face_elm.py`. The ELM hidden layer W is stored in",
        "float16 instead of float64; nothing is retrained.",
        "",
        f"- Size: {(face_dir / 'elm_weights-001.npz').stat().st_size / 1e9:.2f} GB -> {out.stat().st_size / 1e9:.2f} GB",
        f"- FER-2013 test images compared: {len(y)} ({args.check} per emotion)",
        f"- Largest score difference: {diff:.5f}",
        f"- AUC with the original file: {auc(y, original):.4f}",
        f"- AUC with the compact file: {auc(y, compact):.4f}",
        "",
    ]
    (ROOT / "results" / "face_compact.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
