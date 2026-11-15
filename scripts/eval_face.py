#!/usr/bin/env python3
"""Record the saved face pipeline's behaviour on FER-2013 test images.

    python scripts/eval_face.py

Feeds each 48x48 image directly as the face crop (the detector is bypassed, so
every image is scored) and writes results/face_eval.md. This measures the saved
pipeline whether or not the model is switched on for the demo.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# This script records the ORIGINAL saved pipeline's behaviour (the chance result),
# independent of whether the 2026 ELM re-fit is present.
os.environ["SAFESPACE_FACE_MODE"] = "original"

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safespace.models.face import FaceModel  # noqa: E402
from safespace.sim.metrics import auc  # noqa: E402

TEST_CSV = ROOT / "data" / "raw" / "fer2013" / "test.csv"
RESULTS = ROOT / "results"
NAMES = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}
STRESSED = {0, 1, 2, 4}
PER_CLASS = 100


def main() -> None:
    import cv2

    RESULTS.mkdir(exist_ok=True)
    df = pd.read_csv(TEST_CSV)
    df = pd.concat([g.sample(min(len(g), PER_CLASS), random_state=0) for _, g in df.groupby("emotion")]).reset_index(drop=True)
    emo = df["emotion"].to_numpy()
    grays = [np.array(p.split(), dtype=np.uint8).reshape(48, 48) for p in df["pixels"]]

    model = FaceModel().load()
    scores = np.array([model._score_crop(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)) for g in grays])

    y = np.isin(emo, list(STRESSED)).astype(int)
    a = auc(y > 0.5, scores)
    majority = max(y.mean(), 1 - y.mean())
    by_emotion = {NAMES[k]: float(scores[emo == k].mean()) for k in range(7)}

    md = [
        "# Face model evaluation (FER-2013 test images)",
        "",
        "Model: the owner's ResNet50 + ViT + ELM in `artifacts/face/`. Produced by `scripts/eval_face.py`.",
        f"Scored {len(scores)} images ({PER_CLASS} per emotion, detector bypassed).",
        "Stressed = angry, disgust, fear, sad; relaxed = happy, surprise, neutral.",
        "",
        f"- **AUC (stressed vs relaxed): {a:.4f}** (0.50 is a coin flip)",
        f"- Always-guess-majority accuracy: {majority:.4f}",
        "",
        "## Mean score by true emotion",
        "",
        "| emotion | mean score |",
        "|---|---:|",
        *[f"| {name} | {score:.4f} |" for name, score in by_emotion.items()],
        "",
        "The scores barely vary by emotion and the AUC is near 0.50. The saved ViT",
        "feature extractor holds untrained weights (all normalisation scales exactly",
        "1, all offsets and biases exactly 0), so most of the ELM's inputs carry no",
        "signal. See docs/models/face.md. The model ships switched off by default.",
        "",
    ]
    (RESULTS / "face_eval.md").write_text("\n".join(md))
    print(f"AUC {a:.4f} | majority {majority:.4f} | by emotion: " + ", ".join(f"{k} {v:.2f}" for k, v in by_emotion.items()))


if __name__ == "__main__":
    main()
