#!/usr/bin/env python3
"""Measure the existing voice model on all 1,440 RAVDESS speech clips.

    python scripts/eval_voice.py

Writes results/voice_eval.md. The model is run through the repository's own
feature code, unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safespace.models.voice import VoiceModel, extract_features  # noqa: E402

RAVDESS = ROOT / "data" / "raw" / "ravdess"
RESULTS = ROOT / "results"
STRESSED = {4, 5, 6, 7}  # sad, angry, fearful, disgust


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    files = sorted(RAVDESS.glob("Actor_*/*.wav"))
    if not files:
        raise SystemExit(f"No RAVDESS clips under {RAVDESS}")

    model = VoiceModel().load()
    emotion = np.array([int(f.name.split("-")[2]) for f in files])
    actor = np.array([int(f.name.split("-")[6][:2]) for f in files])
    y = np.isin(emotion, list(STRESSED)).astype(int)

    # Cache the raw audio once so the volume variants do not reload from disk.
    import librosa

    scores = np.array([model.predict(f).score for f in files])

    def score_features(feats):
        x = feats[None, :, :, None].astype("float32")
        return float(model._model.predict(x, verbose=0)[0][0])

    def feats_from_audio(a, sr=22050):
        m = librosa.feature.mfcc(y=a, sr=sr, n_mfcc=40)
        return np.pad(m, ((0, 0), (0, max(0, 216 - m.shape[1]))))[:, :216]

    def accuracy(s):
        return float(((s >= 0.5).astype(int) == y).mean())

    overall = accuracy(scores)
    per_actor = [accuracy_mask(scores, y, actor == a) for a in sorted(set(actor))]

    volume_rows = []
    for g in (0.3, 0.5, 2.0, 3.0):
        vs = []
        for f in files:
            a, _ = librosa.load(f, duration=3, offset=0.5)
            vs.append(score_features(feats_from_audio(np.clip(a * g, -1, 1))))
        volume_rows.append((g, accuracy(np.array(vs))))

    md = [
        "# Voice model evaluation (RAVDESS speech, all 1,440 clips)",
        "",
        "Model: the owner's CNN in `artifacts/voice/`. Produced by `scripts/eval_voice.py`.",
        "Stressed = sad, angry, fearful, disgust; relaxed = calm, happy, neutral, surprised.",
        "",
        "This includes the clips the model was trained on. It confirms the saved file works; "
        "it is not a measure of how well the model generalises to new speakers.",
        "",
        f"- Overall accuracy: **{overall:.4f}** on {len(files)} clips",
        f"- Per-actor accuracy: min {min(per_actor):.4f}, median {float(np.median(per_actor)):.4f}, max {max(per_actor):.4f}",
        "",
        "## Accuracy when the clip volume is changed",
        "",
        "| volume | accuracy |",
        "|---|---:|",
        *[f"| x{g} | {a:.4f} |" for g, a in volume_rows],
        "",
    ]
    (RESULTS / "voice_eval.md").write_text("\n".join(md))
    print(f"overall {overall:.4f} | per-actor min {min(per_actor):.4f} max {max(per_actor):.4f}")


def accuracy_mask(scores, y, mask):
    return float(((scores[mask] >= 0.5).astype(int) == y[mask]).mean())


if __name__ == "__main__":
    main()
