#!/usr/bin/env python3
"""Fit per-modality calibration (raw score -> probability).

    python scripts/fit_calibration.py

Wrist is calibrated on its leave-one-subject-out scores; text on a random half of
its Dreaddit test scores, reported on the other half. Voice and face have no
clean held-out data at this layer (the voice actors overlap train/test, and the
face score is already a probability from its own re-fit), so they use the
identity mapping. Writes artifacts/calibration.json and results/calibration.md.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safespace.models.base import artifacts_dir  # noqa: E402
from safespace.models.calibration import apply_platt, brier, fit_platt  # noqa: E402

RESULTS = ROOT / "results"


def read_csv_cols(path, score_col, label_col):
    scores, labels = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            s = row[score_col]
            if s == "" or s == "nan":
                continue
            scores.append(float(s))
            labels.append(int(float(row[label_col])))
    return np.array(scores), np.array(labels)


def main() -> None:
    cal = {"voice": {"a": 1.0, "b": 0.0}, "face": {"a": 1.0, "b": 0.0}}
    report = ["# Per-modality calibration", "", "Produced by `scripts/fit_calibration.py`.", ""]

    # Wrist: fit on all out-of-fold scores.
    wrist_csv = RESULTS / "wrist_loso.csv"
    if wrist_csv.exists():
        s, y = read_csv_cols(wrist_csv, "oof_score", "target")
        a, b = fit_platt(s, y)
        cal["wrist"] = {"a": a, "b": b}
        after = np.array([apply_platt(v, a, b) for v in s])
        report += [f"- **wrist**: Brier {brier(s, y):.4f} -> {brier(after, y):.4f} (a={a:.3f}, b={b:.3f})"]
    else:
        cal["wrist"] = {"a": 1.0, "b": 0.0}
        report += ["- wrist: no LOSO scores found; identity mapping"]

    # Text: fit on a random half, report on the other.
    text_csv = RESULTS / "text_eval.csv"
    if text_csv.exists():
        s, y = read_csv_cols(text_csv, "score", "label")
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(s))
        half = len(s) // 2
        fit_i, test_i = idx[:half], idx[half:]
        a, b = fit_platt(s[fit_i], y[fit_i])
        cal["text"] = {"a": a, "b": b}
        before = brier(s[test_i], y[test_i])
        after = brier(np.array([apply_platt(v, a, b) for v in s[test_i]]), y[test_i])
        report += [f"- **text**: Brier {before:.4f} -> {after:.4f} on held-out half (a={a:.3f}, b={b:.3f})"]
    else:
        cal["text"] = {"a": 1.0, "b": 0.0}
        report += ["- text: no eval scores found; identity mapping"]

    report += [
        "- **voice**: identity mapping (actors overlap train/test, no clean held-out data)",
        "- **face**: identity mapping (score already a probability from its own re-fit)",
        "",
    ]

    # Calm prior: each model's average calibrated score on its not-stressed held-out
    # examples. The web app starts a new person's baseline there (v3 calm_prior).
    calm_sources = {
        "wrist": (RESULTS / "wrist_loso.csv", "oof_score", "target"),
        "text": (RESULTS / "text_eval.csv", "score", "label"),
        "voice": (ROOT / "data" / "cache" / "voice_scores.csv", "score", "stressed"),
        "face": (ROOT / "data" / "cache" / "face_scores.csv", "score", "stressed"),
    }
    report += ["Calm prior (average calibrated score on not-stressed held-out examples):", ""]
    for m, (path, score_col, label_col) in calm_sources.items():
        if not path.exists():
            report += [f"- {m}: no scores found; none set"]
            continue
        s, y = read_csv_cols(path, score_col, label_col)
        calm = np.array([apply_platt(v, cal[m]["a"], cal[m]["b"]) for v in s[y == 0]])
        cal[m]["calm_prior"] = round(float(calm.mean()), 4)
        report += [f"- **{m}**: {cal[m]['calm_prior']:.4f} (n = {len(calm)})"]
    report += [""]

    (artifacts_dir() / "calibration.json").write_text(json.dumps(cal, indent=2))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "calibration.md").write_text("\n".join(report))
    print("calibration:", {k: (round(v["a"], 3), round(v["b"], 3)) for k, v in cal.items()})


if __name__ == "__main__":
    main()
