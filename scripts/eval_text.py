#!/usr/bin/env python3
"""Measure the existing text model on the Dreaddit test set.

    python scripts/eval_text.py

Writes results/text_eval.csv and results/text_eval.md.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safespace.models.text import TextModel  # noqa: E402
from safespace.sim.metrics import auc  # noqa: E402

TEST_CSV = ROOT / "data" / "raw" / "dreaddit" / "dreaddit-test.csv"
PROBES = ROOT / "tests" / "data" / "text_probes.txt"
RESULTS = ROOT / "results"


def bootstrap_ci(labels, scores, metric, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(labels))
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        try:
            vals.append(metric(labels[s], scores[s]))
        except Exception:
            pass
    return 1.96 * float(np.std(vals)) if vals else float("nan")


def acc(labels, scores):
    return float(((scores >= 0.5).astype(int) == labels).mean())


def f1(labels, scores):
    pred = (scores >= 0.5).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    model = TextModel().load()

    ids, labels, scores = [], [], []
    with open(TEST_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = model.predict(row["text"])
            ids.append(row.get("id", len(ids)))
            labels.append(int(row["label"]))
            scores.append(r.score if r.available else 0.5)

    labels = np.array(labels)
    scores = np.array(scores)
    with open(RESULTS / "text_eval.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "score"])
        w.writerows(zip(ids, labels, scores))

    majority = max(labels.mean(), 1 - labels.mean())
    a, fa = acc(labels, scores), bootstrap_ci(labels, scores, acc)
    fv, ff = f1(labels, scores), bootstrap_ci(labels, scores, f1)
    av, af = auc(labels > 0.5, scores), bootstrap_ci(labels, scores, lambda l, s: auc(l > 0.5, s))

    probe_lines = []
    for line in PROBES.read_text().splitlines():
        if line.strip():
            probe_lines.append(f"| {model.predict(line).score:.4f} | {line} |")

    md = [
        "# Text model evaluation (Dreaddit test set)",
        "",
        f"Model: the owner's fine-tuned BERT in `artifacts/text/`. Produced by `scripts/eval_text.py`.",
        f"Test set: {len(labels)} posts.",
        "",
        "The model's original training split is unknown, so some of these posts may have been seen in training.",
        "",
        "| metric | value | 95% interval |",
        "|---|---:|---:|",
        f"| accuracy at 0.5 | {a:.4f} | +/- {fa:.4f} |",
        f"| F1 | {fv:.4f} | +/- {ff:.4f} |",
        f"| AUC | {av:.4f} | +/- {af:.4f} |",
        f"| always-guess-majority accuracy | {majority:.4f} | |",
        "",
        "## Probe sentences (negated, calm, and neutral)",
        "",
        "These are not from Dreaddit; they test the model's weak spots by hand.",
        "",
        "| score | sentence |",
        "|---:|---|",
        *probe_lines,
        "",
    ]
    (RESULTS / "text_eval.md").write_text("\n".join(md))
    print(f"accuracy {a:.4f} +/- {fa:.4f} | F1 {fv:.4f} | AUC {av:.4f} | majority {majority:.4f}")


if __name__ == "__main__":
    main()
