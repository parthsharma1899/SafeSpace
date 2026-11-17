#!/usr/bin/env python3
"""Train the wrist stress model on WESAD and evaluate it leave-one-subject-out.

    DYLD_FALLBACK_LIBRARY_PATH=... python scripts/train_wrist.py

(See PLAN.md A.4 for the DYLD prefix needed on macOS.)

Writes results/wrist_loso.md, results/wrist_loso.csv, and the final model
artifacts/wrist/wrist_xgb.json (trained on all 15 subjects).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safespace.data.wesad import subject_paths, load_subject  # noqa: E402
from safespace.models.base import artifacts_dir  # noqa: E402
from safespace.models.wrist_windows import ACC_FEATURE, FEATURES, subject_windows  # noqa: E402
from safespace.sim.metrics import auc  # noqa: E402

WESAD = ROOT / "data" / "raw" / "wesad" / "WESAD"
RESULTS = ROOT / "results"
SEED = 42


def make_model(scale_pos_weight):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED,
        eval_metric="logloss", scale_pos_weight=scale_pos_weight,
    )


def to_matrix(windows, feature_names):
    X = np.array([[w.features[f] for f in feature_names] for w in windows], dtype=float)
    y = np.array([w.target for w in windows], dtype=int)
    subj = np.array([w.subject for w in windows])
    return X, y, subj


def loso(windows, feature_names):
    """Leave-one-subject-out. Returns per-subject rows and out-of-fold scores."""
    X, y, subj = to_matrix(windows, feature_names)
    subjects = sorted(set(subj), key=lambda s: int(s[1:]))
    rows, oof = [], np.full(len(y), np.nan)
    for s in subjects:
        tr, te = subj != s, subj == s
        pos, neg = int(y[tr].sum()), int((1 - y[tr]).sum())
        model = make_model(neg / max(pos, 1))
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        oof[te] = p
        pred = (p >= 0.5).astype(int)
        a = auc(y[te] > 0.5, p)
        rows.append({
            "subject": s, "n": int(te.sum()), "stress_windows": int(y[te].sum()),
            "accuracy": float((pred == y[te]).mean()),
            "majority": float(max(y[te].mean(), 1 - y[te].mean())),
            "auc": a if a is not None else float("nan"),
        })
    return rows, oof, y, subj


def f1(y, p):
    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    paths = subject_paths(WESAD)
    if not paths:
        raise SystemExit(f"No WESAD subjects under {WESAD}")

    windows = []
    for p in paths:
        windows.extend(subject_windows(load_subject(p)))
    print(f"{len(paths)} subjects, {len(windows)} windows ({sum(w.target for w in windows)} stress)")

    rows, oof, y, subj = loso(windows, FEATURES)
    overall_acc = float(((oof >= 0.5).astype(int) == y).mean())
    overall_auc = auc(y > 0.5, oof)
    overall_f1 = f1(y, oof)
    majority = float(max(y.mean(), 1 - y.mean()))

    # Same LOSO with acc_std added, reported separately (not the shipped model).
    _, oof_acc, y2, _ = loso(windows, FEATURES + [ACC_FEATURE])
    acc_variant_auc = auc(y2 > 0.5, oof_acc)
    acc_variant_acc = float(((oof_acc >= 0.5).astype(int) == y2).mean())

    # Feature importance from a model trained on all subjects.
    X, yall, _ = to_matrix(windows, FEATURES)
    final = make_model(int((1 - yall).sum()) / max(int(yall.sum()), 1))
    final.fit(X, yall)
    importances = dict(zip(FEATURES, final.feature_importances_))

    out_dir = artifacts_dir() / "wrist"
    out_dir.mkdir(parents=True, exist_ok=True)
    final.save_model(str(out_dir / "wrist_xgb.json"))

    with open(RESULTS / "wrist_loso.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["subject", "window_start_s", "label", "target", "oof_score"])
        for win, sc in zip(windows, oof):
            w.writerow([win.subject, f"{win.start_s:.0f}", win.label, win.target, f"{sc:.4f}"])

    md = [
        "# Wrist model: leave-one-subject-out evaluation (WESAD)",
        "",
        "Model: XGBoost on wrist heart-rate, HRV and skin-conductance features, each",
        "relative to the subject's own resting baseline. Produced by `scripts/train_wrist.py`.",
        "Stress (label 2) is the positive class; baseline (1) and amusement (3) are negative.",
        "",
        f"- Windows: {len(windows)} from {len(paths)} subjects ({int(y.sum())} stress)",
        f"- **Overall out-of-fold accuracy: {overall_acc:.4f}** (majority baseline {majority:.4f})",
        f"- Overall AUC: {overall_auc:.4f}",
        f"- Overall F1: {overall_f1:.4f}",
        "",
        "With `acc_std` added (movement; NOT the shipped model, since the WESAD stress",
        f"task is done standing and the others seated): accuracy {acc_variant_acc:.4f}, AUC {acc_variant_auc:.4f}.",
        "",
        "## Per-subject",
        "",
        "| subject | windows | stress | accuracy | majority | AUC |",
        "|---|---:|---:|---:|---:|---:|",
        *[f"| {r['subject']} | {r['n']} | {r['stress_windows']} | {r['accuracy']:.3f} | {r['majority']:.3f} | {r['auc']:.3f} |" for r in rows],
        "",
        "## Feature importance (model trained on all subjects)",
        "",
        "| feature | importance |",
        "|---|---:|",
        *[f"| {k} | {v:.3f} |" for k, v in sorted(importances.items(), key=lambda kv: -kv[1])],
        "",
    ]
    (RESULTS / "wrist_loso.md").write_text("\n".join(md))
    print(f"accuracy {overall_acc:.4f} (majority {majority:.4f}) | AUC {overall_auc:.4f} | max importance {max(importances.values()):.3f}")
    if overall_auc < 0.80 or overall_acc <= majority or max(importances.values()) > 0.60:
        print("WARNING: a pass condition is not met (AUC>=0.80, acc>majority, max importance<=0.60). See PLAN.md 3A.4.")


if __name__ == "__main__":
    main()
