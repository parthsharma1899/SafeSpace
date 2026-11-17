"""Metrics defined in docs/protocol.md."""

from __future__ import annotations

from typing import Optional

import numpy as np

EPISODE, CALM, ALARM = 0.5, 0.3, 0.7


def auc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """Area under the ROC curve from ranks, with ties sharing their average rank.

    Returns None when only one class is present.
    """
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = np.asarray(scores)[order]
    ranks = np.empty(len(scores))
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def half_time(fused: np.ndarray, at: int, before: float, after: float, horizon: int = 60) -> float:
    """Steps after `at` until the fused score has covered half of the change from before to after.

    Returns `horizon` if it never does within the horizon.
    """
    halfway = (before + after) / 2
    rising = after > before
    for k in range(horizon):
        if at + k >= len(fused):
            break
        v = fused[at + k]
        if not np.isnan(v) and ((v >= halfway) if rising else (v <= halfway)):
            return float(k)
    return float(horizon)


def evaluate(truth: np.ndarray, fused: np.ndarray) -> dict[str, Optional[float]]:
    """Score one session. `fused` holds NaN where the method produced nothing."""
    ok = ~np.isnan(fused)
    out: dict[str, Optional[float]] = {"failures": float((~ok).sum())}
    if not ok.any():
        return {**out, "mae": None, "bias": None, "auc": None, "false_alarm": None}
    err = fused[ok] - truth[ok]
    calm = ok & (truth < CALM)
    out["mae"] = float(np.abs(err).mean())
    out["bias"] = float(err.mean())
    out["auc"] = auc(truth[ok] > EPISODE, fused[ok])
    out["false_alarm"] = float((fused[calm] > ALARM).mean()) if calm.any() else None
    return out
