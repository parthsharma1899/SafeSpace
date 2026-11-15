"""Platt scaling: turn a model's raw score into a calibrated probability.

A raw score of 0.7 does not necessarily mean "70% likely stressed". Platt
scaling fits a logistic curve from held-out raw scores to their true labels, so
the calibrated number can be read as a probability. `fit_platt` returns two
numbers (a, b); `apply_platt` applies them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

_EPS = 1e-4


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def fit_platt(scores, labels) -> tuple[float, float]:
    """Fit calibrated = sigmoid(a * logit(score) + b). Returns (a, b)."""
    from sklearn.linear_model import LogisticRegression

    x = _logit(np.asarray(scores, dtype=float)).reshape(-1, 1)
    y = np.asarray(labels, dtype=int)
    if len(np.unique(y)) < 2:
        return 1.0, 0.0
    lr = LogisticRegression(C=1e6)
    lr.fit(x, y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def apply_platt(score: float, a: float, b: float) -> float:
    z = a * _logit(np.array([score]))[0] + b
    return float(1.0 / (1.0 + np.exp(-z)))


def brier(scores, labels) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    return float(np.mean((scores - labels) ** 2))


def load_calibration(path: Path) -> Mapping[str, tuple[float, float]]:
    data = json.loads(Path(path).read_text())
    return {k: (v["a"], v["b"]) for k, v in data.items()}
