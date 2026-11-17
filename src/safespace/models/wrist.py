"""Wrist stress model: XGBoost on pulse and skin-conductance features.

Unlike the legacy model, this needs the person's own resting baseline. Call
`calibrate(...)` with at least 120 s of quiet rest once, then `predict(...)` with
at least 60 s of signal. Without a baseline, or with too short or too noisy a
window, it returns unavailable rather than a number.

Inputs are pulse (BVP-like) and skin conductance (EDA-like) sample arrays with
their sampling rates; they are resampled to the model's 64 Hz and 4 Hz. The
model was trained on the Empatica E4; its calibration on the owner's ESP32
wearable is unvalidated until the Phase 4 pilot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import ModalityScore, artifacts_dir
from .wrist_features import ACC_HZ, BVP_HZ, EDA_HZ, window_features
from .wrist_windows import FEATURES, _relative

CAL_MIN_S, PRED_MIN_S = 120, 60


def _resample(x, src_hz, dst_hz):
    x = np.asarray(x, dtype=float).ravel()
    if src_hz == dst_hz or len(x) == 0:
        return x
    n = int(round(len(x) * dst_hz / src_hz))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)


@dataclass
class Baseline:
    features: dict
    eda_ref: float


class WristModel:
    name = "wrist"

    def __init__(self, model_path=None):
        self._path = model_path or (artifacts_dir() / "wrist" / "wrist_xgb.json")
        self._model = None

    def load(self) -> "WristModel":
        if self._model is None:
            from xgboost import XGBClassifier

            m = XGBClassifier()
            m.load_model(str(self._path))
            self._model = m
        return self

    def calibrate(self, pulse, conductance, rate_pulse, rate_conductance, acc=None, rate_acc=None) -> Optional[Baseline]:
        bvp = _resample(pulse, rate_pulse, BVP_HZ)
        if len(bvp) / BVP_HZ < CAL_MIN_S:
            return None
        eda = _resample(conductance, rate_conductance, EDA_HZ)
        acc_arr = self._acc(acc, rate_acc, len(bvp))
        eda_ref = float(np.mean(eda)) if len(eda) else 1.0
        feat = window_features(bvp, eda, acc_arr, eda_ref=eda_ref)
        if feat is None:
            return None
        return Baseline(features=feat, eda_ref=eda_ref)

    def predict(self, pulse, conductance, rate_pulse, rate_conductance, baseline: Optional[Baseline], acc=None, rate_acc=None) -> ModalityScore:
        if baseline is None:
            return ModalityScore(None, False, "no resting baseline")
        bvp = _resample(pulse, rate_pulse, BVP_HZ)
        if len(bvp) / BVP_HZ < PRED_MIN_S:
            return ModalityScore(None, False, "window shorter than 60 s")
        eda = _resample(conductance, rate_conductance, EDA_HZ)
        acc_arr = self._acc(acc, rate_acc, len(bvp))
        feat = window_features(bvp, eda, acc_arr, eda_ref=baseline.eda_ref)
        if feat is None:
            return ModalityScore(None, False, "signal too noisy")
        self.load()
        rel = _relative(feat, baseline.features)
        x = np.array([[rel[f] for f in FEATURES]], dtype=float)
        score = float(self._model.predict_proba(x)[0, 1])
        return ModalityScore(round(score, 4), True)

    @staticmethod
    def _acc(acc, rate_acc, n_bvp):
        if acc is None:
            return np.zeros((max(1, n_bvp * ACC_HZ // BVP_HZ), 3))
        a = np.asarray(acc, dtype=float).reshape(-1, 3)
        if rate_acc and rate_acc != ACC_HZ:
            idx = np.linspace(0, len(a) - 1, int(round(len(a) * ACC_HZ / rate_acc)))
            a = np.stack([np.interp(idx, np.arange(len(a)), a[:, k]) for k in range(3)], axis=1)
        return a
