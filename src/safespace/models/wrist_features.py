"""Per-window features from wrist BVP, EDA and ACC.

One window is 60 seconds of wrist signal. `window_features` returns a dict of
features, or None when the pulse signal is too noisy to trust (fewer than 30
usable beat-to-beat intervals).

The features are deliberately physiological (heart rate and its variability,
skin-conductance level and responses), each later expressed relative to the same
person's resting baseline, so the model cannot learn a per-session shortcut such
as absolute temperature.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage, signal

BVP_HZ, EDA_HZ, ACC_HZ = 64, 4, 32
MIN_INTERVALS = 30


def _bandpass(x, lo, hi, fs, order=3):
    sos = signal.butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def _lowpass(x, cutoff, fs, order=2):
    sos = signal.butter(order, cutoff, btype="low", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def _beat_intervals(bvp: np.ndarray) -> np.ndarray:
    filtered = _bandpass(bvp.ravel(), 0.7, 3.5, BVP_HZ)
    peaks, _ = signal.find_peaks(filtered, distance=int(0.3 * BVP_HZ), prominence=0.3 * filtered.std())
    intervals = np.diff(peaks) / BVP_HZ
    return intervals[(intervals >= 0.33) & (intervals <= 1.5)]


def window_features(bvp, eda, acc, eda_ref: float) -> Optional[dict]:
    bvp = np.asarray(bvp, dtype=float).ravel()
    eda = np.asarray(eda, dtype=float).ravel()
    acc = np.asarray(acc, dtype=float).reshape(-1, 3)
    minutes = len(bvp) / BVP_HZ / 60.0

    intervals = _beat_intervals(bvp)
    if len(intervals) < MIN_INTERVALS:
        return None

    hr = 60.0 / intervals.mean()
    sdnn = intervals.std(ddof=1) * 1000.0
    rmssd = np.sqrt(np.mean(np.diff(intervals) ** 2)) * 1000.0

    eda_f = _lowpass(eda, 1.0, EDA_HZ) if len(eda) > 12 else eda
    eda_mean = float(eda_f.mean())
    t_min = np.arange(len(eda_f)) / EDA_HZ / 60.0
    eda_slope = float(np.polyfit(t_min, eda_f, 1)[0]) if len(eda_f) > 1 else 0.0

    baseline = ndimage.median_filter(eda_f, size=max(1, int(8 * EDA_HZ)))
    phasic = eda_f - baseline
    scr_peaks, props = signal.find_peaks(phasic, prominence=max(1e-9, 0.01 * eda_ref))
    scr_rate = len(scr_peaks) / minutes if minutes > 0 else 0.0
    scr_amp = float(props["prominences"].mean()) if len(scr_peaks) else 0.0

    mag = np.sqrt((acc ** 2).sum(axis=1))
    acc_std = float(mag.std())

    return {
        "hr": float(hr),
        "sdnn": float(sdnn),
        "rmssd": float(rmssd),
        "eda_mean": eda_mean,
        "eda_slope": eda_slope,
        "scr_rate": float(scr_rate),
        "scr_amp": scr_amp,
        "acc_std": acc_std,
    }
