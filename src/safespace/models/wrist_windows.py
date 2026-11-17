"""Turn one WESAD subject into labelled feature windows, each relative to that
person's own resting baseline.

A window is 60 s, stepped by 30 s, taken from the longest unbroken run of each of
labels 1 (baseline), 2 (stress) and 3 (amusement). The first 120 s of the
baseline run is the calibration period: its features are the reference every
other window is divided by, and no window overlapping it is emitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .wrist_features import window_features
from ..data.wesad import ACC_HZ, BVP_HZ, EDA_HZ, LABEL_HZ, WesadSubject

WINDOW_S, STEP_S, CAL_S = 60, 30, 120
POSITIVE_LABEL = 2
USED_LABELS = (1, 2, 3)

# The seven inputs the shipped model uses, in order. acc_std is stored but excluded.
FEATURES = ["hr_rel", "sdnn_rel", "rmssd_rel", "eda_mean_rel", "eda_slope_norm", "scr_rate_diff", "scr_amp_norm"]
ACC_FEATURE = "acc_std_rel"


@dataclass
class Window:
    subject: str
    label: int
    target: int
    start_s: float
    features: dict  # FEATURES + ACC_FEATURE


def _longest_run(label: np.ndarray, value: int) -> Optional[tuple[int, int]]:
    mask = label == value
    if not mask.any():
        return None
    best = (0, 0)
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j
        else:
            i += 1
    return best  # [start, end) in label samples


def _slice(sig, hz, a_s, b_s):
    return sig[int(a_s * hz):int(b_s * hz)]


def _extract(subj: WesadSubject, a_s: float, b_s: float, eda_ref: float):
    return window_features(
        _slice(subj.bvp, BVP_HZ, a_s, b_s),
        _slice(subj.eda, EDA_HZ, a_s, b_s),
        _slice(subj.acc, ACC_HZ, a_s, b_s),
        eda_ref=eda_ref,
    )


def _relative(f: dict, b: dict) -> dict:
    return {
        "hr_rel": f["hr"] / b["hr"] - 1,
        "sdnn_rel": f["sdnn"] / b["sdnn"] - 1 if b["sdnn"] else 0.0,
        "rmssd_rel": f["rmssd"] / b["rmssd"] - 1 if b["rmssd"] else 0.0,
        "eda_mean_rel": f["eda_mean"] / b["eda_mean"] - 1 if b["eda_mean"] else 0.0,
        "eda_slope_norm": f["eda_slope"] / b["eda_mean"] if b["eda_mean"] else 0.0,
        "scr_rate_diff": f["scr_rate"] - b["scr_rate"],
        "scr_amp_norm": f["scr_amp"] / b["eda_mean"] if b["eda_mean"] else 0.0,
        "acc_std_rel": f["acc_std"] / b["acc_std"] - 1 if b["acc_std"] else 0.0,
    }


def subject_windows(subj: WesadSubject) -> list[Window]:
    base_run = _longest_run(subj.label, 1)
    if base_run is None:
        return []
    cal_start = base_run[0] / LABEL_HZ
    cal_end = cal_start + CAL_S
    base_end_s = base_run[1] / LABEL_HZ
    if base_end_s - cal_end < WINDOW_S:
        return []  # not enough baseline beyond the calibration period

    cal_eda = _slice(subj.eda, EDA_HZ, cal_start, cal_end)
    eda_ref = float(np.mean(cal_eda)) if len(cal_eda) else 1.0
    baseline_feat = _extract(subj, cal_start, cal_end, eda_ref)
    if baseline_feat is None:
        return []

    windows: list[Window] = []
    for value in USED_LABELS:
        run = _longest_run(subj.label, value)
        if run is None:
            continue
        run_start_s, run_end_s = run[0] / LABEL_HZ, run[1] / LABEL_HZ
        start = run_start_s
        while start + WINDOW_S <= run_end_s:
            overlaps_cal = (value == 1) and (start < cal_end) and (start + WINDOW_S > cal_start)
            if not overlaps_cal:
                f = _extract(subj, start, start + WINDOW_S, eda_ref)
                if f is not None:
                    windows.append(Window(
                        subject=subj.subject,
                        label=value,
                        target=int(value == POSITIVE_LABEL),
                        start_s=start,
                        features=_relative(f, baseline_feat),
                    ))
            start += STEP_S
    return windows
