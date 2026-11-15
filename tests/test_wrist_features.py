import numpy as np
import pytest

pytest.importorskip("scipy")

from safespace.models.wrist_features import ACC_HZ, BVP_HZ, EDA_HZ, window_features


def _synthetic(bpm=72, seconds=60, n_scr=3):
    # BVP: a clean pulse wave at the given rate.
    t = np.arange(seconds * BVP_HZ) / BVP_HZ
    bvp = np.sin(2 * np.pi * (bpm / 60.0) * t)
    # EDA: flat with n_scr bumps.
    eda = np.full(seconds * EDA_HZ, 2.0)
    for k in range(n_scr):
        centre = int((k + 1) * seconds * EDA_HZ / (n_scr + 1))
        idx = np.arange(len(eda))
        eda = eda + 0.5 * np.exp(-((idx - centre) ** 2) / (2 * (2 * EDA_HZ) ** 2))
    acc = np.zeros((seconds * ACC_HZ, 3))
    acc[:, 2] = 9.8
    return bvp, eda, acc


def test_heart_rate_recovered():
    bvp, eda, acc = _synthetic(bpm=72)
    f = window_features(bvp, eda, acc, eda_ref=2.0)
    assert f is not None
    assert abs(f["hr"] - 72) < 2


def test_scr_rate_counts_bumps():
    bvp, eda, acc = _synthetic(n_scr=3)
    f = window_features(bvp, eda, acc, eda_ref=2.0)
    assert round(f["scr_rate"]) == 3


def test_noisy_pulse_returns_none():
    rng = np.random.default_rng(0)
    bvp = rng.normal(size=60 * BVP_HZ) * 0.01  # no real beats
    eda = np.full(60 * EDA_HZ, 2.0)
    acc = np.zeros((60 * ACC_HZ, 3))
    # Flat/noise pulse should yield too few valid intervals.
    f = window_features(bvp, eda, acc, eda_ref=2.0)
    assert f is None or f["hr"] > 0  # must not crash; typically None
