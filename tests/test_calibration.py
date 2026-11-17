import numpy as np
import pytest

pytest.importorskip("sklearn")

from safespace.models.calibration import apply_platt, brier, fit_platt


def test_platt_improves_brier_on_a_miscalibrated_score():
    rng = np.random.default_rng(0)
    # True probability is a sharp function of a latent; the raw score is a
    # squashed, over-confident version of it.
    latent = rng.normal(size=4000)
    p_true = 1 / (1 + np.exp(-latent))
    labels = (rng.random(4000) < p_true).astype(int)
    raw = np.clip(0.5 + 0.1 * latent, 0, 1)  # under-confident raw score

    a, b = fit_platt(raw, labels)
    calibrated = np.array([apply_platt(s, a, b) for s in raw])
    assert brier(calibrated, labels) < brier(raw, labels)
    assert calibrated.min() >= 0 and calibrated.max() <= 1


def test_single_class_is_identity():
    assert fit_platt([0.2, 0.3, 0.4], [1, 1, 1]) == (1.0, 0.0)


def test_apply_is_monotonic():
    a, b = 1.5, -0.2
    xs = np.linspace(0.01, 0.99, 50)
    ys = [apply_platt(x, a, b) for x in xs]
    assert all(ys[i] <= ys[i + 1] + 1e-12 for i in range(len(ys) - 1))
