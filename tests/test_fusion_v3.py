"""Behaviour v3 is designed to have (the project's residual-persistence intent)."""

import numpy as np
import pytest

from safespace.fusion import ResidualPersistenceV3, V3Config

MODS = ["wrist", "text", "voice", "face"]


def run(fuser, rows):
    out = None
    for r in rows:
        out = fuser.step(r)
    return out


def calm(rng, level=0.2, noise=0.03):
    return {m: float(np.clip(level + rng.normal(0, noise), 0, 1)) for m in MODS}


def test_first_reading_is_the_plain_average():
    r = ResidualPersistenceV3(MODS).step({"wrist": 0.2, "text": 0.4, "voice": 0.6, "face": 0.8})
    assert r.score == pytest.approx(0.5)
    assert r.details["_v3"]["change"] == pytest.approx(0.0)
    assert set(r.weights.values()) == {0.25}


def test_a_spike_that_stays_takes_over_and_a_one_off_spike_does_not():
    rng = np.random.default_rng(0)
    stays, once = ResidualPersistenceV3(MODS), ResidualPersistenceV3(MODS)
    for f in (stays, once):
        run(f, [calm(rng) for _ in range(30)])
    spike = lambda: {**calm(rng), "wrist": 0.9}
    first = stays.step(spike())
    once.step(spike())
    # One spike moves the score but the signal has not yet earned extra weight.
    assert first.weights["wrist"] == pytest.approx(0.25, abs=0.05)
    assert first.score > first.details["_v3"]["normal"] + 0.07  # a noticeable nudge
    # If it stays, its weight grows and the score follows it.
    for _ in range(3):
        later = stays.step(spike())
    assert later.weights["wrist"] > 0.6
    assert later.score > 0.75
    # If it does not stay, the score goes back to normal.
    back = run(once, [calm(rng) for _ in range(4)])
    assert back.score < 0.3


def test_persistent_stress_is_not_learned_as_normal():
    rng = np.random.default_rng(1)
    f = ResidualPersistenceV3(MODS)
    run(f, [calm(rng) for _ in range(30)])
    high = run(f, [calm(rng, level=0.8) for _ in range(200)])
    assert high.score > 0.7
    assert high.details["wrist"]["baseline"] < 0.35  # the baseline did not drift up to 0.8


def test_recovery_counts_as_change():
    rng = np.random.default_rng(2)
    f = ResidualPersistenceV3(MODS)
    run(f, [calm(rng) for _ in range(30)])
    run(f, [{**calm(rng), "wrist": 0.9} for _ in range(10)])
    after = run(f, [calm(rng) for _ in range(6)])
    assert after.score < 0.3


def test_a_person_s_steady_high_signal_does_not_dominate_at_rest():
    """Someone whose text model always reads 0.8 on a calm day: no change, no extra weight."""
    rng = np.random.default_rng(3)
    f = ResidualPersistenceV3(MODS)
    r = run(f, [{**calm(rng), "text": float(0.8 + rng.normal(0, 0.03))} for _ in range(40)])
    assert max(r.weights.values()) < 0.3
    assert abs(r.details["_v3"]["change"]) < 0.05


def test_a_stuck_sensor_is_treated_as_missing():
    rng = np.random.default_rng(4)
    f = ResidualPersistenceV3(MODS)
    run(f, [calm(rng) for _ in range(20)])
    r = run(f, [{**calm(rng), "face": 0.97} for _ in range(14)])  # exactly constant: stuck
    assert "face" not in r.weights
    assert "face" in r.details["_v3"]["stuck"]
    assert r.score < 0.3


def test_cap_limits_one_signal_s_share():
    rng = np.random.default_rng(5)
    f = ResidualPersistenceV3(MODS, V3Config(cap=0.6, stuck_window=0))
    run(f, [calm(rng) for _ in range(30)])
    r = run(f, [{**calm(rng), "voice": 0.9} for _ in range(8)])
    assert r.weights["voice"] == pytest.approx(0.6)
    assert sum(r.weights.values()) == pytest.approx(1.0)


def test_calm_prior_starts_the_baseline_and_the_person_takes_over():
    f = ResidualPersistenceV3(["text"], V3Config(calm_prior={"text": 0.3}, stuck_window=0))
    first = f.step({"text": 0.6})
    assert first.details["text"]["baseline"] == pytest.approx(0.3)
    for v in [0.6, 0.58, 0.62, 0.6, 0.61, 0.59]:
        last = f.step({"text": v})
    assert last.details["text"]["baseline"] > 0.5  # learned this person's normal
    assert last.details["_v3"]["established"] == 1.0


def test_a_calm_steady_signal_is_not_mistaken_for_a_stuck_one():
    rng = np.random.default_rng(6)
    f = ResidualPersistenceV3(MODS)
    r = run(f, [calm(rng, noise=0.02) for _ in range(60)])
    assert r.details["_v3"]["stuck"] == []
    assert set(r.weights) == set(MODS)


def test_tuned_v3_matches_the_tuning_file():
    import json
    from pathlib import Path

    from safespace.fusion.v3 import TUNED_V3

    data = json.loads((Path(__file__).resolve().parents[1] / "results" / "fusion_tuning.json").read_text())
    for key, value in data["v3"].items():
        assert getattr(TUNED_V3, key) == value
