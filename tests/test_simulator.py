import numpy as np
import pytest

from safespace.sim import MODALITIES, SCENARIOS, generate, methods, run_session, score_method
from safespace.sim.metrics import auc, evaluate, half_time
from safespace.sim.simulator import STEP_DOWN, STEP_UP


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_sessions_are_valid_and_repeatable(name):
    a, b = generate(SCENARIOS[name], 7), generate(SCENARIOS[name], 7)
    assert np.array_equal(a.truth, b.truth)
    assert set(a.scores) == set(MODALITIES)
    assert a.truth.min() >= 0 and a.truth.max() <= 1
    for m in MODALITIES:
        assert np.array_equal(a.scores[m], b.scores[m], equal_nan=True)
        seen = a.scores[m][~np.isnan(a.scores[m])]
        assert seen.min() >= 0 and seen.max() <= 1
    assert not np.array_equal(a.truth, generate(SCENARIOS[name], 8).truth) or name == "steps"


def test_same_seed_gives_the_same_truth_across_scenarios():
    assert np.array_equal(generate(SCENARIOS["clean"], 3).truth, generate(SCENARIOS["stuck"], 3).truth)


def test_gaps_match_the_requested_availability():
    share = {m: [] for m in MODALITIES}
    for seed in range(60):
        s = generate(SCENARIOS["missing"], seed)
        for m in MODALITIES:
            share[m].append(np.mean(~np.isnan(s.scores[m])))
    assert np.mean(share["wrist"]) == pytest.approx(0.95, abs=0.04)
    assert np.mean(share["face"]) == pytest.approx(0.70, abs=0.05)
    assert np.mean(share["voice"]) == pytest.approx(0.30, abs=0.05)
    assert np.mean(share["text"]) == pytest.approx(0.08, abs=0.02)


def test_faults_are_applied():
    stuck = generate(SCENARIOS["stuck"], 1)
    (m,) = stuck.faulty
    assert np.mean(stuck.scores[m] > 0.9) > 0.35
    flip = generate(SCENARIOS["flipping"], 1)
    assert set(np.unique(flip.scores[flip.faulty[0]])) == {0.0, 1.0}
    clean, biased = generate(SCENARIOS["clean"], 1), generate(SCENARIOS["biased"], 1)
    m = biased.faulty[0]
    assert np.mean(biased.scores[m] - clean.scores[m]) > 0.2


def test_step_scenario_changes_exactly_where_reaction_time_is_measured():
    z = generate(SCENARIOS["steps"], 0).truth
    assert z[STEP_UP - 1] == 0.2 and z[STEP_UP] == 0.85
    assert z[STEP_DOWN - 1] == 0.85 and z[STEP_DOWN] == 0.2


def test_auc():
    assert auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(np.array([1, 1, 0, 0]), np.array([0.1, 0.2, 0.8, 0.9])) == 0.0
    assert auc(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    assert auc(np.array([1, 1]), np.array([0.2, 0.3])) is None


def test_half_time():
    fused = np.array([0.2, 0.2, 0.3, 0.6, 0.85])
    assert half_time(fused, 1, 0.2, 0.85) == 2.0
    assert half_time(np.full(100, 0.2), 1, 0.2, 0.85, horizon=30) == 30.0


def test_evaluate_counts_missing_output_and_false_alarms():
    truth = np.array([0.1, 0.1, 0.9, 0.9])
    out = evaluate(truth, np.array([np.nan, 0.8, 0.9, 0.9]))
    assert out["failures"] == 1
    assert out["false_alarm"] == 1.0
    assert out["bias"] == pytest.approx(0.7 / 3)


def test_every_method_runs_on_every_scenario():
    for name, scenario in SCENARIOS.items():
        session = generate(scenario, 0)
        for label, factory in methods().items():
            fused = run_session(factory(session), session)
            ok = ~np.isnan(fused)
            assert ok.sum() > 200, (name, label)
            assert fused[ok].min() >= 0 and fused[ok].max() <= 1


def test_v1_substitutions_are_reported():
    rows = score_method(methods()["v1"], SCENARIOS["missing"], range(3))
    assert all(r["filled_steps"] > 0 for r in rows)
