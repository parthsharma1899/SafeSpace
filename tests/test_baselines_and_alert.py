import pytest

from safespace.fusion import AlertRule, GuardedPaperWMSS, Instant, OracleKalman, Smoothed, StaticWeights, stress_level

MODS = ["wrist", "text", "voice", "face"]


def test_instant_rules():
    scores = {"wrist": 0.1, "text": 0.2, "voice": 0.9}
    assert Instant("mean").step(scores).score == pytest.approx(0.4)
    assert Instant("median").step(scores).score == pytest.approx(0.2)
    assert Instant("max").step(scores).score == pytest.approx(0.9)
    with pytest.raises(ValueError):
        Instant("mode")


def test_static_weights_renormalise_over_present_modalities():
    fuser = StaticWeights({"wrist": 3, "text": 1})
    assert fuser.step({"wrist": 1.0, "text": 0.0}).score == pytest.approx(0.75)
    assert fuser.step({"text": 0.4}).score == pytest.approx(0.4)


def test_smoothing_respects_elapsed_time():
    one_long = Smoothed(Instant("mean"), rate=0.3)
    two_short = Smoothed(Instant("mean"), rate=0.3)
    for f in (one_long, two_short):
        f.step({"wrist": 0.0})
    a = one_long.step({"wrist": 1.0}, dt=2.0).score
    two_short.step({"wrist": 1.0})
    b = two_short.step({"wrist": 1.0}).score
    assert a == pytest.approx(b) == pytest.approx(1 - 0.7**2)


def test_oracle_prefers_the_precise_modality():
    fuser = OracleKalman({"wrist": 0.001, "face": 0.1})
    result = fuser.step({"wrist": 0.2, "face": 0.8})
    assert result.weights["wrist"] > 0.95
    assert result.score < 0.25


def test_guarded_v1_counts_its_substitutions():
    fuser = GuardedPaperWMSS(MODS)
    assert fuser.step({}).score is None
    fuser.step({"wrist": 0.5, "text": 0.5})
    assert fuser.filled_steps == 1
    fuser.step(dict.fromkeys(MODS, 0.9))
    for _ in range(8):
        fuser.step(dict.fromkeys(MODS, 0.9))
    result = fuser.step(dict.fromkeys(MODS, 0.1))
    assert fuser.zero_weight_steps == 1
    assert result.score == pytest.approx(0.1)


def test_paper_thresholds():
    assert stress_level(0.39) == "Low"
    assert stress_level(0.4) == "Medium"
    assert stress_level(0.7) == "Medium"
    assert stress_level(0.71) == "High"


def test_alert_rises_at_once_and_clears_slowly():
    rule = AlertRule(clear_time=3.0)
    assert rule.update(0.2) == "Low"
    assert rule.update(0.8) == "High"
    assert rule.update(0.2) == "High"
    assert rule.update(0.2) == "High"
    assert rule.update(0.2) == "Medium"
    assert rule.update(None) == "Medium"
    for _ in range(2):
        assert rule.update(0.2) == "Medium"
    assert rule.update(0.2) == "Low"


def test_alert_does_not_clear_on_a_score_just_under_the_threshold():
    rule = AlertRule(margin=0.05, clear_time=2.0)
    rule.update(0.9)
    for _ in range(10):
        assert rule.update(0.68) == "High"
