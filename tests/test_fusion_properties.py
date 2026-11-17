"""Requirements 1 to 5 of docs/protocol.md, checked on many random inputs."""

import copy

import numpy as np
import pytest

import json
from pathlib import Path

from safespace.fusion import TUNED, Instant, OracleKalman, ResidualPersistenceV2, Smoothed, StaticWeights, V2Config

MODS = ["wrist", "text", "voice", "face"]


def make_fusers():
    return [
        ResidualPersistenceV2(MODS),
        ResidualPersistenceV2(MODS, TUNED, name="v2-tuned"),
        ResidualPersistenceV2(MODS, V2Config(carry=6.0), name="v2-carry"),
        ResidualPersistenceV2(MODS, V2Config(use_variance=False), name="v2-no-variance"),
        ResidualPersistenceV2(MODS, V2Config(use_offset=False), name="v2-no-offset"),
        ResidualPersistenceV2(MODS, V2Config(adaptive_gain=False), name="v2-fixed-gain"),
        ResidualPersistenceV2(MODS, V2Config(reliability={"wrist": 3.0, "face": 0.5}), name="v2-reliability"),
        Instant("mean"),
        Instant("median"),
        Instant("max"),
        Smoothed(Instant("mean")),
        Smoothed(Instant("median")),
        StaticWeights({"wrist": 2, "text": 1, "voice": 1, "face": 0.5}),
        OracleKalman({"wrist": 0.01, "text": 0.04, "voice": 0.02, "face": 0.05}),
    ]


FUSERS = make_fusers()
IDS = [f.name for f in FUSERS]


def random_step(rng, p_missing=0.3):
    """One time step: each modality is missing, None, NaN, or a score, sometimes an extreme one."""
    scores = {}
    for m in MODS:
        u = rng.random()
        if u < p_missing / 3:
            continue
        if u < 2 * p_missing / 3:
            scores[m] = None
        elif u < p_missing:
            scores[m] = float("nan")
        else:
            scores[m] = float(rng.choice([0.0, 1.0, rng.random(), rng.random()]))
    return scores


@pytest.mark.parametrize("fuser", FUSERS, ids=IDS)
def test_bounded_and_never_fails(fuser):
    rng = np.random.default_rng(1)
    for _ in range(60):
        fuser.reset()
        seen_any = False
        for _ in range(40):
            scores = random_step(rng)
            result = fuser.step(scores, dt=float(rng.choice([0.25, 1.0, 1.0, 5.0])))
            seen_any = seen_any or any(v is not None and v == v for v in scores.values())
            if seen_any:
                assert result.score is not None and 0.0 <= result.score <= 1.0
            else:
                assert result.score is None
            if result.weights:
                assert sum(result.weights.values()) == pytest.approx(1.0)
                assert all(w >= 0 for w in result.weights.values())


@pytest.mark.parametrize("fuser", FUSERS, ids=IDS)
def test_all_zero_and_all_one_inputs(fuser):
    """The inputs that crashed v1."""
    for value in (0.0, 1.0):
        fuser.reset()
        for _ in range(5):
            assert fuser.step(dict.fromkeys(MODS, value)).score == pytest.approx(value)
    fuser.reset()
    for _ in range(10):
        fuser.step(dict.fromkeys(MODS, 0.9))
    assert 0.0 <= fuser.step(dict.fromkeys(MODS, 0.2)).score <= 0.9


@pytest.mark.parametrize("fuser", FUSERS, ids=IDS)
def test_agreement_is_passed_through(fuser):
    rng = np.random.default_rng(2)
    for value in rng.random(20):
        fuser.reset()
        for _ in range(8):
            assert fuser.step(dict.fromkeys(MODS, float(value))).score == pytest.approx(float(value))


@pytest.mark.parametrize("fuser", FUSERS, ids=IDS)
def test_raising_a_score_never_lowers_the_result(fuser):
    rng = np.random.default_rng(3)
    for _ in range(300):
        fuser.reset()
        for _ in range(int(rng.integers(0, 15))):
            fuser.step(random_step(rng), dt=float(rng.choice([0.5, 1.0, 3.0])))
        scores = random_step(rng, p_missing=0.2)
        present = [m for m, v in scores.items() if v is not None and v == v]
        if not present:
            continue
        m = str(rng.choice(present))
        raised = dict(scores)
        raised[m] = min(1.0, scores[m] + float(rng.random()) * (1 - scores[m]))
        twin = copy.deepcopy(fuser)
        low = fuser.step(scores).score
        high = twin.step(raised).score
        assert high >= low - 1e-12


@pytest.mark.parametrize("fuser", FUSERS, ids=IDS)
def test_missing_modality_is_left_out_not_invented(fuser):
    fuser.reset()
    result = fuser.step({"wrist": 0.2, "text": None, "voice": float("nan")})
    assert set(result.weights) == {"wrist"}
    assert result.available == ("wrist",)
    assert result.score == pytest.approx(0.2)


@pytest.mark.parametrize("fuser", FUSERS, ids=IDS)
def test_holds_last_score_when_nothing_is_available(fuser):
    fuser.reset()
    assert fuser.step({}).score is None
    first = fuser.step({"wrist": 0.6, "face": 0.4}).score
    held = fuser.step({"wrist": None})
    assert held.score == pytest.approx(first)
    assert held.available == ()


@pytest.mark.parametrize("fuser", FUSERS, ids=IDS)
def test_rejects_invalid_scores(fuser):
    fuser.reset()
    with pytest.raises(TypeError):
        fuser.step({"wrist": "Error: No face detected"})
    with pytest.raises(ValueError):
        fuser.step({"wrist": 1.2})
    with pytest.raises(ValueError):
        fuser.step({"wrist": -0.1})


@pytest.mark.parametrize("fuser", [f for f in FUSERS if f.name.startswith("v2")], ids=[n for n in IDS if n.startswith("v2")])
def test_v2_treats_rises_and_falls_alike(fuser):
    """Feeding 1 - score must give 1 - result: no built-in tilt towards high or low."""
    rng = np.random.default_rng(4)
    for _ in range(40):
        mirror = copy.deepcopy(fuser)
        fuser.reset()
        mirror.reset()
        for _ in range(30):
            scores = {m: v for m, v in random_step(rng).items() if v is not None and v == v}
            a = fuser.step(scores).score
            b = mirror.step({m: 1 - v for m, v in scores.items()}).score
            if a is not None:
                assert a + b == pytest.approx(1.0)


def test_v2_turns_down_a_modality_that_stays_away_from_the_others():
    fuser = ResidualPersistenceV2(MODS)
    for _ in range(30):
        result = fuser.step({"wrist": 0.95, "text": 0.2, "voice": 0.2, "face": 0.2})
    assert result.weights["wrist"] < 0.1
    assert result.score < 0.3


def test_v2_turns_down_a_jumpy_modality():
    rng = np.random.default_rng(5)
    fuser = ResidualPersistenceV2(MODS)
    for _ in range(200):
        result = fuser.step({"wrist": 0.5, "text": 0.5, "voice": 0.5, "face": float(rng.integers(0, 2))})
    assert result.weights["face"] < 0.1


def test_v2_follows_a_shared_change_in_both_directions():
    fuser = ResidualPersistenceV2(MODS)
    for _ in range(20):
        fuser.step(dict.fromkeys(MODS, 0.2))
    up = fuser.step(dict.fromkeys(MODS, 0.9)).score
    assert up > 0.75
    for _ in range(20):
        fuser.step(dict.fromkeys(MODS, 0.9))
    down = fuser.step(dict.fromkeys(MODS, 0.2)).score
    assert down < 0.35


def test_v2_carry_keeps_a_recent_reading_and_then_lets_it_go():
    fuser = ResidualPersistenceV2(MODS, V2Config(carry=4.0))
    fuser.step({"wrist": 0.3, "text": 0.9})
    soon = fuser.step({"wrist": 0.3})
    assert "text" in soon.weights and soon.available == ("wrist",)
    assert soon.weights["text"] < soon.weights["wrist"]
    for _ in range(4):
        later = fuser.step({"wrist": 0.3})
    assert "text" not in later.weights


def test_tuned_matches_the_tuning_file():
    """The named TUNED constant must equal the v2 block chosen by the benchmark."""
    data = json.loads((Path(__file__).resolve().parents[1] / "results" / "fusion_tuning.json").read_text())
    for key, value in data["v2"].items():
        assert getattr(TUNED, key) == value
