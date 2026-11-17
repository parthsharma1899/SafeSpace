"""v1 must be the paper's algorithm exactly: same numbers as the original code and as Table 15."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from safespace.fusion import MissingModalityError, PaperWMSS, ZeroTotalWeightError

FIVE = ["biomedical", "behavioral", "text", "voice", "facial"]
LEGACY = Path(__file__).resolve().parents[1] / "legacy" / "app" / "fusion_engine.py"

# Table 15 of the paper draft: (residual, persistence, final weight) per modality, per timestamp.
TABLE_15 = [
    [(0.0000, 0.9999, 0.3498), (0.0000, 0.8020, 0.2806), (0.0000, 0.2964, 0.1037), (0.0000, 0.3799, 0.1329), (0.0000, 0.3800, 0.1330)],
    [(-0.6999, 0.6999, 0.0000), (-0.1778, 0.7258, 0.1199), (0.4441, 0.4867, 0.5871), (0.0000, 0.3799, 0.1465), (0.0000, 0.3800, 0.1465)],
    [(0.2100, 0.7900, 0.2956), (0.0561, 0.7499, 0.2034), (0.3342, 0.6299, 0.3255), (0.0000, 0.3799, 0.0877), (0.0000, 0.3800, 0.0878)],
    [(-0.5530, 0.5530, 0.0000), (-0.1427, 0.6887, 0.1513), (0.2193, 0.7239, 0.5255), (0.0000, 0.3799, 0.1616), (0.0000, 0.3800, 0.1616)],
    [(-0.3871, 0.3871, 0.0000), (0.0821, 0.7239, 0.5089), (-0.2174, 0.6307, 0.0686), (0.0000, 0.3799, 0.2112), (0.0000, 0.3800, 0.2113)],
]


def test_reproduces_paper_table_15():
    """Scores are recovered as residual + persistence; the weights must match the table."""
    engine = PaperWMSS(FIVE)
    for row in TABLE_15:
        scores = {m: min(1.0, max(0.0, r + p)) for m, (r, p, _) in zip(FIVE, row)}
        result = engine.step(scores)
        for m, (residual, persistence, weight) in zip(FIVE, row):
            assert result.weights[m] == pytest.approx(weight, abs=2e-3)
            assert result.details[m]["residual"] == pytest.approx(residual, abs=2e-3)
            assert result.details[m]["persistence"] == pytest.approx(persistence, abs=2e-3)


@pytest.mark.skipif(not LEGACY.exists(), reason="original fusion_engine.py not present")
def test_identical_to_original_code():
    spec = importlib.util.spec_from_file_location("legacy_fusion_engine", LEGACY)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)

    rng = np.random.default_rng(0)
    for _ in range(200):
        old, new = legacy.FusionEngine(), PaperWMSS(FIVE)
        for _ in range(12):
            scores = {m: float(v) for m, v in zip(FIVE, rng.random(5).round(rng.integers(1, 5)))}
            try:
                expected = old.run_batch([scores])[0]
            except ZeroDivisionError:
                with pytest.raises(ZeroTotalWeightError):
                    new.step(scores)
                continue
            got = new.step(scores)
            assert round(got.score, 4) == expected["WMSS"]
            assert got.dominant == expected["Dominant Modality"]
            for m in FIVE:
                assert round(got.weights[m], 4) == expected["Normalized Weights"][m]


def test_first_step_is_the_contraharmonic_mean():
    """With no history the weights are proportional to the scores, so WMSS = sum(s^2) / sum(s)."""
    scores = {"a": 0.1, "b": 0.1, "c": 0.1, "d": 0.9}
    values = np.array(list(scores.values()))
    result = PaperWMSS(list(scores)).step(scores)
    assert result.score == pytest.approx((values**2).sum() / values.sum())
    assert result.score > values.mean()


def test_more_stress_can_lower_the_score():
    """Known flaw kept on record: three modalities getting more stressed lowers v1's output."""
    mods = ["a", "b", "c", "d"]
    calm_others = PaperWMSS(mods).step({"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.9}).score
    tenser_others = PaperWMSS(mods).step({"a": 0.1, "b": 0.1, "c": 0.1, "d": 0.9}).score
    assert calm_others == pytest.approx(0.9)
    assert tenser_others == pytest.approx(0.7)
    assert tenser_others < calm_others


def test_falling_score_gets_zero_weight():
    """Known flaw kept on record: the modality that reports recovery is ignored."""
    mods = ["a", "b", "c", "d"]
    engine = PaperWMSS(mods)
    for _ in range(10):
        engine.step(dict.fromkeys(mods, 0.8))
    result = engine.step({"a": 0.8, "b": 0.8, "c": 0.8, "d": 0.1})
    assert result.weights["d"] == 0
    assert result.score == pytest.approx(0.8)


def test_fails_when_everything_calms_down_at_once():
    mods = ["a", "b", "c", "d"]
    engine = PaperWMSS(mods)
    for _ in range(10):
        engine.step(dict.fromkeys(mods, 0.9))
    with pytest.raises(ZeroTotalWeightError):
        engine.step(dict.fromkeys(mods, 0.2))
    with pytest.raises(ZeroTotalWeightError):
        PaperWMSS(mods).step(dict.fromkeys(mods, 0.0))


def test_requires_every_modality():
    with pytest.raises(MissingModalityError):
        PaperWMSS(["a", "b"]).step({"a": 0.5})
    with pytest.raises(MissingModalityError):
        PaperWMSS(["a", "b"]).step({"a": 0.5, "b": None})


def test_error_text_is_rejected_instead_of_fused():
    """The original facial model returned strings such as 'Error: No face detected'."""
    engine = PaperWMSS(["a", "b"])
    with pytest.raises(TypeError):
        engine.step({"a": 0.5, "b": "Error: No face detected"})
    assert engine.step({"a": 0.5, "b": 0.5}).score == pytest.approx(0.5)
