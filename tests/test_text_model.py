import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from safespace.models import artifacts_dir
from safespace.models.text import TextModel

HAVE_WEIGHTS = (artifacts_dir() / "text" / "stress_classifier_epoch2" / "config.json").exists()
needs_weights = pytest.mark.skipif(not HAVE_WEIGHTS, reason="text weights not in artifacts/")


def test_empty_text_is_unavailable():
    # No weights needed: empty input is rejected before the model loads.
    r = TextModel().predict("   ")
    assert r.available is False
    assert r.reason == "no text"


@needs_weights
def test_clear_stress_sentence():
    r = TextModel().predict("I'm really overwhelmed by all the tasks.")
    assert r.available is True
    assert r.score == pytest.approx(0.9432, abs=0.001)
