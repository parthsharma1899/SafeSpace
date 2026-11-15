import os
from pathlib import Path

import pytest

from safespace.models import ModalityScore, artifacts_dir


def test_available_score_must_be_in_range():
    ModalityScore(0.5, True)
    ModalityScore(None, False, "no face detected")
    with pytest.raises(ValueError):
        ModalityScore(1.5, True)
    with pytest.raises(ValueError):
        ModalityScore(None, True)        # available but no score
    with pytest.raises(ValueError):
        ModalityScore(0.5, False)        # unavailable but has a score


def test_unavailable_carries_a_reason():
    s = ModalityScore(None, False, "clip is silent")
    assert s.available is False
    assert s.score is None
    assert s.reason == "clip is silent"


def test_artifacts_dir_honours_the_environment(monkeypatch):
    monkeypatch.setenv("SAFESPACE_ARTIFACTS", "/tmp/somewhere")
    assert artifacts_dir() == Path("/tmp/somewhere")
    monkeypatch.delenv("SAFESPACE_ARTIFACTS", raising=False)
    assert artifacts_dir().name == "artifacts"
