import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("tensorflow")

from safespace.models import artifacts_dir
from safespace.models.face import FaceModel

FACE_DIR = artifacts_dir() / "face"
HAVE_WEIGHTS = (FACE_DIR / "elm_weights-001.npz").exists()
HAVE_REFIT = (FACE_DIR / "refit" / "elm_B.npy").exists() and (FACE_DIR / "refit" / "platt.json").exists()
needs = pytest.mark.skipif(not HAVE_WEIGHTS, reason="face weights not in artifacts/")


def test_env_0_forces_off(monkeypatch):
    monkeypatch.setenv("SAFESPACE_ENABLE_FACE", "0")
    r = FaceModel().predict("whatever.jpg")
    assert r.available is False
    assert "disabled" in r.reason


def test_default_enabled_matches_refit_presence(monkeypatch):
    monkeypatch.delenv("SAFESPACE_ENABLE_FACE", raising=False)
    assert FaceModel().enabled is HAVE_REFIT


@needs
def test_blank_image_has_no_face(tmp_path, monkeypatch):
    import cv2

    monkeypatch.setenv("SAFESPACE_ENABLE_FACE", "1")
    blank = tmp_path / "blank.jpg"
    cv2.imwrite(str(blank), np.full((200, 200, 3), 200, np.uint8))
    r = FaceModel().predict(str(blank))
    assert r.available is False
    assert r.reason == "no face detected"


@needs
def test_unreadable_image(monkeypatch):
    monkeypatch.setenv("SAFESPACE_ENABLE_FACE", "1")
    r = FaceModel().predict("does_not_exist.jpg")
    assert r.available is False
    assert r.reason == "unreadable image"


@pytest.mark.skipif(not HAVE_REFIT, reason="refit not present")
def test_refit_scores_a_real_face(tmp_path, monkeypatch):
    """A synthetic face-like image, when a face is found, scores in [0, 1]."""
    import cv2

    monkeypatch.delenv("SAFESPACE_ENABLE_FACE", raising=False)
    monkeypatch.delenv("SAFESPACE_FACE_MODE", raising=False)
    model = FaceModel().load()
    # Score a crop directly (bypassing detection) to confirm the refit path returns a probability.
    crop = np.random.default_rng(0).integers(0, 255, (120, 120, 3), dtype=np.uint8)
    s = model._score_crop(crop)
    assert 0.0 <= s <= 1.0
