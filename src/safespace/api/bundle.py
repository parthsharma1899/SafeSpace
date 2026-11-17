"""Holds the four modality models for the API and applies calibration.

`RealBundle` loads the trained models once; `StubBundle` returns fixed scores so
the API can be tested without any weights. Both expose the same methods, each
returning a ModalityScore.
"""

from __future__ import annotations

import json
from typing import Optional

from ..models.base import ModalityScore, artifacts_dir
from ..models.calibration import apply_platt


def _load_calibration() -> dict:
    path = artifacts_dir() / "calibration.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _calibrated(score: ModalityScore, ab) -> ModalityScore:
    if not score.available or ab is None:
        return score
    a, b = ab
    if (a, b) == (1.0, 0.0):
        return score
    return ModalityScore(round(apply_platt(score.score, a, b), 4), True)


def installed() -> dict[str, bool]:
    """Which models' weight files are present in artifacts/ (see scripts/download_models.py)."""
    a = artifacts_dir()
    face = a / "face"
    return {
        "wrist": (a / "wrist" / "wrist_xgb.json").is_file(),
        "text": (a / "text" / "stress_classifier_epoch2" / "model.safetensors").is_file(),
        "voice": (a / "voice" / "speech_model.keras").is_file(),
        "face": (face / "refit" / "elm_B.npy").is_file()
        and (face / "resnet50_features").is_dir()
        and ((face / "elm_hidden_f16.npz").is_file() or (face / "elm_weights-001.npz").is_file()),
    }


NOT_INSTALLED = "model not installed"


class RealBundle:
    """The trained models. A model whose weights are missing reports every input
    as unavailable instead of stopping the server."""

    is_stub = False

    def __init__(self):
        from ..models.face import FaceModel
        from ..models.text import TextModel
        from ..models.voice import VoiceModel
        from ..models.wrist import WristModel

        self._text = TextModel()
        self._voice = VoiceModel()
        self._face = FaceModel()
        self._wrist = WristModel()
        self._on = installed()
        self._on["face"] = self._on["face"] and self._face.enabled
        cal = _load_calibration()
        self._cal = {m: (cal.get(m, {}).get("a", 1.0), cal.get(m, {}).get("b", 0.0)) for m in ("wrist", "text", "voice", "face")}

    def load(self) -> "RealBundle":
        for name, model in (("text", self._text), ("voice", self._voice), ("wrist", self._wrist), ("face", self._face)):
            if self._on[name]:
                model.load()
        return self

    def health(self) -> dict:
        return {m: "ready" if on else "not installed" for m, on in self._on.items()}

    def score_text(self, text) -> ModalityScore:
        if not self._on["text"]:
            return ModalityScore(None, False, NOT_INSTALLED)
        return _calibrated(self._text.predict(text), self._cal["text"])

    def score_voice(self, path) -> ModalityScore:
        if not self._on["voice"]:
            return ModalityScore(None, False, NOT_INSTALLED)
        return _calibrated(self._voice.predict(path), self._cal["voice"])

    def score_face(self, path) -> ModalityScore:
        if not self._on["face"]:
            return ModalityScore(None, False, NOT_INSTALLED)
        return _calibrated(self._face.predict(path), self._cal["face"])

    def calibrate_wrist(self, pulse, conductance, rate_pulse, rate_conductance):
        if not self._on["wrist"]:
            return None
        return self._wrist.calibrate(pulse, conductance, rate_pulse, rate_conductance)

    def score_wrist(self, pulse, conductance, rate_pulse, rate_conductance, baseline) -> ModalityScore:
        if not self._on["wrist"]:
            return ModalityScore(None, False, NOT_INSTALLED)
        return _calibrated(self._wrist.predict(pulse, conductance, rate_pulse, rate_conductance, baseline), self._cal["wrist"])


class StubBundle:
    """Fixed, weight-free models for tests. `scores` maps a modality to a number
    (or None to make it report unavailable)."""

    is_stub = True

    def __init__(self, scores: Optional[dict] = None, face_enabled: bool = True):
        self.scores = scores or {"wrist": 0.6, "text": 0.5, "voice": 0.4, "face": 0.3}
        self.face_enabled = face_enabled

    def load(self):
        return self

    def health(self) -> dict:
        # "demo": these are fixed sample scores, not trained models.
        return {"wrist": "demo", "text": "demo", "voice": "demo", "face": "demo" if self.face_enabled else "disabled"}

    def _score(self, modality) -> ModalityScore:
        v = self.scores.get(modality)
        if v is None:
            return ModalityScore(None, False, "stub unavailable")
        return ModalityScore(float(v), True)

    def score_text(self, text) -> ModalityScore:
        if text is None or not str(text).strip():
            return ModalityScore(None, False, "no text")
        return self._score("text")

    def score_voice(self, path) -> ModalityScore:
        return self._score("voice")

    def score_face(self, path) -> ModalityScore:
        if not self.face_enabled:
            return ModalityScore(None, False, "face model disabled")
        return self._score("face")

    def calibrate_wrist(self, pulse, conductance, rate_pulse, rate_conductance):
        return object()  # a non-None baseline token

    def score_wrist(self, pulse, conductance, rate_pulse, rate_conductance, baseline) -> ModalityScore:
        if baseline is None:
            return ModalityScore(None, False, "no resting baseline")
        return self._score("wrist")
