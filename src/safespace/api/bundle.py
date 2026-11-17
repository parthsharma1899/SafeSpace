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


class RealBundle:
    def __init__(self):
        from ..models.face import FaceModel
        from ..models.text import TextModel
        from ..models.voice import VoiceModel
        from ..models.wrist import WristModel

        self._text = TextModel()
        self._voice = VoiceModel()
        self._face = FaceModel()
        self._wrist = WristModel()
        cal = _load_calibration()
        self._cal = {m: (cal.get(m, {}).get("a", 1.0), cal.get(m, {}).get("b", 0.0)) for m in ("wrist", "text", "voice", "face")}

    def load(self) -> "RealBundle":
        self._text.load()
        self._voice.load()
        self._wrist.load()
        if self._face.enabled:
            self._face.load()
        return self

    def health(self) -> dict:
        def ok(path_exists):
            return "ready" if path_exists else "unavailable"

        return {
            "wrist": ok((artifacts_dir() / "wrist" / "wrist_xgb.json").exists()),
            "text": ok((artifacts_dir() / "text" / "stress_classifier_epoch2" / "config.json").exists()),
            "voice": ok((artifacts_dir() / "voice" / "speech_model.keras").exists()),
            "face": "ready" if self._face.enabled else "disabled",
        }

    def score_text(self, text) -> ModalityScore:
        return _calibrated(self._text.predict(text), self._cal["text"])

    def score_voice(self, path) -> ModalityScore:
        return _calibrated(self._voice.predict(path), self._cal["voice"])

    def score_face(self, path) -> ModalityScore:
        return _calibrated(self._face.predict(path), self._cal["face"])

    def calibrate_wrist(self, pulse, conductance, rate_pulse, rate_conductance):
        return self._wrist.calibrate(pulse, conductance, rate_pulse, rate_conductance)

    def score_wrist(self, pulse, conductance, rate_pulse, rate_conductance, baseline) -> ModalityScore:
        return _calibrated(self._wrist.predict(pulse, conductance, rate_pulse, rate_conductance, baseline), self._cal["wrist"])


class StubBundle:
    """Fixed, weight-free models for tests. `scores` maps a modality to a number
    (or None to make it report unavailable)."""

    def __init__(self, scores: Optional[dict] = None, face_enabled: bool = True):
        self.scores = scores or {"wrist": 0.6, "text": 0.5, "voice": 0.4, "face": 0.3}
        self.face_enabled = face_enabled

    def load(self):
        return self

    def health(self) -> dict:
        return {"wrist": "ready", "text": "ready", "voice": "ready", "face": "ready" if self.face_enabled else "disabled"}

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
