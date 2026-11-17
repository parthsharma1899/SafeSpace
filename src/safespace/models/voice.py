"""Voice stress model: the owner's CNN on MFCCs, unchanged.

The feature code below is a character-for-character copy of `extract_features`
in legacy/app/models/voice_model.py, so the wrapper and the original give
identical scores. Weights live in artifacts/voice/speech_model.keras.

A clip shorter than 1 second, or one that is essentially silent, is treated as
missing rather than scored.
"""

from __future__ import annotations

import numpy as np

from .base import ModalityScore, artifacts_dir

_MIN_SECONDS = 1.0
_SILENCE_PEAK = 0.001


def extract_features(file_path, max_len=216):
    import librosa

    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    padded = np.pad(mfccs, ((0, 0), (0, max(0, max_len - mfccs.shape[1]))), mode='constant')
    return padded[:, :max_len]


class VoiceModel:
    name = "voice"

    def __init__(self, model_path=None):
        self._path = model_path or (artifacts_dir() / "voice" / "speech_model.keras")
        self._model = None

    def load(self) -> "VoiceModel":
        if self._model is None:
            from tensorflow.keras.models import load_model

            self._model = load_model(str(self._path))
        return self

    def _availability(self, file_path) -> str:
        import librosa

        audio, sr = librosa.load(file_path, sr=None)
        if len(audio) / sr < _MIN_SECONDS:
            return "clip shorter than 1 s"
        if float(np.max(np.abs(audio))) < _SILENCE_PEAK:
            return "clip is silent"
        return ""

    def predict(self, file_path) -> ModalityScore:
        reason = self._availability(file_path)
        if reason:
            return ModalityScore(None, False, reason)
        self.load()
        features = extract_features(file_path)
        features = np.expand_dims(features, axis=-1)
        features = np.expand_dims(features, axis=0)
        score = self._model.predict(features, verbose=0)[0][0]
        return ModalityScore(round(float(score), 4), True)
