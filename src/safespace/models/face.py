"""Face stress model: the owner's ResNet50 + ViT features into an ELM.

Two scoring paths share the owner's frozen ResNet, ViT and ELM hidden layer:

- **refit** (default when the refit files exist): the ELM output layer was
  re-solved on FER-2013 in 2026, because the ELM's saved output weights produced
  chance scores. Images are prepared as in training the re-fit (ResNet via
  resnet50.preprocess_input, ViT scaled to [-1, 1]); the raw ELM output is mapped
  to a probability by a logistic fitted on the FER-2013 validation split. Test
  AUC 0.70. The saved ViT extractor holds untrained weights, so this path is
  driven mainly by the ResNet features; see docs/models/face.md.
- **original**: the paper's exact computation (/255 for both extractors, the
  saved output weights). Kept for reference; it scores at chance.

The model is enabled by default when the refit files are present. Set
SAFESPACE_ENABLE_FACE=0 to force it off, or =1 to force it on. The scoring path
is chosen by SAFESPACE_FACE_MODE ("auto" = refit if available, "original", or
"refit"); it defaults to "auto".
"""

from __future__ import annotations

import json
import os

import numpy as np

from .base import ModalityScore, artifacts_dir

_ON = {"1", "true", "yes", "on"}
_OFF = {"0", "false", "no", "off"}


def _safe_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


class FaceModel:
    name = "face"

    def __init__(self, face_dir=None):
        self._dir = face_dir or (artifacts_dir() / "face")
        self._c = None

    def _refit_available(self) -> bool:
        return (self._dir / "refit" / "elm_B.npy").exists() and (self._dir / "refit" / "platt.json").exists()

    @property
    def enabled(self) -> bool:
        env = os.environ.get("SAFESPACE_ENABLE_FACE", "").strip().lower()
        if env in _OFF:
            return False
        if env in _ON:
            return True
        return self._refit_available()  # default: on only when a working refit exists

    @property
    def use_refit(self) -> bool:
        mode = os.environ.get("SAFESPACE_FACE_MODE", "auto").strip().lower()
        if mode == "original":
            return False
        if mode == "refit":
            return True
        return self._refit_available()  # "auto"

    def load(self) -> "FaceModel":
        if self._c is not None:
            return self
        import cv2
        import tensorflow as tf

        elm = np.load(self._dir / "elm_weights-001.npz")
        resnet = tf.saved_model.load(str(self._dir / "resnet50_features"))
        vit = tf.saved_model.load(str(self._dir / "vit_features"))
        c = {
            "W": elm["W"], "b": elm["b"],
            "resnet": resnet.signatures["serving_default"],
            "vit": vit.signatures["serving_default"],
            "cascade": cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"),
        }
        if self._refit_available():
            c["mean"] = np.load(self._dir / "refit" / "scaler_mean.npy")
            c["std"] = np.load(self._dir / "refit" / "scaler_std.npy")
            c["B"] = np.load(self._dir / "refit" / "elm_B.npy")
            platt = json.loads((self._dir / "refit" / "platt.json").read_text())
            c["platt"] = (platt["a"], platt["b"])
        scale = np.load(self._dir / "features" / "scaler_scale.npy")
        scale[scale == 0] = 1.0
        c["orig_mean"] = np.load(self._dir / "features" / "scaler_mean.npy")
        c["orig_scale"] = scale
        c["orig_B"] = elm["B"]
        self._c = c
        return self

    def _features(self, face_bgr, refit: bool):
        import cv2
        import tensorflow as tf

        rgb = cv2.cvtColor(cv2.resize(face_bgr, (224, 224)), cv2.COLOR_BGR2RGB).astype("float32")
        if refit:
            from tensorflow.keras.applications.resnet50 import preprocess_input

            r_in = tf.convert_to_tensor(preprocess_input(rgb[None].copy()))
            v_in = tf.convert_to_tensor(rgb[None] / 127.5 - 1.0)
        else:
            r_in = v_in = tf.convert_to_tensor(rgb[None] / 255.0)
        r = next(iter(self._c["resnet"](r_in).values())).numpy().flatten()
        v = next(iter(self._c["vit"](v_in).values())).numpy().flatten()
        return np.concatenate([r, v])

    def _score_crop(self, face_bgr) -> float:
        c = self._c
        refit = self.use_refit
        f = self._features(face_bgr, refit)
        if refit:
            fn = (f - c["mean"]) / c["std"]
            z = fn @ c["W"] + c["b"]
            raw = (np.maximum(0.01 * z, z) @ c["B"]).item()
            a, b = c["platt"]
            return float(1.0 / (1.0 + np.exp(-(a * raw + b))))
        fn = _safe_divide(f - c["orig_mean"], c["orig_scale"])
        z = fn @ c["W"] + c["b"]
        return float(np.clip((np.maximum(0.01 * z, z) @ c["orig_B"]).item(), 0, 1))

    def predict(self, image_path) -> ModalityScore:
        if not self.enabled:
            return ModalityScore(None, False, "face model disabled")
        import cv2

        self.load()
        img = cv2.imread(str(image_path))
        if img is None:
            return ModalityScore(None, False, "unreadable image")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self._c["cascade"].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return ModalityScore(None, False, "no face detected")
        x, y, w, h = faces[0]
        return ModalityScore(round(self._score_crop(img[y:y + h, x:x + w]), 4), True)
