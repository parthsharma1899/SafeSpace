"""Text stress model: the owner's fine-tuned BERT classifier, unchanged.

Weights live in artifacts/text/stress_classifier_epoch2/. The score is the
model's probability for class index 1 ("stressed"). Empty or whitespace-only
text is treated as missing, because the model returns about 0.41 for an empty
string, which is not a meaningful stress reading.
"""

from __future__ import annotations

from .base import ModalityScore, artifacts_dir

_MAX_TOKENS = 512


class TextModel:
    name = "text"

    def __init__(self, model_dir=None):
        self._dir = model_dir or (artifacts_dir() / "text" / "stress_classifier_epoch2")
        self._tokenizer = None
        self._model = None

    def load(self) -> "TextModel":
        if self._model is not None:
            return self
        import torch  # noqa: F401
        from transformers import BertForSequenceClassification, BertTokenizer

        self._tokenizer = BertTokenizer.from_pretrained(str(self._dir))
        self._model = BertForSequenceClassification.from_pretrained(str(self._dir))
        self._model.eval()
        return self

    def predict(self, text: str) -> ModalityScore:
        if text is None or not text.strip():
            return ModalityScore(None, False, "no text")
        self.load()
        import torch

        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=_MAX_TOKENS, padding=True)
        with torch.no_grad():
            logits = self._model(**inputs).logits
            prob = torch.nn.functional.softmax(logits, dim=1)[0][1].item()
        return ModalityScore(round(float(prob), 4), True)
