import sys
import types

import pytest

from safespace.api import support


def test_low_and_medium_are_fixed(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert support.support_message("Low", "wrist", "steady") == support.LOW
    assert support.support_message("Medium", "text", "rising") == support.MEDIUM


def test_high_without_key_uses_fallback(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert support.support_message("High", "voice", "rising") == support.HIGH_FALLBACK


def _install_fake_genai(monkeypatch, capture, behaviour):
    """Install a fake `google.genai` module. `behaviour` runs with the prompt."""
    genai = types.ModuleType("google.genai")

    class FakeClient:
        def __init__(self, api_key=None):
            pass

        class models:  # noqa: N801
            pass

    def generate_content(self, model=None, contents=None):
        capture["model"] = model
        capture["contents"] = contents
        return behaviour(contents)

    FakeClient.models = types.SimpleNamespace(generate_content=None)

    class Models:
        def generate_content(self, model=None, contents=None):
            capture["model"] = model
            capture["contents"] = contents
            return behaviour(contents)

    class Client:
        def __init__(self, api_key=None):
            self.models = Models()

    genai.Client = Client
    google = sys.modules.get("google") or types.ModuleType("google")
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setattr(google, "genai", genai, raising=False)


def test_high_with_key_generates_and_sends_no_user_data(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")
    capture = {}
    _install_fake_genai(monkeypatch, capture, lambda c: types.SimpleNamespace(text="Take a slow breath."))

    msg = support.support_message("High", "wrist", "rising")
    assert msg == "Take a slow breath."
    # The request must contain only level/modality/trend, never user content.
    sent = capture["contents"]
    assert "wrist" in sent and "rising" in sent
    assert "do not diagnose" in sent.lower()
    # It is exactly the instruction template filled in -- nothing else leaks.
    assert sent == support._INSTRUCTION.format(modality="wrist", trend="rising")
    assert capture["model"] == "gemini-test"


def test_high_with_key_but_api_error_falls_back(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")

    def boom(_contents):
        raise RuntimeError("api down")

    _install_fake_genai(monkeypatch, {}, boom)
    assert support.support_message("High", "face", "steady") == support.HIGH_FALLBACK


def test_high_with_key_but_no_model_falls_back(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    _install_fake_genai(monkeypatch, {}, lambda c: types.SimpleNamespace(text="should not be used"))
    assert support.support_message("High", "voice", "rising") == support.HIGH_FALLBACK
