import io

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from safespace.api.app import create_app
from safespace.api.bundle import StubBundle


def client(**kwargs):
    return TestClient(create_app(StubBundle(**kwargs)))


def test_health_lists_modalities():
    r = client().get("/health")
    assert r.status_code == 200
    assert set(r.json()["models"]) == {"wrist", "text", "voice", "face"}


def test_create_session_returns_id():
    c = client()
    sid = c.post("/sessions").json()["session_id"]
    assert len(sid) == 32
    assert all(ch in "0123456789abcdef" for ch in sid)


def test_unknown_session_is_404():
    c = client()
    assert c.get("/sessions/deadbeef").status_code == 404
    assert c.post("/sessions/deadbeef/observations", data={"text": "hi"}).status_code == 404


def test_observation_fuses_available_modalities():
    c = client()
    sid = c.post("/sessions").json()["session_id"]
    r = c.post(f"/sessions/{sid}/observations", data={"text": "I feel stressed"})
    body = r.json()
    assert r.status_code == 200
    assert body["modalities"]["text"]["available"] is True
    assert 0.0 <= body["fused"] <= 1.0
    assert body["level"] in ("Low", "Medium", "High")
    assert set(body["weights"]) == {"text"}


def test_missing_modality_is_not_invented():
    # Rule 8: a modality whose model is unavailable is passed to fusion as missing.
    c = client(scores={"wrist": None, "text": 0.5, "voice": 0.4, "face": 0.3})
    sid = c.post("/sessions").json()["session_id"]
    wrist_json = '{"pulse": [0,1,0], "conductance": [1,1,1], "rate_pulse": 64, "rate_conductance": 4}'
    # No baseline set, so wrist is unavailable; it must not appear as a weighted modality.
    r = c.post(f"/sessions/{sid}/observations", data={"text": "hi", "wrist": wrist_json})
    body = r.json()
    assert body["modalities"]["wrist"]["available"] is False
    assert "wrist" not in body["weights"]


def test_disabled_face_is_unavailable():
    c = client(face_enabled=False)
    assert c.get("/health").json()["models"]["face"] == "disabled"


def test_oversized_upload_is_413():
    c = client()
    sid = c.post("/sessions").json()["session_id"]
    big = io.BytesIO(b"\0" * (16 * 1024 * 1024 + 1))
    r = c.post(f"/sessions/{sid}/observations", files={"audio": ("big.wav", big, "audio/wav")})
    assert r.status_code == 413


def test_two_sessions_are_independent():
    c = client()
    a = c.post("/sessions").json()["session_id"]
    b = c.post("/sessions").json()["session_id"]
    c.post(f"/sessions/{a}/observations", data={"text": "stressed"})
    assert len(c.get(f"/sessions/{a}").json()["history"]) == 1
    assert len(c.get(f"/sessions/{b}").json()["history"]) == 0


def test_baseline_then_wrist_scores():
    c = client()
    sid = c.post("/sessions").json()["session_id"]
    base = c.post(f"/sessions/{sid}/baseline", json={"pulse": [0, 1], "conductance": [1, 1], "rate_pulse": 64, "rate_conductance": 4})
    assert base.json()["ok"] is True
    wrist_json = '{"pulse": [0,1,0], "conductance": [1,1,1], "rate_pulse": 64, "rate_conductance": 4}'
    body = c.post(f"/sessions/{sid}/observations", data={"wrist": wrist_json}).json()
    assert body["modalities"]["wrist"]["available"] is True
