"""Signed-in accounts: Supabase token checks, consent, saved scores, deletion."""

import io
import sqlite3
import time

import pytest

pytest.importorskip("fastapi")
jwt = pytest.importorskip("jwt")
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from safespace.api import bundle as bundle_mod  # noqa: E402
from safespace.api.app import _bundle_from_env, create_app  # noqa: E402
from safespace.api.auth import SupabaseVerifier, public_config  # noqa: E402
from safespace.api.bundle import StubBundle  # noqa: E402
from safespace.api.store import AccountStore  # noqa: E402
from safespace.api.tracker import BASELINE_READINGS  # noqa: E402

URL = "https://example-project.supabase.co"
KEY = ec.generate_private_key(ec.SECP256R1())
OTHER_KEY = ec.generate_private_key(ec.SECP256R1())


def token(sub="user-a", key=KEY, iss=f"{URL}/auth/v1", aud="authenticated", exp_in=3600):
    now = int(time.time())
    claims = {"sub": sub, "aud": aud, "iss": iss, "exp": now + exp_in, "iat": now, "email": f"{sub}@example.com"}
    return jwt.encode(claims, key, algorithm="ES256")


def auth(sub="user-a", **kw):
    return {"Authorization": f"Bearer {token(sub, **kw)}"}


@pytest.fixture
def db(tmp_path):
    return AccountStore(tmp_path / "accounts.db")


@pytest.fixture
def c(db):
    verifier = SupabaseVerifier(URL, keys=lambda _token: KEY.public_key())
    return TestClient(create_app(StubBundle(), db, verifier=verifier))


def checkin(c, sub="user-a", text="busy week"):
    return c.post("/me/checkins", data={"text": text}, headers=auth(sub))


# ---- sign-in ------------------------------------------------------------------

def test_account_endpoints_need_a_token(c):
    assert c.get("/me").status_code == 401
    assert c.get("/me", headers={"Authorization": "Basic abc"}).status_code == 401
    assert c.get("/me", headers={"Authorization": "Bearer not-a-jwt"}).status_code == 401


@pytest.mark.parametrize("bad", [
    {"key": OTHER_KEY},                       # signed by someone else
    {"exp_in": -10},                          # expired
    {"iss": "https://evil.supabase.co/auth/v1"},  # another project
    {"aud": "anon"},                          # not a signed-in user
])
def test_invalid_tokens_are_rejected(c, bad):
    assert c.get("/me", headers=auth(**bad)).status_code == 401


def test_unreachable_key_server_is_503_not_expired(db):
    """Regression: a failed fetch of Supabase's keys (e.g. no root certificates on
    macOS Python) was reported as an expired sign-in."""

    def unreachable(_token):
        raise jwt.PyJWKClientConnectionError("certificate verify failed")

    c = TestClient(create_app(StubBundle(), db, verifier=SupabaseVerifier(URL, keys=unreachable)))
    r = c.get("/me", headers=auth())
    assert r.status_code == 503


def test_key_fetch_uses_a_certificate_bundle():
    certifi = pytest.importorskip("certifi")
    from safespace.api.auth import _ssl_context

    ctx = _ssl_context()
    assert ctx.verify_mode.name == "CERT_REQUIRED"
    assert ctx.cert_store_stats()["x509_ca"] > 0, f"no CA certificates loaded (certifi at {certifi.where()})"


def test_without_supabase_configured_accounts_are_off():
    c = TestClient(create_app(StubBundle(), AccountStore(":memory:"), verifier=None))
    assert c.get("/me", headers=auth()).status_code == 503
    # Guest check-ins still work.
    sid = c.post("/sessions").json()["session_id"]
    assert c.post(f"/sessions/{sid}/observations", data={"text": "hi"}).status_code == 200


def test_public_config_never_contains_secrets(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", URL)
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "sb_publishable_abc")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "sb_secret_never")
    cfg = public_config()
    assert cfg == {"auth": {"url": URL, "publishable_key": "sb_publishable_abc"}}
    assert "never" not in str(cfg)


# ---- consent and saved scores ---------------------------------------------------

def test_scores_are_saved_only_after_consent(c):
    me = c.get("/me", headers=auth()).json()
    assert me == {"consented": False, "checkins": 0, "baseline_readings": BASELINE_READINGS, "latest": None}
    assert checkin(c).status_code == 403

    assert c.post("/me/consent", headers=auth()).json()["consented"] is True
    r = checkin(c)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["fused"] <= 1 and body["checkin_number"] == 1
    assert body["baseline_ready"] is False and body["baseline_progress"] == 1
    assert c.get("/me", headers=auth()).json()["checkins"] == 1


def test_only_scores_are_stored(c, db):
    c.post("/me/consent", headers=auth())
    c.post(
        "/me/checkins", data={"text": "my private words"}, headers=auth(),
        files={"audio": ("v.wav", io.BytesIO(b"RIFFvoice"), "audio/wav"), "image": ("f.jpg", io.BytesIO(b"jpegbytes"), "image/jpeg")},
    )
    con = sqlite3.connect(db.path)
    dump = "\n".join(con.iterdump())
    assert "my private words" not in dump and "RIFFvoice" not in dump and "jpegbytes" not in dump
    assert "user-a@example.com" not in dump  # no email either
    cols = [r[1] for r in con.execute("PRAGMA table_info(account_checkins)")]
    assert cols == ["id", "user_id", "t", "wrist", "text", "voice", "face"]


def test_people_see_only_their_own_history(c):
    for sub in ("user-a", "user-b"):
        c.post("/me/consent", headers=auth(sub))
    checkin(c, "user-a")
    checkin(c, "user-a")
    checkin(c, "user-b")
    assert len(c.get("/me/history", headers=auth("user-a")).json()["checkins"]) == 2
    assert len(c.get("/me/history", headers=auth("user-b")).json()["checkins"]) == 1


def test_baseline_builds_over_check_ins(c):
    c.post("/me/consent", headers=auth())
    results = [checkin(c).json() for _ in range(BASELINE_READINGS)]
    assert [r["baseline_progress"] for r in results] == list(range(1, BASELINE_READINGS + 1))
    assert [r["baseline_ready"] for r in results] == [False] * (BASELINE_READINGS - 1) + [True]
    latest = c.get("/me", headers=auth()).json()["latest"]
    assert latest["baseline_ready"] is True and latest["change"] is not None
    assert latest["baseline_counts"] == {"text": BASELINE_READINGS}


def test_each_signal_learns_its_own_normal(c):
    c.post("/me/consent", headers=auth())
    for _ in range(BASELINE_READINGS):
        checkin(c)
    voice = {"audio": ("v.wav", io.BytesIO(b"x"), "audio/wav")}
    body = c.post("/me/checkins", data={"text": "hi"}, files=voice, headers=auth()).json()
    # Text is learned; voice is new, so the reading is not yet compared with a full baseline.
    assert body["baseline_counts"] == {"text": BASELINE_READINGS, "voice": 1}
    assert body["baseline_ready"] is False and body["baseline_progress"] == 1


def test_a_check_in_with_nothing_readable_is_not_saved(db):
    verifier = SupabaseVerifier(URL, keys=lambda _token: KEY.public_key())
    c = TestClient(create_app(StubBundle(scores={"text": None}), db, verifier=verifier))
    c.post("/me/consent", headers=auth())
    body = checkin(c).json()
    assert body["fused"] is None
    assert c.get("/me", headers=auth()).json()["checkins"] == 0


def test_delete_my_data_removes_everything(c):
    c.post("/me/consent", headers=auth())
    checkin(c)
    checkin(c)
    assert c.delete("/me", headers=auth()).json() == {"deleted_checkins": 2}
    me = c.get("/me", headers=auth()).json()
    assert me["consented"] is False and me["checkins"] == 0 and me["latest"] is None
    assert checkin(c).status_code == 403  # must agree again before anything is saved


# ---- demo mode when no weights are installed -------------------------------------

def test_demo_mode_when_no_weights(monkeypatch, tmp_path):
    monkeypatch.setenv("SAFESPACE_ARTIFACTS", str(tmp_path))
    monkeypatch.delenv("SAFESPACE_API_STUB", raising=False)
    assert bundle_mod.installed() == {"wrist": False, "text": False, "voice": False, "face": False}
    b = _bundle_from_env()
    assert b.is_stub
    c = TestClient(create_app(b, AccountStore(":memory:"), verifier=None))
    health = c.get("/health").json()
    assert health["demo"] is True
    assert set(health["models"].values()) == {"demo"}  # never claims real models are ready
    assert c.get("/config").json()["demo"] is True
