"""SafeSpace API: a stress check-in backend.

Create the app with `create_app(bundle, accounts, verifier)`, passing a model
bundle (RealBundle in production, StubBundle in tests), an account store and a
sign-in verifier. Run it with:

    uvicorn safespace.api.app:app --host 127.0.0.1 --port 8080

It binds to localhost and has no debug mode. Settings come from the environment
or the repo-root `.env` (see `.env.example`).

Two ways to check in:
- **Signed in** (`/me/...`): the person signs in with Supabase and agrees to have
  their scores saved. Each check-in's scores are kept, and the whole history is
  replayed through v3 so their baseline, residuals and persistence build across
  visits (tracker.py).
- **Guest** (`/sessions/...`): nothing is saved; scores live in memory for 30
  minutes. Also used by the wearable bridge.

Uploaded audio and images are written to a temporary file that is deleted before
the response is sent. Only scores and times are ever stored.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..models.base import ModalityScore
from .auth import AuthError, AuthUnavailable, Identity, bearer_token, public_config, verifier_from_env
from .bundle import RealBundle, StubBundle, installed
from .config import load_env
from .sessions import SessionStore
from .store import AccountStore
from .supabase_store import StoreUnavailable, SupabaseStore
from .support import support_message
from .tracker import BASELINE_READINGS, replay

MAX_UPLOAD = 16 * 1024 * 1024
log = logging.getLogger("safespace.api")
_UNSET = object()


async def _to_tempfile(upload: UploadFile, suffix: str) -> str:
    """Stream an upload to a temp file, enforcing the size cap. Caller deletes it."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    size = 0
    try:
        with os.fdopen(fd, "wb") as f:
            while True:
                chunk = await upload.read(1 << 20)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(status_code=413, detail="upload too large")
                f.write(chunk)
    except Exception:
        _safe_unlink(path)
        raise
    return path


def _safe_unlink(path: Optional[str]) -> None:
    if path:
        try:
            os.unlink(path)
        except OSError:
            pass


def _trend(history: list[dict]) -> str:
    if len(history) < 2 or history[-1]["fused"] is None or history[-2]["fused"] is None:
        return "steady"
    d = history[-1]["fused"] - history[-2]["fused"]
    return "rising" if d > 0.03 else "falling" if d < -0.03 else "steady"


def store_from_env():
    """Where signed-in people's scores are kept: SQLite locally (default), or
    Supabase's Postgres with SAFESPACE_STORE=supabase (the hosted app)."""
    if os.environ.get("SAFESPACE_STORE", "sqlite").strip().lower() == "supabase":
        url = os.environ.get("SUPABASE_URL", "").strip()
        key = os.environ.get("SUPABASE_PUBLISHABLE_KEY", "").strip()
        if not (url and key):
            raise RuntimeError("SAFESPACE_STORE=supabase needs SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY")
        return SupabaseStore(url, key)
    return AccountStore()


def create_app(bundle=None, accounts=None, verifier=_UNSET) -> FastAPI:
    # RealBundle() is cheap to construct (it loads weights only on .load(), which
    # runs on startup), so importing this module never loads the heavy models.
    bundle = bundle if bundle is not None else RealBundle()
    accounts = accounts if accounts is not None else store_from_env()
    verifier = verifier_from_env() if verifier is _UNSET else verifier
    sessions = SessionStore()

    @asynccontextmanager
    async def lifespan(_app):
        bundle.load()
        yield

    app = FastAPI(title="SafeSpace", docs_url=None, redoc_url=None, lifespan=lifespan)

    @app.exception_handler(StoreUnavailable)
    async def store_down(_request, exc):
        log.error("%s", exc)
        return JSONResponse(status_code=503, content={"detail": "saved data is unavailable right now"})

    # Allow the local dev/preview frontend (any localhost port) to call the API.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- sign-in --------------------------------------------------------------

    def signed_in(authorization: Optional[str] = Header(None)) -> Identity:
        if verifier is None:
            raise HTTPException(status_code=503, detail="sign-in is not set up on this server")
        try:
            return verifier.verify(bearer_token(authorization))
        except AuthUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))
        except AuthError as e:
            raise HTTPException(status_code=401, detail=str(e), headers={"WWW-Authenticate": "Bearer"})

    # ---- scoring the inputs of one check-in ---------------------------------

    async def score_inputs(text, wrist, audio, image, wrist_baseline) -> dict[str, dict]:
        modalities: dict[str, dict] = {}

        def record(name, score):
            modalities[name] = {"available": score.available, "score": score.score, "reason": score.reason}

        def safely(fn, *args, reason):
            # One unreadable input (e.g. an audio format the decoder can't open) must
            # mark only that signal unavailable, never fail the whole check-in.
            try:
                return fn(*args)
            except Exception:
                return ModalityScore(None, False, reason)

        if text is not None:
            record("text", safely(bundle.score_text, text, reason="could not process the text"))
        if wrist is not None:
            try:
                w = json.loads(wrist)
                args = (w["pulse"], w["conductance"], float(w["rate_pulse"]), float(w["rate_conductance"]), wrist_baseline)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                raise HTTPException(status_code=422, detail="wrist must be JSON with pulse, conductance, rate_pulse, rate_conductance")
            record("wrist", safely(bundle.score_wrist, *args, reason="could not process the wearable signal"))

        audio_path = image_path = None
        try:
            if audio is not None and audio.filename:
                audio_path = await _to_tempfile(audio, os.path.splitext(audio.filename)[1] or ".wav")
                record("voice", safely(bundle.score_voice, audio_path, reason="could not read the recording"))
            if image is not None and image.filename:
                image_path = await _to_tempfile(image, os.path.splitext(image.filename)[1] or ".jpg")
                record("face", safely(bundle.score_face, image_path, reason="could not read the photo"))
        finally:
            _safe_unlink(audio_path)
            _safe_unlink(image_path)
        return modalities

    def respond(modalities: dict[str, dict], history: list[dict]) -> dict:
        latest = history[-1]
        message = support_message(latest["estimate_level"], latest["dominant"], _trend(history)) if latest["estimate_level"] else ""
        return {
            **{k: v for k, v in latest.items() if k != "dominant"},
            "checkin_number": len(history),
            "modalities": modalities,
            "carried": [],
            "message": message,
        }

    def summary(item: dict) -> dict:
        return {k: item[k] for k in ("t", "fused", "change", "baseline_ready", "baseline_progress", "baseline_counts", "estimate_level")}

    # ---- general --------------------------------------------------------------

    @app.get("/health")
    def health():
        return {"models": bundle.health(), "demo": bundle.is_stub, "sign_in": verifier is not None}

    @app.get("/config")
    def config():
        return {**public_config(), "baseline_readings": BASELINE_READINGS, "demo": bundle.is_stub}

    # ---- signed-in accounts: scores are saved and the baseline builds -----------

    @app.get("/me")
    def me(who: Identity = Depends(signed_in)):
        account = accounts.get(who)
        history = replay(accounts.checkins(who))
        return {
            "consented": account.consented is not None,
            "checkins": account.checkins,
            "baseline_readings": BASELINE_READINGS,
            "latest": summary(history[-1]) if history else None,
        }

    @app.post("/me/consent")
    def give_consent(who: Identity = Depends(signed_in)):
        account = accounts.consent(who)
        return {"consented": True, "checkins": account.checkins}

    @app.post("/me/checkins")
    async def account_checkin(
        who: Identity = Depends(signed_in),
        text: Optional[str] = Form(None),
        audio: Optional[UploadFile] = File(None),
        image: Optional[UploadFile] = File(None),
    ):
        if accounts.get(who).consented is None:
            raise HTTPException(status_code=403, detail="agree to saving your scores first")
        modalities = await score_inputs(text, None, audio, image, None)
        scores = {m: d["score"] for m, d in modalities.items() if d["available"]}
        if not scores:
            return {"fused": None, "modalities": modalities, "checkin_number": accounts.get(who).checkins, "message": ""}
        accounts.add_checkin(who, time.time(), scores)
        return respond(modalities, replay(accounts.checkins(who)))

    @app.get("/me/history")
    def account_history(who: Identity = Depends(signed_in)):
        return {"checkins": [summary(h) for h in replay(accounts.checkins(who))]}

    @app.delete("/me")
    def delete_my_data(who: Identity = Depends(signed_in)):
        return {"deleted_checkins": accounts.delete(who)}

    # ---- guest sessions: kept in memory only (also the wearable bridge) ---------

    @app.post("/sessions")
    def create_session():
        return {"session_id": sessions.create().session_id}

    def _session(session_id: str):
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown session")
        return session

    @app.post("/sessions/{session_id}/baseline")
    async def set_baseline(session_id: str, payload: dict):
        session = _session(session_id)
        try:
            baseline = bundle.calibrate_wrist(
                payload["pulse"], payload["conductance"],
                float(payload["rate_pulse"]), float(payload["rate_conductance"]),
            )
        except (KeyError, TypeError, ValueError):
            raise HTTPException(status_code=422, detail="baseline needs pulse, conductance, rate_pulse, rate_conductance")
        if baseline is None:
            return {"ok": False, "reason": "need at least 120 s of resting signal"}
        session.baseline = baseline
        return {"ok": True}

    @app.post("/sessions/{session_id}/observations")
    async def observe(
        session_id: str,
        text: Optional[str] = Form(None),
        wrist: Optional[str] = Form(None),
        audio: Optional[UploadFile] = File(None),
        image: Optional[UploadFile] = File(None),
    ):
        session = _session(session_id)
        modalities = await score_inputs(text, wrist, audio, image, session.baseline)
        scores = {m: d["score"] for m, d in modalities.items() if d["available"]}
        if not scores:
            return {"fused": None, "modalities": modalities, "checkin_number": len(session.checkins), "message": ""}
        session.checkins.append((time.time(), scores))
        response = respond(modalities, replay(session.checkins))
        response["t"] = round(response["t"] - session.created, 2)
        session.history.append(response)
        return response

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str):
        return {"history": _session(session_id).history}

    @app.websocket("/sessions/{session_id}/wrist")
    async def wrist_stream(websocket: WebSocket, session_id: str):
        await websocket.accept()
        if sessions.get(session_id) is None:
            await websocket.close(code=4404)
            return
        try:
            while True:
                await websocket.receive_json()  # samples are buffered by the client/bridge
        except WebSocketDisconnect:
            return

    return app


def _bundle_from_env():
    """Real models, unless SAFESPACE_API_STUB=1 or no weights are installed at all,
    in which case the app runs in demo mode with fixed scores (and says so)."""
    if os.environ.get("SAFESPACE_API_STUB") == "1" or not any(installed().values()):
        return StubBundle()
    return RealBundle()


# Module-level app for `uvicorn safespace.api.app:app`. Models load on startup, not at import.
load_env()
app = create_app(_bundle_from_env())
