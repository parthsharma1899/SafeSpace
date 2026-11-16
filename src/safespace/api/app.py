"""SafeSpace API: a stress check-in backend.

Create the app with `create_app(bundle)`, passing a model bundle (RealBundle in
production, StubBundle in tests). Run it with:

    uvicorn safespace.api.app:app --host 127.0.0.1 --port 8080

It binds to localhost and has no debug mode. Uploaded audio and images are
written to a temporary file that is deleted before the response is sent; nothing
a user sends is logged or kept on disk.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from .bundle import RealBundle, StubBundle
from .sessions import SessionStore

MAX_UPLOAD = 16 * 1024 * 1024
MODALITIES = ("wrist", "text", "voice", "face")

_FIXED = {
    "Low": "You seem to be doing fine. Keep going.",
    "Medium": "Things look a little tense. A few slow breaths or a short break might help.",
    "High": "This looks like a stressful moment. It may help to pause, breathe slowly, and reach out to someone you trust.",
}


def _support_message(level: str, top_modality: Optional[str], trend: str) -> str:
    # Phase 7 replaces this with a Gemini call for the High case; the fixed text
    # here is exactly that call's fallback.
    try:
        from .support import support_message

        return support_message(level, top_modality, trend)
    except Exception:
        return _FIXED.get(level, _FIXED["Medium"])


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


def create_app(bundle=None) -> FastAPI:
    # RealBundle() is cheap to construct (it loads weights only on .load(), which
    # runs on startup), so importing this module never loads the heavy models.
    bundle = bundle if bundle is not None else RealBundle()
    store = SessionStore()

    @asynccontextmanager
    async def lifespan(_app):
        bundle.load()
        yield

    app = FastAPI(title="SafeSpace", docs_url=None, redoc_url=None, lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"models": bundle.health()}

    @app.post("/sessions")
    def create_session():
        return {"session_id": store.create().session_id}

    def _require(session_id: str):
        session = store.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown session")
        return session

    @app.post("/sessions/{session_id}/baseline")
    async def set_baseline(session_id: str, payload: dict):
        session = _require(session_id)
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
        session = _require(session_id)
        now = time.monotonic()
        modalities: dict[str, dict] = {}

        def record(name, score):
            modalities[name] = {"available": score.available, "score": score.score, "reason": score.reason}

        if text is not None:
            record("text", bundle.score_text(text))
        if wrist is not None:
            try:
                w = json.loads(wrist)
                score = bundle.score_wrist(w["pulse"], w["conductance"], float(w["rate_pulse"]), float(w["rate_conductance"]), session.baseline)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                raise HTTPException(status_code=422, detail="wrist must be JSON with pulse, conductance, rate_pulse, rate_conductance")
            record("wrist", score)

        audio_path = image_path = None
        try:
            if audio is not None and audio.filename:
                audio_path = await _to_tempfile(audio, os.path.splitext(audio.filename)[1] or ".wav")
                record("voice", bundle.score_voice(audio_path))
            if image is not None and image.filename:
                image_path = await _to_tempfile(image, os.path.splitext(image.filename)[1] or ".jpg")
                record("face", bundle.score_face(image_path))
        finally:
            _safe_unlink(audio_path)
            _safe_unlink(image_path)

        # A modality is passed to the fusion only when it produced a real score.
        scores = {m: d["score"] for m, d in modalities.items() if d["available"]}
        dt = session.dt_since_last(now)
        result = session.fuser.step(scores, dt=dt)
        level = session.alert.update(result.score, dt=dt)
        session._last_obs_time = now

        trend = "steady"
        fused_detail = result.details.get("_fused", {})
        if fused_detail.get("residual", 0) > 0.03:
            trend = "rising"
        elif fused_detail.get("residual", 0) < -0.03:
            trend = "falling"
        message = _support_message(level, result.dominant, trend) if result.score is not None else ""

        response = {
            "t": round(now - session.created, 2),
            "modalities": modalities,
            "fused": None if result.score is None else round(result.score, 4),
            "weights": {k: round(v, 4) for k, v in result.weights.items()},
            "spread": None if result.spread is None else round(result.spread, 4),
            "level": level,
            "message": message,
        }
        session.history.append(response)
        return response

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str):
        session = _require(session_id)
        return {"history": session.history}

    @app.websocket("/sessions/{session_id}/wrist")
    async def wrist_stream(websocket: WebSocket, session_id: str):
        await websocket.accept()
        session = store.get(session_id)
        if session is None:
            await websocket.close(code=4404)
            return
        try:
            while True:
                await websocket.receive_json()  # samples are buffered by the client/bridge
        except WebSocketDisconnect:
            return

    return app


# Module-level app for `uvicorn safespace.api.app:app`. Models load on startup,
# not at import. Set SAFESPACE_API_STUB=1 to serve fixed stub scores with no weights.
app = create_app(StubBundle()) if os.environ.get("SAFESPACE_API_STUB") == "1" else create_app()
