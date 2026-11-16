"""Per-session fusion state for the API.

Each check-in session has its own fusion engine, alert rule, resting baseline and
history. Sessions expire after 30 minutes of inactivity. Nothing a user sends is
written to disk here; only the resulting scores and timings are held in memory.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Optional

from ..fusion import AlertRule, ResidualPersistenceV2, TUNED
from ..sim.simulator import MODALITIES

DT_SCALE = 30.0           # one "step" of fusion time = 30 real seconds
SESSION_TTL = 30 * 60.0   # seconds of inactivity before a session is dropped


@dataclass
class Session:
    session_id: str
    fuser: ResidualPersistenceV2
    alert: AlertRule
    created: float
    last_seen: float
    baseline: object = None  # a wrist Baseline, or None
    history: list = field(default_factory=list)
    _last_obs_time: Optional[float] = None

    def dt_since_last(self, now: float) -> float:
        if self._last_obs_time is None:
            return 1.0
        return max(1e-6, (now - self._last_obs_time) / DT_SCALE)

    def touch(self, now: float) -> None:
        self.last_seen = now


class SessionStore:
    def __init__(self, ttl: float = SESSION_TTL, clock=time.monotonic):
        self._ttl = ttl
        self._clock = clock
        self._sessions: dict[str, Session] = {}

    def _prune(self) -> None:
        now = self._clock()
        dead = [sid for sid, s in self._sessions.items() if now - s.last_seen > self._ttl]
        for sid in dead:
            del self._sessions[sid]

    def create(self) -> Session:
        self._prune()
        now = self._clock()
        sid = secrets.token_hex(16)
        session = Session(
            session_id=sid,
            fuser=ResidualPersistenceV2(MODALITIES, TUNED),
            alert=AlertRule(),
            created=now,
            last_seen=now,
        )
        self._sessions[sid] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        self._prune()
        session = self._sessions.get(session_id)
        if session is not None:
            session.touch(self._clock())
        return session

    def __len__(self) -> int:
        return len(self._sessions)
