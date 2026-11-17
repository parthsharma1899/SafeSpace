"""Guest sessions, kept in memory only (guest check-ins and the wearable bridge).

A session holds its check-ins' scores in memory and is dropped after 30 minutes
of inactivity. Signed-in accounts, whose scores are saved, live in `store.py`.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Optional

SESSION_TTL = 30 * 60.0


@dataclass
class Session:
    session_id: str
    created: float
    last_seen: float
    baseline: object = None  # a wrist Baseline, or None
    checkins: list = field(default_factory=list)  # (time, scores)
    history: list = field(default_factory=list)   # responses, for GET /sessions/{id}

    def touch(self, now: float) -> None:
        self.last_seen = now


class SessionStore:
    def __init__(self, ttl: float = SESSION_TTL, clock=time.time):
        self._ttl = ttl
        self._clock = clock
        self._sessions: dict[str, Session] = {}

    def _prune(self) -> None:
        now = self._clock()
        for sid in [sid for sid, s in self._sessions.items() if now - s.last_seen > self._ttl]:
            del self._sessions[sid]

    def create(self) -> Session:
        self._prune()
        now = self._clock()
        session = Session(session_id=secrets.token_hex(16), created=now, last_seen=now)
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        self._prune()
        session = self._sessions.get(session_id)
        if session is not None:
            session.touch(self._clock())
        return session

    def __len__(self) -> int:
        return len(self._sessions)
