"""Per-account storage for the web app.

An account is a signed-in Supabase user, identified only by the user id in
their sign-in token; no name or email is stored here. For each account we keep
when they agreed to have their scores saved, and one row per check-in with each
signal's score (empty if that signal was not used). The words, recordings and
photos a person sends are never written here; they are deleted as soon as they
are scored. Deleting an account's data removes every row.

SQLite, at data/app/safespace.db (or $SAFESPACE_DB). One server process.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SIGNALS = ("wrist", "text", "voice", "face")


def default_db_path() -> Path:
    env = os.environ.get("SAFESPACE_DB")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "data" / "app" / "safespace.db"


@dataclass(frozen=True)
class Account:
    user_id: str
    consented: Optional[float]  # when they agreed to have scores saved; None if not yet
    checkins: int


class AccountStore:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else default_db_path()
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS accounts ("
                " user_id TEXT PRIMARY KEY, created REAL NOT NULL, consented REAL)"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS account_checkins ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " user_id TEXT NOT NULL REFERENCES accounts(user_id) ON DELETE CASCADE,"
                " t REAL NOT NULL, wrist REAL, text REAL, voice REAL, face REAL)"
            )
            self._db.execute("CREATE INDEX IF NOT EXISTS account_checkins_user ON account_checkins(user_id, t)")

    def get(self, user_id: str) -> Account:
        """The account's state; an account never seen before has no consent and no check-ins."""
        with self._lock:
            row = self._db.execute("SELECT consented FROM accounts WHERE user_id = ?", (user_id,)).fetchone()
            count = self._db.execute("SELECT COUNT(*) FROM account_checkins WHERE user_id = ?", (user_id,)).fetchone()[0]
        return Account(user_id=user_id, consented=row[0] if row else None, checkins=count)

    def consent(self, user_id: str, now: Optional[float] = None) -> Account:
        now = time.time() if now is None else now
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO accounts (user_id, created, consented) VALUES (?, ?, ?)"
                " ON CONFLICT(user_id) DO UPDATE SET consented = COALESCE(accounts.consented, excluded.consented)",
                (user_id, now, now),
            )
        return self.get(user_id)

    def delete(self, user_id: str) -> int:
        """Remove the account's consent record and every saved check-in. Returns check-ins removed."""
        with self._lock, self._db:
            n = self._db.execute("SELECT COUNT(*) FROM account_checkins WHERE user_id = ?", (user_id,)).fetchone()[0]
            self._db.execute("DELETE FROM accounts WHERE user_id = ?", (user_id,))
        return n

    def add_checkin(self, user_id: str, t: float, scores: dict[str, float]) -> None:
        values = [scores.get(s) for s in SIGNALS]
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO account_checkins (user_id, t, wrist, text, voice, face) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, t, *values),
            )

    def checkins(self, user_id: str) -> list[tuple[float, dict[str, float]]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT t, wrist, text, voice, face FROM account_checkins WHERE user_id = ? ORDER BY t, id", (user_id,)
            ).fetchall()
        return [(row[0], {s: v for s, v in zip(SIGNALS, row[1:]) if v is not None}) for row in rows]
