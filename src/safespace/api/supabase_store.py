"""Per-account storage in Supabase's Postgres, for the hosted app.

Same interface and data as store.AccountStore (scores and times only), kept in
the tables created by deploy/supabase.sql. Every request is made *as the
signed-in person*: it carries the project's publishable key and that person's
own access token, so Postgres row-level security lets it see and change only
their rows. The server holds no Supabase secret.

Selected with SAFESPACE_STORE=supabase (the hosted Space sets this).
"""

from __future__ import annotations

from typing import Optional

import httpx

from .store import SIGNALS, Account


class StoreUnavailable(Exception):
    """Supabase could not be reached or refused the request."""


class SupabaseStore:
    def __init__(self, url: str, publishable_key: str, transport: Optional[httpx.BaseTransport] = None):
        self._rest = url.rstrip("/") + "/rest/v1"
        self._key = publishable_key
        self._http = httpx.Client(timeout=10.0, transport=transport)

    def _call(self, who, method: str, table: str, *, params=None, json=None, prefer: Optional[str] = None) -> httpx.Response:
        headers = {"apikey": self._key, "Authorization": f"Bearer {who.token}"}
        if prefer:
            headers["Prefer"] = prefer
        try:
            r = self._http.request(method, f"{self._rest}/{table}", params=params, json=json, headers=headers)
        except httpx.HTTPError as e:
            raise StoreUnavailable(f"cannot reach Supabase: {e}") from e
        if r.status_code >= 400:
            raise StoreUnavailable(f"Supabase {method} {table}: {r.status_code} {r.text[:200]}")
        return r

    @staticmethod
    def _total(r: httpx.Response) -> int:
        # Content-Range: "0-4/5" or "*/0"
        return int(r.headers.get("content-range", "*/0").rsplit("/", 1)[-1])

    def get(self, who) -> Account:
        consent = self._call(who, "GET", "consents", params={"select": "consented_at"}).json()
        count = self._total(self._call(who, "HEAD", "checkins", params={"select": "id"}, prefer="count=exact"))
        return Account(user_id=who.user_id, consented=1.0 if consent else None, checkins=count)

    def consent(self, who, now: Optional[float] = None) -> Account:
        self._call(who, "POST", "consents", params={"on_conflict": "user_id"}, json={}, prefer="resolution=ignore-duplicates,return=minimal")
        return self.get(who)

    def delete(self, who) -> int:
        mine = {"user_id": f"eq.{who.user_id}"}
        removed = self._total(self._call(who, "DELETE", "checkins", params=mine, prefer="count=exact,return=minimal"))
        self._call(who, "DELETE", "consents", params=mine, prefer="return=minimal")
        return removed

    def add_checkin(self, who, t: float, scores: dict[str, float]) -> None:
        row = {"t": t, **{s: scores.get(s) for s in SIGNALS}}
        self._call(who, "POST", "checkins", json=row, prefer="return=minimal")

    def checkins(self, who) -> list[tuple[float, dict[str, float]]]:
        rows = self._call(who, "GET", "checkins", params={"select": "t," + ",".join(SIGNALS), "order": "t.asc,id.asc"}).json()
        return [(r["t"], {s: r[s] for s in SIGNALS if r[s] is not None}) for r in rows]
