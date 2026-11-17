"""A small in-memory stand-in for Supabase's REST API (PostgREST), for tests.

Implements only what SupabaseStore uses, and enforces what deploy/supabase.sql's
row-level security does: every request must carry the publishable key and a
user's token, and sees only that user's rows.
"""

from __future__ import annotations

import json
from urllib.parse import parse_qs

import httpx
import jwt

KEY = "sb_publishable_test"


class FakePostgrest:
    def __init__(self):
        self.consents: dict[str, dict] = {}
        self.checkins: list[dict] = []
        self.requests: list[httpx.Request] = []
        self._next_id = 1

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.headers.get("apikey") != KEY:
            return httpx.Response(401, json={"message": "Invalid API key"})
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer "):
            return httpx.Response(401, json={"message": "no token"})
        uid = jwt.decode(auth[7:], options={"verify_signature": False})["sub"]  # PostgREST's auth.uid()
        table = request.url.path.rsplit("/", 1)[-1]
        q = {k: v[0] for k, v in parse_qs(request.url.query.decode()).items()}
        prefer = request.headers.get("prefer", "")

        if table == "consents":
            if request.method == "GET":
                return httpx.Response(200, json=[{"consented_at": "now"}] if uid in self.consents else [])
            if request.method == "POST":
                self.consents.setdefault(uid, {"user_id": uid})
                return httpx.Response(201)
            if request.method == "DELETE":
                if q.get("user_id") == f"eq.{uid}":
                    self.consents.pop(uid, None)
                return httpx.Response(204)

        if table == "checkins":
            mine = [r for r in self.checkins if r["user_id"] == uid]
            if request.method == "HEAD":
                return httpx.Response(200, headers={"content-range": f"*/{len(mine)}"})
            if request.method == "GET":
                cols = q["select"].split(",")
                rows = sorted(mine, key=lambda r: (r["t"], r["id"]))
                return httpx.Response(200, json=[{c: r[c] for c in cols} for r in rows])
            if request.method == "POST":
                row = json.loads(request.content)
                self.checkins.append({"id": self._next_id, "user_id": uid, **row})
                self._next_id += 1
                return httpx.Response(201)
            if request.method == "DELETE":
                gone = [r for r in mine if q.get("user_id") == f"eq.{uid}"]
                self.checkins = [r for r in self.checkins if r not in gone]
                headers = {"content-range": f"*/{len(gone)}"} if "count=exact" in prefer else {}
                return httpx.Response(204, headers=headers)

        return httpx.Response(404, json={"message": f"unexpected {request.method} {table}"})
