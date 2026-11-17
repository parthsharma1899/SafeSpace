"""Sign-in: checks the access token that Supabase Auth gives a signed-in user.

The web app signs people in with Supabase (Google, or email and password). Each request
from a signed-in user carries `Authorization: Bearer <access token>`. This module
verifies that token against the project's public signing keys, published at
`<SUPABASE_URL>/auth/v1/.well-known/jwks.json`, so the server needs no Supabase
secret. A valid token yields the user's id (the token's `sub`), which keys their
saved scores.

Configuration (environment or the repo-root `.env`, see config.py):
- SUPABASE_URL: the project URL, e.g. https://abcd.supabase.co. Without it,
  sign-in is off and the account endpoints answer 503.
- SUPABASE_PUBLISHABLE_KEY: the public key the browser uses to talk to Supabase.
  The server only passes it on to the web app (GET /config); it is not a secret.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import jwt

ALGORITHMS = ["ES256", "RS256"]
AUDIENCE = "authenticated"


class AuthError(Exception):
    """The request has no valid sign-in token."""


@dataclass(frozen=True)
class Identity:
    user_id: str
    email: Optional[str] = None


class SupabaseVerifier:
    """Verifies Supabase access tokens with the project's public keys.

    `keys` lets tests supply a key resolver in place of fetching the JWKS:
    a callable taking the token and returning the public key.
    """

    def __init__(self, url: str, keys=None):
        self.url = url.rstrip("/")
        self.issuer = f"{self.url}/auth/v1"
        if keys is None:
            client = jwt.PyJWKClient(f"{self.issuer}/.well-known/jwks.json", cache_keys=True, lifespan=3600)
            keys = lambda token: client.get_signing_key_from_jwt(token).key  # noqa: E731
        self._keys = keys

    def verify(self, token: str) -> Identity:
        try:
            key = self._keys(token)
            claims = jwt.decode(
                token, key, algorithms=ALGORITHMS, audience=AUDIENCE, issuer=self.issuer,
                options={"require": ["exp", "sub", "aud", "iss"]},
            )
        except Exception as e:  # any failure, including an unreachable key server, means "not signed in"
            raise AuthError("invalid or expired sign-in") from e
        return Identity(user_id=str(claims["sub"]), email=claims.get("email"))


def bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise AuthError("sign-in required")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthError("sign-in required")
    return token.strip()


def verifier_from_env() -> Optional[SupabaseVerifier]:
    url = os.environ.get("SUPABASE_URL", "").strip()
    return SupabaseVerifier(url) if url else None


def public_config() -> dict:
    """What the web app needs to show the sign-in screen. Nothing secret."""
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_PUBLISHABLE_KEY", "").strip()
    return {"auth": {"url": url, "publishable_key": key} if url and key else None}
