"""One server for the whole app: the API under /api and the built web app at /.

Used by the hosted Space, so the browser sees a single HTTPS address (needed for
the camera and microphone, and no cross-origin calls):

    uvicorn safespace.api.serve:app --host 0.0.0.0 --port 7860

The web app is read from SAFESPACE_WEB_DIR, default web/dist (`npm run build`).
For local development, run the API and the Vite dev server separately instead
(README, "Run the app").
"""

from __future__ import annotations

import os
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

from .app import app as api

ROOT = Path(__file__).resolve().parents[3]
WEB_DIR = Path(os.environ.get("SAFESPACE_WEB_DIR", ROOT / "web" / "dist"))


class Site:
    """Sends /api/... to the API (with the prefix removed) and everything else to the web app."""

    def __init__(self, api_app, web_dir: Path):
        self.api = api_app
        self.web = None
        if (web_dir / "index.html").is_file():
            # Wrapped in an app so a missing file is a plain 404 response, not an exception.
            self.web = Starlette(routes=[Mount("/", app=StaticFiles(directory=str(web_dir), html=True))])

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":  # model loading lives in the API's lifespan
            return await self.api(scope, receive, send)
        path = scope.get("path", "")
        if path == "/api" or path.startswith("/api/"):
            scope = dict(scope, path=path[4:] or "/", raw_path=(path[4:] or "/").encode())
            return await self.api(scope, receive, send)
        if self.web is None:
            return await PlainTextResponse("web app not built: run `npm run build` in web/", status_code=404)(scope, receive, send)
        return await self.web(scope, receive, send)


app = Site(api, WEB_DIR)
