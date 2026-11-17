"""The combined server used by the hosted Space: /api -> API, everything else -> web app."""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from safespace.api.app import create_app  # noqa: E402
from safespace.api.bundle import StubBundle  # noqa: E402
from safespace.api.serve import Site  # noqa: E402
from safespace.api.store import AccountStore  # noqa: E402


@pytest.fixture
def site(tmp_path):
    (tmp_path / "index.html").write_text("<!doctype html><title>SafeSpace</title>")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "app.js").write_text("console.log(1)")
    api = create_app(StubBundle(), AccountStore(":memory:"), verifier=None)
    return TestClient(Site(api, tmp_path))


def test_api_is_served_under_api(site):
    r = site.get("/api/health")
    assert r.status_code == 200 and "models" in r.json()
    sid = site.post("/api/sessions").json()["session_id"]
    assert site.post(f"/api/sessions/{sid}/observations", data={"text": "hi"}).status_code == 200


def test_web_app_is_served_at_root(site):
    assert "SafeSpace" in site.get("/").text
    assert site.get("/assets/app.js").status_code == 200
    assert site.get("/health").status_code == 404  # the API is only under /api


def test_missing_web_build_is_explained(tmp_path):
    api = create_app(StubBundle(), AccountStore(":memory:"), verifier=None)
    r = TestClient(Site(api, tmp_path / "nope")).get("/")
    assert r.status_code == 404 and "npm run build" in r.text
