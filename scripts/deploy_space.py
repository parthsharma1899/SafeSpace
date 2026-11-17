#!/usr/bin/env python3
"""Maintainers only: publish the app to its Hugging Face Docker Space.

    hf auth login                      # an account that owns the Space
    python scripts/deploy_space.py     # create (first time) or update the Space

Creates the Space if needed (public, Docker), sets its public variables from
the environment or `.env` (SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY, GEMINI_MODEL),
and uploads only what the Docker build needs (deploy/space/ plus src/, web/,
pyproject.toml and the model download script). The Space then builds and starts.

Secrets are never set by this script. Add them yourself in the Space's
Settings -> Variables and secrets:
- HF_TOKEN: a read-only token that can read the model repo (models download at start-up)
- GEMINI_API_KEY: optional, for generated support messages

Space id: $SAFESPACE_SPACE, default parth18s/safespace.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safespace.api.config import load_env  # noqa: E402

SPACE = os.environ.get("SAFESPACE_SPACE", "parth18s/safespace")
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
SKIP_DIRS = {"node_modules", "dist", "__pycache__", ".pytest_cache"}


def files_to_upload() -> dict[str, Path]:
    """Path in the Space -> local file."""
    out = {
        "Dockerfile": ROOT / "deploy" / "space" / "Dockerfile",
        "README.md": ROOT / "deploy" / "space" / "README.md",
        "start.sh": ROOT / "deploy" / "space" / "start.sh",
        "pyproject.toml": ROOT / "pyproject.toml",
        "scripts/download_models.py": ROOT / "scripts" / "download_models.py",
        "scripts/model_manifest.json": ROOT / "scripts" / "model_manifest.json",
    }
    for top in ("src", "web"):
        for p in sorted((ROOT / top).rglob("*")):
            rel = p.relative_to(ROOT)
            if p.is_file() and not SKIP_DIRS.intersection(rel.parts) and not p.name.endswith((".pyc", ".DS_Store")) \
                    and not any(part.endswith(".egg-info") for part in rel.parts):
                out[rel.as_posix()] = p
    return out


def main() -> int:
    load_env()
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_PUBLISHABLE_KEY", "").strip()
    if not (url and key):
        print("Set SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY (in .env) first.", file=sys.stderr)
        return 1

    api = HfApi()
    api.create_repo(SPACE, repo_type="space", space_sdk="docker", private=False, exist_ok=True)

    existing = {v for v in api.get_space_variables(SPACE)}
    api.add_space_variable(SPACE, "SUPABASE_URL", url)
    api.add_space_variable(SPACE, "SUPABASE_PUBLISHABLE_KEY", key)
    if "GEMINI_MODEL" not in existing:
        api.add_space_variable(SPACE, "GEMINI_MODEL", os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL)

    files = files_to_upload()
    remote = set(api.list_repo_files(SPACE, repo_type="space"))
    stale = sorted(remote - set(files) - {".gitattributes"})
    ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v)) for k, v in files.items()]
    ops += [CommitOperationDelete(path_in_repo=f) for f in stale]
    api.create_commit(SPACE, repo_type="space", operations=ops, commit_message="Deploy SafeSpace")

    host = SPACE.replace("/", "-").replace("_", "-").lower()
    print(f"Uploaded {len(files)} files ({len(stale)} removed) to https://huggingface.co/spaces/{SPACE}")
    print(f"The Space is building; the app will be at https://{host}.hf.space")
    print("Secrets to add in its settings if not done yet: HF_TOKEN (read-only), GEMINI_API_KEY (optional).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
