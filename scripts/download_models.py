#!/usr/bin/env python3
"""Download the trained model weights into artifacts/ and check them.

    python scripts/download_models.py     # all four models (about 1.5 GB)
    python scripts/download_models.py --models text,voice

Files come from the Hugging Face repo in $SAFESPACE_MODELS_REPO (default
parth18s/safespace-fusion-models) at the revision recorded in
scripts/model_manifest.json, and each one is checked against the size and
SHA-256 listed there. The repo is public: no account or login is needed.
Use is limited to non-commercial research by the datasets' terms (README).

Without the weights the API still runs, in demo mode with fixed scores.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safespace.models.base import artifacts_dir  # noqa: E402
from safespace.models.hub import MODEL_PATTERNS, check, read_manifest, repo_id  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="wrist,text,voice,face", help="comma-separated: wrist,text,voice,face")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    wanted = ["calibration"] + [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in wanted if m not in MODEL_PATTERNS]
    if unknown:
        ap.error(f"unknown model(s): {', '.join(unknown)}")

    manifest = read_manifest()
    entries = [e for e in manifest["files"] if e["model"] in wanted]
    dest = artifacts_dir()
    dest.mkdir(parents=True, exist_ok=True)
    total = sum(e["size"] for e in entries) / 1e9
    print(f"Downloading {len(entries)} files ({total:.2f} GB) from {repo_id()} into {dest}")

    try:
        snapshot_download(
            repo_id=repo_id(),
            revision=manifest.get("revision"),
            local_dir=dest,
            allow_patterns=[e["path"] for e in entries],
        )
    except (RepositoryNotFoundError, GatedRepoError):
        print(
            f"\nCould not open the model repo {repo_id()}. Check your internet connection and\n"
            "that the repo id is right; if it has been made private, sign in with\n"
            "`hf auth login` using an account that has access.",
            file=sys.stderr,
        )
        return 1

    print("Checking files ...")
    problems = check(dest, entries)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"All {len(entries)} files match scripts/model_manifest.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
