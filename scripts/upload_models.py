#!/usr/bin/env python3
"""Maintainers only: publish artifacts/ to the Hugging Face model repo.

    hf auth login                               # an account with write access
    python scripts/upload_models.py             # upload, then record the revision
    python scripts/upload_models.py --squash    # also erase the repo's earlier history

Uploads every file the app needs (src/safespace/models/hub.py), plus the model
card docs/models/hub_card.md as README.md, in one commit. Remote files that are
not part of that set are deleted, except those under legacy/ (the original 2025
files, kept for reference). Files the Hub already holds are not sent again.

Then writes scripts/model_manifest.json (path, size, SHA-256 and the repo
revision) for download_models.py to check against; commit it to git.

--squash replaces the repo's history with a single commit, so deleted files
(and anything that should never have been uploaded) cannot be recovered from
older revisions. It cannot be undone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safespace.models.base import artifacts_dir  # noqa: E402
from safespace.models.hub import MANIFEST, local_files, repo_id, sha256  # noqa: E402

CARD = ROOT / "docs" / "models" / "hub_card.md"
KEEP = {"README.md", ".gitattributes"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--squash", action="store_true", help="erase earlier history after uploading (irreversible)")
    args = ap.parse_args()

    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    api = HfApi()
    repo = repo_id()
    src = artifacts_dir()
    groups = local_files(src)
    missing = [m for m, files in groups.items() if not files]
    if missing:
        print(f"artifacts/ has no files for: {', '.join(missing)}", file=sys.stderr)
        return 1

    entries = []
    for model, files in groups.items():
        for rel in files:
            p = src / rel
            entries.append({"path": rel, "model": model, "size": p.stat().st_size, "sha256": sha256(p)})
    wanted = {e["path"] for e in entries}

    remote = set(api.list_repo_files(repo))
    stale = sorted(f for f in remote - wanted - KEEP if not f.startswith("legacy/"))
    ops = [CommitOperationAdd(path_in_repo=e["path"], path_or_fileobj=str(src / e["path"])) for e in entries]
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(CARD)))
    ops += [CommitOperationDelete(path_in_repo=f) for f in stale]

    print(f"{repo}: uploading {len(entries)} files ({sum(e['size'] for e in entries) / 1e9:.2f} GB), deleting {len(stale)}")
    for f in stale:
        print(f"  delete {f}")
    api.create_commit(repo, operations=ops, commit_message="Publish SafeSpace models (artifacts/ layout)")

    if args.squash:
        api.super_squash_history(repo, commit_message="SafeSpace models")
        print("History squashed to one commit.")

    revision = api.model_info(repo).sha
    MANIFEST.write_text(json.dumps({"repo": repo, "revision": revision, "files": entries}, indent=2) + "\n")
    print(f"Wrote {MANIFEST.relative_to(ROOT)} at revision {revision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
