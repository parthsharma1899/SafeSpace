#!/usr/bin/env python3
import argparse

from huggingface_hub import HfApi, create_repo

from model_artifacts import ARTIFACT_PATTERNS, IGNORED_UPLOAD_PATTERNS, MODELS_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload local SafeSpace model artifacts to Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face repo id, for example username/safespace-fusion-models.",
    )
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=("model", "dataset", "space"),
        help="Hugging Face repository type.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create the Hugging Face repo as public. Defaults to private.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not MODELS_DIR.exists():
        raise SystemExit(f"Models directory not found: {MODELS_DIR}")

    print(f"Ensuring Hugging Face repo exists: {args.repo_id}")
    create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=not args.public,
        exist_ok=True,
    )

    print(f"Uploading model artifacts from {MODELS_DIR}...")
    HfApi().upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=MODELS_DIR,
        allow_patterns=ARTIFACT_PATTERNS,
        ignore_patterns=IGNORED_UPLOAD_PATTERNS,
        commit_message="Upload SafeSpace model artifacts",
    )
    print("Upload complete.")


if __name__ == "__main__":
    main()
