#!/usr/bin/env python3
import argparse

from huggingface_hub import snapshot_download

from model_artifacts import MODELS_DIR, REQUIRED_FILES, missing_required_files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download SafeSpace model artifacts from a Hugging Face Hub repo."
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
    return parser.parse_args()


def main():
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model artifacts from {args.repo_id} into {MODELS_DIR}...")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        local_dir=MODELS_DIR,
        allow_patterns=REQUIRED_FILES,
    )

    missing = missing_required_files()
    if missing:
        print("Download finished, but these required files are still missing:")
        for path in missing:
            print(f"  - {path}")
        raise SystemExit(1)

    print("Model artifacts are ready.")


if __name__ == "__main__":
    main()
