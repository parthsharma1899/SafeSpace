#!/usr/bin/env python3
from model_artifacts import MODELS_DIR, REQUIRED_FILES, missing_required_files


def sizeof(path):
    size = path.stat().st_size
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} {unit}"
        size /= 1024


def main():
    missing = missing_required_files()
    if missing:
        print("Missing required model artifacts:")
        for path in missing:
            print(f"  - {path}")
        print()
        print("Download them with:")
        print("  python scripts/download_models.py --repo-id YOUR_HF_USERNAME/safespace-fusion-models")
        raise SystemExit(1)

    print(f"All required model artifacts are present in {MODELS_DIR}:")
    for relative_path in REQUIRED_FILES:
        path = MODELS_DIR / relative_path
        print(f"  - {relative_path} ({sizeof(path)})")


if __name__ == "__main__":
    main()
