# SafeSpace Model Artifacts

This directory contains the lightweight Python loaders that the app imports.
The trained weights and generated model files are intentionally not committed
to normal Git because several are larger than GitHub's regular file limit.

Use the helper scripts from the repository root:

```bash
python scripts/upload_models_to_hf.py --repo-id parth18s/safespace-fusion-models
python scripts/download_models.py --repo-id parth18s/safespace-fusion-models
python scripts/check_models.py
```

For a private Hugging Face repo, log in first:

```bash
hf auth login
```
