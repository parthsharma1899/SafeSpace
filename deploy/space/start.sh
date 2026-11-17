#!/bin/sh
# Start-up for the Hugging Face Space: fetch the model weights (from the private
# model repo, using the Space's HF_TOKEN secret), then serve the app on port 7860.
# If the download fails, the app still starts, in demo mode, and says so.

if python scripts/download_models.py; then
  echo "Models ready."
else
  echo "Model download failed (is the HF_TOKEN secret set and allowed to read the model repo?). Starting in demo mode."
fi

exec uvicorn safespace.api.serve:app --host 0.0.0.0 --port 7860 --proxy-headers --forwarded-allow-ips="*"
