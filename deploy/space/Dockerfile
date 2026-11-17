# SafeSpace on a Hugging Face Docker Space: the API and the web app on one HTTPS
# address. Published by scripts/deploy_space.py; see README "Hosted app".

# 1) Build the web app.
FROM node:20-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# 2) The Python server with the four models' libraries.
FROM python:3.13-slim
# libgomp1: XGBoost (OpenMP). libgl1 + libglib2.0-0: OpenCV.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Spaces run as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    SAFESPACE_STORE=supabase
WORKDIR /home/user/app

# CPU-only PyTorch (the default wheel bundles GPU libraries the Space cannot use).
RUN pip install --no-cache-dir --user --index-url https://download.pytorch.org/whl/cpu torch==2.12.1

COPY --chown=user pyproject.toml README.md ./
COPY --chown=user src ./src
RUN pip install --no-cache-dir --user -e ".[models,api]"

COPY --chown=user scripts/download_models.py scripts/model_manifest.json ./scripts/
COPY --chown=user start.sh ./
COPY --chown=user --from=web /web/dist ./web/dist

EXPOSE 7860
CMD ["sh", "start.sh"]
