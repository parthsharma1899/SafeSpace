#!/usr/bin/env python3
"""Re-fit only the ELM output layer of the face model on FER-2013.

    python scripts/refit_face_elm.py [--limit N] [--batch 32]

Keeps the owner's architecture: the frozen ResNet50 and ViT feature extractors
and the ELM hidden layer (W, b) are used unchanged. Only the output weights B
are re-solved by ridge regression, choosing the ridge strength by AUC on the
validation split. Nothing original is overwritten; new files are written under
artifacts/face/refit/.

ResNet input uses Keras's resnet50.preprocess_input; ViT input is scaled to
[-1, 1]; the two feature vectors are concatenated in that order (ResNet first),
matching the order the saved W expects.

Features are cached to a float16 memmap under data/face_feats/ so the networks
run once per split. With --limit N only N images per emotion are used (a smoke
test). Writes results/face_refit.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safespace.models.base import artifacts_dir  # noqa: E402
from safespace.sim.metrics import auc  # noqa: E402

FER = ROOT / "data" / "raw" / "fer2013"
CACHE = ROOT / "data" / "face_feats"
RESULTS = ROOT / "results"
STRESSED = {0, 1, 2, 4}
LAMBDAS = [0.01, 0.1, 1.0, 10.0, 100.0]
FEAT_DIM = 153344  # 2048 ResNet + 151296 ViT


def load_split(name: str, limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(FER / f"{name}.csv")
    if limit:
        df = pd.concat([g.sample(min(len(g), limit), random_state=0) for _, g in df.groupby("emotion")]).reset_index(drop=True)
    grays = np.stack([np.array(p.split(), dtype=np.uint8).reshape(48, 48) for p in df["pixels"]])
    y = np.isin(df["emotion"].to_numpy(), list(STRESSED)).astype(np.float64)
    return grays, y


def extract_to_memmap(grays, extractors, path: Path, batch: int) -> np.memmap:
    import cv2
    import tensorflow as tf

    resnet, vit = extractors
    from tensorflow.keras.applications.resnet50 import preprocess_input

    n = len(grays)
    mm = np.memmap(path, dtype=np.float16, mode="w+", shape=(n, FEAT_DIM))
    for i in range(0, n, batch):
        chunk = grays[i:i + batch]
        rgb = np.stack([cv2.cvtColor(cv2.resize(g, (224, 224)), cv2.COLOR_GRAY2RGB) for g in chunk]).astype("float32")
        r_in = tf.convert_to_tensor(preprocess_input(rgb.copy()))
        v_in = tf.convert_to_tensor(rgb / 127.5 - 1.0)
        r = next(iter(resnet(r_in).values())).numpy().reshape(len(chunk), -1)
        v = next(iter(vit(v_in).values())).numpy().reshape(len(chunk), -1)
        mm[i:i + len(chunk)] = np.concatenate([r, v], axis=1).astype(np.float16)
    mm.flush()
    return mm


def feature_mean_std(mm: np.memmap, batch: int) -> tuple[np.ndarray, np.ndarray]:
    n = mm.shape[0]
    s = np.zeros(FEAT_DIM, np.float64)
    ss = np.zeros(FEAT_DIM, np.float64)
    for i in range(0, n, batch):
        x = mm[i:i + batch].astype(np.float64)
        s += x.sum(0)
        ss += (x * x).sum(0)
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean**2, 0.0))
    std[std < 1e-6] = 1.0
    return mean, std


def hidden(mm: np.memmap, mean, std, W, b, batch: int) -> np.ndarray:
    n = mm.shape[0]
    H = np.empty((n, W.shape[1]), np.float64)
    for i in range(0, n, batch):
        x = (mm[i:i + batch].astype(np.float64) - mean) / std
        z = x @ W + b
        H[i:i + batch] = np.maximum(0.01 * z, z)
    return H


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="images per emotion (smoke test)")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    import tensorflow as tf

    face_dir = artifacts_dir() / "face"
    elm = np.load(face_dir / "elm_weights-001.npz")
    W, b = elm["W"].astype(np.float64), elm["b"].astype(np.float64)
    resnet = tf.saved_model.load(str(face_dir / "resnet50_features")).signatures["serving_default"]
    vit = tf.saved_model.load(str(face_dir / "vit_features")).signatures["serving_default"]
    CACHE.mkdir(parents=True, exist_ok=True)

    splits = {}
    for name in ("train", "val", "test"):
        grays, y = load_split(name, args.limit)
        mm = extract_to_memmap(grays, (resnet, vit), CACHE / f"{name}.f16", args.batch)
        splits[name] = (mm, y)
        print(f"{name}: {len(y)} images -> features cached")

    mean, std = feature_mean_std(splits["train"][0], args.batch)
    Htr, ytr = hidden(splits["train"][0], mean, std, W, b, args.batch), splits["train"][1]
    Hva, yva = hidden(splits["val"][0], mean, std, W, b, args.batch), splits["val"][1]
    Hte, yte = hidden(splits["test"][0], mean, std, W, b, args.batch), splits["test"][1]

    HtH = Htr.T @ Htr
    Hty = Htr.T @ ytr
    eye = np.eye(HtH.shape[0])

    best = None
    for lam in LAMBDAS:
        B = np.linalg.solve(HtH + lam * eye, Hty)
        va_auc = auc(yva > 0.5, Hva @ B)
        if best is None or va_auc > best[1]:
            best = (lam, va_auc, B)
    lam, va_auc, B = best

    # Map the unbounded raw ELM output to a probability with a logistic fitted on val.
    from sklearn.linear_model import LogisticRegression

    raw_val = Hva @ B
    lr = LogisticRegression(C=1e6).fit(raw_val.reshape(-1, 1), yva.astype(int))
    pa, pb = float(lr.coef_[0][0]), float(lr.intercept_[0])

    te_scores = Hte @ B
    te_prob = 1.0 / (1.0 + np.exp(-(pa * te_scores + pb)))
    te_auc = auc(yte > 0.5, te_scores)  # AUC is invariant to the monotone squashing
    te_acc = float(((te_prob >= 0.5).astype(int) == yte).mean())
    te_brier = float(np.mean((te_prob - yte) ** 2))
    majority = max(yte.mean(), 1 - yte.mean())

    refit_dir = face_dir / "refit"
    refit_dir.mkdir(exist_ok=True)
    np.save(refit_dir / "elm_B.npy", B)
    np.save(refit_dir / "scaler_mean.npy", mean)
    np.save(refit_dir / "scaler_std.npy", std)
    (refit_dir / "platt.json").write_text(__import__("json").dumps({"a": pa, "b": pb}))

    passed = te_auc >= 0.70
    md = [
        "# Face model ELM re-fit (FER-2013)",
        "",
        "Produced by `scripts/refit_face_elm.py`. Only the ELM output layer B was re-solved;",
        "the ResNet, the ViT file and the ELM hidden layer (W, b) are unchanged.",
        f"{'Smoke test (--limit ' + str(args.limit) + ').' if args.limit else 'Full FER-2013 splits.'}",
        "",
        f"- Chosen ridge strength (by validation AUC): lambda = {lam}",
        f"- Validation AUC: {va_auc:.4f}",
        f"- **Test AUC: {te_auc:.4f}** (pass bar >= 0.70: {'PASS' if passed else 'FAIL'})",
        f"- Test accuracy (calibrated, 0.5 cut): {te_acc:.4f}",
        f"- Test Brier (calibrated): {te_brier:.4f}",
        f"- Always-guess-majority accuracy: {majority:.4f}",
        "",
        f"New files: artifacts/face/refit/elm_B.npy, scaler_mean.npy, scaler_std.npy, platt.json.",
        "" if passed else "Test AUC is below 0.70, so the face model stays switched off.",
        "",
    ]
    (RESULTS / "face_refit.md").write_text("\n".join(md))
    print(f"lambda {lam} | val AUC {va_auc:.4f} | TEST AUC {te_auc:.4f} | {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
