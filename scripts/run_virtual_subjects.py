#!/usr/bin/env python3
"""Validation step 2: virtual subjects.

Each modality's real, held-out scores are split into two pools (stressed and not
stressed). A simulated stress curve is generated; at each step, each present
modality's score is drawn from the pool that matches the true state. The model
errors are real; only the pairing across people is artificial.

    python scripts/run_virtual_subjects.py

Writes results/virtual_subjects.md. Voice and face score pools are cached under
data/cache/ (gitignored) because they are slow to compute.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safespace.fusion import GuardedPaperWMSS, Instant, ResidualPersistenceV2, Smoothed, TUNED  # noqa: E402
from safespace.models.base import artifacts_dir  # noqa: E402
from safespace.models.calibration import apply_platt  # noqa: E402
from safespace.sim.metrics import auc, evaluate  # noqa: E402
from safespace.sim.simulator import MODALITIES, SCENARIOS, generate  # noqa: E402

RESULTS = ROOT / "results"
CACHE = ROOT / "data" / "cache"
SEEDS = range(1000, 1200)
STRESSED_EMOTIONS = {4, 5, 6, 7}
STRESSED_FER = {0, 1, 2, 4}


def _calib():
    path = artifacts_dir() / "calibration.json"
    c = json.loads(path.read_text()) if path.exists() else {}
    return {m: (c.get(m, {}).get("a", 1.0), c.get(m, {}).get("b", 0.0)) for m in MODALITIES}


def _apply(scores, ab):
    a, b = ab
    return np.array([apply_platt(float(s), a, b) for s in scores])


def _read_two_col(path, score_col, label_col):
    s, y = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            v = row[score_col]
            if v in ("", "nan"):
                continue
            s.append(float(v)); y.append(int(float(row[label_col])))
    return np.array(s), np.array(y)


def _voice_scores():
    cache = CACHE / "voice_scores.csv"
    if cache.exists():
        return _read_two_col(cache, "score", "stressed")
    from safespace.models.voice import VoiceModel

    files = sorted((ROOT / "data" / "raw" / "ravdess").glob("Actor_*/*.wav"))
    model = VoiceModel().load()
    s, y = [], []
    for fpath in files:
        r = model.predict(str(fpath))
        if r.available:
            s.append(r.score); y.append(int(int(fpath.name.split("-")[2]) in STRESSED_EMOTIONS))
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["score", "stressed"]); w.writerows(zip(s, y))
    return np.array(s), np.array(y)


def _face_scores(per_class=150):
    cache = CACHE / "face_scores.csv"
    if cache.exists():
        return _read_two_col(cache, "score", "stressed")
    import cv2
    import pandas as pd

    from safespace.models.face import FaceModel

    df = pd.read_csv(ROOT / "data" / "raw" / "fer2013" / "test.csv")
    df = pd.concat([g.sample(min(len(g), per_class), random_state=0) for _, g in df.groupby("emotion")]).reset_index(drop=True)
    model = FaceModel().load()
    s, y = [], []
    for _, row in df.iterrows():
        gray = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
        s.append(model._score_crop(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)))
        y.append(int(row["emotion"] in STRESSED_FER))
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["score", "stressed"]); w.writerows(zip(s, y))
    return np.array(s), np.array(y)


def build_pools():
    cal = _calib()
    pools = {}
    ws, wy = _read_two_col(RESULTS / "wrist_loso.csv", "oof_score", "target")
    ws = _apply(ws, cal["wrist"])
    pools["wrist"] = (ws[wy == 1], ws[wy == 0])
    ts, ty = _read_two_col(RESULTS / "text_eval.csv", "score", "label")
    ts = _apply(ts, cal["text"])
    pools["text"] = (ts[ty == 1], ts[ty == 0])
    vs, vy = _voice_scores(); vs = _apply(vs, cal["voice"])
    pools["voice"] = (vs[vy == 1], vs[vy == 0])
    fs, fy = _face_scores(); fs = _apply(fs, cal["face"])
    pools["face"] = (fs[fy == 1], fs[fy == 0])
    return pools


def methods(rates):
    return {
        "v1": lambda: GuardedPaperWMSS(MODALITIES),
        "mean": lambda: Instant("mean"),
        "median": lambda: Instant("median"),
        "mean+ema": lambda: Smoothed(Instant("mean"), rate=rates["mean_ema_rate"]),
        "median+ema": lambda: Smoothed(Instant("median"), rate=rates["median_ema_rate"]),
        "v2": lambda: ResidualPersistenceV2(MODALITIES, TUNED),
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    pools = build_pools()
    rates = json.loads((RESULTS / "fusion_tuning.json").read_text())
    rng = np.random.default_rng(12345)

    agg = {name: [] for name in methods(rates)}
    for seed in SEEDS:
        base = generate(SCENARIOS["missing"], seed)  # reuse its truth curve and presence mask
        truth = base.truth
        # Replace each present modality's value with a draw from the matching real pool.
        drawn = {}
        for m in MODALITIES:
            col = np.full(len(truth), np.nan)
            present = ~np.isnan(base.scores[m])
            for t in np.where(present)[0]:
                stressed, calm = pools[m]
                pool = stressed if truth[t] > 0.5 else calm
                col[t] = pool[rng.integers(len(pool))]
            drawn[m] = col
        for name, factory in methods(rates).items():
            fuser = factory()
            fused = np.full(len(truth), np.nan)
            for t in range(len(truth)):
                step = {m: float(drawn[m][t]) for m in MODALITIES if not np.isnan(drawn[m][t])}
                sc = fuser.step(step).score
                if sc is not None:
                    fused[t] = sc
            agg[name].append(evaluate(truth, fused))

    def mean_of(name, key):
        vals = [r[key] for r in agg[name] if r[key] is not None]
        return float(np.mean(vals)) if vals else float("nan")

    pool_sizes = {m: (len(pools[m][0]), len(pools[m][1])) for m in MODALITIES}
    md = [
        "# Virtual-subject validation",
        "",
        "Produced by `scripts/run_virtual_subjects.py`. Each simulated person combines",
        "real, held-out model scores from different real people, drawn to match a",
        "simulated stress curve (seeds 1000-1199).",
        "",
        "Each simulated person combines scores from different real people. This assumes",
        "the modalities are independent once the stress state is known. Earlier work on",
        "such combined ('chimeric') subjects in biometrics found that results do not",
        "always carry over to real people, so these numbers are supporting evidence only.",
        "",
        "Score pools (stressed / not), real held-out scores per modality: "
        + ", ".join(f"{m} {pool_sizes[m][0]}/{pool_sizes[m][1]}" for m in MODALITIES) + ".",
        "",
        "| method | MAE | bias | episode AUC | false-alarm share |",
        "|---|---:|---:|---:|---:|",
        *[f"| {n} | {mean_of(n,'mae'):.4f} | {mean_of(n,'bias'):+.4f} | {mean_of(n,'auc'):.4f} | {mean_of(n,'false_alarm'):.4f} |" for n in methods(rates)],
        "",
    ]
    (RESULTS / "virtual_subjects.md").write_text("\n".join(md))
    print("\n".join(md[-(len(methods(rates)) + 2):]))


if __name__ == "__main__":
    main()
