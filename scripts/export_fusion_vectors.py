#!/usr/bin/env python3
"""Export reference fusion vectors for the TypeScript port to match.

    python scripts/export_fusion_vectors.py

Writes web/src/fusion/vectors.json: 200 random sequences of 30 steps over the
four modalities (each score missing with probability 0.25), with the Python
outputs of v1 (guarded) and v2 (tuned) at each step. The web test asserts the
TypeScript ports reproduce these to 6 decimal places.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safespace.fusion import GuardedPaperWMSS, ResidualPersistenceV2, TUNED  # noqa: E402

MODALITIES = ("wrist", "text", "voice", "face")
OUT = ROOT / "web" / "src" / "fusion" / "vectors.json"


def main() -> None:
    rng = np.random.default_rng(0)
    sequences = []
    for _ in range(200):
        steps = []
        v1 = GuardedPaperWMSS(MODALITIES)
        v2 = ResidualPersistenceV2(MODALITIES, TUNED)
        for _ in range(30):
            scores = {m: round(float(rng.random()), 6) for m in MODALITIES if rng.random() >= 0.25}
            r1 = v1.step(dict(scores))
            r2 = v2.step(dict(scores))
            # Full precision: the TS port is asserted to match exactly (< 1e-9),
            # which is a stricter check than 6 decimal places and avoids rounding ties.
            steps.append({
                "scores": scores,
                "v1": r1.score,
                "v2": r2.score,
            })
        sequences.append(steps)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tuned = {k: getattr(TUNED, k) for k in ("lam", "var_rate", "prior_std", "floor_std", "base_gain", "surprise_k", "carry")}
    OUT.write_text(json.dumps({"modalities": list(MODALITIES), "tuned": tuned, "sequences": sequences}))
    print(f"wrote {OUT} ({len(sequences)} sequences)")


if __name__ == "__main__":
    main()
