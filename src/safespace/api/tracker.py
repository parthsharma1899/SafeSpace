"""Turns a person's stored check-ins into their current reading, using v3.

The whole history is replayed through the fusion on every request. Histories are
small, and replaying means the stored data is just scores and times: if the
algorithm improves, every user's results are recomputed from the same data.

Time between check-ins: each check-in counts as one reading, plus one more for
every 12 hours since the previous one. So persistence carries over between
check-ins made close together and fades when someone hasn't checked in for a day
or two, while the baseline (their normal) builds up over many check-ins.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Iterable, Optional

from ..fusion import AlertRule, ResidualPersistenceV3, stress_level
from ..fusion.v3 import TUNED_V3, V3Config
from ..models.base import artifacts_dir
from .store import SIGNALS

HOURS_PER_STEP = 12.0
# A signal's baseline is the person's own once it has more than `warmup` readings.
BASELINE_READINGS = TUNED_V3.warmup + 1
# The alert follows the change from the person's normal once it is known.
CHANGE_MEDIUM, CHANGE_HIGH = 0.10, 0.20


def app_config(calm_prior: Optional[dict] = None) -> V3Config:
    """v3 as tuned on the benchmark, with each signal's typical calm value as the
    starting baseline and no stuck-sensor check (each check-in is a fresh reading
    from a model, not a sensor stream)."""
    if calm_prior is None:
        calm_prior = {}
        path = artifacts_dir() / "calibration.json"
        if path.exists():
            data = json.loads(path.read_text())
            calm_prior = {m: v["calm_prior"] for m, v in data.items() if isinstance(v, dict) and "calm_prior" in v}
    return replace(TUNED_V3, calm_prior=calm_prior, stuck_window=0)


def replay(checkins: Iterable[tuple[float, dict[str, float]]], config: Optional[V3Config] = None) -> list[dict]:
    """Run every check-in through v3 in order; return one result per check-in."""
    cfg = config or app_config()
    fuser = ResidualPersistenceV3(SIGNALS, cfg)
    by_score, by_change = AlertRule(), AlertRule(medium=CHANGE_MEDIUM, high=CHANGE_HIGH)
    out, prev_t = [], None
    seen = {m: 0 for m in SIGNALS}
    for t, scores in checkins:
        for m in scores:
            seen[m] += 1
        # Readings so far of the least-seen signal used this time, out of BASELINE_READINGS.
        progress = min((seen[m] for m in scores), default=0)
        dt = 1.0 if prev_t is None else 1.0 + max(0.0, t - prev_t) / 3600.0 / HOURS_PER_STEP
        prev_t = t
        r = fuser.step(scores, dt=dt)
        v3 = r.details.get("_v3", {})
        ready = bool(v3.get("established", 0.0))
        score_level = by_score.update(r.score, dt=dt)
        change_level = by_change.update(v3.get("change"), dt=dt) if r.score is not None else by_change.level
        out.append({
            "t": t,
            "fused": None if r.score is None else round(r.score, 4),
            "normal": None if "normal" not in v3 else round(v3["normal"], 4),
            "change": None if "change" not in v3 else round(v3["change"], 4),
            "baseline_ready": ready,
            "baseline_progress": min(progress, BASELINE_READINGS),
            # Readings so far per signal ever used, capped at BASELINE_READINGS.
            "baseline_counts": {m: min(n, BASELINE_READINGS) for m, n in seen.items() if n},
            "estimate_level": None if r.score is None else stress_level(r.score),
            "level": change_level if ready else score_level,
            "weights": {k: round(v, 4) for k, v in r.weights.items()},
            "spread": None if r.spread is None else round(r.spread, 4),
            "signals": {
                m: {"change": round(d["change"], 4), "baseline": round(d["baseline"], 4), "established": bool(d["established"])}
                for m, d in r.details.items() if m in SIGNALS
            },
            "dominant": r.dominant,
        })
    return out
