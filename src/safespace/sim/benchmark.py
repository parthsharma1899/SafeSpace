"""Run fusion methods over simulated sessions and collect the protocol's metrics."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Iterable, Mapping, Optional

import numpy as np

from ..fusion import Fuser, GuardedPaperWMSS, Instant, OracleKalman, ResidualPersistenceV2, ResidualPersistenceV3, Smoothed, V2Config, V3Config
from .metrics import evaluate, half_time
from .simulator import MODALITIES, SCENARIOS, STEP_DOWN, STEP_UP, Scenario, Session, generate

MethodFactory = Callable[[Session], Fuser]

DEFAULT_SETTINGS = {"mean_ema_rate": 0.3, "median_ema_rate": 0.3, "oracle_process_var": 0.002, "v2": {}, "v3": {}}

# Each v3 improvement switched off on its own, to show what it contributes.
ABLATIONS = {
    "v3 equal weights": dict(alpha=0.0, beta=0.0),
    "v3 without baseline pause": dict(gate=1e9),
    "v3 rises only": dict(symmetric=False),
    "v3 without smoothing": dict(level_rate=1.0),
}


def run_session(fuser: Fuser, session: Session) -> np.ndarray:
    """Feed one session to a method. Steps where it has no score yet are NaN."""
    fuser.reset()
    out = np.full(len(session.truth), np.nan)
    for t in range(len(session.truth)):
        score = fuser.step(session.step_scores(t)).score
        if score is not None:
            out[t] = score
    return out


def methods(settings: Optional[Mapping] = None, include_ablations: bool = True) -> dict[str, MethodFactory]:
    """Every method in the protocol, as factories that build a fresh instance per session."""
    s = {**DEFAULT_SETTINGS, **(settings or {})}
    v2 = V2Config(**s["v2"])
    v3 = V3Config(**s["v3"])
    out: dict[str, MethodFactory] = {
        "v1": lambda session: GuardedPaperWMSS(MODALITIES),
        "mean": lambda session: Instant("mean"),
        "median": lambda session: Instant("median"),
        "max": lambda session: Instant("max"),
        "mean+ema": lambda session: Smoothed(Instant("mean"), rate=s["mean_ema_rate"]),
        "median+ema": lambda session: Smoothed(Instant("median"), rate=s["median_ema_rate"]),
        "oracle": lambda session: OracleKalman(session.noise_var, process_var=s["oracle_process_var"]),
        "v2": lambda session: ResidualPersistenceV2(MODALITIES, v2),
        "v3": lambda session: ResidualPersistenceV3(MODALITIES, v3),
    }
    if include_ablations:
        for name, change in ABLATIONS.items():
            out[name] = lambda session, cfg=replace(v3, **change): ResidualPersistenceV3(MODALITIES, cfg)
    return out


def score_method(factory: MethodFactory, scenario: Scenario, seeds: Iterable[int]) -> list[dict]:
    """One row of metrics per session."""
    rows = []
    for seed in seeds:
        session = generate(scenario, seed)
        fuser = factory(session)
        fused = run_session(fuser, session)
        row = {"scenario": scenario.name, "seed": seed, **evaluate(session.truth, fused)}
        if scenario.stress == "steps":
            row["rise_time"] = half_time(fused, STEP_UP, 0.2, 0.85)
            row["fall_time"] = half_time(fused, STEP_DOWN, 0.85, 0.2)
        if isinstance(fuser, GuardedPaperWMSS):
            row["zero_weight_steps"] = float(fuser.zero_weight_steps)
            row["filled_steps"] = float(fuser.filled_steps)
        rows.append(row)
    return rows


def mean_error(factory: MethodFactory, seeds: Iterable[int], scenarios: Optional[Iterable[str]] = None) -> float:
    """Tuning objective from the protocol: mean absolute error averaged over scenarios."""
    seeds = list(seeds)
    per_scenario = []
    for name in scenarios or SCENARIOS:
        rows = score_method(factory, SCENARIOS[name], seeds)
        per_scenario.append(np.mean([r["mae"] for r in rows if r["mae"] is not None]))
    return float(np.mean(per_scenario))
