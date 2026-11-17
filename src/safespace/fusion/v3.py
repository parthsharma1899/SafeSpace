"""v3: residual-persistence fusion relative to each person's own baseline.

This is the project's fusion idea as originally intended, with the definitions
corrected and guards added. For every signal (wrist, text, voice, face):

    baseline     this person's normal level for the signal
    residual     how far the latest reading is from that normal, in units of
                 the signal's own normal wobble ("is this unusual for *this*
                 signal?")
    persistence  a short running average of the residual: a spike that stays
                 builds persistence, a one-off spike fades

A signal's weight is the paper's Equation 4 with those definitions:

    weight = floor + alpha * |residual| + beta * |persistence|

(each term counts only beyond the signal's normal wobble, and rises and falls
count alike). The weight comes from what the signal has done up to now; the
newest reading moves the score through its own change. So a one-off spike
nudges the score, and a spike that stays takes over on the next reading.

The output has two parts:

    normal = average of the present signals' baselines   (the person's normal)
    change = weighted average of each signal's change from its normal
    score  = normal + change                              (clipped to [0, 1])

With no history the baselines equal the first readings, the change is zero and
the score is the plain average.

Guards (all optional through V3Config):
- the baseline updates only while a signal is near it, so a long stress
  episode is not learned as "normal";
- the first readings can start from a typical calm value for each signal;
- a signal whose recent readings are flat (a stuck sensor) is treated as
  missing;
- one signal's share of the weight can be capped;
- every rate is time-aware, so gaps between readings are handled.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from .base import FusionResult, Scores, clean_scores, per_step_rate, weighted_spread


@dataclass(frozen=True)
class V3Config:
    baseline_rate: float = 0.05
    """How fast the baseline (the person's normal) follows readings, per unit time."""
    persist_rate: float = 0.3
    """How fast persistence follows the residual."""
    level_rate: float = 0.5
    """How fast each signal's recent level follows its readings (smooths noise)."""
    alpha: float = 0.3
    """Weight given to the latest residual (a sudden spike)."""
    beta: float = 1.0
    """Weight given to persistence (a change that has stayed)."""
    floor: float = 1.0
    """Weight every present signal gets regardless of change."""
    deadband: float = 1.0
    """Change smaller than this many units of normal wobble adds no weight."""
    gate: float = 2.0
    """The baseline and wobble only update while persistence is below this."""
    noise_rate: float = 0.05
    """How fast each signal's normal wobble is learned."""
    prior_noise: float = 0.10
    """Normal wobble assumed before any is learned."""
    min_noise: float = 0.03
    """No signal's normal wobble is taken as smaller than this."""
    warmup: int = 5
    """Readings before a signal's baseline counts as established."""
    calm_prior: Mapping[str, float] = field(default_factory=dict)
    """Typical calm value per signal to start the baseline from (else the first reading)."""
    prior_strength: float = 2.0
    """How many readings the calm prior counts as while the baseline is learned."""
    cap: Optional[float] = None
    """Largest share of the weight any one signal may hold (None = no cap)."""
    stuck_window: int = 10
    """Readings checked for a stuck sensor (0 turns the check off)."""
    stuck_std: float = 0.005
    """A signal whose last `stuck_window` readings vary less than this is treated as stuck.
    Kept very small: a calm person's steady readings still vary more than a stuck sensor's."""
    symmetric: bool = True
    """Count falls as well as rises."""


# The settings chosen by scripts/run_fusion_benchmark.py on the tuning seeds (the
# "v3" block of results/fusion_tuning.json). Used by the app. Tuning set alpha to 0:
# persistence alone, not the latest residual, decides how much a signal counts.
TUNED_V3 = V3Config(
    persist_rate=0.4, level_rate=0.6, alpha=0.0, beta=0.5,
    deadband=1.0, baseline_rate=0.02, cap=0.6,
)


class _Signal:
    __slots__ = ("baseline", "noise_var", "residual", "persistence", "level", "n", "age", "recent")

    def __init__(self, prior: Optional[float], prior_noise: float, window: int):
        self.baseline = prior
        self.noise_var = prior_noise ** 2
        self.residual = 0.0
        self.persistence = 0.0
        self.level: Optional[float] = None
        self.n = 0
        self.age = 0.0
        self.recent: deque = deque(maxlen=max(1, window))


def _cap_shares(weights: dict[str, float], cap: float) -> dict[str, float]:
    """Normalise weights to shares, then limit any share to `cap`, redistributing the rest."""
    total = sum(weights.values())
    shares = {m: w / total for m, w in weights.items()}
    if len(shares) * cap < 1.0:
        return shares  # a cap below 1/n cannot be satisfied; leave shares as they are
    fixed: dict[str, float] = {}
    while True:
        over = [m for m in shares if m not in fixed and shares[m] > cap + 1e-12]
        if not over:
            return {**shares, **fixed}
        for m in over:
            fixed[m] = cap
        free = [m for m in shares if m not in fixed]
        left = 1.0 - sum(fixed.values())
        free_total = sum(weights[m] for m in free)
        shares = {m: left * weights[m] / free_total for m in free}


class ResidualPersistenceV3:
    name = "v3"

    def __init__(self, modalities: Sequence[str], config: Optional[V3Config] = None, name: Optional[str] = None):
        self.modalities = tuple(modalities)
        self.config = config or V3Config()
        if name:
            self.name = name
        self.reset()

    def reset(self) -> None:
        cfg = self.config
        self._s = {m: _Signal(cfg.calm_prior.get(m), cfg.prior_noise, cfg.stuck_window) for m in self.modalities}
        self._last: Optional[float] = None

    # ---- helpers -------------------------------------------------------------

    def _excess(self, x: float) -> float:
        x = abs(x) if self.config.symmetric else x
        return max(0.0, x - self.config.deadband)

    def _weight(self, st: _Signal) -> float:
        cfg = self.config
        return cfg.floor + cfg.alpha * self._excess(st.residual) + cfg.beta * self._excess(st.persistence)

    def _stuck(self, st: _Signal) -> bool:
        """Judged from earlier readings only, so the newest reading cannot change it."""
        cfg = self.config
        if cfg.stuck_window <= 0 or len(st.recent) < cfg.stuck_window:
            return False
        vals = list(st.recent)
        mean = sum(vals) / len(vals)
        return math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)) < cfg.stuck_std

    # ---- one step ------------------------------------------------------------

    def step(self, scores: Scores, dt: float = 1.0) -> FusionResult:
        cfg = self.config
        fresh = {m: v for m, v in clean_scores(scores).items() if m in self._s}
        for st in self._s.values():
            if st.n:
                st.age += dt

        # A stuck sensor is treated as missing (its readings still update the stuck check).
        stuck = [m for m in fresh if self._stuck(self._s[m])]
        for m in fresh:
            self._s[m].recent.append(fresh[m])
        present = {m: v for m, v in fresh.items() if m not in stuck}
        if not present:
            return FusionResult(score=self._last, details={"_v3": {"stuck": stuck}})

        # Weights come from each signal's history up to now (fixed within this step).
        raw = {m: self._weight(self._s[m]) for m in present}
        shares = _cap_shares(raw, cfg.cap) if cfg.cap else {m: w / sum(raw.values()) for m, w in raw.items()}

        details: dict[str, dict[str, float]] = {}
        baselines, changes = {}, {}
        for m, s in present.items():
            st = self._s[m]
            gap = st.age if st.n else 1.0
            if st.baseline is None:
                st.baseline = s
            baseline_before = st.baseline
            sigma = max(math.sqrt(st.noise_var), cfg.min_noise)
            residual = (s - baseline_before) / sigma

            st.level = s if st.level is None else st.level + per_step_rate(cfg.level_rate, gap) * (s - st.level)
            st.persistence = residual if st.n == 0 else st.persistence + per_step_rate(cfg.persist_rate, gap) * (residual - st.persistence)
            st.residual = residual

            baselines[m] = baseline_before
            changes[m] = st.level - baseline_before

            # Learn this person's normal: a running mean while warming up, then a slow
            # average that pauses while a change persists.
            st.n += 1
            if st.n <= cfg.warmup:
                k = cfg.prior_strength if m in cfg.calm_prior else 0.0
                st.baseline += (s - st.baseline) / (st.n + k)
            elif abs(st.persistence) < cfg.gate:
                st.baseline += per_step_rate(cfg.baseline_rate, gap) * (s - st.baseline)
            if st.n <= cfg.warmup or abs(residual) < cfg.gate:
                # While warming up, the assumed wobble counts as two earlier readings.
                rate = 1.0 / (st.n + 2) if st.n <= cfg.warmup else per_step_rate(cfg.noise_rate, gap)
                dev = s - baseline_before
                st.noise_var += rate * (dev * dev - st.noise_var)
            st.age = 0.0

            details[m] = {
                "baseline": baseline_before,
                "residual": residual,
                "persistence": st.persistence,
                "level": st.level,
                "change": changes[m],
                "established": float(st.n > cfg.warmup),
            }

        normal = sum(baselines.values()) / len(baselines)
        change = sum(shares[m] * changes[m] for m in present)
        score = min(1.0, max(0.0, normal + change))
        self._last = score
        details["_v3"] = {
            "normal": normal,
            "change": change,
            "established": float(all(self._s[m].n > cfg.warmup for m in present)),
            "stuck": stuck,
        }
        return FusionResult(
            score=score,
            weights=shares,
            available=tuple(present),
            spread=weighted_spread(present, shares),
            details=details,
        )
