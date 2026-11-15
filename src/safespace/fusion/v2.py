"""v2: residual-persistence fusion, redesigned.

v1 used each modality's residual and persistence to set its weight, which made the
weight grow with the score itself. v2 keeps the same two quantities but gives them
different jobs.

Per modality, from its own history:

    persistence_m  its smoothed level (EMA of its scores)
    residual_m     how far a new score lands from that level

Trust in a modality is the inverse of its estimated squared error, which splits into a
variance part and a bias part:

    trust_m = reliability_m / (floor^2 + variance of residual_m + offset_m^2)

    offset_m = persistence_m - median persistence of the other modalities

So a modality loses weight when it is jumpy (large residuals) or when its level stays
away from where the others sit (large offset), not when its score is low.

The fused score itself is also a persistence plus a residual:

    consensus_t = sum(weight_m * score_m)
    fused_t     = fused_{t-1} + gain_t * (consensus_t - fused_{t-1})

`gain_t` is small while the consensus moves within its usual range, and approaches 1
when the consensus residual is large compared with that range. Shared changes are
followed quickly, in either direction; ordinary jitter is smoothed away.

All weights at step t are computed from history up to t-1. Within a step the fused score
is therefore a fixed-weight average of the incoming scores passed through an increasing
function, which is what guarantees that raising a score can never lower the result.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from .base import FusionResult, Scores, clean_scores, per_step_rate, weighted_spread


@dataclass(frozen=True)
class V2Config:
    lam: float = 0.3
    """Rate of each modality's EMA per unit time (the paper's lambda)."""
    var_rate: float = 0.1
    """Rate at which residual variances are tracked."""
    prior_std: float = 0.15
    """Residual standard deviation assumed for a modality with no history yet."""
    floor_std: float = 0.10
    """No modality is ever treated as more precise than this."""
    base_gain: float = 0.3
    """How fast the fused score follows the consensus when nothing unusual is happening."""
    surprise_k: float = 2.5
    """Consensus residual, in units of its usual size, at which the gain is halfway to 1."""
    carry: float = 0.0
    """How long a reading keeps counting after it was made, fading linearly. 0 turns this off."""
    reliability: Mapping[str, float] = field(default_factory=dict)
    """Fixed prior weight per modality, for example from validation accuracy. Default 1."""
    use_variance: bool = True
    use_offset: bool = True
    adaptive_gain: bool = True


class _ModalityState:
    __slots__ = ("ema", "var", "last", "age")

    def __init__(self, prior_var: float):
        self.ema: Optional[float] = None
        self.var = prior_var
        self.last: Optional[float] = None
        self.age = 0.0


class ResidualPersistenceV2:
    name = "v2"

    def __init__(self, modalities: Sequence[str], config: Optional[V2Config] = None, name: Optional[str] = None):
        self.modalities = tuple(modalities)
        self.config = config or V2Config()
        if name:
            self.name = name
        self.reset()

    def reset(self) -> None:
        prior_var = self.config.prior_std**2
        self._state = {m: _ModalityState(prior_var) for m in self.modalities}
        self._fused: Optional[float] = None
        self._fused_var = prior_var / max(1, len(self.modalities))

    def step(self, scores: Scores, dt: float = 1.0) -> FusionResult:
        cfg = self.config
        fresh = {m: v for m, v in clean_scores(scores).items() if m in self._state}
        for st in self._state.values():
            if st.last is not None:
                st.age += dt

        # Readings used at this step: fresh ones at full strength, recent ones fading out.
        used: dict[str, float] = dict(fresh)
        freshness: dict[str, float] = {m: 1.0 for m in fresh}
        if cfg.carry > 0:
            for m, st in self._state.items():
                if m not in fresh and st.last is not None and st.age < cfg.carry:
                    used[m] = st.last
                    freshness[m] = 1.0 - st.age / cfg.carry
        if not used:
            return FusionResult(score=self._fused)

        # Trust comes from history only, so it does not depend on this step's scores.
        known = {m: self._state[m].ema for m in used if self._state[m].ema is not None}
        floor_var = cfg.floor_std**2
        trust, details = {}, {}
        for m in used:
            st = self._state[m]
            variance = st.var if cfg.use_variance else cfg.prior_std**2
            offset = 0.0
            if cfg.use_offset and st.ema is not None:
                others = [v for k, v in known.items() if k != m]
                if others:
                    offset = st.ema - statistics.median(others)
            trust[m] = cfg.reliability.get(m, 1.0) * freshness[m] / (floor_var + variance + offset * offset)
            details[m] = {"variance": variance, "offset": offset}

        total = sum(trust.values())
        weights = {m: trust[m] / total for m in used}
        consensus = sum(weights[m] * used[m] for m in used)

        if self._fused is None:
            self._fused = consensus
            gain, fused_residual = 1.0, 0.0
        else:
            fused_residual = consensus - self._fused
            gain = per_step_rate(cfg.base_gain, dt)
            if cfg.adaptive_gain:
                z2 = fused_residual**2 / (self._fused_var + 1e-12)
                gain += (1 - gain) * z2 / (z2 + cfg.surprise_k**2)
            self._fused += gain * fused_residual
            self._fused_var += per_step_rate(cfg.var_rate, dt) * (fused_residual**2 - self._fused_var)
        self._fused = min(1.0, max(0.0, self._fused))

        # Update each modality's persistence and residual statistics with its fresh reading.
        for m, s in fresh.items():
            st = self._state[m]
            if st.ema is None:
                st.ema, residual = s, 0.0
            else:
                gap = st.age
                residual = s - st.ema
                st.var += per_step_rate(cfg.var_rate, gap) * (residual**2 - st.var)
                st.ema += per_step_rate(cfg.lam, gap) * residual
            st.last, st.age = s, 0.0
            details[m].update(residual=residual, persistence=st.ema)

        details["_fused"] = {"consensus": consensus, "residual": fused_residual, "gain": gain}
        return FusionResult(
            score=self._fused,
            weights=weights,
            available=tuple(fresh),
            spread=weighted_spread(used, weights),
            details=details,
        )


# The settings chosen by scripts/run_fusion_benchmark.py on the tuning seeds.
# Kept as a named constant so the API, the benchmark and the tests all use the
# same v2. These values must match the "v2" block of results/fusion_tuning.json.
TUNED = V2Config(floor_std=0.05, base_gain=0.3, surprise_k=2.5, var_rate=0.05, carry=6.0)
