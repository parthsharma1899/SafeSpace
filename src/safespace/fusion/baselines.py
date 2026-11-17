"""Simple fusion methods that anything more elaborate has to beat."""

from __future__ import annotations

import statistics
from typing import Mapping, Optional, Sequence

from .base import FusionResult, Fuser, Scores, clean_scores, per_step_rate, weighted_spread


class Instant:
    """Combine the scores available at this step with a plain mean, median or max.

    `max` is the "raise the alarm if any signal is high" rule.
    """

    def __init__(self, kind: str = "mean"):
        if kind not in ("mean", "median", "max"):
            raise ValueError(f"unknown kind {kind!r}")
        self.kind = kind
        self.name = kind
        self._last: Optional[float] = None

    def reset(self) -> None:
        self._last = None

    def step(self, scores: Scores, dt: float = 1.0) -> FusionResult:
        x = clean_scores(scores)
        if not x:
            return FusionResult(score=self._last)
        values = list(x.values())
        if self.kind == "mean":
            score = sum(values) / len(values)
        elif self.kind == "median":
            score = statistics.median(values)
        else:
            score = max(values)
        equal = {m: 1 / len(x) for m in x}
        self._last = score
        return FusionResult(score=score, weights=equal, available=tuple(x), spread=weighted_spread(x, equal))


class StaticWeights:
    """Weighted mean with fixed weights, renormalised over whichever modalities are present."""

    def __init__(self, weights: Mapping[str, float], name: str = "static"):
        if any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError("weights must be non-negative and not all zero")
        self.base = dict(weights)
        self.name = name
        self._last: Optional[float] = None

    def reset(self) -> None:
        self._last = None

    def step(self, scores: Scores, dt: float = 1.0) -> FusionResult:
        x = {m: v for m, v in clean_scores(scores).items() if self.base.get(m, 0) > 0}
        if not x:
            return FusionResult(score=self._last)
        total = sum(self.base[m] for m in x)
        weights = {m: self.base[m] / total for m in x}
        score = sum(weights[m] * x[m] for m in x)
        self._last = score
        return FusionResult(score=score, weights=weights, available=tuple(x), spread=weighted_spread(x, weights))


class Smoothed:
    """Any fusion method followed by an exponential moving average over time."""

    def __init__(self, inner: Fuser, rate: float = 0.3, name: Optional[str] = None):
        if not 0 < rate <= 1:
            raise ValueError("rate must be in (0, 1]")
        self.inner = inner
        self.rate = rate
        self.name = name or f"{inner.name}+ema"
        self._level: Optional[float] = None

    def reset(self) -> None:
        self.inner.reset()
        self._level = None

    def step(self, scores: Scores, dt: float = 1.0) -> FusionResult:
        r = self.inner.step(scores, dt)
        if r.available and r.score is not None:
            if self._level is None:
                self._level = r.score
            else:
                self._level += per_step_rate(self.rate, dt) * (r.score - self._level)
        return FusionResult(score=self._level, weights=r.weights, available=r.available, spread=r.spread, details=r.details)


class OracleKalman:
    """A scalar Kalman filter that is told every modality's true noise variance.

    It models stress as a random walk and each score as stress plus independent noise.
    Real systems never know the noise levels, so this is a reference point for how well
    fusion could do under ideal assumptions, not a method to deploy.
    """

    name = "oracle"

    def __init__(self, noise_var: Mapping[str, float], process_var: float = 0.002, modalities: Optional[Sequence[str]] = None):
        self.noise_var = dict(noise_var)
        self.process_var = process_var
        self.modalities = tuple(modalities or noise_var)
        self.reset()

    def reset(self) -> None:
        self._x: Optional[float] = None
        self._p = 1.0

    def step(self, scores: Scores, dt: float = 1.0) -> FusionResult:
        x = {m: v for m, v in clean_scores(scores).items() if m in self.noise_var}
        if self._x is not None:
            self._p += self.process_var * dt
        if not x:
            return FusionResult(score=self._x)
        precision = {m: 1 / max(self.noise_var[m], 1e-9) for m in x}
        total = sum(precision.values())
        weights = {m: precision[m] / total for m in x}
        obs = sum(weights[m] * x[m] for m in x)
        obs_var = 1 / total
        if self._x is None:
            self._x, self._p = obs, obs_var
        else:
            gain = self._p / (self._p + obs_var)
            self._x += gain * (obs - self._x)
            self._p *= 1 - gain
        self._x = min(1.0, max(0.0, self._x))
        return FusionResult(score=self._x, weights=weights, available=tuple(x), spread=weighted_spread(x, weights))
