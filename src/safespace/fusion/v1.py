"""v1: the fusion rule from the 2025 paper draft, ported without changing its arithmetic.

    EMA_t        = lam * S_t + (1 - lam) * EMA_{t-1}
    residual     = S_t - EMA_t
    persistence  = EMA_t
    weight       = max(0, alpha * residual + beta * persistence)
    WMSS         = sum(normalised weight * S_t)

The paper's Equation 4 omits the max(0, .) but its Table 15 and the original code both
apply it, so it is kept here. This class exists as the baseline that newer methods are
measured against; `tests/test_fusion_v1.py` checks it against the original code and
against Table 15.

Two behaviours of the original are preserved on purpose:

- every modality must be present at every step;
- when every weight is zero (all scores zero, or all scores falling together) there is
  nothing to normalise by, and the step fails.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .base import FusionResult, Scores, clean_scores, weighted_spread


class MissingModalityError(ValueError):
    """v1 needs a score from every modality at every step."""


class ZeroTotalWeightError(ZeroDivisionError):
    """Every v1 weight is zero, so the weights cannot be normalised."""


class PaperWMSS:
    name = "v1"

    def __init__(self, modalities: Sequence[str], lam: float = 0.3, alpha: float = 0.7, beta: float = 0.3):
        self.modalities = tuple(modalities)
        self.lam, self.alpha, self.beta = lam, alpha, beta
        self.reset()

    def reset(self) -> None:
        self.ema: Optional[dict[str, float]] = None

    def step(self, scores: Scores, dt: float = 1.0) -> FusionResult:
        """Fuse one time step. `dt` is accepted for interface compatibility and ignored:
        v1 counts steps, not time."""
        x = clean_scores(scores)
        missing = [m for m in self.modalities if m not in x]
        if missing:
            raise MissingModalityError(f"v1 needs every modality at every step; missing {missing}")

        if self.ema is None:
            self.ema = {m: x[m] for m in self.modalities}

        new_ema, raw, details = {}, {}, {}
        for m in self.modalities:
            s = x[m]
            ema = self.lam * s + (1 - self.lam) * self.ema[m]
            residual = s - ema
            raw[m] = max(0, self.alpha * residual + self.beta * ema)
            new_ema[m] = ema
            details[m] = {"ema": ema, "residual": residual, "persistence": ema, "raw_weight": raw[m]}

        total = sum(raw.values())
        if total == 0:
            raise ZeroTotalWeightError("all v1 weights are zero")

        weights = {m: raw[m] / total for m in self.modalities}
        score = sum(x[m] * weights[m] for m in self.modalities)
        self.ema = new_ema
        return FusionResult(
            score=score,
            weights=weights,
            available=self.modalities,
            spread=weighted_spread(x, weights),
            details=details,
        )


class GuardedPaperWMSS:
    """v1 made runnable on inputs it cannot handle, for benchmarking only.

    A missing score is replaced by that modality's last value (or, before it has ever
    reported, by the mean of the scores that are present). A zero-weight step returns the
    plain mean. Both substitutions are counted so they can be reported next to the results.
    """

    name = "v1"

    def __init__(self, modalities: Sequence[str], **kwargs):
        self.modalities = tuple(modalities)
        self._inner = PaperWMSS(modalities, **kwargs)
        self.reset()

    def reset(self) -> None:
        self._inner.reset()
        self._last: dict[str, float] = {}
        self._last_score: Optional[float] = None
        self.filled_steps = 0
        self.zero_weight_steps = 0

    def step(self, scores: Scores, dt: float = 1.0) -> FusionResult:
        x = clean_scores(scores)
        present = {m: x[m] for m in self.modalities if m in x}
        if not present and not self._last:
            return FusionResult(score=None)

        filled = dict(present)
        fallback = sum(present.values()) / len(present) if present else None
        for m in self.modalities:
            if m not in filled:
                filled[m] = self._last.get(m, fallback if fallback is not None else self._last_score)
        if len(present) < len(self.modalities):
            self.filled_steps += 1
        self._last.update(present)

        try:
            result = self._inner.step(filled)
        except ZeroTotalWeightError:
            self.zero_weight_steps += 1
            mean = sum(filled.values()) / len(filled)
            equal = {m: 1 / len(filled) for m in filled}
            result = FusionResult(score=mean, weights=equal, available=tuple(present), spread=weighted_spread(filled, equal))
        self._last_score = result.score
        return result
