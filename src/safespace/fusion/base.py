"""Types shared by every fusion method."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Optional, Protocol

Scores = Mapping[str, Optional[float]]


@dataclass(frozen=True)
class FusionResult:
    """Output of one fusion step.

    score      Fused stress score in [0, 1]. None only if no modality has ever reported.
    weights    Share of each modality in this step's score. Sums to 1, or empty when
               nothing was available.
    available  Modalities that reported a score at this step.
    spread     Weighted standard deviation of the modality scores around their
               consensus: how much the modalities disagree. None when nothing was used.
    details    Per-modality internals (EMA, residual and so on), for inspection.
    """

    score: Optional[float]
    weights: dict[str, float] = field(default_factory=dict)
    available: tuple[str, ...] = ()
    spread: Optional[float] = None
    details: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def dominant(self) -> Optional[str]:
        """Modality with the largest weight at this step."""
        if not self.weights:
            return None
        return max(self.weights, key=self.weights.get)


class Fuser(Protocol):
    """A fusion method: takes the scores available at one time step, returns one score."""

    name: str

    def reset(self) -> None: ...

    def step(self, scores: Scores, dt: float = 1.0) -> FusionResult: ...


def clean_scores(scores: Scores) -> dict[str, float]:
    """Return the usable scores from one time step.

    A modality is treated as missing when its key is absent or its value is None or NaN.
    Anything else must be a real number in [0, 1]; a string such as an error message, or
    a number outside the range, is a bug in the caller and raises instead of being fused.
    """
    out: dict[str, float] = {}
    for name, value in scores.items():
        if value is None:
            continue
        if isinstance(value, (str, bytes, bool)):
            raise TypeError(f"score for {name!r} must be a number in [0, 1], got {value!r}")
        try:
            x = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"score for {name!r} must be a number in [0, 1], got {value!r}") from exc
        if math.isnan(x):
            continue
        if not 0.0 <= x <= 1.0:
            raise ValueError(f"score for {name!r} must be in [0, 1], got {x}")
        out[name] = x
    return out


def per_step_rate(rate: float, dt: float) -> float:
    """Convert a smoothing rate defined per unit time into the rate for a gap of dt.

    With dt = 1 this returns rate unchanged. Larger gaps let more of the old value go, so
    an estimate that has not been updated for a while follows the next reading more closely.
    """
    if dt <= 0:
        return 0.0
    return 1.0 - (1.0 - rate) ** dt


def weighted_spread(values: Mapping[str, float], weights: Mapping[str, float]) -> float:
    centre = sum(weights[m] * values[m] for m in weights)
    return math.sqrt(max(0.0, sum(weights[m] * (values[m] - centre) ** 2 for m in weights)))
