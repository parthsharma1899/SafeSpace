"""Turning a fused score into a stress level.

`stress_level` is the paper's rule, applied to a single score. `AlertRule` adds memory:
the level goes up as soon as the score crosses a threshold, and comes back down only
after the score has stayed clearly below it for a while. That keeps the estimate itself
unbiased while still making the alert quick to raise and slow to clear.
"""

from __future__ import annotations

from typing import Optional

LEVELS = ("Low", "Medium", "High")


def stress_level(score: float, medium: float = 0.4, high: float = 0.7) -> str:
    """Paper thresholds: below 0.4 is Low, up to and including 0.7 is Medium, above is High."""
    if score < medium:
        return "Low"
    if score <= high:
        return "Medium"
    return "High"


class AlertRule:
    def __init__(self, medium: float = 0.4, high: float = 0.7, margin: float = 0.05, clear_time: float = 3.0):
        if not 0 <= medium < high <= 1:
            raise ValueError("need 0 <= medium < high <= 1")
        self.medium, self.high = medium, high
        self.margin, self.clear_time = margin, clear_time
        self.reset()

    def reset(self) -> None:
        self._level = 0
        self._below_for = 0.0

    @property
    def level(self) -> str:
        return LEVELS[self._level]

    def update(self, score: Optional[float], dt: float = 1.0) -> str:
        """Feed the latest fused score; returns the current level. A missing score holds the level."""
        if score is None:
            return self.level
        target = LEVELS.index(stress_level(score, self.medium, self.high))
        if target >= self._level:
            self._level, self._below_for = target, 0.0
            return self.level

        # Step down one level at a time, and only after a sustained stay below the
        # threshold of the current level by a clear margin.
        threshold = self.high if self._level == 2 else self.medium
        if score < threshold - self.margin:
            self._below_for += dt
            if self._below_for >= self.clear_time:
                self._level -= 1
                self._below_for = 0.0
        else:
            self._below_for = 0.0
        return self.level
