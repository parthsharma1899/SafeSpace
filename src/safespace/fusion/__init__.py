"""Fusion methods: the paper's original rule (v1), simple baselines, and the redesign (v2)."""

from .alert import AlertRule, stress_level
from .base import FusionResult, Fuser, Scores, clean_scores
from .baselines import Instant, OracleKalman, Smoothed, StaticWeights
from .v1 import GuardedPaperWMSS, MissingModalityError, PaperWMSS, ZeroTotalWeightError
from .v2 import TUNED, ResidualPersistenceV2, V2Config

__all__ = [
    "AlertRule",
    "FusionResult",
    "Fuser",
    "GuardedPaperWMSS",
    "Instant",
    "MissingModalityError",
    "OracleKalman",
    "PaperWMSS",
    "ResidualPersistenceV2",
    "Scores",
    "Smoothed",
    "StaticWeights",
    "TUNED",
    "V2Config",
    "ZeroTotalWeightError",
    "clean_scores",
    "stress_level",
]
