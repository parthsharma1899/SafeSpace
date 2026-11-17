"""Score-level simulator and benchmark for fusion methods."""

from .benchmark import methods, run_session, score_method
from .simulator import MODALITIES, SCENARIOS, Fault, ModalitySpec, Scenario, Session, generate

__all__ = [
    "MODALITIES",
    "SCENARIOS",
    "Fault",
    "ModalitySpec",
    "Scenario",
    "Session",
    "generate",
    "methods",
    "run_session",
    "score_method",
]
