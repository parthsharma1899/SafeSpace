"""Score-level simulator for testing fusion methods against a known answer.

A session has a hidden true-stress curve. Each modality reports that curve through its own
noise, bias, delay, gaps and faults. Nothing here imitates raw faces, voices or sensor
traces: the simulator stands in for "whatever the four models output", so it can say how a
fusion rule behaves under stated assumptions, and nothing about real people.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Optional

import numpy as np

MODALITIES = ("wrist", "text", "voice", "face")


@dataclass(frozen=True)
class ModalitySpec:
    noise: float = 0.12
    """Standard deviation of the noise added to the score."""
    bias: float = 0.0
    """Constant offset added to the score."""
    lag: float = 0.0
    """Time constant, in steps, of this modality's delayed view of true stress."""
    sensitivity: float = 1.0
    """1 follows true stress fully; 0 never moves from the resting level."""
    availability: float = 1.0
    """Long-run share of steps on which the modality reports."""
    run_length: float = 1.0
    """Average length of a present or absent stretch; 1 means independent from step to step."""


@dataclass(frozen=True)
class Fault:
    kind: str
    """'stuck' holds the score at `value`; 'flip' outputs random 0 or 1; 'offset' adds `value`."""
    modality: Optional[str] = None
    """None picks one modality at random for each session."""
    start: tuple[float, float] = (0.0, 0.0)
    """Range from which the start is drawn, as a fraction of the session."""
    length: tuple[float, float] = (1.0, 1.0)
    """Range from which the length is drawn, as a fraction of the session."""
    value: float = 1.0
    probability: float = 1.0
    """Chance that the fault occurs in a given session."""


@dataclass(frozen=True)
class Scenario:
    name: str
    specs: Mapping[str, ModalitySpec] = field(default_factory=lambda: {m: ModalitySpec() for m in MODALITIES})
    faults: tuple[Fault, ...] = ()
    stress: str = "episodes"
    """'episodes', 'calm' or 'steps'."""
    steps: int = 240
    lone_detector: bool = False
    """If set, one random modality keeps full sensitivity and noise 0.08 while the rest go nearly blind."""
    personal_offset_std: float = 0.0
    """If set, each modality gets a constant per-person offset (std of the offsets; they average to zero)."""
    leader: bool = False
    """If set, one random modality responds fully and at once; the others late (lag 6) and partly (60%)."""
    lone_or_stuck: bool = False
    """If set, half the sessions have a lone detector and half have one modality stuck near 1.0."""


@dataclass
class Session:
    truth: np.ndarray
    scores: dict[str, np.ndarray]
    scenario: Scenario
    noise_var: dict[str, float]
    faulty: tuple[str, ...] = ()

    def step_scores(self, t: int) -> dict[str, float]:
        return {m: float(v[t]) for m, v in self.scores.items() if not np.isnan(v[t])}


STEP_UP, STEP_DOWN = 80, 160


def true_stress(kind: str, steps: int, rng: np.random.Generator) -> np.ndarray:
    base = rng.uniform(0.08, 0.25)
    if kind == "steps":
        z = np.full(steps, 0.2)
        z[STEP_UP:STEP_DOWN] = 0.85
        return z

    z = np.full(steps, base)
    if kind == "episodes":
        t = int(rng.integers(10, 40))
        while t < steps - 20:
            peak = rng.uniform(0.6, 0.95)
            rise, plateau, decay = int(rng.integers(2, 7)), int(rng.integers(15, 41)), rng.uniform(8, 20)
            ramp = base + (peak - base) * 0.5 * (1 - np.cos(np.pi * np.arange(1, rise + 1) / rise))
            tail_len = int(decay * 5)
            tail = base + (peak - base) * np.exp(-np.arange(1, tail_len + 1) / decay)
            episode = np.concatenate([ramp, np.full(plateau, peak), tail])[: steps - t]
            z[t : t + len(episode)] = np.maximum(z[t : t + len(episode)], episode)
            t += rise + plateau + int(rng.integers(40, 90))
    elif kind != "calm":
        raise ValueError(f"unknown stress kind {kind!r}")

    wander = np.zeros(steps)
    for i in range(1, steps):
        wander[i] = 0.9 * wander[i - 1] + rng.normal(0, 0.02)
    return np.clip(z + wander, 0.0, 1.0)


def _presence(spec: ModalitySpec, steps: int, rng: np.random.Generator) -> np.ndarray:
    if spec.availability >= 1:
        return np.ones(steps, dtype=bool)
    p, run = spec.availability, max(1.0, spec.run_length)
    stay_present, stay_absent = 1 - (1 - p) / run, 1 - p / run
    present = np.empty(steps, dtype=bool)
    present[0] = rng.random() < p
    for i in range(1, steps):
        stay = stay_present if present[i - 1] else stay_absent
        present[i] = present[i - 1] if rng.random() < stay else not present[i - 1]
    return present


def generate(scenario: Scenario, seed: int) -> Session:
    # The true-stress curve is drawn first, so a given seed yields the same curve in every
    # scenario of the same stress kind and scenarios can be compared session by session.
    rng = np.random.default_rng(seed)
    steps = scenario.steps
    truth = true_stress(scenario.stress, steps, rng)
    specs = dict(scenario.specs)

    # New random draws happen only when a scenario switches these on, so the
    # original scenarios generate exactly the same sessions as before.
    stuck_half = False
    if scenario.lone_or_stuck:
        stuck_half = bool(rng.random() < 0.5)
    if scenario.lone_detector or (scenario.lone_or_stuck and not stuck_half):
        detector = str(rng.choice(list(specs)))
        specs = {m: replace(s, sensitivity=1.0, noise=0.08) if m == detector else replace(s, sensitivity=0.15, noise=0.10) for m, s in specs.items()}
    if scenario.leader:
        lead = str(rng.choice(list(specs)))
        specs = {m: s if m == lead else replace(s, lag=6.0, sensitivity=0.6) for m, s in specs.items()}
    offsets = dict.fromkeys(specs, 0.0)
    if scenario.personal_offset_std > 0:
        drawn = rng.normal(0, scenario.personal_offset_std, len(specs))
        offsets = dict(zip(specs, drawn - drawn.mean()))
    faults = list(scenario.faults)
    if stuck_half:
        faults.append(Fault("stuck", start=(0.1, 0.4), length=(0.4, 0.5), value=0.97))

    rest = truth[0]
    scores: dict[str, np.ndarray] = {}
    for m, spec in specs.items():
        view = truth.copy()
        if spec.lag > 0:
            a = 1 - np.exp(-1 / spec.lag)
            for i in range(1, steps):
                view[i] = view[i - 1] + a * (truth[i] - view[i - 1])
        s = rest + spec.sensitivity * (view - rest) + spec.bias + offsets[m] + rng.normal(0, spec.noise, steps)
        s = np.clip(s, 0.0, 1.0)
        s[~_presence(spec, steps, rng)] = np.nan
        scores[m] = s

    faulty = []
    for fault in faults:
        if rng.random() >= fault.probability:
            continue
        m = fault.modality or str(rng.choice(list(specs)))
        start = int(rng.uniform(*fault.start) * steps)
        end = min(steps, start + max(1, int(rng.uniform(*fault.length) * steps)))
        seg = slice(start, end)
        present = ~np.isnan(scores[m][seg])
        if fault.kind == "stuck":
            values = np.clip(fault.value + rng.normal(0, 0.01, end - start), 0, 1)
        elif fault.kind == "flip":
            values = rng.integers(0, 2, end - start).astype(float)
        elif fault.kind == "offset":
            values = np.clip(np.nan_to_num(scores[m][seg]) + fault.value, 0, 1)
        else:
            raise ValueError(f"unknown fault kind {fault.kind!r}")
        scores[m][seg] = np.where(present, values, np.nan)
        faulty.append(m)

    return Session(truth=truth, scores=scores, scenario=scenario, noise_var={m: s.noise**2 for m, s in specs.items()}, faulty=tuple(faulty))


def _specs(**overrides: ModalitySpec) -> dict[str, ModalitySpec]:
    return {m: overrides.get(m, ModalitySpec()) for m in MODALITIES}


_UNEQUAL = _specs(wrist=ModalitySpec(noise=0.05), voice=ModalitySpec(noise=0.10), text=ModalitySpec(noise=0.20), face=ModalitySpec(noise=0.30))
_GAPS = dict(
    wrist=dict(availability=0.95, run_length=20),
    face=dict(availability=0.70, run_length=8),
    voice=dict(availability=0.30, run_length=5),
    text=dict(availability=0.08, run_length=1),
)

SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in [
        Scenario("clean"),
        Scenario("unequal", specs=_UNEQUAL),
        Scenario("missing", specs={m: ModalitySpec(**_GAPS[m]) for m in MODALITIES}),
        Scenario("stuck", faults=(Fault("stuck", start=(0.1, 0.4), length=(0.4, 0.5), value=0.97),)),
        Scenario("flipping", faults=(Fault("flip"),)),
        Scenario("biased", faults=(Fault("offset", value=0.3),)),
        Scenario("calm", stress="calm"),
        Scenario("steps", stress="steps"),
        Scenario("minority", lone_detector=True),
        Scenario(
            "mixed",
            specs={
                "wrist": ModalitySpec(noise=0.08, lag=2.0, **_GAPS["wrist"]),
                "voice": ModalitySpec(noise=0.12, **_GAPS["voice"]),
                "text": ModalitySpec(noise=0.18, **_GAPS["text"]),
                "face": ModalitySpec(noise=0.22, bias=0.10, **_GAPS["face"]),
            },
            faults=(Fault("stuck", start=(0.1, 0.7), length=(0.1, 0.2), value=0.97, probability=0.5),),
        ),
        # Added 2026-11-17 (see the addendum in docs/protocol.md).
        Scenario("personal", personal_offset_std=0.15),
        Scenario("leader", leader=True),
        Scenario("lone_or_stuck", lone_or_stuck=True),
    ]
}
