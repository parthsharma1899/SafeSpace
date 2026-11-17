"""Load WESAD wrist (Empatica E4) signals and labels.

Each SX.pkl is a Python 2 pickle of a dict. It is read with a restricted
unpickler that allows only NumPy array construction, so loading an untrusted
file cannot execute arbitrary code. Only wrist signals and labels are read;
chest signals and temperature are ignored.

Sampling rates (Empatica E4): BVP 64 Hz, EDA 4 Hz, ACC 32 Hz. Labels are at
700 Hz; 1 = baseline, 2 = stress, 3 = amusement (other values are ignored).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BVP_HZ, EDA_HZ, ACC_HZ, LABEL_HZ = 64, 4, 32, 700

_ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


@dataclass
class WesadSubject:
    subject: str
    bvp: np.ndarray    # (n, 1) at 64 Hz
    eda: np.ndarray    # (n, 1) at 4 Hz
    acc: np.ndarray    # (n, 3) at 32 Hz
    label: np.ndarray  # (n,) at 700 Hz


def load_subject(path) -> WesadSubject:
    path = Path(path)
    with open(path, "rb") as f:
        d = _RestrictedUnpickler(f, encoding="latin1").load()
    wrist = d["signal"]["wrist"]
    return WesadSubject(
        subject=path.stem,
        bvp=np.asarray(wrist["BVP"], dtype=np.float64).reshape(-1, 1),
        eda=np.asarray(wrist["EDA"], dtype=np.float64).reshape(-1, 1),
        acc=np.asarray(wrist["ACC"], dtype=np.float64).reshape(-1, 3),
        label=np.asarray(d["label"]).ravel().astype(int),
    )


def subject_paths(root) -> list[Path]:
    """All SX.pkl files under a WESAD folder, sorted by subject number."""
    root = Path(root)
    paths = list(root.glob("S*/S*.pkl"))
    return sorted(paths, key=lambda p: int(p.stem[1:]))
