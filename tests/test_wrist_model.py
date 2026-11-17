from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

from safespace.data.wesad import load_subject, subject_paths
from safespace.models import artifacts_dir
from safespace.models.wrist import WristModel
from safespace.models.wrist_windows import CAL_S, subject_windows
from safespace.data.wesad import BVP_HZ, EDA_HZ, LABEL_HZ

ROOT = Path(__file__).resolve().parents[1]
WESAD = ROOT / "data" / "raw" / "wesad" / "WESAD"
HAVE_MODEL = (artifacts_dir() / "wrist" / "wrist_xgb.json").exists()
HAVE_DATA = (WESAD / "S2" / "S2.pkl").exists()
needs = pytest.mark.skipif(not (HAVE_MODEL and HAVE_DATA), reason="WESAD data or trained wrist model missing")


@pytest.mark.skipif(not HAVE_DATA, reason="WESAD data missing")
def test_windows_have_both_classes_for_s2():
    windows = subject_windows(load_subject(WESAD / "S2" / "S2.pkl"))
    targets = {w.target for w in windows}
    assert targets == {0, 1}


@needs
def test_stress_scores_higher_than_baseline_for_s2():
    subj = load_subject(WESAD / "S2" / "S2.pkl")
    windows = subject_windows(subj)
    # Calibrate on this subject's calibration period (first 120 s of baseline run).
    import numpy as np

    from safespace.models.wrist_windows import _longest_run

    base = _longest_run(subj.label, 1)
    cal_start = base[0] / LABEL_HZ
    pulse = subj.bvp[int(cal_start * BVP_HZ):int((cal_start + CAL_S) * BVP_HZ)]
    cond = subj.eda[int(cal_start * EDA_HZ):int((cal_start + CAL_S) * EDA_HZ)]
    model = WristModel().load()
    baseline = model.calibrate(pulse, cond, BVP_HZ, EDA_HZ)
    assert baseline is not None

    from safespace.models.wrist_windows import FEATURES

    stress = [w for w in windows if w.target == 1]
    calm = [w for w in windows if w.target == 0]
    xs = np.array([[w.features[f] for f in FEATURES] for w in stress])
    xc = np.array([[w.features[f] for f in FEATURES] for w in calm])
    ss = model._model.predict_proba(xs)[:, 1].mean()
    sc = model._model.predict_proba(xc)[:, 1].mean()
    assert ss > sc
