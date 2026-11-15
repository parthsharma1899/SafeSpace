from pathlib import Path

import pytest

pytest.importorskip("librosa")
pytest.importorskip("tensorflow")

from safespace.models import artifacts_dir
from safespace.models.voice import VoiceModel

ROOT = Path(__file__).resolve().parents[1]
RAVDESS = ROOT / "data" / "raw" / "ravdess"
HAVE_WEIGHTS = (artifacts_dir() / "voice" / "speech_model.keras").exists()
CLIPS = sorted(RAVDESS.glob("Actor_*/*.wav"))[:20]

needs = pytest.mark.skipif(not (HAVE_WEIGHTS and CLIPS), reason="voice weights or RAVDESS clips missing")


@needs
def test_wrapper_matches_the_original_function():
    """The wrapper must reproduce the original predict_stress to 4 decimal places."""
    import importlib.util

    legacy_path = ROOT / "safespace_fusion" / "models" / "voice_model.py"
    spec = importlib.util.spec_from_file_location("legacy_voice", legacy_path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)

    model = VoiceModel().load()
    for clip in CLIPS:
        assert model.predict(str(clip)).score == pytest.approx(legacy.predict_stress(str(clip)), abs=1e-4)
