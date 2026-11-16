import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "tests" / "data" / "wearable_sample.jsonl"

spec = importlib.util.spec_from_file_location("wearable_bridge", ROOT / "scripts" / "wearable_bridge.py")
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


def test_parse_sample_extracts_pulse_and_conductance():
    s = bridge.parse_sample('{"BVP": 512, "GSR": 2100, "Temp": 31.2}')
    assert s["pulse"] == 512.0 and s["conductance"] == 2100.0

    assert bridge.parse_sample("") is None
    assert bridge.parse_sample("not json") is None
    assert bridge.parse_sample('{"Temp": 31.2}') is None  # no BVP/GSR


def test_forwards_all_sample_lines():
    sent = []
    n = bridge.forward_samples(SAMPLE.read_text().splitlines(), sent.append)
    assert n == 20
    assert len(sent) == 20
    assert all("pulse" in s and "conductance" in s for s in sent)


def test_skips_bad_lines_in_a_stream():
    lines = ['{"BVP": 1, "GSR": 2}', "garbage", "", '{"BVP": 3, "GSR": 4}']
    sent = []
    assert bridge.forward_samples(lines, sent.append) == 2
