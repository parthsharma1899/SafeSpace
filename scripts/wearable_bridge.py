#!/usr/bin/env python3
"""Forward ESP32 wearable samples to the SafeSpace API's wrist WebSocket.

The device emits one JSON object per line, e.g.
    {"BVP": 512, "GSR": 2100, "Temp": 31.2, "AccelX": 0.1, "AccelY": 0.0, "AccelZ": 9.8}

Only BVP (pulse) and GSR (skin conductance) are forwarded. The device address is
always given on the command line, never hard-coded:

    # from a WebSocket device
    python scripts/wearable_bridge.py --session <id> --api ws://127.0.0.1:8080 --ws ws://192.168.1.50:81
    # from a serial device
    python scripts/wearable_bridge.py --session <id> --api ws://127.0.0.1:8080 --serial /dev/tty.usbserial-0001 --baud 115200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Callable, Iterable, Optional


def parse_sample(line: str) -> Optional[dict]:
    """Return {'pulse', 'conductance', 't'} from one device line, or None if unusable."""
    line = line.strip()
    if not line:
        return None
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        return None
    if "BVP" not in d or "GSR" not in d:
        return None
    try:
        return {"pulse": float(d["BVP"]), "conductance": float(d["GSR"]), "t": float(d.get("t", time.time()))}
    except (TypeError, ValueError):
        return None


def forward_samples(lines: Iterable[str], send: Callable[[dict], None]) -> int:
    """Parse each line and send the valid samples. Returns how many were forwarded."""
    count = 0
    for line in lines:
        sample = parse_sample(line)
        if sample is not None:
            send(sample)
            count += 1
    return count


def _serial_lines(port: str, baud: int):
    import serial

    with serial.Serial(port, baud, timeout=1) as ser:
        while True:
            raw = ser.readline()
            if raw:
                yield raw.decode("utf-8", errors="ignore")


def _ws_device_lines(url: str):
    from websockets.sync.client import connect

    with connect(url) as ws:
        while True:
            yield ws.recv()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", required=True, help="API session id")
    ap.add_argument("--api", required=True, help="API base WebSocket URL, e.g. ws://127.0.0.1:8080")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ws", help="device WebSocket URL")
    src.add_argument("--serial", help="device serial port")
    ap.add_argument("--baud", type=int, default=115200)
    args = ap.parse_args(argv)

    from websockets.sync.client import connect

    device = _ws_device_lines(args.ws) if args.ws else _serial_lines(args.serial, args.baud)
    sink_url = f"{args.api.rstrip('/')}/sessions/{args.session}/wrist"
    with connect(sink_url) as sink:
        def send(sample):
            sink.send(json.dumps(sample))

        n = forward_samples(device, send)
    print(f"forwarded {n} samples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
