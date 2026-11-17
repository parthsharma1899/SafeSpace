"""The supportive message shown with a stress level.

Low and Medium are fixed text, as in the paper. High is fixed too, unless a
Gemini API key is configured, in which case a short message is generated. The
request to Gemini contains only the stress level, the top contributing modality
and the trend -- never the user's text, audio, image or sensor data. Any failure,
missing key or slow reply (> 5 s) falls back to the fixed High message.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Optional

LOW = "You seem to be doing fine. Keep going."
MEDIUM = "Things look a little tense. A few slow breaths or a short break might help."
HIGH_FALLBACK = "This looks like a stressful moment. It may help to pause, breathe slowly, and reach out to someone you trust."

_INSTRUCTION = (
    "Write two or three short, warm sentences for someone whose stress estimate is "
    "currently high. The signal contributing most is {modality} and the estimate is "
    "{trend}. Do not give medical advice, do not diagnose, and do not mention data "
    "or sensors."
)

_TIMEOUT_S = 5.0


def _ask_gemini(prompt: str) -> str:
    from google import genai

    model = os.environ.get("GEMINI_MODEL")
    if not model:
        # No model id configured; cannot call. Fall back. (See PLAN.md D.5: the
        # owner supplies GEMINI_API_KEY and GEMINI_MODEL together.)
        raise RuntimeError("GEMINI_MODEL not set")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = client.models.generate_content(model=model, contents=prompt)
    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError("empty response")
    return text


def support_message(level: str, top_modality: Optional[str], trend: str) -> str:
    if level == "Low":
        return LOW
    if level == "Medium":
        return MEDIUM
    # High:
    if not os.environ.get("GEMINI_API_KEY"):
        return HIGH_FALLBACK
    prompt = _INSTRUCTION.format(modality=top_modality or "unclear", trend=trend)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_ask_gemini, prompt).result(timeout=_TIMEOUT_S)
    except (FutureTimeout, Exception):
        return HIGH_FALLBACK
