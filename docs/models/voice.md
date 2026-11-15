# Voice model card

## What it is

The owner's CNN. A 3-second clip is turned into 40 MFCC features over 216 frames
and classified; the probability of "stressed" is the score. Weights are in
`artifacts/voice/speech_model.keras`. Kept unchanged, including its exact feature
extraction.

## Training data

RAVDESS (24 actors speaking two fixed sentences in several acted emotions).
Stressed = sad, angry, fearful, disgust; relaxed = calm, happy, neutral,
surprised. See `docs/datasets.md`.

## Evaluation

Measured by `scripts/eval_voice.py` on all 1,440 RAVDESS clips. See
`results/voice_eval.md`.

- Overall accuracy: **0.9424**
- Per-actor accuracy: min 0.8333, median 0.9500, max 1.0000
- Robust to loudness: 0.9347 to 0.9417 as the volume is scaled from 0.3x to 3x

## Known limits

- **Acted emotion, not real stress.** RAVDESS actors perform emotions on cue.
- **Same speakers in training and test.** The 1,440 clips include the clips the
  model trained on, and the paper's split kept the same 24 actors on both sides,
  so this number measures that the file works, not how it generalises to new
  speakers. The paper reported 82.76%.
- Clips shorter than 1 second, or silent clips, are treated as missing.
- English, two fixed sentences; spontaneous speech is untested.

## Calibration

No clean held-out data exists (the actors overlap train and test), so the voice
score uses the identity mapping. See `results/calibration.md`.
