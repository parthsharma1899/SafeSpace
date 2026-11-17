# Voice model evaluation (RAVDESS speech, all 1,440 clips)

Model: the owner's CNN in `artifacts/voice/`. Produced by `scripts/eval_voice.py`.
Stressed = sad, angry, fearful, disgust; relaxed = calm, happy, neutral, surprised.

This includes the clips the model was trained on. It confirms the saved file works; it is not a measure of how well the model generalises to new speakers.

- Overall accuracy: **0.9424** on 1440 clips
- Per-actor accuracy: min 0.8333, median 0.9500, max 1.0000

## Accuracy when the clip volume is changed

| volume | accuracy |
|---|---:|
| x0.3 | 0.9347 |
| x0.5 | 0.9361 |
| x2.0 | 0.9417 |
| x3.0 | 0.9403 |
