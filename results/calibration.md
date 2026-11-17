# Per-modality calibration

Produced by `scripts/fit_calibration.py`.

- **wrist**: Brier 0.0825 -> 0.0788 (a=0.631, b=-0.088)
- **text**: Brier 0.0750 -> 0.0730 on held-out half (a=1.267, b=-0.395)
- **voice**: identity mapping (actors overlap train/test, no clean held-out data)
- **face**: identity mapping (score already a probability from its own re-fit)

Calm prior (average calibrated score on not-stressed held-out examples):

- **wrist**: 0.1135 (n = 670)
- **text**: 0.1331 (n = 346)
- **voice**: 0.1722 (n = 672)
- **face**: 0.4071 (n = 450)
