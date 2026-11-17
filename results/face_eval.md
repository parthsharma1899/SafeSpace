# Face model evaluation (FER-2013 test images)

Model: the owner's ResNet50 + ViT + ELM in `artifacts/face/`. Produced by `scripts/eval_face.py`.
Scored 656 images (100 per emotion, detector bypassed).
Stressed = angry, disgust, fear, sad; relaxed = happy, surprise, neutral.

- **AUC (stressed vs relaxed): 0.5064** (0.50 is a coin flip)
- Always-guess-majority accuracy: 0.5427

## Mean score by true emotion

| emotion | mean score |
|---|---:|
| angry | 0.6143 |
| disgust | 0.6220 |
| fear | 0.5853 |
| happy | 0.5792 |
| sad | 0.5648 |
| surprise | 0.6249 |
| neutral | 0.5607 |

The scores barely vary by emotion and the AUC is near 0.50. The saved ViT
feature extractor holds untrained weights (all normalisation scales exactly
1, all offsets and biases exactly 0), so most of the ELM's inputs carry no
signal. See docs/models/face.md. The model ships switched off by default.
