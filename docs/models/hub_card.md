---
license: other
license_name: mixed-dataset-terms
tags:
  - stress-detection
  - multimodal
  - research
---

# SafeSpace models

Trained weights for [SafeSpace](https://github.com/parthsharma1899/SafeSpace), a
research prototype that estimates stress from wrist signals, text, voice and a
face photo, and fuses them over time into one score.

**Research use only.** Not a diagnostic tool. Facial expression and acted speech
are proxies for stress, not stress itself. Each model's limits are in its card
under `docs/models/` in the code repository.

## Download

No account or login is needed. From the SafeSpace code repository:

```bash
python scripts/download_models.py      # all four models, about 1.5 GB
```

The script downloads into `artifacts/` and checks every file against the sizes
and SHA-256 hashes in `scripts/model_manifest.json`. Then follow the code
repository's README to run the app locally.

**Terms of use:** non-commercial research and personal use only. The weights
were trained on datasets with their own terms (below); RAVDESS in particular is
CC BY-NC-SA 4.0, which rules out commercial use.

## Files

The layout matches the code's `artifacts/` folder.

| Path | Model | What it is |
|---|---|---|
| `calibration.json` | all | Per-model score calibration (Platt a, b) and each model's average score on calm examples, used as a new person's starting baseline |
| `wrist/wrist_xgb.json` | wrist | XGBoost on baseline-relative BVP, EDA and motion features; WESAD, leave-one-subject-out AUC 0.95 |
| `text/stress_classifier_epoch2/` | text | Fine-tuned BERT classifier; Dreaddit |
| `voice/speech_model.keras` | voice | CNN on MFCC features; RAVDESS speech |
| `face/resnet50_features/`, `face/vit_features/` | face | Frozen ResNet50 and ViT feature extractors (TensorFlow SavedModel). The ViT holds untrained weights; see the face card |
| `face/elm_hidden_f16.npz` | face | ELM hidden layer (W, b, original B), with W stored in float16: 0.61 GB instead of 2.45 GB, scores within 0.0002 of the original |
| `face/refit/` | face | ELM output layer re-fitted on FER-2013 (test AUC 0.70), its feature scaler and probability mapping |
| `face/features/` | face | The original feature scaler, for the original (reference) scoring path |
| `legacy/` | - | The original 2025 files, kept for reference and not used by the app |

## Training data and terms

The weights inherit the terms of the data they were trained on. Check these
before any use beyond private research.

| Model | Data | Terms |
|---|---|---|
| wrist | WESAD (Schmidt et al., 2018) | Research dataset; cite the paper. No redistribution terms given; ask the authors before publishing weights |
| text | Dreaddit (Turcan & McKeown, 2019) | No licence stated; Reddit content terms apply |
| voice | RAVDESS (Livingstone & Russo, 2018) | CC BY-NC-SA 4.0: non-commercial, attribution, share-alike |
| face | FER-2013 (Kaggle) | No clear licence; Kaggle competition rules |
