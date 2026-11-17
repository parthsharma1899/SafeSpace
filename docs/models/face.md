# Face model card

## What it is

The owner's design: a face photo is passed through two frozen feature
extractors, a ResNet50 and a Vision Transformer (ViT), their outputs are
concatenated, and an Extreme Learning Machine (ELM) maps the result to a stress
score. Weights are in `artifacts/face/`.

The architecture (CNN + ViT features into an ELM) is the project's claimed
novelty for face-based stress detection and is kept unchanged.

## Status: on, using a 2026 re-fit of the ELM output layer

As originally saved, the pipeline scored at chance (AUC 0.5064 on FER-2013 test;
see `results/face_eval.md`). The saved ViT feature extractor holds **untrained**
weights: all 25 of its normalisation scale arrays are exactly 1, and all
normalisation offsets and biases are exactly 0, so most of the ELM's inputs
carried no signal. The ResNet extractor, by contrast, holds real pretrained
weights. `features/scaler_scale.npy` is all zeros. The original training
notebook is lost, so the correct ViT cannot be recovered.

Rather than retrain, only the ELM's **output layer** was re-solved on FER-2013,
keeping the owner's ResNet, ViT file and ELM hidden layer unchanged
(`scripts/refit_face_elm.py`). See `results/face_refit.md`:

- Chosen ridge strength (by validation AUC): lambda = 100
- **Test AUC: 0.7025** (the ship-on bar was 0.70)
- The raw ELM output is mapped to a probability by a logistic fitted on the
  validation split.

The model is enabled by default when these re-fit files are present. Set
`SAFESPACE_ENABLE_FACE=0` to turn it off; `SAFESPACE_FACE_MODE=original` runs the
paper's original (chance) path for comparison.

## Training data

FER-2013 (facial-expression images, 7 emotion classes), with emotions 0, 1, 2, 4
(angry, disgust, fear, sad) treated as stressed and 3, 5, 6 (happy, surprise,
neutral) as relaxed. See `docs/datasets.md`.

## Known limits

- **Facial expression is not stress**; at best this is a proxy.
- **Modest and ResNet-driven.** AUC 0.70 is well above chance but not strong,
  and because the saved ViT is untrained, the signal comes almost entirely from
  the ResNet features. The claimed ViT contribution is not functioning as the
  paper describes; the paper's 72.69% compared a 2-class result against prior
  7-class work, and its own Table 6 shows ViT + ELM alone at 44%.
- The re-fit used FER-2013's published train/val/test split, so it is not
  contaminated, but FER-2013 is posed expressions, not stress.
