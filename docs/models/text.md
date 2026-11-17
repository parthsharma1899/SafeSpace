# Text model card

## What it is

The owner's fine-tuned BERT sequence classifier. A sentence is tokenised and the
model outputs a probability for "stressed" (class 1), used directly as the stress
score. Weights are in `artifacts/text/stress_classifier_epoch2/`. Kept unchanged.

## Training data

Dreaddit (Turcan & McKeown, 2019): Reddit posts labelled stressed or not. See
`docs/datasets.md`.

## Evaluation

Measured by `scripts/eval_text.py` on the Dreaddit test set (715 posts). See
`results/text_eval.md`.

- Accuracy at 0.5: **0.9077** (+/- 0.0217), against a majority baseline of 0.5161
- F1: 0.9115
- AUC: 0.9669 (+/- 0.0115)

## Known limits

- **The training split is unknown.** Some test posts may have been seen during
  training, so the accuracy above is likely optimistic. The paper reported 79.72%
  on a smaller (143-post) test set.
- **Weak on negation.** "I am not stressed at all, life is great." scores 0.60,
  and "Today was calm, no stress at all." scores 0.41. See the probe table in
  `results/text_eval.md`.
- English Reddit posts only; not validated on spoken or typed check-in text.
- Empty or whitespace-only input is treated as missing, not scored.

## Calibration

Platt scaling fitted on a random half of the Dreaddit test scores and reported on
the other half; see `results/calibration.md`.
