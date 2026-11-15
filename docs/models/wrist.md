# Wrist model card

## What it is

A gradient-boosted tree (XGBoost) that scores acute stress from wrist signals:
pulse (BVP), skin conductance (EDA) and movement, summarised over 60-second
windows. Every feature is expressed relative to the same person's resting
baseline, so the model reads a change from rest, not an absolute level.

This keeps the owner's design (XGBoost on wrist signals) but is retrained from
scratch; the legacy `biomedical_model.pkl` (the paper's 99.82% model, trained on
one subject with a random split) is not used. Weights are in
`artifacts/wrist/wrist_xgb.json`.

## Features (each relative to the person's 120 s resting baseline)

Heart rate and heart-rate variability (SDNN, RMSSD) from BVP; skin-conductance
level, slope, response rate and response amplitude from EDA. Movement (`acc_std`)
is computed but excluded from the shipped model, because in WESAD the stress task
is performed standing and the others seated, so movement would be a shortcut.

## Training data

WESAD (Schmidt et al., 2018): 15 subjects wearing an Empatica E4, in baseline,
stress (Trier Social Stress Test) and amusement conditions. Stress is the
positive class; baseline and amusement are negative. See `docs/datasets.md`.

## Evaluation

Leave-one-subject-out (train on 14 people, test on the 15th, repeated), by
`scripts/train_wrist.py`. See `results/wrist_loso.md`.

- 981 windows from 15 subjects (311 stress)
- **Out-of-fold accuracy: 0.8899** (majority baseline 0.6830)
- AUC: 0.9512, F1: 0.8218
- No single feature dominates (top importance 0.446, skin-conductance response
  amplitude), unlike the legacy model, which was effectively a single
  skin-temperature cutoff.

## Known limits

- Lab-induced stress in 15 adults on one device (Empatica E4). Real-world
  generalisation is untested.
- **The device matters.** The model is validated on the E4's BVP and EDA units.
  The owner's ESP32 wearable uses different sensors (a GSR sensor and a MAX30102
  pulse sensor), so a per-person resting baseline is required and the mapping to
  the E4's units is unvalidated until the Phase 4 pilot.
- Needs at least 120 s of quiet rest to calibrate and at least 60 s per reading;
  without a baseline, or on too-noisy signal, it returns "unavailable".

## Calibration

Platt scaling fitted on the out-of-fold scores; Brier 0.0825 -> 0.0788. See
`results/calibration.md`.
