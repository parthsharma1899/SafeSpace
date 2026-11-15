# Wrist model: leave-one-subject-out evaluation (WESAD)

Model: XGBoost on wrist heart-rate, HRV and skin-conductance features, each
relative to the subject's own resting baseline. Produced by `scripts/train_wrist.py`.
Stress (label 2) is the positive class; baseline (1) and amusement (3) are negative.

- Windows: 981 from 15 subjects (311 stress)
- **Overall out-of-fold accuracy: 0.8899** (majority baseline 0.6830)
- Overall AUC: 0.9512
- Overall F1: 0.8218

With `acc_std` added (movement; NOT the shipped model, since the WESAD stress
task is done standing and the others seated): accuracy 0.8940, AUC 0.9459.

## Per-subject

| subject | windows | stress | accuracy | majority | AUC |
|---|---:|---:|---:|---:|---:|
| S2 | 63 | 19 | 0.698 | 0.698 | 0.911 |
| S3 | 64 | 20 | 0.750 | 0.688 | 0.784 |
| S4 | 64 | 20 | 1.000 | 0.688 | 1.000 |
| S5 | 65 | 20 | 0.985 | 0.692 | 1.000 |
| S6 | 65 | 20 | 0.985 | 0.692 | 1.000 |
| S7 | 65 | 20 | 0.954 | 0.692 | 0.999 |
| S8 | 65 | 21 | 0.785 | 0.677 | 0.998 |
| S9 | 65 | 20 | 0.938 | 0.692 | 0.990 |
| S10 | 68 | 23 | 0.971 | 0.662 | 0.996 |
| S11 | 66 | 21 | 0.939 | 0.682 | 0.975 |
| S13 | 66 | 21 | 0.909 | 0.682 | 0.995 |
| S14 | 66 | 21 | 0.924 | 0.682 | 0.990 |
| S15 | 66 | 21 | 0.742 | 0.682 | 0.990 |
| S16 | 66 | 21 | 0.894 | 0.682 | 1.000 |
| S17 | 67 | 23 | 0.866 | 0.657 | 0.991 |

## Feature importance (model trained on all subjects)

| feature | importance |
|---|---:|
| scr_amp_norm | 0.446 |
| hr_rel | 0.154 |
| eda_mean_rel | 0.133 |
| rmssd_rel | 0.087 |
| scr_rate_diff | 0.080 |
| eda_slope_norm | 0.051 |
| sdnn_rel | 0.049 |
