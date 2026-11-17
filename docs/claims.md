# Claims list

Every number and claim in the 2025 paper draft, with its status and the script
that will regenerate or replace it. The `Script` column is filled as scripts
are written. No claim may appear in the revised paper unless its script exists
and its status is resolved.

Status values: `reproduced`, `to re-measure`, `to replace`, `not reproduced`,
`not valid`, `unsupported`, `to correct`, `to check`, `to build`, `removed`.

| Where in the paper draft | Claim | Status | Script |
|---|---|---|---|
| Abstract | The fusion is better than previous multimodal approaches | partly supported, narrowed: v3 (the paper's idea, re-implemented relative to each person's baseline) beats v1 in all 13 scenarios and is the best method tested when one signal leads; overall it ties a smoothed average, and on real model outputs paired across people it trails v2 and the smoothed average. Claim should be "tracks lasting change from a person's baseline and wins when one signal leads", not "uniformly most accurate". | scripts/run_fusion_benchmark.py, scripts/run_virtual_subjects.py |
| Table 15, section 4.3 | Real-time evaluation of the fusion | not valid: wrist inputs were random numbers; voice and face inputs were one repeated file | scripts/run_fusion_benchmark.py |
| Table 8, Table 9, Fig. 6 | Wrist model 99.82% | replaced: retrained on all 15 WESAD subjects, leave-one-subject-out accuracy 0.8899 (majority 0.6830) / AUC 0.9512, no single-feature dominance (top 0.446), no temperature. The 99.82% was one subject with a random split. | scripts/train_wrist.py |
| Table 6, Table 7, Fig. 5 | Face model 72.69%, better than prior work | not reproduced as saved (AUC 0.5064, chance; saved ViT untrained). ELM output layer re-fit in 2026 reaches AUC 0.7025 on FER-2013 test, ResNet-driven. Prior work was 7-class. | scripts/eval_face.py, scripts/refit_face_elm.py |
| Table 10, Table 11, Fig. 7 | Voice model 82.76%, better than prior work | re-measured: 94.24% on all 1,440 RAVDESS clips, but these include training clips and the paper's split shares actors, so it is not a generalisation figure. Loudness-robust (0.93-0.94 at 0.3x-3x). | scripts/eval_voice.py |
| Table 12, Table 13, Fig. 8 | Text model 79.72%, better than prior work | re-measured: 90.77% acc / 0.9669 AUC on all 715 Dreaddit test posts, but the training split is unknown so some may be seen in training (likely inflated). Weak on negation ("I am not stressed at all" scores 0.60). | scripts/eval_text.py |
| Section 3.2.3 and Table 5 | The text model is described two different ways | to correct | (paper edit) |
| Section 3.2.1 | Face images scaled to [-1, 1], ELM outputs emotion classes | to correct: the code scales to [0, 1] and outputs one number | (paper edit) |
| Equation 4, Algorithm 1 | Weight formula without a clip at zero | to correct: the code clips at zero; v3 keeps the equation's form but measures residual and persistence from the person's baseline and counts falls (docs/fusion.md) | docs/fusion.md |
| Section 4.1 | "learning rate of x" | to correct: placeholder left in | (paper edit) |
| Section 4.4 | Voice 80%, face 74.86% | to correct: contradicts 82.76% and 72.69% elsewhere | (paper edit) |
| Title, section 3.4, Algorithm 1 | Gemini support messages | to build in Phase 7 | src/safespace/api/support.py |
| Keywords, Table 2 | Explainable AI, real-world evaluation, imbalance handling | unsupported | (paper edit) |
| Table 14, section 3.2.5 | Behavioral model | removed | (none) |
| Contributions | A ViT for stress from faces is new | to check: at least one earlier paper does this | (literature check) |
