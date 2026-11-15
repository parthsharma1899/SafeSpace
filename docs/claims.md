# Claims list

Every number and claim in the 2025 paper draft, with its status and the script
that will regenerate or replace it. The `Script` column is filled as scripts
are written. No claim may appear in the revised paper unless its script exists
and its status is resolved.

Status values: `reproduced`, `to re-measure`, `to replace`, `not reproduced`,
`not valid`, `unsupported`, `to correct`, `to check`, `to build`, `removed`.

| Where in the paper draft | Claim | Status | Script |
|---|---|---|---|
| Abstract | The fusion is better than previous multimodal approaches | unsupported: the draft has no fusion experiment | scripts/run_fusion_benchmark.py |
| Table 15, section 4.3 | Real-time evaluation of the fusion | not valid: wrist inputs were random numbers; voice and face inputs were one repeated file | scripts/run_fusion_benchmark.py |
| Table 8, Table 9, Fig. 6 | Wrist model 99.82% | to replace: one subject, random split, model is a temperature cutoff | scripts/train_wrist.py |
| Table 6, Table 7, Fig. 5 | Face model 72.69%, better than prior work | not reproduced: saved pipeline is at chance; the saved ViT is untrained; prior work was 7-class | scripts/eval_face.py |
| Table 10, Table 11, Fig. 7 | Voice model 82.76%, better than prior work | to re-measure: the file works; same actors in train and test; prior work was 8-class | scripts/eval_voice.py |
| Table 12, Table 13, Fig. 8 | Text model 79.72%, better than prior work | to re-measure: 143 test posts, margin about 6.6 points either way | scripts/eval_text.py |
| Section 3.2.3 and Table 5 | The text model is described two different ways | to correct | (paper edit) |
| Section 3.2.1 | Face images scaled to [-1, 1], ELM outputs emotion classes | to correct: the code scales to [0, 1] and outputs one number | (paper edit) |
| Equation 4, Algorithm 1 | Weight formula without a clip at zero | to correct: the code and Table 15 clip at zero | docs/fusion.md |
| Section 4.1 | "learning rate of x" | to correct: placeholder left in | (paper edit) |
| Section 4.4 | Voice 80%, face 74.86% | to correct: contradicts 82.76% and 72.69% elsewhere | (paper edit) |
| Title, section 3.4, Algorithm 1 | Gemini support messages | to build in Phase 7 | src/safespace/api/support.py |
| Keywords, Table 2 | Explainable AI, real-world evaluation, imbalance handling | unsupported | (paper edit) |
| Table 14, section 3.2.5 | Behavioral model | removed | (none) |
| Contributions | A ViT for stress from faces is new | to check: at least one earlier paper does this | (literature check) |
