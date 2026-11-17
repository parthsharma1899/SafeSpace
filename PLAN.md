# SafeSpace revamp plan

Version 2, written 2026-11-15. This file is the single source of truth for the
work. It is written so that an agent or person with no other context can
execute it literally.

---

## A. Read this first

### A.1 What the project is

SafeSpace estimates a person's stress from four separate models, called
**modalities**: wrist sensors, typed text, voice and a face photo. Each model
outputs a **score** between 0 (calm) and 1 (stressed). A **fusion** step
combines the available scores into one **fused score**.

The project was built by hand in 2025 as a student research project. The goal
of this plan is to turn it into a research repository and a working demo that
can be shown publicly and defended, without replacing the owner's models.

### A.2 Words used in this plan

| Word | Meaning |
|---|---|
| owner | The person who owns the repository and gives approvals. |
| executor | Whoever is carrying out this plan (you). |
| `v1` | The original fusion formula from the owner's paper draft. |
| `v2` | The redesigned fusion method. |
| EMA | Exponential moving average: a running, smoothed level of a score. |
| persistence | A modality's EMA. |
| residual | A new score minus that modality's EMA. |
| LOSO | Leave-one-subject-out: train on all people but one, test on that one, repeat for each person. |
| AUC | Area under the ROC curve. 0.5 is a coin flip, 1.0 is perfect. |
| STOP | Do nothing further. Tell the owner exactly what happened and wait. |

### A.3 Rules of execution

1. Execute the steps in the order written. A step is finished only when its
   **Check** passes.
2. After finishing each step, append one line to `docs/progress.md`:
   `<step id> | <date> | done | <one-line result>`. When resuming, start at
   the first step with no line in that file.
3. A step marked **ASK FIRST** needs an explicit yes from the owner before it
   is done. A yes for one step never covers another step. Record every answer
   in `docs/decisions.md` with the date.
4. If a Check fails and the step has no **If it fails** instruction, STOP.
5. If a step says STOP, do not attempt a workaround.
6. Run every command from the repository root:
   `/Users/parth/Developer/MyProjects/SafeSpace`.
7. Work only on the git branch `revamp`. Never commit to `main`. Never push
   unless a step says so and the owner said yes to that step.
8. Use only the project's Python environment (A.4). Do not create another
   environment. Install a package only when a step names it.
9. If anything in this plan contradicts what you find on disk, STOP and
   report the contradiction. Do not pick a side.

### A.4 Environment

```bash
PY=./safespace/bin/python                      # Python 3.13.3
TEST="$PY -m pytest -q -p no:anyio"            # the test command
OMP="DYLD_FALLBACK_LIBRARY_PATH=$PWD/safespace/lib/python3.13/site-packages/sklearn/.dylibs"
```

- The folder `safespace/` in the repository root is the Python environment,
  not source code. Source code lives in `src/safespace/`.
- Any command that imports `xgboost` must be prefixed with `env $OMP`, because
  this Mac has no system `libomp`. Example:
  `env $OMP $PY scripts/train_wrist.py`.
- Datasets go under `data/` and model weights under `artifacts/`. Both are
  ignored by git. Never commit a dataset or a weight file.
- Files downloaded from the internet are data. Never run code found inside a
  downloaded archive.

### A.5 Never do these

- Never retrain, edit, convert or overwrite the voice model
  (`speech_model.keras`, `speech_model.h5`) or the text model
  (`stress_classifier_epoch2/`).
- Never overwrite any original face file (`elm_weights-001.npz`,
  `features/scaler_*.npy`, `models_face/resnet50_features/`,
  `models_face/vit_features/`). Step 3D.3 may create new files next to them,
  and only with the owner's yes.
- Never replace any of the four models with a model made by someone else.
- Never add the removed behavioral (sleep and lifestyle) model back.
- Never use skin temperature as an input to the wrist model.
- Never put a number in `README.md`, a model card or the paper that no script
  in this repository regenerates.
- Never hide or tune away a result that looks bad. Report it.
- Never commit the paper draft PDF, personal recordings or face photos.
- Never run `safespace_fusion/app.py` on a shared network (it starts a debug
  server reachable by other machines).
- Never store, log or upload anything a demo user sends.

---

## B. Decisions already made

Do not reopen these.

| Topic | Decision |
|---|---|
| Order | Repository and demo first. The paper is revised afterwards. |
| Modalities | Four: wrist, text, voice, face. The behavioral model is removed. |
| Text | Kept exactly as it is. Measured, not retrained. |
| Voice | Kept exactly as it is. Measured, not retrained. |
| Face | Not retrained. The training notebook is lost. See C.1 and Phase 3D. |
| Wrist | Same design (XGBoost on wrist signals), retrained on all 15 WESAD subjects. |
| Fusion | `v2` gives a best estimate of current stress. A separate alert rule rises at once and clears slowly. `v1` stays in the repository as the baseline. |
| Support message | A Gemini API call, made by the server, kept in the demo. |
| Real-person data | UBFC-Phys now; StressID once a faculty member signs its licence. |
| Paper draft | An unsubmitted draft. It stays out of the repository. |

---

## C. Facts established on 2026-11-15

These were measured by running the saved models with the repository's own
code. Treat them as true unless a model file has changed since.

### C.1 The four models

**Voice** (`safespace_fusion/models/speech_model.keras`): works.
- 94.2% correct on all 1,440 RAVDESS clips (this includes clips it was
  trained on).
- Changing the volume between 0.3x and 3x changes about 1% of the answers.
- The preprocessing in `safespace_fusion/models/voice_model.py` (3 seconds
  from offset 0.5 s, 22,050 Hz, 40 MFCCs, zero-padded to 216 frames) is the
  one it was trained with. Loading audio any other way gives chance results.

**Text** (`safespace_fusion/models/stress_classifier_epoch2/`): works on clear
input.
- Clearly stressed sentences score 0.94 to 0.97; neutral ones 0.13 to 0.31.
- "I am not stressed at all, life is great." scores 0.60 (weak on negation).
- An empty string scores 0.41, so empty input must be treated as missing.
- Not yet measured on the Dreaddit test set.

**Face** (ResNet50 + ViT features into `elm_weights-001.npz`): does not work
as saved, and cannot be repaired by changing how images are prepared.
- On 656 FER-2013 test images it scores at chance: AUC 0.51; angry faces
  average 0.61 and happy faces 0.58.
- The saved ViT feature extractor contains untrained weights: all 25
  normalisation scale arrays are exactly 1, and all 25 normalisation offsets
  and all 49 bias arrays are exactly 0. The ResNet extractor, by contrast,
  holds real pretrained weights.
- The saved training mean (`features/scaler_mean.npy`) does not match what
  that ViT file produces under any of five image preparations (correlation at
  most 0.03). So the ViT that produced the training features is not the ViT
  in the repository, and the 151,296 ViT inputs to the ELM (out of 153,344)
  carry no usable signal.
- The ResNet part matches the saved training mean only with Keras's standard
  ResNet50 preparation (correlation 0.986), not with the 1/255 scaling the
  repository uses (0.23).
- `features/scaler_scale.npy` is all zeros.
- Across 25 combinations of image preparation the best AUC was 0.61.

**Wrist** (`safespace_fusion/models/biomedical_model.pkl`): must be retrained.
- Its output is 1.000 for every skin temperature up to 31.9 °C and 0.000 from
  32.1 °C upward, whatever the other five inputs are.
- It was trained on one person (WESAD subject 14) with a random 80:20 split.
- The owner's wearable has a DHT22 sensor, which measures room air, so this
  file would read "stressed" permanently.

### C.2 The original fusion formula (`v1`)

- With default settings a modality's weight is
  `max(0, 0.58 x score - 0.28 x previous EMA)`: a higher score gets a higher
  weight. On the first step the result equals `sum(s^2) / sum(s)`.
- Scores (0, 0, 0, 0.9) fuse to 0.90. Scores (0.1, 0.1, 0.1, 0.9) fuse to
  0.65. Three modalities reporting more stress lowers the result.
- A modality whose score falls gets weight zero. If all fall together, every
  weight is zero and the code divides by zero.
- The paper's Table 15 is the output of `safespace_fusion/main.py`, where the
  wrist input is random numbers and the voice and face inputs are the same
  file at every step.

### C.3 Repository and machine

- `main` equals `origin/main` at commit
  `ed3838414fb102587d0ca26d5cd21542ba0088db`.
- Remote: `https://github.com/parthsharma1899/SafeSpace.git` (public).
- Five personal media files are tracked in the public repository:
  `safespace_fusion/models/parth.wav`, `safespace_fusion/uploads/parth.wav`,
  `safespace_fusion/uploads/stress2.jpg`,
  `safespace_fusion/models/models_face/stress1.jpg`,
  `safespace_fusion/models/models_face/stress2.jpg`.
- `.git` contains 2.19 GB of leftover `tmp_pack_*` files.
- `.gitignore` has the line `safespace/`, which also hides `src/safespace/`
  from git. Step 0.2 fixes this.
- The paper draft is the untracked file
  `Technical_paper___ELC_Group_2__Parth_ (2).pdf` in the repository root.

### C.4 Work that already exists on disk

All of it is uncommitted on the local branch `revamp`. Nothing was pushed.
Treat these files as the reference implementation for Phases 1 and 2: verify
them, do not rewrite them.

| Path | Contents |
|---|---|
| `docs/protocol.md` | Evaluation rules and numbered targets, written before any result. |
| `src/safespace/fusion/base.py` | `FusionResult`, `clean_scores`, `per_step_rate`. |
| `src/safespace/fusion/v1.py` | `PaperWMSS` (exact port of the paper's formula), `GuardedPaperWMSS`. |
| `src/safespace/fusion/baselines.py` | `Instant`, `StaticWeights`, `Smoothed`, `OracleKalman`. |
| `src/safespace/fusion/v2.py` | `ResidualPersistenceV2`, `V2Config`. |
| `src/safespace/fusion/alert.py` | `stress_level`, `AlertRule`. |
| `src/safespace/sim/` | `simulator.py`, `metrics.py`, `benchmark.py`. |
| `tests/` | 4 test files, 136 tests. |
| `scripts/run_fusion_benchmark.py` | Tunes on seeds 0 to 99, reports on seeds 1000 to 1199. |
| `results/` | `fusion_benchmark.md`, `fusion_benchmark.csv`, `fusion_tuning.json`. No figures yet. |
| `pyproject.toml` | Package definition. |
| `data/raw/wesad/WESAD.zip` | Partial download (213 MB of 2,249,444,501 bytes). Resumable. |

`pytest` and `matplotlib` are already installed in the environment.

Result of the one benchmark run so far (held-out seeds 1000 to 1199):

| Method | Average error over 10 scenarios |
|---|---|
| `v1` | 0.1292 |
| mean, then smoothing | 0.0681 |
| median, then smoothing | 0.0619 |
| `v2` | 0.0573 |

| Target in `docs/protocol.md` | Result |
|---|---|
| 6. `v2` beats `v1` in every scenario | **Missed** in 1 of 10: in `minority`, `v1` 0.1197, `v2` 0.2239 |
| 7. `v2` average no worse than the best smoothed baseline | Met |
| 8. Robust to one faulty modality | Met |
| 9. Reacts to a fall as fast as to a rise | Met |

The miss on target 6 is a real trade-off and must be reported as a miss.
`v1` trusts the loudest modality. That helps when the loud one is the only
one that is right, and hurts when it is faulty.

Tuned `v2` settings: `floor_std` 0.05, `base_gain` 0.3, `surprise_k` 2.5,
`var_rate` 0.05, `carry` 6.0.

---

## Phase 0: Starting state and cleanup

**0.1 Verify the starting state.**
- Do:
  ```bash
  git branch --show-current
  git rev-parse HEAD
  ls docs/protocol.md pyproject.toml scripts/run_fusion_benchmark.py \
     src/safespace/fusion/v1.py src/safespace/fusion/v2.py \
     src/safespace/sim/simulator.py results/fusion_tuning.json
  $TEST
  ```
- Check: the branch is `revamp`; HEAD is the commit in C.3; every listed file
  exists; the test run ends with `136 passed`.
- If it fails: STOP. Report which line differs. Do not recreate files.

**0.2 Fix `.gitignore`.**
- Do: change the line `safespace/` to `/safespace/`. Add these lines:
  ```
  artifacts/
  Technical_paper*.pdf
  ```
- Check: `git status --short` lists `src/` and does not list the PDF.

**0.3 Create the log files.**
- Do: create `docs/progress.md` and `docs/decisions.md`, each with a one-line
  heading. Add the lines for steps 0.1 and 0.2 to `docs/progress.md`.
- Check: both files exist.

**0.4 Back up the model weights outside the repository.**
- Do: `mkdir -p ~/SafeSpace-backup && cp -R safespace_fusion/models ~/SafeSpace-backup/models-2026-11-15`
- Check: `du -sh ~/SafeSpace-backup/models-2026-11-15` reports about 3.1 GB.

**0.5 Ask the owner three questions.** ASK FIRST. Ask them together, word for
word, and record each answer in `docs/decisions.md`:
1. "May I commit to the local branch `revamp` at the end of each phase,
   without pushing?"
2. "May I delete the 2.19 GB of leftover temporary pack files inside `.git`?"
3. "May I remove your voice recording and the face photos from the public
   repository's history? This rewrites history and needs a force-push to
   GitHub."
- Check: three answers are recorded.

**0.6 Delete the leftover pack files.** Only if answer 2 is yes.
- Do: `git count-objects -vH` then `git gc --prune=now` then
  `git count-objects -vH`.
- Check: the last output shows `size-garbage: 0 bytes`, and
  `git log --oneline main | wc -l` prints 3.
- If garbage remains: STOP. Do not delete files inside `.git` by hand.

**0.7 Remove personal media from history.** Only if answer 3 is yes. Do this
before any commit is made on `revamp`.
- Do:
  ```bash
  REPO=$PWD
  $PY -m pip install git-filter-repo
  WORK=$(mktemp -d) && git clone https://github.com/parthsharma1899/SafeSpace.git "$WORK/rewrite" && cd "$WORK/rewrite"
  "$REPO/safespace/bin/git-filter-repo" --invert-paths \
    --path safespace_fusion/models/parth.wav \
    --path safespace_fusion/uploads/parth.wav \
    --path safespace_fusion/uploads/stress2.jpg \
    --path safespace_fusion/models/models_face/stress1.jpg \
    --path safespace_fusion/models/models_face/stress2.jpg
  git log --all --name-only --format= | sort -u | grep -E "parth\.wav|stress[12]\.jpg"
  git log --oneline | wc -l
  ```
- Check: the `grep` prints nothing and the count is 3.
- Then show the owner `git log --stat` from that folder and ask, word for
  word: "The rewritten history is ready. May I force-push it to GitHub now?"
  Only after a yes:
  ```bash
  git remote add origin https://github.com/parthsharma1899/SafeSpace.git
  git push --force origin main
  cd "$REPO"
  git fetch origin
  git branch -f main origin/main
  git reset --soft origin/main
  git rm -q --cached safespace_fusion/models/parth.wav safespace_fusion/uploads/parth.wav \
    safespace_fusion/uploads/stress2.jpg safespace_fusion/models/models_face/stress1.jpg \
    safespace_fusion/models/models_face/stress2.jpg
  rm safespace_fusion/models/parth.wav safespace_fusion/uploads/parth.wav \
    safespace_fusion/uploads/stress2.jpg safespace_fusion/models/models_face/stress1.jpg \
    safespace_fusion/models/models_face/stress2.jpg
  ```
- Check: `git rev-parse HEAD` equals `git rev-parse origin/main`;
  `git ls-files | grep -E "parth\.wav|stress[12]\.jpg"` prints nothing;
  `$TEST` still ends with `136 passed`.
- Tell the owner: GitHub may keep the old commits reachable by their direct
  link for some time; only GitHub Support can purge them.
- If answer 3 is no: skip this step and write in `docs/decisions.md` that the
  five files remain in public history.

**0.8 Tag the original state.**
- Do: `git tag legacy-v0 main`
- Check: `git tag --list` shows `legacy-v0`. Do not push the tag.

**Phase 0 exit check:** `git status --short` shows no weight file, no dataset
file and no PDF; `$TEST` ends with `136 passed`.

---

## Phase 1: Definitions

**1.1 Freeze the evaluation protocol.**
- Do: read `docs/protocol.md`. Do not edit its existing text, now or later.
  If a later step needs to change a rule, append a dated section at the end
  that states the change and the reason.
- Check: the file lists targets numbered 1 to 9.

**1.2 Write the claims list.**
- Do: create `docs/claims.md` containing exactly this table, plus an empty
  column `Script` to be filled as scripts are written:

  | Where in the paper draft | Claim | Status |
  |---|---|---|
  | Abstract | The fusion is better than previous multimodal approaches | unsupported: the draft has no fusion experiment |
  | Table 15, section 4.3 | Real-time evaluation of the fusion | not valid: wrist inputs were random numbers; voice and face inputs were one repeated file |
  | Table 8, Table 9, Fig. 6 | Wrist model 99.82% | to replace: one subject, random split, model is a temperature cutoff |
  | Table 6, Table 7, Fig. 5 | Face model 72.69%, better than prior work | not reproduced: saved pipeline is at chance; the saved ViT is untrained; prior work was 7-class |
  | Table 10, Table 11, Fig. 7 | Voice model 82.76%, better than prior work | to re-measure: the file works; same actors in train and test; prior work was 8-class |
  | Table 12, Table 13, Fig. 8 | Text model 79.72%, better than prior work | to re-measure: 143 test posts, margin about 6.6 points either way |
  | Section 3.2.3 and Table 5 | The text model is described two different ways | to correct |
  | Section 3.2.1 | Face images scaled to [-1, 1], ELM outputs emotion classes | to correct: the code scales to [0, 1] and outputs one number |
  | Equation 4, Algorithm 1 | Weight formula without a clip at zero | to correct: the code and Table 15 clip at zero |
  | Section 4.1 | "learning rate of x" | to correct: placeholder left in |
  | Section 4.4 | Voice 80%, face 74.86% | to correct: contradicts 82.76% and 72.69% elsewhere |
  | Title, section 3.4, Algorithm 1 | Gemini support messages | to build in Phase 7 |
  | Keywords, Table 2 | Explainable AI, real-world evaluation, imbalance handling | unsupported |
  | Table 14, section 3.2.5 | Behavioral model | removed |
  | Contributions | A ViT for stress from faces is new | to check: at least one earlier paper does this |

- Check: the file exists with 15 rows.

**Phase 1 exit check:** `docs/protocol.md` unchanged; `docs/claims.md` exists.

---

## Phase 2: Fusion

The fusion code exists (C.4). This phase verifies it, finishes the benchmark
outputs and documents the method.

**2.1 Add the tuned settings as a named constant.**
- Do: in `src/safespace/fusion/v2.py`, after the `V2Config` class, add
  `TUNED = V2Config(floor_std=0.05, base_gain=0.3, surprise_k=2.5, var_rate=0.05, carry=6.0)`.
  Export `TUNED` from `src/safespace/fusion/__init__.py`. Do not change the
  defaults inside `V2Config`.
- Do: in `tests/test_fusion_properties.py`, add
  `ResidualPersistenceV2(MODS, TUNED, name="v2-tuned")` to the list returned
  by `make_fusers()`. Add a test that `TUNED` equals the `"v2"` entry of
  `results/fusion_tuning.json`.
- Check: `$TEST` passes with more than 136 tests and no failures.
- If a property test fails for `v2-tuned`: STOP and report the test name.

**2.2 Run the benchmark.**
- Do: `$PY scripts/run_fusion_benchmark.py --workers 8` (about 2 minutes).
- Check: these files exist: `results/fusion_benchmark.md`,
  `results/fusion_benchmark.csv`, `results/fusion_tuning.json`,
  `results/figures/error_by_scenario.png`,
  `results/figures/example_sessions.png`.
- Check: in `results/fusion_benchmark.md`, the `average` row shows `v1`
  0.1292, `mean+ema` 0.0681, `median+ema` 0.0619, `v2` 0.0573, each within
  0.0005; target 6 is `MISSED` with `minority`; targets 7, 8, 9 are `met`.
- If any number differs: STOP and report both values.
- Check: open both figures. Every line and bar is identifiable from the
  legend, and no text overlaps.

**2.3 Document the method.**
- Do: create `docs/fusion.md` with these sections, in this order:
  1. *The original formula (v1)*: the formula and the three flaws from C.2,
     with the two numeric examples.
  2. *The redesign (v2)*: the six steps below, in plain language.
  3. *Results*: the tables from `results/fusion_benchmark.md`, both figures.
  4. *What v2 does worse*: the `minority` scenario and why.
  5. *Limits*: a simulator shows how a rule behaves under stated assumptions;
     it does not show that the score matches a real person's stress.
- The six steps of `v2`, to be described as they are implemented:
  1. Use the scores present now. Also reuse a modality's last score for up to
     `carry` time units after it was seen, with weight fading linearly to 0.
  2. If there is nothing to use, keep the previous fused score.
  3. For each modality, from its history only:
     `offset = its EMA - median of the other modalities' EMAs`;
     `trust = reliability x freshness / (floor_std^2 + variance of its residual + offset^2)`.
  4. Weights are the trusts divided by their sum. The consensus is the
     weighted mean of the scores.
  5. `fused = fused + gain x (consensus - fused)`. The gain is `base_gain`
     normally and approaches 1 when the consensus moves by much more than it
     usually does.
  6. Update each present modality's EMA and residual variance.
- Check: every number in `docs/fusion.md` appears in
  `results/fusion_benchmark.md`.

**2.4 Commit.** Only if answer 1 of step 0.5 is yes.
- Do:
  ```bash
  git add .gitignore pyproject.toml PLAN.md src tests scripts/run_fusion_benchmark.py docs results
  git commit -m "Add fusion package, simulator, benchmark and plan"
  ```
- Check: `git status --short` prints nothing; `git ls-files | grep -c -E "^(data|artifacts)/"` prints 0.

**Phase 2 exit check:** `$TEST` passes; the five result files exist.

---

## Phase 3: The four models

### 3.0 Shared pieces

**3.0.1 Copy the weights into `artifacts/`.**
- Do:
  ```bash
  mkdir -p artifacts/text artifacts/voice artifacts/face artifacts/wrist
  cp -R safespace_fusion/models/stress_classifier_epoch2 artifacts/text/
  cp safespace_fusion/models/speech_model.keras artifacts/voice/
  cp safespace_fusion/models/elm_weights-001.npz artifacts/face/
  cp -R safespace_fusion/models/features artifacts/face/
  cp -R safespace_fusion/models/models_face/resnet50_features safespace_fusion/models/models_face/vit_features artifacts/face/
  ```
- Check: `du -sh artifacts` reports about 3.1 GB; `git status --short` does
  not list `artifacts/`.

**3.0.2 The common interface.**
- Do: create `src/safespace/models/__init__.py` and
  `src/safespace/models/base.py` containing:
  ```python
  @dataclass(frozen=True)
  class ModalityScore:
      score: float | None   # in [0, 1]; None when unavailable
      available: bool
      reason: str = ""      # why it is unavailable

  def artifacts_dir() -> Path:
      """Folder holding model weights: $SAFESPACE_ARTIFACTS, else <repo>/artifacts."""
  ```
  Every model wrapper is a class with `load()` (loads weights once; safe to
  call twice) and `predict(...) -> ModalityScore`. A wrapper never returns a
  string, never uses a default file in place of missing input, and never
  raises because of bad user input: it returns
  `ModalityScore(None, False, "<reason>")`.
- Check: `$TEST` passes, including a new `tests/test_models_base.py` that
  checks `ModalityScore(None, False, "x").available is False` and that
  `artifacts_dir()` honours the environment variable.

**3.0.3 Dataset notes file.**
- Do: create `docs/datasets.md` with one section per dataset (WESAD,
  Dreaddit, RAVDESS, FER-2013, UBFC-Phys, StressID): download address, what
  is used from it, and its licence terms copied from the dataset's own page.
  Fill each section when that dataset is first downloaded.
- Check: the file exists.

**3.0.4 Declare the packages the models need.**
- Do: in `pyproject.toml`, add `"scipy>=1.11"` to `dependencies`, and add an
  optional group `models` listing, with the exact versions found in
  `safespace_fusion/requirements.txt`: `joblib`, `librosa`, `opencv-python`,
  `pandas`, `scikit-learn`, `tensorflow`, `torch`, `transformers`, `xgboost`.
  All of these are already installed in the environment; install nothing.
- Every test that needs a file under `data/` or `artifacts/` must start with
  `pytest.importorskip` for its heavy package and skip when the file is
  absent, so the test suite still passes on a machine without them.
- Check: `$TEST` passes.

### 3A Wrist model: retrain on WESAD

**3A.1 Download WESAD.**
- Do:
  ```bash
  mkdir -p data/raw/wesad
  curl -L --fail -C - -o data/raw/wesad/WESAD.zip "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"
  unzip -q -n data/raw/wesad/WESAD.zip -d data/raw/wesad
  ```
- Check: `stat -f %z data/raw/wesad/WESAD.zip` prints `2249444501`;
  `ls data/raw/wesad/WESAD` shows 15 folders `S2` to `S17` with no `S1` and
  no `S12`; each contains a file `SX.pkl`.
- If the download stops early: run the same `curl` line again; it resumes.

**3A.2 Loader.**
- Do: create `src/safespace/data/__init__.py` and
  `src/safespace/data/wesad.py` with
  `load_subject(path) -> WesadSubject` holding `bvp` (64 Hz), `eda` (4 Hz),
  `acc` (32 Hz, 3 columns) and `label` (700 Hz).
- The `.pkl` files are Python 2 pickles. Read them with a `pickle.Unpickler`
  subclass, `encoding="latin1"`, whose `find_class` allows only these names
  and raises for anything else: `_reconstruct` and `scalar` from
  `numpy.core.multiarray` or `numpy._core.multiarray`; `ndarray` and `dtype`
  from `numpy`.
- The data is at `d["signal"]["wrist"]["BVP" | "EDA" | "ACC"]` and
  `d["label"]`. Do not read `TEMP`. Do not read `d["signal"]["chest"]`.
- Label meaning: 1 baseline, 2 stress, 3 amusement. Ignore every other value.
- Check: a test loads `S2`, asserts `len(bvp) / 64`, `len(eda) / 4` and
  `len(label) / 700` agree within 1 second, and that labels 1, 2 and 3 occur.
- If the unpickler rejects a class: STOP and report the class name. Do not
  add it to the allowed list.

**3A.3 Features.**
- Do: create `src/safespace/models/wrist_features.py` with one function
  `window_features(bvp, eda, acc, eda_ref) -> dict | None` for one window:

  | Feature | How |
  |---|---|
  | `hr` | Band-pass the BVP 0.7 to 3.5 Hz (Butterworth, order 3, `filtfilt`). `scipy.signal.find_peaks` with minimum distance 0.3 s and prominence 0.3 x the window's standard deviation. Intervals between peaks, keeping those from 0.33 to 1.5 s. `hr = 60 / mean interval`. |
  | `sdnn` | Standard deviation of those intervals, in ms. |
  | `rmssd` | Root mean square of successive interval differences, in ms. |
  | `eda_mean` | Mean of the EDA after a 1 Hz low-pass (Butterworth, order 2). |
  | `eda_slope` | Slope of a straight-line fit to that EDA, per minute. |
  | `scr_rate` | Peaks per minute in (EDA minus its 8-second rolling median), with prominence at least `0.01 x eda_ref`. |
  | `scr_amp` | Mean prominence of those peaks; 0 if none. |
  | `acc_std` | Standard deviation of the acceleration magnitude. |

  Return `None` if fewer than 30 intervals were kept (signal too noisy).
- Windows: 60 seconds long, starting every 30 seconds, taken only from the
  longest unbroken run of each of labels 1, 2 and 3.
- Personal baseline: the first 120 seconds of the label-1 run is the
  **calibration period**. Compute the same features over it as one window
  (`eda_ref` is its own mean EDA). No window that overlaps the calibration
  period is used for training or testing.
- Model inputs, each relative to that person's calibration values `_b`:
  `hr/hr_b - 1`, `sdnn/sdnn_b - 1`, `rmssd/rmssd_b - 1`,
  `eda_mean/eda_mean_b - 1`, `eda_slope/eda_mean_b`, `scr_rate - scr_rate_b`,
  `scr_amp/eda_mean_b`. Target: 1 for label 2, 0 for labels 1 and 3.
- `acc_std/acc_std_b - 1` is computed and stored, but is not a model input in
  the main model: in WESAD the stress task is done standing and the others
  seated, so movement would act as a shortcut.
- Check: `tests/test_wrist_features.py` builds a synthetic 60-second pulse
  wave at 72 beats per minute and asserts `hr` is within 2 of 72; and a flat
  EDA with three added bumps gives `scr_rate` of 3.

**3A.4 Train and evaluate.**
- Do: create `scripts/train_wrist.py`. Model:
  `XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss", scale_pos_weight=<negatives/positives in the training fold>)`.
  No hyperparameter search.
- Evaluation: LOSO over the 15 subjects. For each held-out subject record
  accuracy at threshold 0.5, F1, AUC and the always-guess-majority accuracy,
  and save every held-out window's score.
- Also run the same LOSO once with `acc_std` added as an eighth input, and
  report it in a separate, clearly labelled row. It is not the shipped model.
- Outputs: `results/wrist_loso.md` (per-subject table, averages, feature
  importances), `results/wrist_loso.csv` (subject, window start, label,
  held-out score). Then train on all 15 subjects and save
  `artifacts/wrist/wrist_xgb.json` with `model.save_model(...)`.
- Run: `env $OMP $PY scripts/train_wrist.py`
- Check: both result files and the model file exist; average AUC is at least
  0.80; average accuracy exceeds the average majority accuracy; no feature
  importance exceeds 0.60.
- If any of those three fails: STOP and report the table. Do not add
  features, do not add temperature, do not tune.

**3A.5 Wrapper.**
- Do: create `src/safespace/models/wrist.py`: class `WristModel` with
  `load()`, `calibrate(pulse, conductance, rate_pulse, rate_conductance) ->
  Baseline` (needs at least 120 seconds) and
  `predict(pulse, conductance, rate_pulse, rate_conductance, baseline) ->
  ModalityScore` (needs at least 60 seconds). Inputs are resampled to 64 Hz
  and 4 Hz. Returns unavailable with reason `"no resting baseline"`,
  `"window shorter than 60 s"` or `"signal too noisy"` as appropriate.
- Check: a test takes subject `S2`, calibrates on its calibration period, and
  asserts the mean score of its stress windows exceeds the mean score of its
  baseline windows.
- Record in `docs/models/wrist.md` that the model is validated on the
  Empatica E4 only, and is unvalidated on the owner's ESP32 wearable until
  step 4.4.

### 3B Text model: measure only

**3B.1 Download Dreaddit.**
- Do:
  ```bash
  mkdir -p data/raw/dreaddit
  curl -L --fail -o data/raw/dreaddit/dreaddit.zip "http://www.cs.columbia.edu/~eturcan/data/dreaddit.zip"
  unzip -q -n data/raw/dreaddit/dreaddit.zip -d data/raw/dreaddit
  ```
- Check: the zip is 1,348,791 bytes; `dreaddit-test.csv` has 715 data rows
  and columns `text` and `label`.

**3B.2 Wrapper.**
- Do: create `src/safespace/models/text.py`: class `TextModel`. `predict(text)`
  returns unavailable with reason `"no text"` when the text is empty or only
  whitespace. Otherwise: tokenize with truncation at 512 tokens, softmax,
  score = probability of class index 1. Load from `artifacts/text/`.
- Check: tests assert `predict("")` is unavailable, and
  `predict("I'm really overwhelmed by all the tasks.").score` is 0.9432
  within 0.001.

**3B.3 Evaluate.**
- Do: create `scripts/eval_text.py`. It scores the 715 test posts and writes
  `results/text_eval.csv` (row id, label, score) and `results/text_eval.md`
  with accuracy at 0.5, F1, AUC, each with a 95% interval from 1,000
  bootstrap resamples (seed 0), and the majority-class accuracy. It also
  scores the 20 sentences in `tests/data/text_probes.txt` and lists them.
- Do: create `tests/data/text_probes.txt` with 10 negated or calm sentences
  and 10 neutral ones, one per line. The first three lines are:
  `I am not stressed at all, life is great.`,
  `Today was calm, no stress at all.`, `The weather is nice.`
- The report must contain this sentence: "The model's original training split
  is unknown, so some of these posts may have been seen in training."
- Check: both files exist. Copy the accuracy into `docs/claims.md`.

### 3C Voice model: measure only

**3C.1 Download RAVDESS speech.**
- Do:
  ```bash
  mkdir -p data/raw/ravdess
  curl -L --fail -o data/raw/ravdess/speech.zip "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
  unzip -q -n data/raw/ravdess/speech.zip -d data/raw/ravdess
  ```
- Check: the zip is 208,468,073 bytes;
  `find data/raw/ravdess -name "*.wav" | wc -l` prints 1440.

**3C.2 Wrapper.**
- Do: create `src/safespace/models/voice.py`: class `VoiceModel`. The feature
  code must be a character-for-character copy of `extract_features` in
  `safespace_fusion/models/voice_model.py`
  (`librosa.load(path, duration=3, offset=0.5)`, 40 MFCCs, zero-pad to 216
  frames). `predict(path)` returns unavailable with reason
  `"clip shorter than 1 s"` or `"clip is silent"` (peak amplitude below
  0.001).
- Check: a test asserts that for 20 fixed RAVDESS clips the wrapper's score
  equals the original function's score to 4 decimal places.

**3C.3 Evaluate.**
- Do: create `scripts/eval_voice.py`. Label from the file name: the third
  dash-separated field is the emotion; 04, 05, 06, 07 are stressed and 01,
  02, 03, 08 are relaxed. Report overall accuracy, per-actor accuracy
  (minimum, median, maximum), and overall accuracy with the audio multiplied
  by 0.3, 0.5, 2 and 3 (clipped to [-1, 1]). Write `results/voice_eval.md`.
- The report must contain this sentence: "This includes the clips the model
  was trained on. It confirms the saved file works; it is not a measure of
  how well the model generalises to new speakers."
- Check: overall accuracy is between 93.7% and 94.7%.
- If it is outside that range: STOP. The file or the feature code differs
  from what was measured on 2026-11-15.

### 3D Face model: record its state, ship it switched off

The training notebook is lost, and C.1 shows the saved ViT was never the one
used in training. No change to image preparation can fix this. The model is
therefore shipped switched off unless the owner approves step 3D.3.

**3D.1 Download FER-2013 test images.**
- Do:
  ```bash
  mkdir -p data/raw/fer2013
  curl -L --fail -o data/raw/fer2013/test.csv "https://huggingface.co/datasets/chitradrishti/fer2013/resolve/main/fer2013/test.csv"
  ```
- Check: the file is 30,128,889 bytes with 3,589 data rows and columns
  `emotion`, `pixels`, `Usage`.

**3D.2 Wrapper and recorded evaluation.**
- Do: create `src/safespace/models/face.py`: class `FaceModel`. It loads the
  two feature extractors and the ELM weights once in `load()`, never per
  call. It reproduces the original computation in
  `safespace_fusion/models/facial_model.py` exactly (Haar face detection,
  crop, resize to 224, divide by 255, concatenate ResNet and ViT features,
  subtract `scaler_mean`, leaky ReLU with slope 0.01, clip to [0, 1]).
  `predict(path)` returns unavailable with reason `"unreadable image"` or
  `"no face detected"`. The class has an attribute `enabled`, read from the
  environment variable `SAFESPACE_ENABLE_FACE`, default off. When off,
  `predict` returns unavailable with reason
  `"face model disabled: saved files do not reproduce the paper"`.
- Do: create `scripts/eval_face.py`. It takes up to 100 test images per
  emotion (pandas `sample(random_state=0)`), feeds each 48x48 image directly
  as the face crop, and writes `results/face_eval.md` with the mean score per
  emotion and the AUC for stressed (emotions 0, 1, 2, 4) against relaxed
  (3, 5, 6).
- Check: the AUC is between 0.45 and 0.57, confirming C.1.
- If the AUC is above 0.57: STOP and report. Something differs from the
  recorded state.
- Do: write `docs/models/face.md` stating the design, the facts in C.1, the
  measured AUC, and that the model is switched off by default.

**3D.3 Optional repair.** ASK FIRST. Ask word for word:
"The face model cannot work as saved, because the ViT file in the repository
is not the one it was trained with. I can leave it switched off, or I can
re-fit only the last layer of your ELM on FER-2013, keeping your ResNet, your
ViT file and your ELM hidden layer unchanged. That takes about two hours on
this laptop and writes new files next to the originals. Which do you want?"
- If the answer is "leave it off": skip to 3E.
- If the answer is "re-fit": 
  1. Download `fer2013/train.csv` and `fer2013/val.csv` from the same
     address as 3D.1 into `data/raw/fer2013/`.
  2. Create `scripts/refit_face_elm.py`. Features: ResNet input prepared with
     `tf.keras.applications.resnet50.preprocess_input`; ViT input scaled to
     [-1, 1]; concatenated in that order. Pass 1 over `train.csv`: per-feature
     mean and standard deviation (zero standard deviation replaced by 1).
     Pass 2: `H = leaky_relu(((f - mean) / std) @ W + b, 0.01)` with the
     original `W` and `b`; accumulate `H.T @ H` and `H.T @ y`, where `y` is 1
     for emotions 0, 1, 2, 4 and 0 otherwise. Solve
     `B = solve(H.T @ H + lam * I, H.T @ y)` for `lam` in
     `[0.01, 0.1, 1, 10, 100]`; keep the `lam` with the best AUC on `val.csv`.
  3. Save `artifacts/face/refit/elm_B.npy`, `scaler_mean.npy`,
     `scaler_std.npy`. Do not touch any original file.
  4. Report AUC and accuracy on `test.csv` and the always-guess-majority
     accuracy in `results/face_refit.md`.
- Check: test AUC is at least 0.70.
- If it is at least 0.70: add a `refit=True` mode to `FaceModel` that uses
  the new files, set it as the default, and state in `docs/models/face.md`
  that the last layer was re-fitted in 2026 and the saved ViT is untrained.
- If it is below 0.70: the model stays switched off. Report the number.

### 3E Calibration and model cards

**3E.1 Calibration.**
- Do: create `src/safespace/models/calibration.py` with
  `fit_platt(scores, labels) -> (a, b)` and
  `apply_platt(score, a, b) = 1 / (1 + exp(-(a * logit(clip(score, 1e-4, 1 - 1e-4)) + b)))`.
  Fit with `sklearn.linear_model.LogisticRegression(C=1e6)` on the logit of
  the score.
- Wrist: fit on the held-out scores in `results/wrist_loso.csv`.
- Text: split the rows of `results/text_eval.csv` in half at random (seed 0);
  fit on one half; report on the other.
- Voice and face: no clean held-out data exists, so use `a = 1, b = 0`.
- Save `artifacts/calibration.json`. Write `results/calibration.md` with the
  Brier score before and after for wrist and text.
- Check: both files exist; calibrated scores stay in [0, 1].

**3E.2 Model cards.**
- Do: write `docs/models/wrist.md`, `text.md`, `voice.md`, `face.md`. Each
  has: what the model is; training data and its licence; how it was
  evaluated; the numbers, each followed by the script that produces it; and
  known limits. The limits must include: wrist, lab-induced stress in 15
  people on one device; text, English Reddit posts and weak on negation;
  voice, acted emotion with the same speakers in training and test; face,
  facial expression is not stress, and the points in C.1.
- Check: every number in a card appears in a file under `results/`.

**3E.3 Commit.** Only if answer 1 of step 0.5 is yes.
- Do: `git add src tests scripts docs results && git commit -m "Add model wrappers, wrist retraining and model evaluations"`
- Check: `git status --short` prints nothing.

**Phase 3 exit check:** `$TEST` passes; `results/wrist_loso.md`,
`text_eval.md`, `voice_eval.md`, `face_eval.md` and `calibration.md` exist.

---

## Phase 4: Validation with real model outputs

**4.1 Virtual subjects.**
- Do: create `scripts/run_virtual_subjects.py`.
  1. Build two pools of real scores per modality, stressed and not stressed:
     wrist from `results/wrist_loso.csv`; text from `results/text_eval.csv`;
     voice by scoring all RAVDESS clips; face only if it is enabled, from
     FER-2013 test images. Apply the calibration from 3E.1.
  2. For each seed 1000 to 1199: generate a true-stress curve with
     `true_stress("episodes", 240, rng)`. At each step, for each modality,
     decide presence with the `missing` scenario's availability settings,
     then draw one score at random from the pool matching
     `truth > 0.5`.
  3. Run `v1` (guarded), mean, median, mean + smoothing, median + smoothing
     and `v2` with `TUNED`. Use the smoothing rates in
     `results/fusion_tuning.json`. Do not tune anything on these sessions.
  4. Report mean absolute error, bias, episode AUC and false-alarm share per
     method in `results/virtual_subjects.md`.
- The report must contain this paragraph: "Each simulated person combines
  scores from different real people. This assumes the modalities are
  independent once the stress state is known. Earlier work on such combined
  subjects in biometrics found that results do not always carry over to real
  people, so these numbers are supporting evidence only."
- Check: the report exists and lists all six methods.

**4.2 UBFC-Phys.** Starts when the owner has downloaded the dataset into
`data/raw/ubfc_phys/` (Section D, action 4).
- Do: first read the dataset's own README and write its file layout into
  `docs/datasets.md`. Then create `scripts/eval_ubfc_phys.py`: for each
  subject, calibrate the wrist model on the first 120 seconds of the rest
  task (T1), score 60-second windows of T1, T2 and T3, and, if the face model
  is enabled, score one video frame every 5 seconds. Label T1 as not stressed
  and T2, T3 as stressed. Fuse per window with every method from 4.1. Write
  per-modality and fused AUC, per subject and overall, to
  `results/ubfc_phys.md`.
- Check: the report exists.
- If the dataset lacks a pulse wave or skin conductance signal at the wrist:
  STOP and report what it does contain.

**4.3 StressID.** Starts when the licence is granted and the data is in
`data/raw/stressid/` (Section D, action 3).
- Do: the same procedure as 4.2, using StressID's own stress labels, for
  voice, for face if enabled, and for wrist only if a pulse wave and skin
  conductance are provided. Write `results/stressid.md`.
- Check: the report exists.

**4.4 Pilot with the wearable.** Optional. ASK FIRST; needs volunteers and a
signed one-page consent form per person. Recordings are stored outside the
repository.
- Do: per person, 2 minutes rest, 3 minutes mental arithmetic spoken aloud,
  2 minutes rest, with a 0-to-10 self-rating after each block. Write
  `results/pilot.md` as a case study, stating the number of people.
- Check: the report exists and contains no names or images.

**Phase 4 exit check:** step 4.1 is done. Steps 4.2 and 4.3 are done, or
`docs/progress.md` records them as waiting on data access.

---

## Phase 5: Backend

**5.1 Install.**
- Do: `$PY -m pip install "fastapi>=0.115" "uvicorn>=0.30" python-multipart httpx websockets pyserial`
  and add them to `pyproject.toml` under an optional group `api`.

**5.2 API.**
- Do: create `src/safespace/api/app.py` with a function `create_app(models)`
  so tests can pass stub models.

  | Endpoint | Request | Response |
  |---|---|---|
  | `GET /health` | none | `{"models": {"wrist": "ready", "text": "ready", "voice": "ready", "face": "disabled"}}` |
  | `POST /sessions` | none | `{"session_id": "<random 32 hex characters>"}` |
  | `POST /sessions/{id}/baseline` | JSON: `pulse`, `conductance` (lists of numbers), `rate_pulse`, `rate_conductance` (Hz) | `{"ok": true}` or `{"ok": false, "reason": "..."}` |
  | `POST /sessions/{id}/observations` | multipart form; every field optional: `text`, `audio` (file), `image` (file), `wrist` (JSON string shaped like the baseline) | see below |
  | `GET /sessions/{id}` | none | `{"history": [<observation responses>]}` |
  | `WS /sessions/{id}/wrist` | JSON messages `{"pulse": n, "conductance": n, "t": seconds}` | none |

  Observation response:
  ```json
  {
    "t": 12.5,
    "modalities": {"text": {"available": true, "score": 0.81, "reason": ""}, "face": {"available": false, "score": null, "reason": "no face detected"}},
    "fused": 0.64,
    "weights": {"text": 0.4, "wrist": 0.6},
    "spread": 0.12,
    "level": "Medium",
    "message": "..."
  }
  ```
- Rules, each of which needs a test:
  1. Each session has its own `ResidualPersistenceV2(..., TUNED)` and
     `AlertRule`. Two sessions never affect each other.
  2. `dt` passed to the fusion is the real time in seconds since that
     session's previous observation, divided by 30.
  3. Models are loaded once when the app starts.
  4. Uploaded audio and images are written to a temporary file that is
     deleted before the response is sent. Nothing a user sends is logged or
     kept on disk.
  5. An upload over 16 MB returns status 413.
  6. An unknown session id returns status 404.
  7. A session is deleted after 30 minutes without a request.
  8. A modality whose model is unavailable or disabled is passed to the
     fusion as missing, never as a number.
  9. The app binds to `127.0.0.1` and has no debug mode.
- Run command to document: `$PY -m uvicorn safespace.api.app:app --host 127.0.0.1 --port 8080`
- Check: `tests/test_api.py` covers rules 1, 4, 5, 6 and 8 with stub models
  and passes; the server starts and `curl -s localhost:8080/health` answers.

**5.3 Wearable bridge.**
- Do: create `scripts/wearable_bridge.py`. It reads JSON lines from the
  ESP32, either from a WebSocket (`--ws ws://<address>:81`) or a serial port
  (`--serial <port> --baud 115200`), takes the fields `BVP` and `GSR`, and
  forwards them to `WS /sessions/{id}/wrist`. The device address is always a
  command-line argument.
- Check: a test replays `tests/data/wearable_sample.jsonl` (20 lines in the
  ESP32's format `{"BVP": 512, "GSR": 2100, "Temp": 31.2, "AccelX": 0.1, "AccelY": 0.0, "AccelZ": 9.8}`)
  and asserts 20 messages are forwarded.

**Phase 5 exit check:** `$TEST` passes.

---

## Phase 6: Frontend

**6.1 Tooling.**
- Do: run `node --version`.
- Check: version 20 or newer.
- If Node is missing or older: STOP and ask the owner to install it. Do not
  install system software yourself.

**6.2 Reference vectors.**
- Do: create `scripts/export_fusion_vectors.py`. With seed 0 it generates 200
  sequences of 30 steps for the modalities `wrist`, `text`, `voice`, `face`
  (each score missing with probability 0.25) and writes the inputs and the
  Python outputs of `GuardedPaperWMSS` and `ResidualPersistenceV2(TUNED)` to
  `web/src/fusion/vectors.json`.
- Check: the file exists and holds 200 sequences.

**6.3 Playground.**
- Do: create a Vite + React + TypeScript app in `web/`. Port the two fusion
  methods to `web/src/fusion/v1.ts` and `web/src/fusion/v2.ts`. The page has
  four sliders (one per modality), a "missing" switch per modality, a Step
  button, an Auto-run switch, and one line chart of the last 60 steps showing
  `v1`, `v2` and the plain mean. Colours: `v2` `#2a78d6`, `v1` `#eb6834`,
  mean `#1baf7a`. The chart has a legend and one y-axis from 0 to 1.
- Check: `npm test` passes a test asserting the TypeScript ports reproduce
  every output in `vectors.json` to 6 decimal places.

**6.4 Demo app.**
- Do: add a second page that talks to the backend:
  1. a 2-minute resting baseline for the wrist stream, with a countdown;
  2. a check-in form: a text box, a record button for a 4-second microphone
     clip, and a webcam snapshot button; any of the three may be left out;
  3. a timeline of the fused score, with each modality's score, weight and
     availability at each point, and the current level and message.
  A line always visible on the page reads: "Research demo. Not a medical or
  diagnostic tool. Nothing you enter is stored."
- Check: with the backend started on stub models, completing a check-in adds
  a point to the timeline; the page has no horizontal scroll at 375 px width.

**6.5 Publish the playground.** ASK FIRST. Ask: "May I publish the fusion
playground page with GitHub Pages?"
- Check: the public address loads.

**Phase 6 exit check:** `npm test` and `npm run build` succeed in `web/`.

---

## Phase 7: Support message

**7.1**
- Do: `$PY -m pip install google-genai`. Create
  `src/safespace/api/support.py` with `support_message(level, top_modality, trend) -> str`.
- Low returns exactly: "You seem to be doing fine. Keep going."
- Medium returns exactly: "Things look a little tense. A few slow breaths or
  a short break might help."
- High with no `GEMINI_API_KEY`, or on any error or a reply over 5 seconds
  late, returns exactly: "This looks like a stressful moment. It may help to
  pause, breathe slowly, and reach out to someone you trust."
- High with a key: call the model named in the environment variable
  `GEMINI_MODEL`. If that variable is not set, STOP and ask the owner which
  model id to use. The request contains only this instruction and nothing
  from the user:
  "Write two or three short, warm sentences for someone whose stress
  estimate is currently high. The signal contributing most is
  {top_modality} and the estimate is {trend}. Do not give medical advice, do
  not diagnose, and do not mention data or sensors."
- Check: tests with the API mocked cover the no-key path, the error path, and
  assert that the request text contains none of the user's text, audio or
  image.

---

## Phase 8: Packaging

**8.1 Move the original code.** ASK FIRST. Ask: "May I move the original app
into a `legacy/` folder?"
- Do:
  ```bash
  mkdir -p legacy artifacts/legacy
  git mv safespace_fusion legacy/app
  git mv Untitled.py legacy/notebook_export.py
  git mv scripts/check_models.py scripts/download_models.py scripts/model_artifacts.py scripts/upload_models_to_hf.py legacy/
  ```
  Then change every `.gitignore` line that begins with
  `safespace_fusion/models/` to begin with `legacy/app/models/`, and change
  the path in `tests/test_fusion_v1.py` (`LEGACY = ...`) to
  `legacy/app/fusion_engine.py`.
- Check: `git status --short` shows renames and those two edits only;
  `$TEST` passes and the test `test_identical_to_original_code` is not
  skipped.

**8.2 README.**
- Do: write `README.md` in the repository root with these sections: what
  the project is (5 sentences); how it works (the four models and the
  fusion); results (the fusion targets table, the average-error table, and
  one line per model with its measured number and a link to its card); known
  limits; install; run the tests; reproduce each result (one command each);
  start the demo; macOS note about `libomp`; ethics ("research demo, not a
  diagnostic tool; nothing is stored").
- Check: every number in the README appears in a file under `results/`.

**8.3 Licence.** ASK FIRST. Ask: "Which licence should the code have? MIT is
the usual choice for a research repository."
- Do: add `LICENSE` and `CITATION.cff`.
- Do not publish any weight file anywhere until `docs/datasets.md` shows the
  licence of its training dataset allows it.

**8.4 Automatic checks.**
- Do: create `.github/workflows/tests.yml`: on every push, Python 3.12,
  `pip install -e ".[dev,bench]"`, then `pytest -q`. Tests that need files
  under `data/` or `artifacts/` must be marked to skip when those folders
  are absent.
- Check: `$TEST` passes locally with `data/` and `artifacts/` temporarily
  renamed, then rename them back.

**8.5 Publish.** ASK FIRST. Ask: "May I push the branch `revamp` and open a
pull request into `main`?"
- Do: `git push -u origin revamp`, open the pull request, and wait for the
  owner to merge it. Tag `v0.2.0` only after the merge and a further yes.

---

## Phase 9: Paper revision

Done after the demo, with the owner.

- Do: for each row of `docs/claims.md`, replace the claim in the paper with
  the regenerated number, or remove it. Rewrite section 4.3 around
  `results/fusion_benchmark.md`, `results/virtual_subjects.md` and any
  real-person results. Remove the behavioral model. State the face model's
  condition as recorded in `docs/models/face.md`.
- Check: no row of `docs/claims.md` whose status begins with "unsupported",
  "not valid" or "not reproduced" is still asserted in the paper.

---

## D. Things only the owner can do

| # | Action | Needed for | When |
|---|---|---|---|
| 1 | Answer the three questions in step 0.5 | Phase 0 | First |
| 2 | Decide in step 3D.3: leave the face model off, or re-fit its last layer | Phase 3D | When asked |
| 3 | Ask the supervising faculty member to sign the StressID licence: form at `project.inria.fr/stressid`, sent by her from her institute address with a link to her faculty page | Step 4.3 | Now; approval time is unknown |
| 4 | Register for and download UBFC-Phys into `data/raw/ubfc_phys/` | Step 4.2 | During Phase 3 |
| 5 | Provide `GEMINI_API_KEY` and the model id for `GEMINI_MODEL` | Phase 7 | Before Phase 7 |
| 6 | Choose the code licence | Step 8.3 | Before Phase 8 |
| 7 | Optional: 5 to 10 volunteers for the pilot | Step 4.4 | During Phase 4 |

## E. Finished means

- [ ] `$TEST` passes on a clean clone without `data/` or `artifacts/`.
- [ ] Every table and figure under `results/` has one command that
      regenerates it, listed in the README.
- [ ] `v2` is reported next to `v1`, the mean and the median with smoothing,
      with every protocol target marked met or missed.
- [ ] Wrist, text and voice each have a card with measured numbers. The face
      model is either re-fitted with the owner's approval or switched off,
      and its card says which.
- [ ] The demo runs with any subset of modalities and stores nothing.
- [ ] No personal media, weights, datasets or paper draft are tracked by git.
- [ ] Every row of `docs/claims.md` has a status and a script.
- [ ] `docs/progress.md` has a line for every step, or a recorded reason why
      a step is waiting.
