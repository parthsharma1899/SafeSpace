# Evaluation protocol for the fusion method

This page was written before any benchmark was run. It fixes what is measured,
what the fusion methods are compared against, and what counts as success, so
that the results cannot shape the rules after the fact.

## What is being estimated

A person's stress level over the last few minutes, as a number between 0
(calm) and 1 (highly stressed).

The estimate is fused from up to four modality scores, each already a number
between 0 and 1: wrist physiology, text, voice and face. Any of them can be
missing at any moment. A fusion method receives the scores available at each
time step and returns one fused score.

## Methods compared

| Name | What it does |
|---|---|
| `v1` | The rule from the 2025 paper draft: weight = max(0, alpha x residual + beta x persistence), ported without changing its arithmetic. |
| `mean`, `median`, `max` | Plain combinations of the scores available at that step. |
| `mean+ema`, `median+ema` | The same, followed by an exponential moving average over time. |
| `oracle` | A Kalman filter that is told each modality's true noise level. It is a reference for how well any method could do, not a competitor. |
| `v2` | The redesigned residual-persistence method described in `docs/fusion.md`. |
| `v2` ablations | `v2` with one component switched off at a time. |

`v1` cannot handle a missing score, and it divides by zero when every weight
is zero. In the benchmark it is wrapped so that a missing score is replaced by
that modality's last value, and a zero-weight step falls back to the plain
mean. Both events are counted and reported.

## Simulated scenarios

Each scenario simulates one person for 240 time steps. A hidden "true stress"
curve is generated first; each modality then reports that curve through its
own noise, bias, delay, gaps and faults.

| Scenario | What it tests |
|---|---|
| `clean` | All four modalities equally reliable. |
| `unequal` | Reliability differs a lot between modalities. |
| `missing` | Realistic gaps: voice and text are present only some of the time. |
| `stuck` | One modality is stuck near 1.0 for a long stretch. |
| `flipping` | One modality outputs only 0 or 1, at random. |
| `biased` | One modality reads consistently too high. |
| `calm` | True stress stays low throughout (measures false alarms). |
| `steps` | Abrupt rise and abrupt fall (measures reaction time in each direction). |
| `minority` | Only one modality detects the stress; the other three miss it. This is designed to hurt methods that trust the majority. |
| `mixed` | Unequal reliability, gaps and occasional faults together. |

## Metrics

- **Error**: mean absolute error between the fused score and true stress.
- **Bias**: mean signed error. Positive means the method reads too high.
- **Episode detection**: area under the ROC curve for separating time steps
  with true stress above 0.5 from those below.
- **False alarms**: share of calm time steps (true stress below 0.3) where
  the fused score is above 0.7.
- **Reaction time**: in `steps`, the number of steps until the fused score
  has covered half of an abrupt change, measured separately for the rise and
  the fall.
- **Failures**: number of steps where the method could not produce a score.

## Tuning rule

Every method with a tunable setting (`v2`, and the smoothing strength of
`mean+ema` and `median+ema`) is tuned on seeds 0 to 99 only, by minimising
mean error averaged over all scenarios. Reported numbers come from seeds 1000
to 1199, which are never used for tuning. `v1` uses the paper's settings
(lambda 0.3, alpha 0.7, beta 0.3).

## What counts as success for v2

Requirements that must hold for every possible input (enforced by tests):

1. The fused score is always between 0 and 1.
2. It never fails, for any combination of scores and missing modalities.
3. If every modality reports the same value from the start, the fused score
   is that value.
4. Raising any one modality's score never lowers the fused score at that step.
5. A missing modality is left out; it is never replaced with a made-up value.

Targets on the held-out seeds:

6. `v2` has lower error than `v1` in every scenario.
7. Averaged over scenarios, `v2` has error no higher than the better of
   `mean+ema` and `median+ema`.
8. In `stuck`, `flipping` and `biased`, the error `v2` gains relative to
   `clean` is less than half of what plain `mean` gains.
9. In `steps`, `v2`'s reaction time for the fall is within 1.5 times its
   reaction time for the rise.

A target that is missed is reported as missed.

## Limits of this benchmark

The simulator tests how a fusion rule behaves under stated assumptions. It
cannot show that the fused score matches a real person's stress. That needs
recordings of several modalities from the same people, which is the next
stage of validation.
