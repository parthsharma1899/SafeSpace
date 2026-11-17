# The fusion method

Each of the four models outputs a stress score between 0 and 1. Fusion combines
the scores available at each moment into one score over time. SafeSpace's main
method is **v3**. This page explains the idea behind it, how it evolved from the
paper's original formula (v1) through an intermediate redesign (v2), and how all
three compare.

Results numbers are copied from [`results/fusion_benchmark.md`](../results/fusion_benchmark.md)
and [`results/virtual_subjects.md`](../results/virtual_subjects.md). The two
small v1 arithmetic examples are properties of the formula, asserted in
[`tests/test_fusion_v1.py`](../tests/test_fusion_v1.py).

## The idea

SafeSpace is meant for **continuous monitoring**. Every person has a baseline,
a normal level, for each signal. Against that baseline:

- a sudden jump in one signal is a **residual**: something may have changed;
- if the jump stays, it is **persistence**: the change is real, not noise, and
  that signal should count for more.

A one-off spike is probably noise; a change that lasts is information. The
fused score should follow the signals that have moved away from the person's
normal and stayed there.

## v3: the idea, implemented

For each signal, v3 keeps:

| Quantity | Meaning |
|---|---|
| baseline | this person's normal level for the signal |
| residual | how far the latest reading is from that normal, in units of the signal's own normal wobble (is this unusual *for this signal*?) |
| persistence | a short running average of the residual: a spike that stays builds it up, a one-off spike fades |

A signal's weight is the paper's Equation 4, with those definitions:

```
weight = floor + alpha * |residual| + beta * |persistence|
```

Only change beyond the signal's normal wobble adds weight, and rises and falls
count alike. The weight comes from what the signal has done so far; the newest
reading moves the score through its own change. So a one-off spike nudges the
score, and a spike that stays takes over on the next reading.

The output has two parts:

```
normal = average of the present signals' baselines     (the person's normal)
change = weighted average of each signal's change from its normal
score  = normal + change                                (clipped to [0, 1])
```

With no history, the baselines equal the first readings, the change is zero and
the score is the plain average.

### What changed from v1, and why

| | v1 (paper) | v3 |
|---|---|---|
| Baseline | none: the moving average included the current score | each person's normal per signal |
| Residual | score minus its moving average | score minus the baseline, in units of the signal's normal wobble |
| Persistence | the moving average of the raw score | a running average of the residual |
| Weight | `max(0, 0.7 * residual + 0.3 * persistence)` | `floor + alpha * |residual| + beta * |persistence|`, from history |
| Falls | a falling signal got weight zero | rises and falls count alike |

v1's formula did not implement its own idea. Persistence was the score's
level, not a change from normal, so a signal that is always high for someone
got extra weight forever. The weight grew with the score itself, which made
v1 read too high: scores (0, 0, 0, 0.9) fuse to 0.90, but (0.1, 0.1, 0.1, 0.9)
fuse to 0.70, so three signals reporting more stress lowered the result. A
falling signal got weight zero, so recovery was ignored, and if every signal
fell together the weights summed to zero and the formula failed.

### Guards added to v3

1. **The baseline pauses during a lasting change**, so a long stress episode is
   not learned as "normal".
2. **A new person's baseline can start from a typical calm value** per signal
   (the app uses each model's average score on calm examples) and becomes the
   person's own after 5 readings.
3. **Residuals are measured in each signal's own units of normal wobble**, so a
   jump that is ordinary for a noisy signal is not mistaken for a spike.
4. **A deadband:** change within a signal's normal wobble adds no weight, so at
   rest every signal counts equally.
5. **Falls count as well as rises.**
6. **Each signal's recent level is smoothed** before it is combined, which keeps
   the score steady.
7. **A stuck-sensor check** treats a signal whose last 10 readings are flat as
   missing (on for sensor streams, off for app check-ins), and **one signal's
   share can be capped**; tuning chose a cap of 60%.
8. **Every rate is time-aware**, so gaps between readings are handled.
9. **Two outputs:** the score (absolute, comparable across people) and the
   change from the person's normal, which drives the alert once the baseline
   is learned.

Tuning set `alpha` to 0: persistence alone, not the latest residual, decides how
much a signal counts. The residual still matters, as what persistence averages.

## How it was tested

A simulator generates sessions with a known true stress curve; each signal
reports it through its own noise, bias, delays, gaps and faults. Every method's
score is compared with the truth. Settings are tuned on seeds 0-99 and every
number below comes from seeds 1000-1199. The rules and targets were written
before the results (`docs/protocol.md`).

The first ten scenarios mostly assume all signals track the same stress. Three
were added for v3's assumptions: a personal offset per signal (`personal`), one
signal leading the others (`leader`), and sessions that have either a lone
detector or a stuck sensor (`lone_or_stuck`).

## Results

Mean absolute error against the true stress (lower is better):

| scenario | v1 | mean | smoothed mean | v2 | v3 |
|---|---:|---:|---:|---:|---:|
| clean | 0.0698 | 0.0458 | 0.0344 | 0.0324 | 0.0358 |
| unequal | 0.1026 | 0.0646 | 0.0454 | 0.0324 | 0.0493 |
| missing | 0.1132 | 0.0678 | 0.0462 | 0.0470 | 0.0527 |
| stuck | 0.1838 | 0.0862 | 0.0803 | 0.0379 | 0.1403 |
| flipping | 0.2305 | 0.1298 | 0.0913 | 0.0388 | 0.0868 |
| biased | 0.1518 | 0.0743 | 0.0738 | 0.0452 | 0.0759 |
| calm | 0.0893 | 0.0443 | 0.0283 | 0.0292 | 0.0311 |
| steps | 0.0710 | 0.0452 | 0.0305 | 0.0253 | 0.0327 |
| minority | 0.1197 | 0.1830 | 0.1789 | 0.2216 | 0.1192 |
| mixed | 0.1602 | 0.0890 | 0.0724 | 0.0653 | 0.0812 |
| personal | 0.1005 | 0.0464 | 0.0367 | 0.0407 | 0.0393 |
| leader | 0.1059 | 0.1104 | 0.1042 | 0.1197 | 0.0859 |
| lone_or_stuck | 0.1507 | 0.1376 | 0.1325 | 0.1350 | 0.1294 |
| average | 0.1269 | 0.0865 | 0.0735 | 0.0670 | 0.0738 |

Targets for v3, set before the run:

| target | result |
|---|---|
| meets requirements 1-5 (bounded, never fails, passes agreement through, more stress never lowers the score, missing signals left out) | met |
| beats v1 in every scenario | met (all 13) |
| average error no higher than the plain mean | met (0.0738 vs 0.0865) |
| beats the plain mean where one signal leads | met (minority 0.1192 vs 0.1830; leader 0.0859 vs 0.1104) |
| reacts to a fall within 1.5x its reaction to a rise | **missed**: rise 0.02 steps, fall 0.04 steps |

The missed target is about speed: both reactions are well under one step, so
the ratio is not meaningful at that scale, but it is reported as missed.

What the results show:

- **v3 is the best method when one signal leads.** In `leader` it beats every
  method tested (0.0859), and in `minority` it matches v1 (0.1192 vs 0.1197),
  far ahead of v2 (0.2216) and the averages. This is what the persistence
  weighting is for.
- **Its weak spot is a stuck sensor.** A signal stuck high looks exactly like a
  lasting real change, so v3 trusts it (0.1403 in `stuck`, against 0.0379 for
  v2). The 60% cap limits the damage; the stuck check catches a perfectly flat
  signal but not the simulator's slightly wobbling one.
- **Overall, v3 ties the smoothed average** (0.0738 vs 0.0735). v2 has the best
  average (0.0670) because most scenarios assume every signal agrees, which is
  what v2 rewards.
- **v3 reads close to the truth** (bias +0.0128, against +0.0840 for v1) and
  rarely false-alarms (0.0056 of calm steps, against 0.0689 for v1).

What each part contributes (each switched off on its own; average error 0.0738
with everything on):

| v3 variant | average | minority | leader | stuck |
|---|---:|---:|---:|---:|
| equal weights (no persistence weighting) | 0.0745 | 0.1794 | 0.1045 | 0.0790 |
| no baseline pause | 0.0739 | 0.1759 | 0.1028 | 0.0800 |
| rises only | 0.0737 | 0.1189 | 0.0859 | 0.1405 |
| no smoothing | 0.0844 | 0.1236 | 0.0929 | 0.1461 |

The persistence weighting is what makes v3 win when one signal leads, at the
cost of the stuck case. Smoothing helps everywhere. The baseline pause and
counting falls barely change these 240-step sessions; the pause matters in long
episodes (`tests/test_fusion_v3.py::test_persistent_stress_is_not_learned_as_normal`).

### On real model outputs (virtual subjects)

Real held-out scores from the four trained models, drawn to match a simulated
stress curve (`scripts/run_virtual_subjects.py`). Each simulated person combines
scores from different real people, so this also assumes the signals agree.

| method | MAE | bias | episode AUC | false-alarm share |
|---|---:|---:|---:|---:|
| v1 | 0.2301 | +0.1870 | 0.8874 | 0.1624 |
| mean | 0.1624 | -0.0305 | 0.9535 | 0.0101 |
| smoothed mean | 0.1257 | -0.0312 | 0.9922 | 0.0002 |
| v2 | 0.1252 | -0.0265 | 0.9830 | 0.0019 |
| v3 | 0.1384 | -0.0099 | 0.9878 | 0.0005 |

v3 is well ahead of v1 and the plain mean and is the least biased method, but
here it trails v2 and the smoothed mean.

## v2, for reference

v2 kept residual and persistence but used them to decide how much to *trust*
each signal: a signal was trusted less if it was jumpy or sat away from the
others. That rewards agreement, which is why it wins when every signal tracks
the same stress and loses badly when one signal is the only one that is right.
It is kept in `src/safespace/fusion/v2.py` as a comparison.

## Limits

- The simulator shows how a rule behaves under stated assumptions; the virtual
  subjects add real model error but still pair signals from different people.
  Neither shows that the score matches one real person's stress. That needs
  several signals recorded from the same people (UBFC-Phys, StressID).
- Whether "one signal leads" or "all signals agree" is closer to real people is
  exactly what those datasets will decide, and it determines whether v3's
  strength or its weak spot matters more.
- In the web app, the baseline builds over a person's check-ins. Until a
  person has 5 check-ins, a result is close to the plain average.
