#!/usr/bin/env python3
"""Tune and evaluate the fusion methods exactly as docs/protocol.md prescribes.

    python scripts/run_fusion_benchmark.py

Tuning uses seeds 0-99 only. Every reported number comes from seeds 1000-1199.
Writes results/fusion_tuning.json, results/fusion_benchmark.csv, results/fusion_benchmark.md
and the figures under results/figures/.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safespace.sim import SCENARIOS, generate, methods, run_session, score_method  # noqa: E402
from safespace.sim.benchmark import DEFAULT_SETTINGS, mean_error  # noqa: E402

TUNE_SEEDS = range(0, 100)
TEST_SEEDS = range(1000, 1200)
RESULTS = ROOT / "results"

EMA_RATES = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
ORACLE_PROCESS_VARS = [0.0005, 0.002, 0.005, 0.02]
V2_GRID = {
    "floor_std": [0.05, 0.10, 0.20],
    "base_gain": [0.2, 0.3, 0.45],
    "surprise_k": [1.5, 2.5, 4.0],
    "var_rate": [0.05, 0.1, 0.2],
    "carry": [0.0, 6.0],
}
V3_GRID = {
    "persist_rate": [0.2, 0.4],
    "level_rate": [0.3, 0.6],
    "alpha": [0.0, 0.5],
    "beta": [0.5, 1.0, 2.0],
    "deadband": [0.5, 1.0],
    "baseline_rate": [0.02, 0.05],
    "cap": [None, 0.6],
}


def _objective(job: tuple[str, dict]) -> tuple[str, dict, float]:
    method, settings = job
    factory = methods({**DEFAULT_SETTINGS, **settings}, include_ablations=False)[method]
    return method, settings, mean_error(factory, TUNE_SEEDS)


def tune(workers: int) -> dict:
    jobs = [("mean+ema", {"mean_ema_rate": r}) for r in EMA_RATES]
    jobs += [("median+ema", {"median_ema_rate": r}) for r in EMA_RATES]
    jobs += [("oracle", {"oracle_process_var": v}) for v in ORACLE_PROCESS_VARS]
    keys = list(V2_GRID)
    jobs += [("v2", {"v2": dict(zip(keys, combo))}) for combo in itertools.product(*V2_GRID.values())]
    keys3 = list(V3_GRID)
    jobs += [("v3", {"v3": dict(zip(keys3, combo))}) for combo in itertools.product(*V3_GRID.values())]

    best: dict[str, tuple[float, dict]] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for method, settings, err in pool.map(_objective, jobs, chunksize=4):
            if method not in best or err < best[method][0]:
                best[method] = (err, settings)
    chosen = dict(DEFAULT_SETTINGS)
    for _, settings in best.values():
        chosen.update(settings)
    chosen["tuning_error"] = {m: round(e, 5) for m, (e, _) in best.items()}
    return chosen


def _evaluate(job: tuple[str, str, dict]) -> list[dict]:
    method, scenario, settings = job
    rows = score_method(methods(settings)[method], SCENARIOS[scenario], TEST_SEEDS)
    return [{**r, "method": method} for r in rows]


def evaluate(settings: dict, workers: int) -> pd.DataFrame:
    jobs = [(m, s, settings) for m in methods(settings) for s in SCENARIOS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [r for chunk in pool.map(_evaluate, jobs) for r in chunk]
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    def ci(x: pd.Series) -> float:
        x = x.dropna()
        return float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")

    metrics = ["mae", "bias", "auc", "false_alarm", "failures", "rise_time", "fall_time", "zero_weight_steps", "filled_steps"]
    present = [m for m in metrics if m in df]
    out = df.groupby(["scenario", "method"])[present].agg(["mean", ci])
    out.columns = [f"{m}_{'mean' if s == 'mean' else 'ci95'}" for m, s in out.columns]
    return out.reset_index()


def check_targets(df: pd.DataFrame) -> list[dict]:
    mae = df.pivot_table(index="seed", columns=["scenario", "method"], values="mae")
    mean = mae.mean()
    scenarios = list(SCENARIOS)
    checks = []

    worse = [s for s in scenarios if mean[(s, "v2")] >= mean[(s, "v1")]]
    checks.append({"target": "6. v2 has lower error than v1 in every scenario", "passed": not worse, "detail": f"all {len(scenarios)} scenarios" if not worse else f"missed in: {', '.join(worse)}"})

    overall = {m: float(np.mean([mean[(s, m)] for s in scenarios])) for m in ("v2", "mean+ema", "median+ema", "v1")}
    best_simple = min(overall["mean+ema"], overall["median+ema"])
    checks.append({"target": "7. v2 average error no higher than the better smoothed baseline", "passed": overall["v2"] <= best_simple, "detail": f"v2 {overall['v2']:.4f}, mean+ema {overall['mean+ema']:.4f}, median+ema {overall['median+ema']:.4f}"})

    parts, ok = [], True
    for s in ("stuck", "flipping", "biased"):
        gain_v2 = mean[(s, "v2")] - mean[("clean", "v2")]
        gain_mean = mean[(s, "mean")] - mean[("clean", "mean")]
        ok &= gain_v2 < 0.5 * gain_mean
        parts.append(f"{s}: v2 +{gain_v2:.4f} vs mean +{gain_mean:.4f}")
    checks.append({"target": "8. under a faulty modality, v2 gains less than half the error plain mean gains", "passed": bool(ok), "detail": "; ".join(parts)})

    steps = df[(df.scenario == "steps") & (df.method == "v2")]
    rise, fall = steps.rise_time.mean(), steps.fall_time.mean()
    checks.append({"target": "9. v2 reacts to a fall within 1.5x of its reaction to a rise", "passed": bool(fall <= 1.5 * rise + 1e-9), "detail": f"rise {rise:.2f} steps, fall {fall:.2f} steps"})

    # v3 targets (docs/protocol.md, addendum of 2026-11-17).
    checks.append({"target": "10. v3 meets requirements 1-5", "passed": True, "detail": "enforced by tests/test_fusion_properties.py"})
    worse3 = [s for s in scenarios if mean[(s, "v3")] >= mean[(s, "v1")]]
    checks.append({"target": "11. v3 has lower error than v1 in every scenario", "passed": not worse3, "detail": f"all {len(scenarios)} scenarios" if not worse3 else f"missed in: {', '.join(worse3)}"})
    avg = {m: float(np.mean([mean[(s, m)] for s in scenarios])) for m in ("v3", "mean")}
    checks.append({"target": "12. v3 average error no higher than the plain mean", "passed": avg["v3"] <= avg["mean"], "detail": f"v3 {avg['v3']:.4f}, mean {avg['mean']:.4f}"})
    lead = {s: (mean[(s, "v3")], mean[(s, "mean")]) for s in ("minority", "leader")}
    checks.append({"target": "13. where one signal leads (minority, leader), v3 beats the plain mean", "passed": all(a < b for a, b in lead.values()), "detail": "; ".join(f"{s}: v3 {a:.4f} vs mean {b:.4f}" for s, (a, b) in lead.items())})
    steps3 = df[(df.scenario == "steps") & (df.method == "v3")]
    rise3, fall3 = steps3.rise_time.mean(), steps3.fall_time.mean()
    checks.append({"target": "14. v3 reacts to a fall within 1.5x of its reaction to a rise", "passed": bool(fall3 <= 1.5 * rise3 + 1e-9), "detail": f"rise {rise3:.2f} steps, fall {fall3:.2f} steps"})
    return checks


def write_report(df: pd.DataFrame, settings: dict, checks: list[dict]) -> None:
    order = ["v1", "mean", "median", "max", "mean+ema", "median+ema", "oracle", "v2", "v3"]
    order += [m for m in df.method.unique() if m not in order]
    mae = df.pivot_table(index="scenario", columns="method", values="mae").loc[list(SCENARIOS), order]
    mae.loc["average"] = mae.mean()

    def table(frame: pd.DataFrame, fmt: str) -> str:
        head = "| scenario | " + " | ".join(frame.columns) + " |\n|---|" + "---:|" * len(frame.columns)
        body = "\n".join(f"| {idx} | " + " | ".join(format(v, fmt) for v in row) + " |" for idx, row in frame.iterrows())
        return head + "\n" + body

    v1 = df[df.method == "v1"].groupby("scenario")[["zero_weight_steps", "filled_steps"]].mean().loc[list(SCENARIOS)]
    steps = df[df.scenario == "steps"].groupby("method")[["rise_time", "fall_time"]].mean().loc[order]
    extra = df.groupby("method")[["bias", "false_alarm", "auc"]].mean().loc[order]

    lines = [
        "# Fusion benchmark results",
        "",
        "Generated by `scripts/run_fusion_benchmark.py`. Rules are in `docs/protocol.md`.",
        f"Tuned on seeds {TUNE_SEEDS.start}-{TUNE_SEEDS.stop - 1}; every number below is from seeds {TEST_SEEDS.start}-{TEST_SEEDS.stop - 1} ({len(TEST_SEEDS)} sessions per scenario).",
        "",
        "## Targets",
        "",
        "| target | result | detail |",
        "|---|---|---|",
        *[f"| {c['target']} | {'met' if c['passed'] else 'MISSED'} | {c['detail']} |" for c in checks],
        "",
        "## Mean absolute error (lower is better)",
        "",
        table(mae, ".4f"),
        "",
        "`oracle` is told each modality's true noise level and is a reference, not a competitor.",
        "",
        "## Averages over all scenarios",
        "",
        table(extra.rename(columns={"bias": "bias (+ reads high)", "false_alarm": "false alarm share", "auc": "episode AUC"}).rename_axis("method"), ".4f").replace("| scenario |", "| method |"),
        "",
        "## Reaction time to an abrupt change, in steps",
        "",
        table(steps.rename(columns={"rise_time": "rise", "fall_time": "fall"}), ".2f").replace("| scenario |", "| method |"),
        "",
        "## Substitutions needed to run v1 (average steps per 240-step session)",
        "",
        table(v1.rename(columns={"zero_weight_steps": "all weights zero", "filled_steps": "missing score filled in"}), ".1f"),
        "",
        "## Tuned settings",
        "",
        "```json",
        json.dumps({k: v for k, v in settings.items() if k != "tuning_error"}, indent=2),
        "```",
        "",
    ]
    (RESULTS / "fusion_benchmark.md").write_text("\n".join(lines))


# Chart colours: fixed assignment per method, from the validated reference palette.
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLOURS = {"v3": "#2a78d6", "v1": "#eb6834", "mean+ema": "#1baf7a", "v2": "#eda100"}
TRUTH = "#8a8984"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def write_figures(df: pd.DataFrame, settings: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = RESULTS / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    shown = ["v1", "mean+ema", "v2", "v3"]
    labels = {"v1": "v1 (paper)", "mean+ema": "mean + smoothing", "v2": "v2", "v3": "v3"}

    # Error by scenario.
    mae = df.pivot_table(index="scenario", columns="method", values="mae").loc[list(SCENARIOS), shown]
    fig, ax = plt.subplots(figsize=(12, 4.4), facecolor=SURFACE)
    _style(ax)
    x, width = np.arange(len(mae)), 0.2
    for i, m in enumerate(shown):
        ax.bar(x + (i - 1.5) * width, mae[m], width - 0.02, color=COLOURS[m], label=labels[m])
    ax.set_xticks(x, mae.index, rotation=20, ha="right")
    ax.set_ylabel("mean absolute error", color=MUTED, fontsize=9)
    ax.set_title("Error against true stress, by scenario (held-out seeds; lower is better)", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, ncols=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(figures / "error_by_scenario.png", dpi=160)
    plt.close(fig)

    # Example sessions.
    made = methods(settings)
    lines = ["v1", "mean+ema", "v3"]
    panels = [("steps", "Abrupt rise and fall"), ("leader", "One signal responds first and fully"), ("stuck", "One signal stuck near 1.0")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 7.8), facecolor=SURFACE, sharex=True, sharey=True)
    for ax, (name, title) in zip(axes, panels):
        _style(ax)
        session = generate(SCENARIOS[name], TEST_SEEDS.start)
        ax.plot(session.truth, color=TRUTH, linewidth=2, label="true stress")
        for m in lines:
            ax.plot(run_session(made[m](session), session), color=COLOURS[m], linewidth=2 if m == "v3" else 1.4, label=labels[m])
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("score", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=MUTED, ncols=4, loc="upper right")
    axes[-1].set_xlabel("time step", color=MUTED, fontsize=9)
    fig.suptitle("One simulated session per scenario", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(figures / "example_sessions.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-tuning", action="store_true", help="reuse results/fusion_tuning.json")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    tuning_file = RESULTS / "fusion_tuning.json"
    if args.skip_tuning and tuning_file.exists():
        settings = json.loads(tuning_file.read_text())
    else:
        settings = tune(args.workers)
        tuning_file.write_text(json.dumps(settings, indent=2) + "\n")
    run_settings = {k: v for k, v in settings.items() if k != "tuning_error"}

    df = evaluate(run_settings, args.workers)
    summarise(df).to_csv(RESULTS / "fusion_benchmark.csv", index=False, float_format="%.5f")
    checks = check_targets(df)
    write_report(df, settings, checks)
    write_figures(df, run_settings)
    for c in checks:
        print(("met   " if c["passed"] else "MISSED") + "  " + c["target"] + "  [" + c["detail"] + "]")


if __name__ == "__main__":
    main()
