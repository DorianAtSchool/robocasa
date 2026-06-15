#!/usr/bin/env python3
"""Summarize and plot one-vs-two robot evaluation results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def latest_stats_path(task_dir: Path) -> Path | None:
    candidates = sorted(task_dir.glob("*/stats.json"))
    return candidates[-1] if candidates else None


def parse_run_start(run_dir: Path) -> datetime | None:
    try:
        return datetime.strptime(run_dir.name, "%Y-%m-%d-%H-%M")
    except ValueError:
        return None


def run_time_bounds(stats_path: Path) -> tuple[datetime | None, datetime]:
    return parse_run_start(stats_path.parent), datetime.fromtimestamp(stats_path.stat().st_mtime)


def estimate_duration_seconds(stats_path: Path) -> float | None:
    start, end = run_time_bounds(stats_path)
    if start is None:
        return None
    return max(0.0, (end - start).total_seconds())


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half_width = z * math.sqrt((p * (1.0 - p) / n) + (z * z / (4.0 * n * n))) / denom
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def collect_condition(log_dir: Path, condition: str) -> dict[str, dict[str, Any]]:
    condition_dir = log_dir / condition
    if not condition_dir.is_dir():
        raise FileNotFoundError(f"Missing condition directory: {condition_dir}")

    results: dict[str, dict[str, Any]] = {}
    for task_dir in sorted(p for p in condition_dir.iterdir() if p.is_dir()):
        stats_path = latest_stats_path(task_dir)
        if stats_path is None:
            continue
        stats = load_json(stats_path)
        num_episodes = int(stats.get("num_episodes", 0))
        success_rate = float(stats.get("success_rate", 0.0))
        successes = int(round(success_rate * num_episodes))
        ci_low, ci_high = wilson_ci(successes, num_episodes)
        duration_sec = estimate_duration_seconds(stats_path)
        run_start, run_end = run_time_bounds(stats_path)
        results[task_dir.name] = {
            "task": task_dir.name,
            "condition": condition,
            "stats_path": str(stats_path),
            "run_dir": str(stats_path.parent),
            "num_episodes": num_episodes,
            "success_rate": success_rate,
            "successes": successes,
            "success_ci_low": ci_low,
            "success_ci_high": ci_high,
            "duration_seconds": duration_sec,
            "duration_minutes": duration_sec / 60.0 if duration_sec is not None else None,
            "run_start": run_start.isoformat() if run_start is not None else None,
            "run_end": run_end.isoformat(),
            "trajectory_init_mode": stats.get("trajectory_init_mode"),
            "missing_trajectory_policy": stats.get("missing_trajectory_policy"),
            "trajectory_path": stats.get("trajectory_path"),
            "spawn_sources": collect_spawn_sources(stats),
        }
    return results


def collect_spawn_sources(stats: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for init in stats.get("trajectory_initializations", []) or []:
        for placement in init.get("placements", []) or []:
            counts[str(placement.get("spawn_source") or "unknown")] += 1
    return counts


def merged_rows(
    baseline: dict[str, dict[str, Any]],
    comparison: dict[str, dict[str, Any]],
    baseline_name: str,
    comparison_name: str,
) -> list[dict[str, Any]]:
    tasks = sorted(set(baseline) | set(comparison))
    rows: list[dict[str, Any]] = []
    for task in tasks:
        base = baseline.get(task)
        comp = comparison.get(task)
        base_rate = base["success_rate"] if base else None
        comp_rate = comp["success_rate"] if comp else None
        base_duration = base.get("duration_minutes") if base else None
        comp_duration = comp.get("duration_minutes") if comp else None
        row = {
            "task": task,
            f"{baseline_name}_successes": base["successes"] if base else "",
            f"{baseline_name}_episodes": base["num_episodes"] if base else "",
            f"{baseline_name}_success_rate": base_rate if base_rate is not None else "",
            f"{baseline_name}_success_ci_low": base["success_ci_low"] if base else "",
            f"{baseline_name}_success_ci_high": base["success_ci_high"] if base else "",
            f"{baseline_name}_duration_minutes": base_duration if base_duration is not None else "",
            f"{comparison_name}_successes": comp["successes"] if comp else "",
            f"{comparison_name}_episodes": comp["num_episodes"] if comp else "",
            f"{comparison_name}_success_rate": comp_rate if comp_rate is not None else "",
            f"{comparison_name}_success_ci_low": comp["success_ci_low"] if comp else "",
            f"{comparison_name}_success_ci_high": comp["success_ci_high"] if comp else "",
            f"{comparison_name}_duration_minutes": comp_duration if comp_duration is not None else "",
            "delta_success_rate": (
                comp_rate - base_rate if base_rate is not None and comp_rate is not None else ""
            ),
            "delta_duration_minutes": (
                comp_duration - base_duration
                if base_duration is not None and comp_duration is not None
                else ""
            ),
            "duration_multiplier": (
                comp_duration / base_duration
                if base_duration not in (None, 0) and comp_duration is not None
                else ""
            ),
            "duration_percent_change": (
                ((comp_duration / base_duration) - 1.0) * 100.0
                if base_duration not in (None, 0) and comp_duration is not None
                else ""
            ),
            f"{baseline_name}_stats_path": base["stats_path"] if base else "",
            f"{comparison_name}_stats_path": comp["stats_path"] if comp else "",
        }
        if base and comp:
            # Conservative independent-binomial CI for the difference in rates.
            row["delta_success_ci_low"] = comp["success_ci_low"] - base["success_ci_high"]
            row["delta_success_ci_high"] = comp["success_ci_high"] - base["success_ci_low"]
        else:
            row["delta_success_ci_low"] = ""
            row["delta_success_ci_high"] = ""
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(condition_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    episodes = sum(int(row["num_episodes"]) for row in condition_results.values())
    successes = sum(int(row["successes"]) for row in condition_results.values())
    ci_low, ci_high = wilson_ci(successes, episodes)
    durations = [
        float(row["duration_minutes"])
        for row in condition_results.values()
        if row.get("duration_minutes") is not None
    ]
    starts = [
        datetime.fromisoformat(str(row["run_start"]))
        for row in condition_results.values()
        if row.get("run_start")
    ]
    ends = [
        datetime.fromisoformat(str(row["run_end"]))
        for row in condition_results.values()
        if row.get("run_end")
    ]
    wall_clock_minutes = (
        max(ends) - min(starts)
    ).total_seconds() / 60.0 if starts and ends else sum(durations)
    return {
        "num_tasks": len(condition_results),
        "num_episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes else 0.0,
        "success_ci_low": ci_low,
        "success_ci_high": ci_high,
        "duration_minutes": wall_clock_minutes,
        "sum_task_duration_minutes": sum(durations),
        "mean_task_duration_minutes": sum(durations) / len(durations) if durations else 0.0,
        "first_run_start": min(starts).isoformat() if starts else None,
        "last_run_end": max(ends).isoformat() if ends else None,
    }


def sort_rows(rows: list[dict[str, Any]], sort_mode: str) -> list[dict[str, Any]]:
    if sort_mode == "delta":
        return sorted(rows, key=lambda row: float(row["delta_success_rate"] or 0.0))
    if sort_mode == "duration_delta":
        return sorted(rows, key=lambda row: float(row["delta_duration_minutes"] or 0.0))
    return sorted(rows, key=lambda row: row["task"])


def fmt_duration(minutes: float) -> str:
    total_seconds = int(round(minutes * 60.0))
    hours, rem = divmod(total_seconds, 3600)
    mins, _ = divmod(rem, 60)
    if hours:
        return f"{hours}h {mins:02d}m"
    return f"{mins}m"


def duration_ratio(comparison_minutes: float, baseline_minutes: float) -> tuple[float, float]:
    if baseline_minutes <= 0:
        return 0.0, 0.0
    multiplier = comparison_minutes / baseline_minutes
    percent_change = (multiplier - 1.0) * 100.0
    return multiplier, percent_change


def fmt_runtime_ratio(comparison_minutes: float, baseline_minutes: float) -> str:
    multiplier, percent_change = duration_ratio(comparison_minutes, baseline_minutes)
    if multiplier <= 0:
        return "n/a"
    return f"{multiplier:.2f}x ({percent_change:+.0f}%)"


def plot_summary(
    rows: list[dict[str, Any]],
    baseline_name: str,
    comparison_name: str,
    baseline_summary: dict[str, Any],
    comparison_summary: dict[str, Any],
    comparison_spawn_sources: Counter[str],
    output_path: Path,
    title: str,
) -> None:
    rows = sort_rows(rows, "delta")
    tasks = [row["task"] for row in rows]
    baseline_rates = [float(row[f"{baseline_name}_success_rate"] or 0.0) * 100 for row in rows]
    comparison_rates = [float(row[f"{comparison_name}_success_rate"] or 0.0) * 100 for row in rows]
    baseline_ci_low = [float(row[f"{baseline_name}_success_ci_low"] or 0.0) * 100 for row in rows]
    baseline_ci_high = [float(row[f"{baseline_name}_success_ci_high"] or 0.0) * 100 for row in rows]
    comparison_ci_low = [float(row[f"{comparison_name}_success_ci_low"] or 0.0) * 100 for row in rows]
    comparison_ci_high = [float(row[f"{comparison_name}_success_ci_high"] or 0.0) * 100 for row in rows]
    deltas = [float(row["delta_success_rate"] or 0.0) * 100 for row in rows]
    delta_ci_low = [float(row["delta_success_ci_low"] or 0.0) * 100 for row in rows]
    delta_ci_high = [float(row["delta_success_ci_high"] or 0.0) * 100 for row in rows]
    baseline_durations = [float(row[f"{baseline_name}_duration_minutes"] or 0.0) for row in rows]
    comparison_durations = [float(row[f"{comparison_name}_duration_minutes"] or 0.0) for row in rows]
    duration_multipliers = [float(row["duration_multiplier"] or 0.0) for row in rows]
    duration_percent_changes = [float(row["duration_percent_change"] or 0.0) for row in rows]

    baseline_color = "#2F6BFF"
    comparison_color = "#F28E2B"
    positive_color = "#2E8B57"
    negative_color = "#C43C39"
    neutral_color = "#6E7781"

    width = max(16.0, 0.62 * len(tasks) + 6.5)
    fig = plt.figure(figsize=(width, 15.0), constrained_layout=True)
    gs = fig.add_gridspec(4, 1, height_ratios=[1.45, 4.1, 2.0, 3.0])
    ax_aggregate = fig.add_subplot(gs[0, 0])
    ax_rates = fig.add_subplot(gs[1, 0])
    ax_delta = fig.add_subplot(gs[2, 0])
    ax_duration = fig.add_subplot(gs[3, 0])

    ax_aggregate.axis("off")
    aggregate_cards = [
        (baseline_name, baseline_summary, baseline_color, 0.06),
        (comparison_name, comparison_summary, comparison_color, 0.36),
    ]
    for label, summary, color, x in aggregate_cards:
        ci_text = f"95% CI {summary['success_ci_low'] * 100:.1f}-{summary['success_ci_high'] * 100:.1f}%"
        ax_aggregate.text(x, 0.78, label.replace("_", " "), transform=ax_aggregate.transAxes, fontsize=12, fontweight="bold", color=color, va="center")
        ax_aggregate.text(x, 0.36, f"{summary['success_rate'] * 100:.1f}%", transform=ax_aggregate.transAxes, fontsize=34, fontweight="bold", color="#111111", va="center")
        ax_aggregate.text(x + 0.112, 0.36, f"{summary['successes']}/{summary['num_episodes']} successes\n{ci_text}", transform=ax_aggregate.transAxes, fontsize=10.5, color="#555555", va="center")
        ax_aggregate.text(x, 0.04, f"runtime {fmt_duration(summary['duration_minutes'])}", transform=ax_aggregate.transAxes, fontsize=12, color="#333333", va="center")

    runtime_multiplier, runtime_pct = duration_ratio(
        comparison_summary["duration_minutes"], baseline_summary["duration_minutes"]
    )
    runtime_color = negative_color if runtime_pct > 0 else positive_color if runtime_pct < 0 else neutral_color
    ax_aggregate.text(0.67, 0.78, "runtime multiplier", transform=ax_aggregate.transAxes, fontsize=12, fontweight="bold", color="#333333", va="center")
    ax_aggregate.text(0.67, 0.36, f"{runtime_multiplier:.2f}x", transform=ax_aggregate.transAxes, fontsize=30, fontweight="bold", color=runtime_color, va="center")
    ax_aggregate.text(0.80, 0.36, f"{runtime_pct:+.1f}%\nwall-clock estimate", transform=ax_aggregate.transAxes, fontsize=10.5, color="#555555", va="center")

    total_spawns = sum(comparison_spawn_sources.values())
    if comparison_spawn_sources:
        source_parts = [f"{source.replace('_', ' ')}: {count}/{total_spawns}" for source, count in comparison_spawn_sources.most_common()]
        spawn_text = "Comparison spawn sources: " + "  |  ".join(source_parts)
    else:
        spawn_text = "Comparison spawn sources: none recorded"
    ax_aggregate.text(0.06, -0.16, spawn_text, transform=ax_aggregate.transAxes, fontsize=10, color="#555555", va="center", ha="left")

    x_positions = list(range(len(tasks)))
    bar_width = 0.38
    base_x = [x - bar_width / 2 for x in x_positions]
    comp_x = [x + bar_width / 2 for x in x_positions]
    base_yerr = [[rate - lo for rate, lo in zip(baseline_rates, baseline_ci_low)], [hi - rate for rate, hi in zip(baseline_rates, baseline_ci_high)]]
    comp_yerr = [[rate - lo for rate, lo in zip(comparison_rates, comparison_ci_low)], [hi - rate for rate, hi in zip(comparison_rates, comparison_ci_high)]]

    ax_rates.bar(base_x, baseline_rates, width=bar_width, label=baseline_name.replace("_", " "), color=baseline_color, alpha=0.95, yerr=base_yerr, capsize=3, ecolor="#333333", linewidth=0.6)
    ax_rates.bar(comp_x, comparison_rates, width=bar_width, label=comparison_name.replace("_", " "), color=comparison_color, alpha=0.95, yerr=comp_yerr, capsize=3, ecolor="#333333", linewidth=0.6)
    ax_rates.set_ylim(0, 112)
    ax_rates.set_ylabel("Success rate (%)")
    ax_rates.set_title("Per-task success rate with Wilson 95% CI", loc="left", fontweight="bold")
    ax_rates.set_xticks(x_positions)
    ax_rates.set_xticklabels(tasks, rotation=45, ha="right", fontsize=9)
    ax_rates.grid(axis="y", alpha=0.18)
    ax_rates.spines[["top", "right"]].set_visible(False)
    ax_rates.legend(loc="upper right", frameon=False, ncols=2)
    for idx, (base, comp, row) in enumerate(zip(baseline_rates, comparison_rates, rows)):
        base_label = f"{int(row[f'{baseline_name}_successes'])}/{int(row[f'{baseline_name}_episodes'])}"
        comp_label = f"{int(row[f'{comparison_name}_successes'])}/{int(row[f'{comparison_name}_episodes'])}"
        ax_rates.text(idx - bar_width / 2, min(max(base, baseline_ci_high[idx]) + 2.0, 108), base_label, va="bottom", ha="center", fontsize=7, rotation=90 if len(tasks) > 14 else 0)
        ax_rates.text(idx + bar_width / 2, min(max(comp, comparison_ci_high[idx]) + 2.0, 108), comp_label, va="bottom", ha="center", fontsize=7, rotation=90 if len(tasks) > 14 else 0)

    colors = [positive_color if value > 0 else negative_color if value < 0 else neutral_color for value in deltas]
    delta_yerr = [[delta - lo for delta, lo in zip(deltas, delta_ci_low)], [hi - delta for delta, hi in zip(deltas, delta_ci_high)]]
    ax_delta.bar(x_positions, deltas, color=colors, width=0.58, yerr=delta_yerr, capsize=3, ecolor="#333333")
    ax_delta.axhline(0, color="#222222", linewidth=0.9)
    max_abs_delta = max([20.0] + [abs(v) for v in delta_ci_low + delta_ci_high])
    ax_delta.set_ylim(-max_abs_delta - 8, max_abs_delta + 8)
    ax_delta.set_ylabel("Delta (pp)")
    ax_delta.set_title("Success-rate delta, two robot minus one robot", loc="left", fontweight="bold")
    ax_delta.set_xticks(x_positions)
    ax_delta.set_xticklabels(tasks, rotation=45, ha="right", fontsize=9)
    ax_delta.grid(axis="y", alpha=0.18)
    ax_delta.spines[["top", "right"]].set_visible(False)
    for idx, value in enumerate(deltas):
        va = "bottom" if value >= 0 else "top"
        y = value + (1.0 if value >= 0 else -1.0)
        ax_delta.text(idx, y, f"{value:+.0f}", va=va, ha="center", fontsize=8)

    ax_duration.bar(base_x, baseline_durations, width=bar_width, label=baseline_name.replace("_", " "), color=baseline_color, alpha=0.95)
    ax_duration.bar(comp_x, comparison_durations, width=bar_width, label=comparison_name.replace("_", " "), color=comparison_color, alpha=0.95)
    ax_duration.set_ylabel("Runtime (min)")
    ax_duration.set_title("Per-task wall-clock runtime (run timestamp to stats write)", loc="left", fontweight="bold")
    ax_duration.set_xticks(x_positions)
    ax_duration.set_xticklabels(tasks, rotation=45, ha="right", fontsize=9)
    ax_duration.grid(axis="y", alpha=0.18)
    ax_duration.spines[["top", "right"]].set_visible(False)
    ax_duration.legend(loc="upper right", frameon=False, ncols=2)
    for idx, (multiplier, pct_change) in enumerate(zip(duration_multipliers, duration_percent_changes)):
        y = max(baseline_durations[idx], comparison_durations[idx]) + 0.6
        color = negative_color if pct_change > 0 else positive_color if pct_change < 0 else neutral_color
        ax_duration.text(
            idx,
            y,
            f"{multiplier:.2f}x\n{pct_change:+.0f}%",
            va="bottom",
            ha="center",
            fontsize=8,
            color=color,
        )
    ax_duration.text(0.0, -0.36, "Timing confidence intervals are not shown because current logs contain one aggregate wall-clock duration per task-condition run, not per-episode durations.", transform=ax_duration.transAxes, fontsize=9, color="#666666", ha="left", va="top")

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path, help="Directory containing condition/task/run stats.")
    parser.add_argument("--baseline-condition", default="one_robot")
    parser.add_argument("--comparison-condition", default="two_robot_traj_or_fallback")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    log_dir = args.log_dir.expanduser().resolve()
    output_dir = args.output_dir or (log_dir / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = collect_condition(log_dir, args.baseline_condition)
    comparison = collect_condition(log_dir, args.comparison_condition)
    rows = merged_rows(baseline, comparison, args.baseline_condition, args.comparison_condition)
    if not rows:
        raise RuntimeError(f"No task results found in {log_dir}")

    baseline_summary = aggregate(baseline)
    comparison_summary = aggregate(comparison)
    comparison_spawn_sources: Counter[str] = Counter()
    for row in comparison.values():
        comparison_spawn_sources.update(row["spawn_sources"])

    summary = {
        "log_dir": str(log_dir),
        "baseline_condition": args.baseline_condition,
        "comparison_condition": args.comparison_condition,
        "baseline": baseline_summary,
        "comparison": comparison_summary,
        "delta_success_rate": comparison_summary["success_rate"] - baseline_summary["success_rate"],
        "delta_duration_minutes": comparison_summary["duration_minutes"] - baseline_summary["duration_minutes"],
        "duration_multiplier": duration_ratio(
            comparison_summary["duration_minutes"], baseline_summary["duration_minutes"]
        )[0],
        "duration_percent_change": duration_ratio(
            comparison_summary["duration_minutes"], baseline_summary["duration_minutes"]
        )[1],
        "comparison_spawn_sources": dict(comparison_spawn_sources),
        "timing_note": "Per-task durations are wall-clock estimates from run directory timestamps to stats.json modification times. Aggregate duration is condition-level wall-clock span from first run start to last stats.json write. Current logs do not contain per-episode runtime samples, so timing confidence intervals are not available.",
        "success_ci_note": "Success-rate confidence intervals use Wilson 95% intervals. Delta intervals are conservative independent-binomial bounds from the two Wilson intervals.",
        "tasks": rows,
    }

    title = args.title or f"Robot interference comparison: {log_dir.name}"
    write_csv(output_dir / "success_comparison.csv", rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    plot_summary(rows, args.baseline_condition, args.comparison_condition, baseline_summary, comparison_summary, comparison_spawn_sources, output_dir / "success_comparison.png", title)

    print(f"Wrote {output_dir / 'success_comparison.png'}")
    print(f"Wrote {output_dir / 'success_comparison.csv'}")
    print(f"Wrote {output_dir / 'summary.json'}")
    print(f"{args.baseline_condition}: {baseline_summary['successes']}/{baseline_summary['num_episodes']} ({baseline_summary['success_rate'] * 100:.1f}%), runtime {fmt_duration(baseline_summary['duration_minutes'])}")
    print(f"{args.comparison_condition}: {comparison_summary['successes']}/{comparison_summary['num_episodes']} ({comparison_summary['success_rate'] * 100:.1f}%), runtime {fmt_duration(comparison_summary['duration_minutes'])}")
    print(f"delta success: {summary['delta_success_rate'] * 100:+.1f} pp")
    print(
        f"runtime multiplier: {summary['duration_multiplier']:.2f}x "
        f"({summary['duration_percent_change']:+.1f}%)"
    )


if __name__ == "__main__":
    main()
