# Robot-Interference Analysis

`report_success_comparison.py` summarizes one-robot vs two-robot evaluation logs and creates a publication-style comparison plot.

The expected log layout is:

```text
<LOG_DIR>/
  one_robot/<TaskName>/<timestamp>/stats.json
  two_robot_traj_or_fallback/<TaskName>/<timestamp>/stats.json
```

Run from the repository root:

```bash
python model_evals/analysis/report_success_comparison.py \
  model_evals/backends/gwp/logs/robot_interference/atomic_seen_5trials
```

Outputs are written to `<LOG_DIR>/analysis/`:

```text
success_comparison.png
success_comparison.csv
summary.json
```

The plot includes:

- aggregate success rates with Wilson 95% confidence intervals
- per-task success rates with Wilson 95% confidence intervals
- success-rate deltas with conservative independent-binomial bounds
- aggregate runtime multiplier and percent change
- per-task runtime bars with multiplier and percent-change labels
- comparison-condition spawn-source counts

Timing is estimated from run directory timestamps and `stats.json` modification times. Current logs do not contain per-episode duration samples, so timing confidence intervals are not shown.

Condition names can be overridden:

```bash
python model_evals/analysis/report_success_comparison.py \
  model_evals/backends/gwp/logs/robot_interference/trajectory_tasks_5idx \
  --baseline-condition one_robot \
  --comparison-condition two_robot_trajectory
```
