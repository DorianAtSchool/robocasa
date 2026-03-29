#!/usr/bin/env python3
"""Local browser for RoboCasa sweep outputs.

Usage:
    python scripts/visualize_sweep_output.py \
        --sweep-dir tmp/sweep_all_tasks_all_trajectories_L11_L42
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

from flask import Flask, abort, render_template_string, request, send_from_directory, url_for

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_trajectories import _iter_completed_runs


IMAGE_COLUMNS = [
    "room_view",
    "top_view",
    "map",
    "agentview_center",
    "agentview_left",
    "agentview_right",
    "wrist",
]


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RoboCasa Sweep Viewer</title>
  <style>
    :root {
      --bg: #f5efe4;
      --panel: #fffaf0;
      --ink: #1f2933;
      --muted: #5b6b79;
      --accent: #8b5e34;
      --border: #decfb7;
      --shadow: 0 14px 40px rgba(31, 41, 51, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
      color: var(--ink);
      background: linear-gradient(180deg, #f5efe4 0%, #ece2d2 100%);
      overflow-x: hidden;
    }
    .page {
      display: grid;
      grid-template-columns: 320px 1fr;
      min-height: 100vh;
      width: 100%;
      overflow-x: clip;
    }
    .sidebar {
      background: linear-gradient(180deg, #23323e 0%, #1b252e 100%);
      color: #f8f3ea;
      padding: 24px;
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
      min-width: 0;
    }
    .sidebar h1 {
      font-size: 1.4rem;
      margin: 0 0 8px;
    }
    .sidebar p, .sidebar label, .sidebar option, .sidebar select, .sidebar input {
      font-family: "Avenir Next", "Helvetica Neue", sans-serif;
    }
    .sidebar form {
      display: grid;
      gap: 12px;
      margin-top: 18px;
    }
    .sidebar input, .sidebar select, .sidebar button {
      width: 100%;
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,0.14);
      background: rgba(255,255,255,0.08);
      color: #f8f3ea;
      padding: 10px 12px;
    }
    .sidebar button {
      background: #c28b54;
      border-color: #c28b54;
      color: #1b252e;
      font-weight: 600;
      cursor: pointer;
    }
    .sidebar a {
      color: #f8f3ea;
      text-decoration: none;
      display: block;
      padding: 8px 10px;
      border-radius: 8px;
      margin-bottom: 4px;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .sidebar a.active {
      background: rgba(255,255,255,0.12);
    }
    .content {
      padding: 28px;
      min-width: 0;
      overflow-x: hidden;
    }
    .hero {
      background: var(--panel);
      border: 1px solid var(--border);
      box-shadow: var(--shadow);
      border-radius: 20px;
      padding: 24px;
      margin-bottom: 20px;
      min-width: 0;
      overflow-x: hidden;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
      margin-top: 16px;
    }
    .metric {
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 14px;
    }
    .metric .label {
      color: var(--muted);
      font-size: 0.85rem;
      font-family: "Avenir Next", "Helvetica Neue", sans-serif;
    }
    .metric .value {
      color: var(--accent);
      font-size: 1.25rem;
      margin-top: 6px;
      font-weight: 700;
      font-family: "Avenir Next", "Helvetica Neue", sans-serif;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .section {
      background: var(--panel);
      border: 1px solid var(--border);
      box-shadow: var(--shadow);
      border-radius: 20px;
      padding: 20px;
      margin-bottom: 20px;
      min-width: 0;
      overflow-x: hidden;
    }
    .step-selector {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 14px;
    }
    .step-selector a {
      text-decoration: none;
      color: var(--ink);
      border: 1px solid var(--border);
      padding: 8px 10px;
      border-radius: 999px;
      background: #fff;
      font-family: "Avenir Next", "Helvetica Neue", sans-serif;
      font-size: 0.9rem;
    }
    .step-selector a.active {
      background: #d9b48b;
      border-color: #d9b48b;
    }
    .image-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 16px;
    }
    .image-card {
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 12px;
      min-height: 220px;
    }
    .image-card h4 {
      margin: 0 0 10px;
      font-family: "Avenir Next", "Helvetica Neue", sans-serif;
    }
    .image-card img {
      width: 100%;
      height: auto;
      border-radius: 10px;
      border: 1px solid #eee2d0;
      background: #f7f0e5;
    }
    .code-block {
      background: #1e252d;
      color: #edf2f7;
      border-radius: 16px;
      padding: 16px;
      overflow: auto;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.9rem;
    }
    .json-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-family: "Avenir Next", "Helvetica Neue", sans-serif;
      table-layout: fixed;
    }
    th, td {
      text-align: left;
      padding: 10px;
      border-bottom: 1px solid #eadfcf;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    code {
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    @media (max-width: 1100px) {
      .page { grid-template-columns: 1fr; }
      .sidebar { position: static; height: auto; }
      .metrics, .image-grid, .json-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="page">
    <aside class="sidebar">
      <h1>RoboCasa Sweep Viewer</h1>
      <p>{{ run_count }} episodes from <code>{{ sweep_dir }}</code></p>
      <form method="get">
        <label for="task">Task</label>
        <select name="task" id="task">
          <option value="">All tasks</option>
          {% for task in tasks %}
          <option value="{{ task }}" {% if task == selected_task %}selected{% endif %}>{{ task }}</option>
          {% endfor %}
        </select>
        <label for="q">Search episode id</label>
        <input id="q" name="q" value="{{ query }}" placeholder="traj_000123">
        <button type="submit">Filter</button>
      </form>
      <div style="margin-top: 22px;">
        {% for item in filtered_runs %}
        <a href="{{ url_for('index', task=selected_task, q=query, episode=item['episode_id']) }}" class="{% if selected_run and item['episode_id'] == selected_run['episode_id'] %}active{% endif %}">
          {{ item['episode_id'] }}
        </a>
        {% endfor %}
      </div>
    </aside>
    <main class="content">
      {% if not selected_run %}
      <section class="hero"><h2>No matching episodes</h2></section>
      {% else %}
      <section class="hero">
        <h2 style="margin: 0 0 6px;">{{ selected_run['task'] }}</h2>
        <div style="color: var(--muted); font-family: 'Avenir Next', 'Helvetica Neue', sans-serif;">{{ selected_run['episode_id'] }}</div>
        <div class="metrics">
          <div class="metric"><div class="label">Steps</div><div class="value">{{ selected_run['num_steps'] }}</div></div>
          <div class="metric"><div class="label">Layout</div><div class="value">{{ selected_run['layout'] }}</div></div>
          <div class="metric"><div class="label">Style</div><div class="value">{{ selected_run['style'] }}</div></div>
          <div class="metric"><div class="label">Seed</div><div class="value">{{ selected_run['seed'] }}</div></div>
          <div class="metric"><div class="label">Run Dir</div><div class="value" style="font-size: 0.95rem;">{{ selected_run['run_dir'] }}</div></div>
        </div>
      </section>

      <section class="section">
        <h3 style="margin-top: 0;">Step Browser</h3>
        <div class="step-selector">
          {% for idx in step_ids %}
          <a href="{{ url_for('index', task=selected_task, q=query, episode=selected_run['episode_id'], step=idx) }}" class="{% if idx == selected_step['step_index'] %}active{% endif %}">
            Step {{ idx }}
          </a>
          {% endfor %}
        </div>
        <div class="metrics" style="margin-top: 18px;">
          <div class="metric"><div class="label">Tool</div><div class="value" style="font-size: 1rem;">{{ selected_step['tool'] or 'n/a' }}</div></div>
          <div class="metric"><div class="label">Robot</div><div class="value">{{ selected_step['robot_idx'] }}</div></div>
          <div class="metric"><div class="label">Success</div><div class="value">{{ 'yes' if selected_step['success'] else 'no' }}</div></div>
        </div>
        <h4>Arguments</h4>
        <div class="code-block">{{ selected_step['args_json'] }}</div>
      </section>

      <section class="section">
        <h3 style="margin-top: 0;">Images</h3>
        <div class="image-grid">
          {% for name, path in selected_step['images'].items() %}
          <div class="image-card">
            <h4>{{ name }}</h4>
            {% if path %}
            <img src="{{ url_for('artifact', relpath=path) }}" alt="{{ name }}">
            {% else %}
            <div style="color: var(--muted); font-family: 'Avenir Next', 'Helvetica Neue', sans-serif;">No image</div>
            {% endif %}
          </div>
          {% endfor %}
        </div>
      </section>

      <section class="section">
        <h3 style="margin-top: 0;">Episode Step Table</h3>
        <table>
          <thead>
            <tr><th>Step</th><th>Tool</th><th>Robot</th><th>Success</th></tr>
          </thead>
          <tbody>
            {% for step in selected_run['steps'] %}
            <tr>
              <td><a href="{{ url_for('index', task=selected_task, q=query, episode=selected_run['episode_id'], step=step['step_index']) }}">Step {{ step['step_index'] }}</a></td>
              <td>{{ step['tool'] }}</td>
              <td>{{ step['robot_idx'] }}</td>
              <td>{{ 'yes' if step['success'] else 'no' }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </section>

      <section class="section">
        <h3 style="margin-top: 0;">Trajectory JSON</h3>
        <div class="json-grid">
          <div>
            <h4>Adapted Trajectory</h4>
            <div class="code-block">{{ selected_run['adapted_trajectory'] }}</div>
          </div>
          <div>
            <h4>Original Trajectory</h4>
            <div class="code-block">{{ selected_run['original_trajectory'] }}</div>
          </div>
        </div>
      </section>

      <section class="section">
        <h3 style="margin-top: 0;">Execution Metadata</h3>
        <div class="code-block">{{ selected_run['execution_metadata'] }}</div>
      </section>
      {% endif %}
    </main>
  </div>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local sweep-output browser")
    parser.add_argument("--sweep-dir", type=str, required=True, help="Path to sweep output directory")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    return parser.parse_args()


def _to_relpath(root: Path, path: Path) -> str | None:
    root = root.resolve()
    path = path.resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return None


def resolve_image_map(root: Path, step: dict) -> dict[str, str | None]:
    images = {name: None for name in IMAGE_COLUMNS}
    for img_path_str in step.get("image_paths") or []:
        img_path = Path(img_path_str)
        if not img_path.is_absolute():
            img_path = (Path.cwd() / img_path).resolve()
        if not img_path.exists() and img_path.suffix == ".png":
            img_path = img_path.with_suffix(".jpg")
        if not img_path.exists():
            continue
        fname = img_path.stem
        for view_token in IMAGE_COLUMNS:
            if f"_{view_token}_" in f"_{fname}_":
                images[view_token] = _to_relpath(root, img_path)
                break
    return images


@lru_cache(maxsize=1)
def load_runs(root_str: str) -> list[dict]:
    root = Path(root_str)
    runs = []
    for run in _iter_completed_runs(root):
        metadata = run["metadata"]
        steps = []
        for step in metadata.get("steps", []):
            args_clean = {
                key: value
                for key, value in (step.get("args") or {}).items()
                if key != "image_paths"
            }
            steps.append({
                "step_index": step.get("step_index", 0),
                "tool": step.get("tool", ""),
                "robot_idx": step.get("robot_idx", 0),
                "success": step.get("success", False),
                "args_json": json.dumps(args_clean, indent=2),
                "images": resolve_image_map(root, step),
            })

        def read_text(name: str) -> str:
            path = run["run_dir"] / name
            return path.read_text() if path.exists() else "{}"

        runs.append({
            "episode_id": run["episode_id"],
            "task": run["task"],
            "task_dir": run["task_dir"],
            "layout": run["layout"],
            "style": run["style"],
            "seed": run["seed"],
            "run_dir": run["run_dir_rel"],
            "num_steps": len(steps),
            "steps": steps,
            "adapted_trajectory": read_text("adapted_trajectory.json"),
            "original_trajectory": read_text("original_trajectory.json"),
            "execution_metadata": read_text("trajectory_execution_metadata.json"),
        })
    return sorted(runs, key=lambda item: item["episode_id"])


def create_app(root: Path) -> Flask:
    app = Flask(__name__)
    app.config["SWEEP_ROOT"] = root.resolve()

    @app.route("/")
    def index():
        runs = load_runs(str(app.config["SWEEP_ROOT"]))
        selected_task = request.args.get("task", "")
        query = request.args.get("q", "").strip()
        filtered_runs = [
            run
            for run in runs
            if (not selected_task or run["task"] == selected_task)
            and query.lower() in run["episode_id"].lower()
        ]

        selected_episode = request.args.get("episode")
        if not selected_episode and filtered_runs:
            selected_episode = filtered_runs[0]["episode_id"]
        selected_run = next((run for run in filtered_runs if run["episode_id"] == selected_episode), None)

        selected_step = None
        step_ids = []
        if selected_run:
            step_ids = [step["step_index"] for step in selected_run["steps"]]
            requested_step = request.args.get("step", type=int)
            if requested_step is None and step_ids:
                requested_step = step_ids[0]
            selected_step = next(
                (step for step in selected_run["steps"] if step["step_index"] == requested_step),
                selected_run["steps"][0] if selected_run["steps"] else None,
            )

        tasks = sorted({run["task"] for run in runs})
        return render_template_string(
            PAGE_TEMPLATE,
            sweep_dir=str(app.config["SWEEP_ROOT"]),
            run_count=len(runs),
            tasks=tasks,
            selected_task=selected_task,
            query=query,
            filtered_runs=filtered_runs,
            selected_run=selected_run,
            selected_step=selected_step,
            step_ids=step_ids,
        )

    @app.route("/artifact/<path:relpath>")
    def artifact(relpath: str):
        root_dir = app.config["SWEEP_ROOT"]
        target = (root_dir / relpath).resolve()
        try:
            target.relative_to(root_dir)
        except ValueError:
            abort(404)
        if not target.exists() or not target.is_file():
            abort(404)
        return send_from_directory(target.parent, target.name)

    return app


def main() -> None:
    args = parse_args()
    root = Path(args.sweep_dir)
    if not (root / "sweep_summary.json").exists():
        raise FileNotFoundError(f"No sweep_summary.json found in {root}")

    app = create_app(root)
    print(f"Serving RoboCasa sweep viewer for {root}")
    print(f"Open http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
