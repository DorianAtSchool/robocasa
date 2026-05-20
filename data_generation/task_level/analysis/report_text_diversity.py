"""Report text and tool-call diversity metrics for generated trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation.task_level.runtime.client import (
    DEFAULT_LOCATION,
    build_raw_google_genai_client,
    load_dotenv_file,
)

try:
    import numpy as np
except ImportError:  # Keep non-embedding metrics and plots dependency-free.
    np = None


DEFAULT_SWEEP_DIR = Path(
    "data_generation/task_level/data/diversity_analysis/"
    "sampling_methods_data_52Tasks_30Trajectories"
)
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
WORD_RE = re.compile(r"\b\w+\b")
PLOT_COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#4f46e5",
    "#be123c",
]


@dataclass(frozen=True)
class TrajectoryRecord:
    method: str
    task: str
    trajectory_id: str
    path: Path
    tools: tuple[str, ...]
    reasoning_texts: tuple[str, ...]
    communication_texts: tuple[str, ...]

    @property
    def reasoning_text(self) -> str:
        return "\n".join(text for text in self.reasoning_texts if text)

    @property
    def communication_text(self) -> str:
        return "\n".join(text for text in self.communication_texts if text)

    @property
    def episode_length(self) -> int:
        return len(self.tools)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute tool-call, reasoning length, communication length, and "
            "Gemini embedding cosine-similarity diversity metrics."
        )
    )
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <sweep-dir>/text_diversity_report.",
    )
    parser.add_argument("--project", default=None)
    parser.add_argument(
        "--quota-project",
        default=None,
        help=(
            "Optional quota/billing project for ADC user credentials. This sets "
            "GOOGLE_CLOUD_QUOTA_PROJECT for the current run."
        ),
    )
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument("--dotenv-path", type=Path, default=None)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-task-type", default="SEMANTIC_SIMILARITY")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=None,
        help="Defaults to <output-dir>/gemini_embedding_cache.jsonl.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Compute non-embedding metrics only.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Do not write SVG plot files.",
    )
    return parser.parse_args()


def resolve_sweep_root(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Sweep directory does not exist: {path}")
    candidates = [path]
    candidates.extend(child for child in path.iterdir() if child.is_dir())
    for candidate in candidates:
        method_dirs = [
            child
            for child in candidate.iterdir()
            if child.is_dir() and child.name != "__MACOSX"
        ]
        if any((method_dir / "summary.json").exists() for method_dir in method_dirs):
            return candidate
    return path


def load_trajectory(path: Path, *, method: str, task: str) -> TrajectoryRecord:
    payload = json.loads(path.read_text(encoding="utf-8"))
    steps = payload.get("steps") or []
    tools: list[str] = []
    reasoning_texts: list[str] = []
    communication_texts: list[str] = []

    for step in steps:
        if not isinstance(step, dict):
            continue
        tool = str(step.get("tool") or "")
        if tool:
            tools.append(tool)
        reasoning = step.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            reasoning_texts.append(reasoning.strip())
        args = step.get("args")
        message = args.get("message") if isinstance(args, dict) else None
        if tool == "communicate" and isinstance(message, str) and message.strip():
            communication_texts.append(message.strip())

    trajectory_id = str(payload.get("trajectory_id") or path.stem)
    composite_task = str(payload.get("composite_task") or task)
    return TrajectoryRecord(
        method=method,
        task=composite_task,
        trajectory_id=trajectory_id,
        path=path,
        tools=tuple(tools),
        reasoning_texts=tuple(reasoning_texts),
        communication_texts=tuple(communication_texts),
    )


def iter_records(sweep_root: Path) -> list[TrajectoryRecord]:
    records: list[TrajectoryRecord] = []
    for method_dir in sorted(sweep_root.iterdir()):
        if not method_dir.is_dir() or method_dir.name == "__MACOSX":
            continue
        for task_dir in sorted(method_dir.iterdir()):
            traj_dir = task_dir / "trajectories"
            if not traj_dir.is_dir():
                continue
            for traj_path in sorted(traj_dir.glob("*.json")):
                records.append(
                    load_trajectory(
                        traj_path,
                        method=method_dir.name,
                        task=task_dir.name,
                    )
                )
    return records


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def summary_stats(values: Iterable[float]) -> dict[str, float | int | None]:
    data = [float(value) for value in values]
    if not data:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
        }
    return {
        "count": len(data),
        "min": min(data),
        "max": max(data),
        "mean": statistics.fmean(data),
        "median": statistics.median(data),
        "std": statistics.pstdev(data),
    }


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counter.values())


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_tool_distribution_rows(records: list[TrajectoryRecord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    method_grouped: dict[str, Counter[str]] = defaultdict(Counter)
    task_grouped: dict[str, Counter[str]] = defaultdict(Counter)
    dataset_counter: Counter[str] = Counter()
    for record in records:
        grouped[(record.method, record.task)].update(record.tools)
        method_grouped[record.method].update(record.tools)
        task_grouped[record.task].update(record.tools)
        dataset_counter.update(record.tools)

    rows: list[dict[str, Any]] = []

    def append_rows(scope: str, method: str, task: str, counter: Counter[str]) -> None:
        total = sum(counter.values())
        for tool, count in sorted(counter.items()):
            rows.append(
                {
                    "scope": scope,
                    "method": method,
                    "task": task,
                    "tool": tool,
                    "count": count,
                    "percentage": count / total if total else None,
                    "total_tool_calls": total,
                    "tool_entropy": entropy(counter),
                }
            )

    for key, counter in sorted(grouped.items()):
        append_rows("method_task", key[0], key[1], counter)
    for method, counter in sorted(method_grouped.items()):
        append_rows("method", method, "__all_tasks__", counter)
    for task, counter in sorted(task_grouped.items()):
        append_rows("task", "__all_methods__", task, counter)
    append_rows("dataset", "__all_methods__", "__all_tasks__", dataset_counter)
    return rows


def build_length_rows(
    records: list[TrajectoryRecord],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_traj_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[TrajectoryRecord]] = defaultdict(list)
    method_grouped: dict[str, list[TrajectoryRecord]] = defaultdict(list)
    task_grouped: dict[str, list[TrajectoryRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.method, record.task)].append(record)
        method_grouped[record.method].append(record)
        task_grouped[record.task].append(record)
        per_traj_rows.append(
            {
                "method": record.method,
                "task": record.task,
                "trajectory_id": record.trajectory_id,
                "episode_length_steps": record.episode_length,
                "tool_calls": len(record.tools),
                "communication_events": len(record.communication_texts),
                "reasoning_word_count": word_count(record.reasoning_text),
                "reasoning_char_count": len(record.reasoning_text),
                "communication_word_count": word_count(record.communication_text),
                "communication_char_count": len(record.communication_text),
                "path": str(record.path),
            }
        )

    aggregate_rows: list[dict[str, Any]] = []

    def append_aggregate(scope: str, method: str, task: str, group: list[TrajectoryRecord]) -> None:
        metrics = {
            "episode_length_steps": [record.episode_length for record in group],
            "communication_events": [len(record.communication_texts) for record in group],
            "reasoning_word_count": [word_count(record.reasoning_text) for record in group],
            "reasoning_char_count": [len(record.reasoning_text) for record in group],
            "communication_word_count": [
                word_count(record.communication_text) for record in group
            ],
            "communication_char_count": [len(record.communication_text) for record in group],
        }
        for metric_name, values in metrics.items():
            stats = summary_stats(values)
            aggregate_rows.append(
                {
                    "scope": scope,
                    "method": method,
                    "task": task,
                    "metric": metric_name,
                    **stats,
                }
            )

    for key, group in sorted(grouped.items()):
        append_aggregate("method_task", key[0], key[1], group)
    for method, group in sorted(method_grouped.items()):
        append_aggregate("method", method, "__all_tasks__", group)
    for task, group in sorted(task_grouped.items()):
        append_aggregate("task", "__all_methods__", task, group)
    append_aggregate("dataset", "__all_methods__", "__all_tasks__", records)
    return per_traj_rows, aggregate_rows


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, path: Path):
        self.path = path
        self.embeddings: dict[str, list[float]] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                self.embeddings[str(item["key"])] = list(item["embedding"])

    def get(self, key: str) -> list[float] | None:
        return self.embeddings.get(key)

    def add_many(self, items: dict[str, list[float]]) -> None:
        if not items:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for key, embedding in items.items():
                if key in self.embeddings:
                    continue
                self.embeddings[key] = embedding
                handle.write(json.dumps({"key": key, "embedding": embedding}) + "\n")


def embedding_key(*, model: str, task_type: str, text: str) -> str:
    return f"gemini:{model}:{task_type}:{text_hash(text)}"


def extract_embedding_values(response_embedding: Any) -> list[float]:
    values = getattr(response_embedding, "values", None)
    if values is None and isinstance(response_embedding, dict):
        values = response_embedding.get("values")
    if values is None and hasattr(response_embedding, "embedding"):
        values = getattr(response_embedding.embedding, "values", None)
    if values is None:
        raise ValueError(f"Could not extract embedding values from {response_embedding!r}")
    return [float(value) for value in values]


def gemini_embed_texts(
    texts: list[str],
    *,
    model: str,
    task_type: str,
    batch_size: int,
    project: str | None,
    location: str,
    cache: EmbeddingCache,
) -> dict[str, list[float]]:
    if batch_size <= 0:
        raise ValueError(f"Embedding batch size must be positive, got {batch_size}")

    unique_texts = sorted(set(text for text in texts if text.strip()))
    result: dict[str, list[float]] = {}
    missing: list[str] = []
    for text in unique_texts:
        key = embedding_key(model=model, task_type=task_type, text=text)
        cached = cache.get(key)
        if cached is None:
            missing.append(text)
        else:
            result[text] = cached

    if missing:
        try:
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is required for Gemini embeddings. Install it and rerun, "
                "or pass --skip-embeddings for length/tool metrics only."
            ) from exc

        client = build_raw_google_genai_client(project=project, location=location)
        config = types.EmbedContentConfig(task_type=task_type)
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            try:
                response = client.models.embed_content(
                    model=model,
                    contents=batch,
                    config=config,
                )
            except Exception as exc:
                message = str(exc)
                if "PERMISSION_DENIED" in message or "aiplatform.endpoints.predict" in message:
                    raise RuntimeError(
                        "Gemini embedding request was denied by Vertex AI.\n"
                        f"Project: {project or '<env GOOGLE_CLOUD_PROJECT>'}\n"
                        f"Location: {location}\n"
                        f"Model: {model}\n"
                        "Required permission: aiplatform.endpoints.predict\n\n"
                        "Use a project where your ADC principal has Vertex AI prediction "
                        "permission, or ask an admin to grant a role such as "
                        "`roles/aiplatform.user` on that project. If you are using "
                        "end-user gcloud credentials, also set an ADC quota project, e.g.:\n"
                        f"  gcloud auth application-default set-quota-project "
                        f"{project or '<project>'}\n\n"
                        "For non-embedding CSV/plot metrics, rerun with --skip-embeddings."
                    ) from exc
                raise
            response_embeddings = getattr(response, "embeddings", None)
            if response_embeddings is None and isinstance(response, dict):
                response_embeddings = response.get("embeddings")
            if response_embeddings is None:
                response_embeddings = [getattr(response, "embedding", response)]
            if len(response_embeddings) != len(batch):
                raise RuntimeError(
                    "Gemini embedding response size mismatch: "
                    f"got {len(response_embeddings)} embeddings for {len(batch)} texts"
                )
            additions: dict[str, list[float]] = {}
            for text, response_embedding in zip(batch, response_embeddings):
                values = extract_embedding_values(response_embedding)
                result[text] = values
                additions[embedding_key(model=model, task_type=task_type, text=text)] = values
            cache.add_many(additions)
    return result


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(
            "Embedding dimension mismatch: "
            f"left has {len(left)} values, right has {len(right)} values"
        )

    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right):
        dot += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


def pairwise_cosine_summary(vectors: list[list[float]]) -> dict[str, Any]:
    if len(vectors) < 2:
        return {
            "num_texts": len(vectors),
            "num_pairs": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError(f"Embedding dimension mismatch in group: {sorted(dimensions)}")
    if np is not None:
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-12)
        values = []
        for index in range(matrix.shape[0] - 1):
            values.append(matrix[index + 1 :] @ matrix[index])
        similarities = np.concatenate(values)
        return {
            "num_texts": len(vectors),
            "num_pairs": int(similarities.size),
            "min": float(np.min(similarities)),
            "max": float(np.max(similarities)),
            "mean": float(np.mean(similarities)),
            "std": float(np.std(similarities)),
        }

    count = 0
    min_value: float | None = None
    max_value: float | None = None
    mean = 0.0
    sum_square_delta = 0.0
    for left_index, left in enumerate(vectors[:-1]):
        for right in vectors[left_index + 1 :]:
            value = cosine_similarity(left, right)
            count += 1
            min_value = value if min_value is None else min(min_value, value)
            max_value = value if max_value is None else max(max_value, value)
            delta = value - mean
            mean += delta / count
            sum_square_delta += delta * (value - mean)
    variance = sum_square_delta / count if count else 0.0
    return {
        "num_texts": int(len(vectors)),
        "num_pairs": count,
        "min": min_value,
        "max": max_value,
        "mean": mean,
        "std": math.sqrt(variance),
    }


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#111827}",
        ".title{font-size:20px;font-weight:700}",
        ".axis{font-size:12px;fill:#4b5563}",
        ".label{font-size:13px}",
        ".small{font-size:11px;fill:#6b7280}",
        ".grid{stroke:#e5e7eb;stroke-width:1}",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="24" y="32">{html.escape(title)}</text>',
    ]


def grouped_bar_svg(
    *,
    title: str,
    categories: list[str],
    series: list[tuple[str, list[float]]],
    y_label: str,
    width: int = 1100,
    height: int = 520,
) -> str:
    margin_left, margin_right, margin_top, margin_bottom = 80, 30, 70, 120
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_value = max([value for _, values in series for value in values] or [1.0])
    max_value = max(max_value, 1.0)
    lines = svg_header(width, height, title)
    lines.append(
        f'<text class="axis" x="20" y="{margin_top + plot_h / 2}" '
        f'transform="rotate(-90 20 {margin_top + plot_h / 2})">{html.escape(y_label)}</text>'
    )
    for tick in range(6):
        value = max_value * tick / 5
        y = margin_top + plot_h - (value / max_value) * plot_h
        lines.append(f'<line class="grid" x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}"/>')
        lines.append(f'<text class="small" x="{margin_left - 8}" y="{y + 4:.1f}" text-anchor="end">{value:.0f}</text>')
    group_w = plot_w / max(len(categories), 1)
    bar_w = min(44, group_w / (len(series) + 1))
    for cat_index, category in enumerate(categories):
        group_x = margin_left + cat_index * group_w
        for series_index, (name, values) in enumerate(series):
            value = values[cat_index]
            bar_h = (value / max_value) * plot_h
            x = group_x + (group_w - bar_w * len(series)) / 2 + series_index * bar_w
            y = margin_top + plot_h - bar_h
            color = PLOT_COLORS[series_index % len(PLOT_COLORS)]
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 2:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
        label_x = group_x + group_w / 2
        lines.append(
            f'<text class="axis" x="{label_x:.1f}" y="{height - 82}" '
            f'text-anchor="end" transform="rotate(-38 {label_x:.1f} {height - 82})">{html.escape(category)}</text>'
        )
    legend_x = margin_left
    for index, (name, _) in enumerate(series):
        x = legend_x + index * 190
        color = PLOT_COLORS[index % len(PLOT_COLORS)]
        lines.append(f'<rect x="{x}" y="48" width="12" height="12" fill="{color}"/>')
        lines.append(f'<text class="label" x="{x + 18}" y="59">{html.escape(name)}</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def horizontal_grouped_bar_svg(
    *,
    title: str,
    categories: list[str],
    series: list[tuple[str, list[float]]],
    x_label: str,
    width: int = 1200,
) -> str:
    row_h = 24
    margin_left, margin_right, margin_top, margin_bottom = 250, 30, 76, 54
    height = margin_top + margin_bottom + max(len(categories), 1) * row_h
    plot_w = width - margin_left - margin_right
    max_value = max([value for _, values in series for value in values] or [1.0])
    max_value = max(max_value, 1.0)
    lines = svg_header(width, height, title)
    for tick in range(6):
        value = max_value * tick / 5
        x = margin_left + (value / max_value) * plot_w
        lines.append(
            f'<line class="grid" x1="{x:.1f}" y1="{margin_top - 12}" '
            f'x2="{x:.1f}" y2="{height - margin_bottom}"/>'
        )
        lines.append(
            f'<text class="small" x="{x:.1f}" y="{height - 26}" '
            f'text-anchor="middle">{value:.0f}</text>'
        )
    lines.append(
        f'<text class="axis" x="{margin_left + plot_w / 2}" y="{height - 6}" '
        f'text-anchor="middle">{html.escape(x_label)}</text>'
    )
    bar_h = max(3, min(8, (row_h - 6) / max(len(series), 1)))
    for cat_index, category in enumerate(categories):
        row_y = margin_top + cat_index * row_h
        lines.append(
            f'<text class="small" x="{margin_left - 10}" y="{row_y + 12}" '
            f'text-anchor="end">{html.escape(category)}</text>'
        )
        for series_index, (_, values) in enumerate(series):
            value = values[cat_index]
            bar_w = (value / max_value) * plot_w
            y = row_y + 2 + series_index * bar_h
            color = PLOT_COLORS[series_index % len(PLOT_COLORS)]
            lines.append(
                f'<rect x="{margin_left}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{bar_h - 1:.1f}" fill="{color}"/>'
            )
    for index, (name, _) in enumerate(series):
        x = margin_left + index * 210
        color = PLOT_COLORS[index % len(PLOT_COLORS)]
        lines.append(f'<rect x="{x}" y="48" width="12" height="12" fill="{color}"/>')
        lines.append(f'<text class="label" x="{x + 18}" y="59">{html.escape(name)}</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def histogram_svg(
    *,
    title: str,
    grouped_values: dict[str, list[int]],
    x_label: str,
    y_label: str = "trajectory count",
    width: int = 1100,
    height: int = 560,
) -> str:
    all_values = [value for values in grouped_values.values() for value in values]
    if not all_values:
        return "\n".join(svg_header(width, height, title) + ["</svg>"]) + "\n"
    min_bin, max_bin = min(all_values), max(all_values)
    bins = list(range(min_bin, max_bin + 1))
    counters = {name: Counter(values) for name, values in grouped_values.items()}
    max_count = max([max(counter.values() or [0]) for counter in counters.values()] or [1])
    margin_left, margin_right, margin_top, margin_bottom = 80, 30, 70, 90
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    lines = svg_header(width, height, title)
    lines.append(
        f'<text class="axis" x="20" y="{margin_top + plot_h / 2}" '
        f'transform="rotate(-90 20 {margin_top + plot_h / 2})">{html.escape(y_label)}</text>'
    )
    bin_group_w = plot_w / max(len(bins), 1)
    methods = sorted(grouped_values)
    bar_w = max(2, min(18, bin_group_w / max(len(methods), 1) - 1))
    for tick in range(6):
        value = max_count * tick / 5
        y = margin_top + plot_h - (value / max_count) * plot_h
        lines.append(f'<line class="grid" x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}"/>')
        lines.append(f'<text class="small" x="{margin_left - 8}" y="{y + 4:.1f}" text-anchor="end">{value:.0f}</text>')
    for bin_index, bin_value in enumerate(bins):
        base_x = margin_left + bin_index * bin_group_w
        for method_index, method in enumerate(methods):
            count = counters[method].get(bin_value, 0)
            bar_h = (count / max_count) * plot_h if max_count else 0
            x = base_x + (bin_group_w - bar_w * len(methods)) / 2 + method_index * bar_w
            y = margin_top + plot_h - bar_h
            color = PLOT_COLORS[method_index % len(PLOT_COLORS)]
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
        if len(bins) <= 35 or bin_index % max(1, len(bins) // 20) == 0:
            x = base_x + bin_group_w / 2
            lines.append(f'<text class="small" x="{x:.1f}" y="{height - 58}" text-anchor="middle">{bin_value}</text>')
    lines.append(f'<text class="axis" x="{width / 2}" y="{height - 22}" text-anchor="middle">{html.escape(x_label)}</text>')
    for index, method in enumerate(methods):
        x = margin_left + index * 160
        color = PLOT_COLORS[index % len(PLOT_COLORS)]
        lines.append(f'<rect x="{x}" y="48" width="12" height="12" fill="{color}"/>')
        lines.append(f'<text class="label" x="{x + 18}" y="59">{html.escape(method)}</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def stacked_tool_svg(
    *,
    title: str,
    records: list[TrajectoryRecord],
    width: int = 1100,
    height: int = 420,
) -> str:
    method_counters: dict[str, Counter[str]] = defaultdict(Counter)
    total_counter: Counter[str] = Counter()
    for record in records:
        method_counters[record.method].update(record.tools)
        total_counter.update(record.tools)
    top_tools = [tool for tool, _ in total_counter.most_common(7)]
    tools = top_tools + ["other"]
    margin_left, margin_right, margin_top = 130, 40, 80
    row_h = 58
    bar_w = width - margin_left - margin_right
    lines = svg_header(width, height, title)
    for method_index, method in enumerate(sorted(method_counters)):
        counter = method_counters[method]
        total = sum(counter.values())
        y = margin_top + method_index * row_h
        lines.append(f'<text class="label" x="{margin_left - 12}" y="{y + 24}" text-anchor="end">{html.escape(method)}</text>')
        x = margin_left
        for tool_index, tool in enumerate(tools):
            count = (
                sum(count for name, count in counter.items() if name not in top_tools)
                if tool == "other"
                else counter.get(tool, 0)
            )
            segment_w = (count / total) * bar_w if total else 0
            color = PLOT_COLORS[tool_index % len(PLOT_COLORS)]
            lines.append(f'<rect x="{x:.1f}" y="{y}" width="{segment_w:.1f}" height="34" fill="{color}"/>')
            if segment_w > 42:
                lines.append(f'<text class="small" x="{x + segment_w / 2:.1f}" y="{y + 22}" text-anchor="middle" fill="#fff">{count / total:.0%}</text>')
            x += segment_w
    legend_y = height - 82
    legend_x = margin_left
    for index, tool in enumerate(tools):
        x = legend_x + (index % 4) * 220
        y = legend_y + (index // 4) * 24
        color = PLOT_COLORS[index % len(PLOT_COLORS)]
        lines.append(f'<rect x="{x}" y="{y}" width="12" height="12" fill="{color}"/>')
        lines.append(f'<text class="small" x="{x + 18}" y="{y + 11}">{html.escape(tool)}</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def write_plots(output_dir: Path, records: list[TrajectoryRecord]) -> None:
    plot_dir = output_dir / "plots"
    methods = sorted({record.method for record in records})
    tasks = sorted({record.task for record in records})
    by_method: dict[str, list[TrajectoryRecord]] = defaultdict(list)
    by_task: dict[str, list[TrajectoryRecord]] = defaultdict(list)
    for record in records:
        by_method[record.method].append(record)
        by_task[record.task].append(record)

    def mean_for(group: list[TrajectoryRecord], metric: str) -> float:
        if metric == "reasoning":
            values = [word_count(record.reasoning_text) for record in group]
        elif metric == "communication":
            values = [word_count(record.communication_text) for record in group]
        elif metric == "episode":
            values = [record.episode_length for record in group]
        else:
            values = [len(record.communication_texts) for record in group]
        return statistics.fmean(values) if values else 0.0

    write_text(
        plot_dir / "mean_text_lengths_by_method.svg",
        grouped_bar_svg(
            title="Mean Text Length by Sampling Method",
            categories=methods,
            series=[
                (
                    "reasoning words",
                    [mean_for(by_method[method], "reasoning") for method in methods],
                ),
                (
                    "communication words",
                    [mean_for(by_method[method], "communication") for method in methods],
                ),
            ],
            y_label="mean words per trajectory",
        ),
    )
    write_text(
        plot_dir / "episode_length_distribution.svg",
        histogram_svg(
            title="Episode Length Distribution",
            grouped_values={
                method: [record.episode_length for record in by_method[method]]
                for method in methods
            },
            x_label="steps per trajectory",
        ),
    )
    write_text(
        plot_dir / "communication_event_distribution.svg",
        histogram_svg(
            title="Communication Event Distribution",
            grouped_values={
                method: [len(record.communication_texts) for record in by_method[method]]
                for method in methods
            },
            x_label="communication events per trajectory",
        ),
    )
    write_text(
        plot_dir / "tool_distribution_by_method.svg",
        stacked_tool_svg(title="Tool Call Distribution by Sampling Method", records=records),
    )
    write_text(
        plot_dir / "mean_text_lengths_by_task.svg",
        horizontal_grouped_bar_svg(
            title="Mean Text Length by Task",
            categories=tasks,
            series=[
                (
                    "reasoning words",
                    [mean_for(by_task[task], "reasoning") for task in tasks],
                ),
                (
                    "communication words",
                    [mean_for(by_task[task], "communication") for task in tasks],
                ),
            ],
            x_label="mean words per trajectory",
        ),
    )
    write_text(
        plot_dir / "mean_episode_length_by_task.svg",
        horizontal_grouped_bar_svg(
            title="Mean Episode Length by Task",
            categories=tasks,
            series=[
                ("steps", [mean_for(by_task[task], "episode") for task in tasks]),
                (
                    "communication events",
                    [mean_for(by_task[task], "communication_events") for task in tasks],
                ),
            ],
            x_label="mean count per trajectory",
        ),
    )
    index = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<title>Text Diversity Report Plots</title>",
        "<style>body{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#111827}"
        "img{display:block;max-width:100%;margin:20px 0 36px;border:1px solid #e5e7eb}"
        "h1{font-size:24px}h2{font-size:18px;margin-top:28px}</style>",
        "</head><body>",
        "<h1>Text Diversity Report Plots</h1>",
    ]
    for filename, title in [
        ("mean_text_lengths_by_method.svg", "Mean Text Length by Sampling Method"),
        ("episode_length_distribution.svg", "Episode Length Distribution"),
        ("communication_event_distribution.svg", "Communication Event Distribution"),
        ("tool_distribution_by_method.svg", "Tool Call Distribution by Sampling Method"),
        ("mean_text_lengths_by_task.svg", "Mean Text Length by Task"),
        ("mean_episode_length_by_task.svg", "Mean Episode Length by Task"),
    ]:
        index.extend(
            [
                f"<h2>{html.escape(title)}</h2>",
                f'<img src="{html.escape(filename)}" alt="{html.escape(title)}">',
            ]
        )
    index.append("</body></html>\n")
    write_text(plot_dir / "index.html", "\n".join(index))


def build_similarity_rows(
    records: list[TrajectoryRecord],
    embeddings: dict[str, list[float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    method_task_groups: dict[tuple[str, str], list[TrajectoryRecord]] = defaultdict(list)
    method_groups: dict[str, list[TrajectoryRecord]] = defaultdict(list)
    task_groups: dict[str, list[TrajectoryRecord]] = defaultdict(list)
    for record in records:
        method_task_groups[(record.method, record.task)].append(record)
        method_groups[record.method].append(record)
        task_groups[record.task].append(record)

    def vectors_for(group: list[TrajectoryRecord], text_kind: str) -> list[list[float]]:
        texts = [
            record.reasoning_text if text_kind == "reasoning" else record.communication_text
            for record in group
        ]
        return [embeddings[text] for text in texts if text.strip() and text in embeddings]

    def append(scope: str, method: str, task: str, group: list[TrajectoryRecord]) -> None:
        for text_kind in ("reasoning", "communication"):
            rows.append(
                {
                    "scope": scope,
                    "method": method,
                    "task": task,
                    "text_kind": text_kind,
                    **pairwise_cosine_summary(vectors_for(group, text_kind)),
                }
            )

    for key, group in sorted(method_task_groups.items()):
        append("method_task", key[0], key[1], group)
    for method, group in sorted(method_groups.items()):
        append("method", method, "__all_tasks__", group)
    for task, group in sorted(task_groups.items()):
        append("task", "__all_methods__", task, group)
    append("dataset", "__all_methods__", "__all_tasks__", records)
    return rows


def write_summary_json(
    path: Path,
    *,
    records: list[TrajectoryRecord],
    sweep_root: Path,
    output_dir: Path,
    embedding_model: str,
    embedding_task_type: str,
    skipped_embeddings: bool,
) -> None:
    method_counts = Counter(record.method for record in records)
    task_counts = Counter(record.task for record in records)
    summary = {
        "sweep_root": str(sweep_root),
        "output_dir": str(output_dir),
        "num_trajectories": len(records),
        "num_methods": len(method_counts),
        "num_tasks": len(task_counts),
        "trajectories_by_method": dict(sorted(method_counts.items())),
        "trajectories_by_task": dict(sorted(task_counts.items())),
        "embedding_model": None if skipped_embeddings else embedding_model,
        "embedding_task_type": None if skipped_embeddings else embedding_task_type,
        "skipped_embeddings": skipped_embeddings,
    }
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.dotenv_path is not None:
        load_dotenv_file(args.dotenv_path)
    else:
        load_dotenv_file()
    if args.quota_project:
        os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"] = args.quota_project

    sweep_root = resolve_sweep_root(args.sweep_dir)
    output_dir = args.output_dir or (sweep_root / "text_diversity_report")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = iter_records(sweep_root)
    if not records:
        raise RuntimeError(f"No trajectory records found under {sweep_root}")

    tool_rows = build_tool_distribution_rows(records)
    per_traj_length_rows, aggregate_length_rows = build_length_rows(records)

    write_csv(
        output_dir / "tool_call_distribution.csv",
        tool_rows,
        [
            "scope",
            "method",
            "task",
            "tool",
            "count",
            "percentage",
            "total_tool_calls",
            "tool_entropy",
        ],
    )
    write_csv(
        output_dir / "trajectory_length_metrics.csv",
        per_traj_length_rows,
        [
            "method",
            "task",
            "trajectory_id",
            "episode_length_steps",
            "tool_calls",
            "communication_events",
            "reasoning_word_count",
            "reasoning_char_count",
            "communication_word_count",
            "communication_char_count",
            "path",
        ],
    )
    write_csv(
        output_dir / "length_metric_summary.csv",
        aggregate_length_rows,
        ["scope", "method", "task", "metric", "count", "min", "max", "mean", "median", "std"],
    )
    if not args.skip_plots:
        write_plots(output_dir, records)

    if not args.skip_embeddings:
        cache_path = args.embedding_cache or (output_dir / "gemini_embedding_cache.jsonl")
        cache = EmbeddingCache(cache_path)
        texts: list[str] = []
        for record in records:
            texts.extend([record.reasoning_text, record.communication_text])
        embeddings = gemini_embed_texts(
            texts,
            model=args.embedding_model,
            task_type=args.embedding_task_type,
            batch_size=args.embedding_batch_size,
            project=args.project,
            location=args.location,
            cache=cache,
        )
        similarity_rows = build_similarity_rows(records, embeddings)
        write_csv(
            output_dir / "embedding_similarity_metrics.csv",
            similarity_rows,
            [
                "scope",
                "method",
                "task",
                "text_kind",
                "num_texts",
                "num_pairs",
                "min",
                "max",
                "mean",
                "std",
            ],
        )

    write_summary_json(
        output_dir / "diversity_metrics_summary.json",
        records=records,
        sweep_root=sweep_root,
        output_dir=output_dir,
        embedding_model=args.embedding_model,
        embedding_task_type=args.embedding_task_type,
        skipped_embeddings=args.skip_embeddings,
    )
    print(f"Wrote diversity metrics for {len(records)} trajectories to {output_dir}")


if __name__ == "__main__":
    main()
