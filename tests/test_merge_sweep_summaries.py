"""Tests for merging per-shard sweep summaries."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
MERGE_SCRIPT_PATH = REPO_ROOT / "scripts" / "merge_sweep_summaries.py"

_MERGE_SPEC = importlib.util.spec_from_file_location(
    "merge_sweep_summaries_script",
    MERGE_SCRIPT_PATH,
)
assert _MERGE_SPEC is not None
assert _MERGE_SPEC.loader is not None
merge_sweep_summaries_script = importlib.util.module_from_spec(_MERGE_SPEC)
_MERGE_SPEC.loader.exec_module(merge_sweep_summaries_script)


class MergeSweepSummariesTests(unittest.TestCase):
    """Validate multi-shard summary merge behavior."""

    def test_merge_sweep_summaries_combines_and_sorts_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            shard0 = temp_path / "shard0.json"
            shard1 = temp_path / "shard1.json"

            base_summary = {
                "input_dir": "/tmp/input",
                "layouts": [11],
                "styles": [34],
                "seeds": [42],
                "scene_combos": 1,
                "gl_backend": "egl",
                "gpu_ids": [0, 1, 2, 3],
                "max_tasks_per_child": None,
                "procs_per_gpu": [12, 12, 12, 12],
                "render_width": 512,
                "render_height": 512,
                "num_shards": 2,
            }
            shard0.write_text(
                json.dumps(
                    {
                        **base_summary,
                        "shard_index": 0,
                        "trajectories": 1,
                        "total": 1,
                        "succeeded": 1,
                        "failed": 0,
                        "results": [
                            {
                                "status": "ok",
                                "task_dir": "task_b",
                                "traj_idx": 1,
                                "layout": 11,
                                "style": 34,
                                "seed": 42,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            shard1.write_text(
                json.dumps(
                    {
                        **base_summary,
                        "shard_index": 1,
                        "trajectories": 1,
                        "total": 1,
                        "succeeded": 1,
                        "failed": 0,
                        "results": [
                            {
                                "status": "ok",
                                "task_dir": "task_a",
                                "traj_idx": 0,
                                "layout": 11,
                                "style": 34,
                                "seed": 42,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            merged = merge_sweep_summaries_script.merge_sweep_summaries(
                [shard0, shard1]
            )

        self.assertEqual(merged["trajectories"], 2)
        self.assertEqual(merged["total"], 2)
        self.assertEqual(
            [(result["task_dir"], result["traj_idx"]) for result in merged["results"]],
            [("task_a", 0), ("task_b", 1)],
        )
        self.assertIsNone(merged["shard_index"])


if __name__ == "__main__":
    unittest.main()
