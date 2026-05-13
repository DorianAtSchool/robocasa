from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from training.bc_task_vlm import calculate_average_tokens_for_tasks as avg_tokens


class _FakeExample:
    def __init__(self, feature: dict):
        self._feature = feature

    def to_feature_dict(self) -> dict:
        return dict(self._feature)


class _FakeProcessor:
    image_token = "<image>"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        del messages, tokenize, add_generation_prompt
        return "prefix <image> middle <image> suffix"

    def tokenizer(
        self, text: str, *, return_attention_mask: bool
    ) -> dict[str, list[int]]:
        del return_attention_mask
        token_count = 3 + text.count(self.image_token)
        return {"attention_mask": [1] * token_count}


class AverageTaskTokenTests(unittest.TestCase):
    def test_count_feature_tokens_tracks_image_tokens_and_images(self):
        with mock.patch.object(
            avg_tokens,
            "_image_token_counts",
            return_value=[4, 6],
        ):
            counts = avg_tokens._count_feature_tokens(
                _FakeProcessor(),
                {
                    "messages": [],
                    "image_paths": ["image_a.png", "image_b.png"],
                },
                image_resolution=1,
                image_size_cache={},
            )

        self.assertEqual(counts.total_tokens, 13)
        self.assertEqual(counts.image_tokens, 10)
        self.assertEqual(counts.images, 2)

    def test_summarize_task_tokens_averages_image_tokens_and_images(self):
        examples = [
            _FakeExample(
                {
                    "trajectory_id": "traj_000000",
                    "messages": [],
                    "image_paths": ["a.png", "b.png"],
                }
            ),
            _FakeExample(
                {
                    "trajectory_id": "traj_000000",
                    "messages": [],
                    "image_paths": ["c.png"],
                }
            ),
            _FakeExample(
                {
                    "trajectory_id": "traj_000001",
                    "messages": [],
                    "image_paths": ["d.png", "e.png", "f.png"],
                }
            ),
        ]

        def fake_image_token_counts(
            processor,
            *,
            image_sources,
            image_resolution,
            image_size_cache,
        ):
            del processor, image_resolution, image_size_cache
            return [10] * len(image_sources)

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir)
            task_root = dataset_root / "task"
            (task_root / "traj_000000").mkdir(parents=True)
            (task_root / "traj_000001").mkdir()

            with mock.patch.object(
                avg_tokens, "_build_examples", return_value=examples
            ):
                with mock.patch.object(
                    avg_tokens,
                    "_image_token_counts",
                    side_effect=fake_image_token_counts,
                ):
                    summary = avg_tokens.summarize_task_tokens(
                        processor=_FakeProcessor(),
                        dataset_root=dataset_root,
                        task_name="task",
                        trajectories_per_task=2,
                        granularity="decentralized",
                        image_resolution=1,
                        image_size_cache={},
                    )

        self.assertEqual(summary.sampled_trajectories, 2)
        self.assertEqual(summary.examples, 3)
        self.assertEqual(summary.mean_image_tokens_per_trajectory, 30.0)
        self.assertEqual(summary.mean_images_per_trajectory, 3.0)
        self.assertEqual(summary.mean_tokens_per_trajectory, 34.5)

    def test_print_table_includes_image_average_columns(self):
        output = StringIO()
        summary = avg_tokens.TaskTokenSummary(
            task_name="task",
            sampled_trajectories=2,
            examples=3,
            mean_tokens_per_trajectory=34.5,
            mean_image_tokens_per_trajectory=30.0,
            mean_images_per_trajectory=3.0,
            stddev_tokens_per_trajectory=1.5,
            min_tokens_per_trajectory=33,
            max_tokens_per_trajectory=36,
        )

        with redirect_stdout(output):
            avg_tokens._print_table(
                model_name_or_path="model",
                dataset_root=Path("/dataset"),
                trajectories_per_task=2,
                granularity="decentralized",
                summaries=[summary],
            )

        rendered = output.getvalue()
        self.assertIn("avg_image_tokens_per_trajectory", rendered)
        self.assertIn("avg_images_per_trajectory", rendered)
        self.assertIn("30.00", rendered)
        self.assertIn("3.00", rendered)


if __name__ == "__main__":
    unittest.main()
