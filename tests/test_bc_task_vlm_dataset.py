from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

from training.bc_task_vlm.dataset import (
    build_example_cache_fingerprint,
    build_example_cache_path,
    build_centralized_examples,
    build_decentralized_examples,
    load_examples_from_cache,
    save_examples_to_cache,
)


DATASET_ROOT = (
    Path(__file__).resolve().parents[1]
    / "data_generation/task_level/data/image/20260404T191734Z"
)
SOURCE_TRAJECTORY_DIR = DATASET_ROOT / "hot_dog_setup" / "traj_000028"


class DecentralizedExampleTests(unittest.TestCase):
    def _build_single_trajectory_dataset_root(self) -> Path:
        temp_root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        trajectory_dir = temp_root / "hot_dog_setup" / "traj_000028"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        for file_name in ("original_trajectory.json", "plan.json", "metadata.json"):
            shutil.copy2(SOURCE_TRAJECTORY_DIR / file_name, trajectory_dir / file_name)
        return temp_root

    def test_decentralized_examples_preserve_supervised_action_count(self):
        dataset_root = self._build_single_trajectory_dataset_root()
        centralized_examples = build_centralized_examples(
            dataset_root=dataset_root,
            task_names=["hot_dog_setup"],
        )
        decentralized_examples = build_decentralized_examples(
            dataset_root=dataset_root,
            task_names=["hot_dog_setup"],
        )

        self.assertTrue(centralized_examples)
        self.assertTrue(decentralized_examples)
        self.assertEqual(
            len(centralized_examples),
            sum(example.num_target_steps for example in decentralized_examples),
        )

    def test_joint_demo_splits_into_one_training_conversation_per_agent(self):
        dataset_root = self._build_single_trajectory_dataset_root()
        decentralized_examples = build_decentralized_examples(
            dataset_root=dataset_root,
            task_names=["hot_dog_setup"],
        )
        trajectory_examples = [
            example
            for example in decentralized_examples
            if example.trajectory_id == "traj_000028"
        ]

        self.assertEqual(
            {example.agent_id for example in trajectory_examples},
            {"agent_0", "agent_1"},
        )

        for example in trajectory_examples:
            roles = [message["role"] for message in example.messages]
            self.assertEqual(roles[0], "system")
            self.assertEqual(
                roles[1:],
                ["user", "assistant"] * example.num_target_steps,
            )

            num_image_placeholders = sum(
                1
                for message in example.messages
                for item in message.get("content", [])
                if isinstance(item, dict) and item.get("type") == "image"
            )
            self.assertEqual(num_image_placeholders, len(example.image_paths))

    def test_decentralized_examples_round_trip_through_cache(self):
        dataset_root = self._build_single_trajectory_dataset_root()
        cache_dir = dataset_root / ".cache"
        decentralized_examples = build_decentralized_examples(
            dataset_root=dataset_root,
            task_names=["hot_dog_setup"],
        )
        fingerprint = build_example_cache_fingerprint(
            dataset_root=dataset_root,
            task_name="hot_dog_setup",
            granularity="decentralized",
        )
        cache_path = build_example_cache_path(
            cache_dir=cache_dir,
            dataset_root=dataset_root,
            task_name="hot_dog_setup",
            granularity="decentralized",
        )

        save_examples_to_cache(
            cache_path=cache_path,
            fingerprint=fingerprint,
            examples=decentralized_examples,
        )
        cached_examples = load_examples_from_cache(
            cache_path=cache_path,
            expected_fingerprint=fingerprint,
            granularity="decentralized",
        )

        self.assertEqual(cached_examples, decentralized_examples)

    def test_cache_invalidates_when_trajectory_inputs_change(self):
        dataset_root = self._build_single_trajectory_dataset_root()
        cache_dir = dataset_root / ".cache"
        decentralized_examples = build_decentralized_examples(
            dataset_root=dataset_root,
            task_names=["hot_dog_setup"],
        )
        fingerprint = build_example_cache_fingerprint(
            dataset_root=dataset_root,
            task_name="hot_dog_setup",
            granularity="decentralized",
        )
        cache_path = build_example_cache_path(
            cache_dir=cache_dir,
            dataset_root=dataset_root,
            task_name="hot_dog_setup",
            granularity="decentralized",
        )
        save_examples_to_cache(
            cache_path=cache_path,
            fingerprint=fingerprint,
            examples=decentralized_examples,
        )

        metadata_path = dataset_root / "hot_dog_setup" / "traj_000028" / "metadata.json"
        metadata_path.write_text(
            metadata_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        updated_fingerprint = build_example_cache_fingerprint(
            dataset_root=dataset_root,
            task_name="hot_dog_setup",
            granularity="decentralized",
        )

        cached_examples = load_examples_from_cache(
            cache_path=cache_path,
            expected_fingerprint=updated_fingerprint,
            granularity="decentralized",
        )

        self.assertIsNone(cached_examples)


if __name__ == "__main__":
    unittest.main()
