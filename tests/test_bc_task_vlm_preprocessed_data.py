from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from training.bc_task_vlm import merge_pretokenized_shards, preprocessed_data
from training.bc_task_vlm.dataset import (
    LazyVisionSFTCollator,
    build_centralized_examples,
    build_decentralized_examples,
    build_split_manifest,
    serialize_pretokenized_tensors,
)
from training.bc_task_vlm.preprocessed_data import (
    PreprocessedFeatureDataset,
    build_or_load_cached_pretokenization_metadata,
    build_preprocessed_dataset_dict,
    build_pretokenization_metadata,
    existing_artifact_image_relpaths_for_examples,
    load_pretokenized_shard,
    load_preprocessed_artifact_from_disk,
    pretokenization_compatibility_reason,
    resolve_hf_dataset_reference,
    save_pretokenized_shard,
    save_preprocessed_artifact,
    stage_artifact_images_for_examples,
)


HAS_DATASETS = importlib.util.find_spec("datasets") is not None
HAS_PILLOW = importlib.util.find_spec("PIL") is not None

DATASET_ROOT = (
    Path(__file__).resolve().parents[1]
    / "data_generation/task_level/data/image/20260404T191734Z"
)
SOURCE_TRAJECTORY_DIR = DATASET_ROOT / "hot_dog_setup" / "traj_000028"


@unittest.skipUnless(
    HAS_DATASETS and HAS_PILLOW,
    "datasets and Pillow are required for preprocessed artifact tests",
)
class PreprocessedArtifactTests(unittest.TestCase):
    def _build_single_trajectory_dataset_root(self) -> Path:
        temp_root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        trajectory_dir = temp_root / "hot_dog_setup" / "traj_000028"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        for file_name in ("original_trajectory.json", "plan.json", "metadata.json"):
            shutil.copy2(SOURCE_TRAJECTORY_DIR / file_name, trajectory_dir / file_name)
        return temp_root

    def test_local_artifact_round_trip_preserves_features(self):
        dataset_root = self._build_single_trajectory_dataset_root()
        train_examples = build_decentralized_examples(
            dataset_root=dataset_root,
            task_names=["hot_dog_setup"],
        )
        validation_examples = build_centralized_examples(
            dataset_root=dataset_root,
            task_names=["hot_dog_setup"],
        )[:2]
        split_manifest = build_split_manifest(
            dataset_root=dataset_root,
            train_examples=train_examples,
            val_examples=validation_examples,
        )
        output_dir = (
            Path(
                self.enterContext(tempfile.TemporaryDirectory(dir=DATASET_ROOT.parent))
            )
            / "artifact"
        )
        progress_messages: list[str] = []
        dataset_dict = build_preprocessed_dataset_dict(
            train_examples=train_examples,
            validation_examples=validation_examples,
            output_dir=output_dir,
            progress_callback=progress_messages.append,
        )

        preprocess_config = {
            "command": "python -m training.bc_task_vlm.preprocess ...",
            "dataset_root": str(dataset_root),
            "train_example_format": "single_turn",
            "preprocessed_format_version": 4,
        }
        save_preprocessed_artifact(
            output_dir=output_dir,
            dataset_dict=dataset_dict,
            split_manifest=split_manifest,
            preprocess_config=preprocess_config,
            progress_callback=progress_messages.append,
        )

        artifact = load_preprocessed_artifact_from_disk(output_dir)
        train_dataset = PreprocessedFeatureDataset(
            artifact.train_split,
            artifact_root=artifact.artifact_root,
        )
        validation_dataset = PreprocessedFeatureDataset(
            artifact.validation_split,
            artifact_root=artifact.artifact_root,
        )

        self.assertEqual(len(train_dataset), len(train_examples))
        self.assertEqual(len(validation_dataset), len(validation_examples))
        self.assertEqual(artifact.split_manifest, split_manifest)
        self.assertEqual(artifact.preprocess_config, preprocess_config)

        original_feature = train_examples[0].to_feature_dict()
        loaded_feature = train_dataset[0]
        loaded_feature_without_images = {
            key: value for key, value in loaded_feature.items() if key != "image_paths"
        }
        self.assertEqual(
            loaded_feature_without_images,
            {
                key: value
                for key, value in original_feature.items()
                if key != "image_paths"
            },
        )
        self.assertEqual(
            len(loaded_feature["image_paths"]), len(original_feature["image_paths"])
        )
        self.assertTrue(loaded_feature["image_paths"])
        self.assertTrue(
            all(
                Path(image_path).is_file()
                for image_path in loaded_feature["image_paths"]
            )
        )
        collated_images = LazyVisionSFTCollator._load_images(
            loaded_feature["image_paths"]
        )
        self.assertTrue(all(image.mode == "RGB" for image in collated_images))
        source_image_path = Path(original_feature["image_paths"][0])
        linked_image_path = Path(loaded_feature["image_paths"][0])
        self.assertGreaterEqual(os.stat(source_image_path).st_nlink, 2)
        self.assertEqual(
            os.stat(source_image_path).st_ino, os.stat(linked_image_path).st_ino
        )

        stored_image_paths = sorted((output_dir / "images").iterdir())
        unique_source_image_paths = {
            image_path
            for example in train_examples + validation_examples
            for image_path in example.image_paths
        }
        self.assertEqual(len(stored_image_paths), len(unique_source_image_paths))
        self.assertTrue(
            any(
                "Staging" in message and "artifact images" in message
                for message in progress_messages
            )
        )
        self.assertTrue(
            any("Serializing train rows" in message for message in progress_messages)
        )
        self.assertTrue(
            any("Saving dataset split data" in message for message in progress_messages)
        )

    def test_artifact_image_resize_writes_resized_copies_without_touching_source(self):
        from PIL import Image

        dataset_root = self._build_single_trajectory_dataset_root()
        train_examples = build_decentralized_examples(
            dataset_root=dataset_root,
            task_names=["hot_dog_setup"],
        )[:1]
        validation_examples: list = []
        output_dir = (
            Path(
                self.enterContext(tempfile.TemporaryDirectory(dir=DATASET_ROOT.parent))
            )
            / "artifact-resized"
        )
        source_image_path = Path(train_examples[0].image_paths[0])
        with Image.open(source_image_path) as image:
            original_source_size = image.size

        dataset_dict = build_preprocessed_dataset_dict(
            train_examples=train_examples,
            validation_examples=validation_examples,
            output_dir=output_dir,
            artifact_image_size=(16, 16),
        )

        self.assertEqual(len(dataset_dict["train"]), len(train_examples))
        artifact_image_path = next((output_dir / "images").iterdir())
        with Image.open(artifact_image_path) as image:
            self.assertEqual(image.size, (16, 16))
        with Image.open(source_image_path) as image:
            self.assertEqual(image.size, original_source_size)
        self.assertNotEqual(
            os.stat(source_image_path).st_ino,
            os.stat(artifact_image_path).st_ino,
        )

    def test_stage_artifact_images_can_resize_before_pretokenization(self):
        from PIL import Image

        dataset_root = self._build_single_trajectory_dataset_root()
        train_examples = build_decentralized_examples(
            dataset_root=dataset_root,
            task_names=["hot_dog_setup"],
        )[:1]
        output_dir = (
            Path(
                self.enterContext(tempfile.TemporaryDirectory(dir=DATASET_ROOT.parent))
            )
            / "artifact-pretokenize-resized"
        )

        image_relpaths_by_source = stage_artifact_images_for_examples(
            examples=train_examples,
            output_dir=output_dir,
            artifact_image_size=(16, 16),
        )

        for image_path in train_examples[0].image_paths:
            artifact_image_path = output_dir / image_relpaths_by_source[image_path]
            with Image.open(artifact_image_path) as image:
                self.assertEqual(image.size, (16, 16))

    def test_existing_artifact_images_can_be_reused_for_resume(self):
        dataset_root = self._build_single_trajectory_dataset_root()
        train_examples = build_decentralized_examples(
            dataset_root=dataset_root,
            task_names=["hot_dog_setup"],
        )[:1]
        output_dir = (
            Path(
                self.enterContext(tempfile.TemporaryDirectory(dir=DATASET_ROOT.parent))
            )
            / "artifact-resume"
        )

        staged_relpaths = stage_artifact_images_for_examples(
            examples=train_examples,
            output_dir=output_dir,
            artifact_image_size=(16, 16),
        )
        resumed_relpaths = existing_artifact_image_relpaths_for_examples(
            examples=train_examples,
            output_dir=output_dir,
        )

        self.assertEqual(resumed_relpaths, staged_relpaths)

    def test_existing_artifact_images_resume_rejects_missing_files(self):
        dataset_root = self._build_single_trajectory_dataset_root()
        train_examples = build_decentralized_examples(
            dataset_root=dataset_root,
            task_names=["hot_dog_setup"],
        )[:1]
        output_dir = (
            Path(
                self.enterContext(tempfile.TemporaryDirectory(dir=DATASET_ROOT.parent))
            )
            / "artifact-resume-missing"
        )

        staged_relpaths = stage_artifact_images_for_examples(
            examples=train_examples,
            output_dir=output_dir,
            artifact_image_size=(16, 16),
        )
        first_relpath = next(iter(staged_relpaths.values()))
        (output_dir / first_relpath).unlink()

        with self.assertRaises(FileNotFoundError):
            existing_artifact_image_relpaths_for_examples(
                examples=train_examples,
                output_dir=output_dir,
            )

    def test_existing_artifact_images_can_be_reused_without_validation(self):
        dataset_root = self._build_single_trajectory_dataset_root()
        train_examples = build_decentralized_examples(
            dataset_root=dataset_root,
            task_names=["hot_dog_setup"],
        )[:1]
        output_dir = (
            Path(
                self.enterContext(tempfile.TemporaryDirectory(dir=DATASET_ROOT.parent))
            )
            / "artifact-resume-unchecked"
        )

        unchecked_relpaths = existing_artifact_image_relpaths_for_examples(
            examples=train_examples,
            output_dir=output_dir,
            validate=False,
        )

        self.assertTrue(unchecked_relpaths)
        first_image_path = train_examples[0].image_paths[0]
        self.assertIn(first_image_path, unchecked_relpaths)

    def test_pretokenized_shard_round_trip_preserves_blobs_and_manifest(self):
        shard_dir = (
            Path(
                self.enterContext(tempfile.TemporaryDirectory(dir=DATASET_ROOT.parent))
            )
            / "pretokenized_shards"
        )

        manifest = save_pretokenized_shard(
            shard_dir=shard_dir,
            shard_index=3,
            num_shards=8,
            train_blobs_by_sample_id={"train-a": b"train-blob"},
            validation_blobs_by_sample_id={"validation-a": b"validation-blob"},
            pretokenization={"processor_family": "qwen", "max_length": 4096},
        )
        shard = load_pretokenized_shard(
            shard_dir=shard_dir,
            shard_index=3,
        )

        self.assertEqual(shard.shard_index, 3)
        self.assertEqual(shard.num_shards, 8)
        self.assertEqual(shard.pretokenization, manifest["pretokenization"])
        self.assertEqual(
            shard.train_blobs_by_sample_id,
            {"train-a": b"train-blob"},
        )
        self.assertEqual(
            shard.validation_blobs_by_sample_id,
            {"validation-a": b"validation-blob"},
        )

    def test_incremental_pretokenized_shard_writer_preserves_chunks(self):
        shard_dir = (
            Path(
                self.enterContext(tempfile.TemporaryDirectory(dir=DATASET_ROOT.parent))
            )
            / "pretokenized_shards"
        )
        progress_messages: list[str] = []

        with preprocessed_data.PretokenizedShardWriter(
            shard_dir=shard_dir,
            shard_index=4,
            num_shards=8,
            pretokenization={"processor_family": "qwen", "max_length": 4096},
            progress_callback=progress_messages.append,
        ) as writer:
            writer.write_split_chunk(
                split_name="train",
                blobs_by_sample_id={"train-a": b"a", "train-b": b"b"},
            )
            writer.write_split_chunk(
                split_name="train",
                blobs_by_sample_id={"train-c": b"c"},
            )
            writer.write_split_chunk(
                split_name="validation",
                blobs_by_sample_id={"validation-a": b"validation-blob"},
            )

        shard = load_pretokenized_shard(
            shard_dir=shard_dir,
            shard_index=4,
        )

        self.assertEqual(
            shard.train_blobs_by_sample_id,
            {"train-a": b"a", "train-b": b"b", "train-c": b"c"},
        )
        self.assertEqual(
            shard.validation_blobs_by_sample_id,
            {"validation-a": b"validation-blob"},
        )
        self.assertEqual(
            shard.manifest["splits"]["train"]["sample_ids"],
            ["train-a", "train-b", "train-c"],
        )
        self.assertTrue(
            any(
                "Wrote 2 train pretokenized samples" in message
                for message in progress_messages
            )
        )

    def test_incremental_pretokenized_shard_writer_removes_stale_temp_files(self):
        shard_dir = (
            Path(
                self.enterContext(tempfile.TemporaryDirectory(dir=DATASET_ROOT.parent))
            )
            / "pretokenized_shards"
        )
        shard_dir.mkdir(parents=True)
        stale_temp_path = shard_dir / ".train-000005.arrow.tmp-12345"
        stale_temp_path.write_bytes(b"stale partial shard")

        with preprocessed_data.PretokenizedShardWriter(
            shard_dir=shard_dir,
            shard_index=5,
            num_shards=8,
            pretokenization={"processor_family": "qwen"},
        ) as writer:
            writer.write_split_chunk(
                split_name="train",
                blobs_by_sample_id={"train-a": b"a"},
            )

        self.assertFalse(stale_temp_path.exists())
        shard = load_pretokenized_shard(
            shard_dir=shard_dir,
            shard_index=5,
        )
        self.assertEqual(shard.train_blobs_by_sample_id, {"train-a": b"a"})

    def test_streamed_merge_output_loads_from_disk(self):
        temp_root = Path(
            self.enterContext(tempfile.TemporaryDirectory(dir=DATASET_ROOT.parent))
        )
        output_dir = temp_root / "artifact"
        shard_dir = output_dir / "pretokenized_shards"
        dataset_root = self._build_single_trajectory_dataset_root()
        train_examples = build_decentralized_examples(
            dataset_root=dataset_root,
            task_names=["hot_dog_setup"],
        )[:2]
        validation_examples = build_centralized_examples(
            dataset_root=dataset_root,
            task_names=["hot_dog_setup"],
        )[:1]
        all_examples = train_examples + validation_examples
        image_relpaths_by_source = preprocessed_data.stage_artifact_images_for_examples(
            examples=all_examples,
            output_dir=output_dir,
            artifact_image_size=(16, 16),
        )

        with preprocessed_data.PretokenizedShardWriter(
            shard_dir=shard_dir,
            shard_index=0,
            num_shards=1,
            pretokenization={"processor_family": "qwen"},
        ) as shard_writer:
            shard_writer.write_split_chunk(
                split_name="train",
                blobs_by_sample_id={
                    example.sample_id: f"train-{index}".encode("utf-8")
                    for index, example in enumerate(train_examples)
                },
            )
            shard_writer.write_split_chunk(
                split_name="validation",
                blobs_by_sample_id={validation_examples[0].sample_id: b"validation-0"},
            )

        with preprocessed_data.StreamedPreprocessedArtifactWriter(
            output_dir=output_dir,
        ) as artifact_writer:
            merge_pretokenized_shards._stream_split_from_shards(
                split_name="train",
                examples_by_sample_id={
                    example.sample_id: example for example in train_examples
                },
                expected_sample_ids={example.sample_id for example in train_examples},
                image_relpaths_by_source=image_relpaths_by_source,
                shard_dir=shard_dir,
                num_shards=1,
                artifact_writer=artifact_writer,
                flush_interval=1,
            )
            merge_pretokenized_shards._stream_split_from_shards(
                split_name="validation",
                examples_by_sample_id={
                    example.sample_id: example for example in validation_examples
                },
                expected_sample_ids={
                    example.sample_id for example in validation_examples
                },
                image_relpaths_by_source=image_relpaths_by_source,
                shard_dir=shard_dir,
                num_shards=1,
                artifact_writer=artifact_writer,
                flush_interval=1,
            )
        split_manifest = build_split_manifest(
            dataset_root=dataset_root,
            train_examples=train_examples,
            val_examples=validation_examples,
        )
        preprocessed_data.write_preprocessed_artifact_sidecars(
            output_dir=output_dir,
            split_manifest=split_manifest,
            preprocess_config={"pretokenized": True},
        )

        loaded = preprocessed_data.load_preprocessed_artifact_from_disk(output_dir)

        self.assertEqual(len(loaded.train_split), 2)
        self.assertEqual(len(loaded.validation_split), 1)
        self.assertEqual(
            loaded.train_split[0][preprocessed_data._PRETOKENIZED_BLOB_COLUMN],
            b"train-0",
        )

    def test_local_artifact_round_trip_preserves_pretokenized_tensors(self):
        dataset_root = self._build_single_trajectory_dataset_root()
        train_examples = build_decentralized_examples(
            dataset_root=dataset_root,
            task_names=["hot_dog_setup"],
        )[:1]
        validation_examples = build_centralized_examples(
            dataset_root=dataset_root,
            task_names=["hot_dog_setup"],
        )[:1]
        split_manifest = build_split_manifest(
            dataset_root=dataset_root,
            train_examples=train_examples,
            val_examples=validation_examples,
        )
        output_dir = (
            Path(
                self.enterContext(tempfile.TemporaryDirectory(dir=DATASET_ROOT.parent))
            )
            / "artifact-pretokenized"
        )
        train_blob = serialize_pretokenized_tensors(
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "labels": [-100, 2, 3],
            }
        )
        validation_blob = serialize_pretokenized_tensors(
            {
                "input_ids": [4, 5],
                "attention_mask": [1, 1],
                "labels": [-100, 5],
            }
        )

        dataset_dict = build_preprocessed_dataset_dict(
            train_examples=train_examples,
            validation_examples=validation_examples,
            output_dir=output_dir,
            train_pretokenized_blobs_by_sample_id={
                train_examples[0].sample_id: train_blob
            },
            validation_pretokenized_blobs_by_sample_id={
                validation_examples[0].sample_id: validation_blob
            },
        )
        save_preprocessed_artifact(
            output_dir=output_dir,
            dataset_dict=dataset_dict,
            split_manifest=split_manifest,
            preprocess_config={
                "command": "python -m training.bc_task_vlm.preprocess ...",
                "dataset_root": str(dataset_root),
                "train_example_format": "single_turn",
                "preprocessed_format_version": 4,
                "pretokenized": True,
                "pretokenization": {
                    "processor_family": "qwen",
                    "processor_class": "FakeQwenProcessor",
                    "processor_name_or_path": "fake-processor",
                    "processor_snapshot_hash": "hash-a",
                    "max_length": None,
                    "image_resolution": 512,
                    "trust_remote_code": False,
                },
            },
        )

        artifact = load_preprocessed_artifact_from_disk(output_dir)
        train_dataset = PreprocessedFeatureDataset(
            artifact.train_split,
            artifact_root=artifact.artifact_root,
        )
        validation_dataset = PreprocessedFeatureDataset(
            artifact.validation_split,
            artifact_root=artifact.artifact_root,
        )

        self.assertEqual(
            train_dataset[0]["pretokenized_tensors"]["input_ids"].tolist(),
            [1, 2, 3],
        )
        self.assertEqual(
            validation_dataset[0]["pretokenized_tensors"]["labels"].tolist(),
            [-100, 5],
        )


class PretokenizationCompatibilityTests(unittest.TestCase):
    def _build_single_trajectory_dataset_root(self) -> Path:
        temp_root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        trajectory_dir = temp_root / "hot_dog_setup" / "traj_000028"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        for file_name in ("original_trajectory.json", "plan.json", "metadata.json"):
            shutil.copy2(SOURCE_TRAJECTORY_DIR / file_name, trajectory_dir / file_name)
        return temp_root

    def test_returns_none_when_artifact_matches_run_config(self):
        reason = pretokenization_compatibility_reason(
            runtime_pretokenization={
                "processor_family": "qwen",
                "processor_class": "FakeQwenProcessor",
                "processor_name_or_path": "processor-a",
                "processor_snapshot_hash": "hash-a",
                "max_length": 4096,
                "image_resolution": 512,
                "trust_remote_code": False,
            },
            preprocess_config={
                "pretokenized": True,
                "pretokenization": {
                    "processor_family": "qwen",
                    "processor_class": "FakeQwenProcessor",
                    "processor_name_or_path": "processor-a",
                    "processor_snapshot_hash": "hash-a",
                    "max_length": 4096,
                    "image_resolution": 512,
                    "trust_remote_code": False,
                },
            },
        )

        self.assertIsNone(reason)

    def test_reports_mismatched_pretokenization_settings(self):
        reason = pretokenization_compatibility_reason(
            runtime_pretokenization={
                "processor_family": "qwen",
                "processor_class": "FakeQwenProcessor",
                "processor_name_or_path": "processor-a",
                "processor_snapshot_hash": "hash-a",
                "max_length": 4096,
                "image_resolution": 512,
                "trust_remote_code": False,
            },
            preprocess_config={
                "pretokenized": True,
                "pretokenization": {
                    "processor_family": "qwen",
                    "processor_class": "FakeQwenProcessor",
                    "processor_name_or_path": "processor-b",
                    "processor_snapshot_hash": "hash-b",
                    "max_length": 2048,
                    "image_resolution": 256,
                    "trust_remote_code": True,
                },
            },
        )

        self.assertIn("processor_name_or_path", reason)
        self.assertIn("processor_snapshot_hash", reason)
        self.assertIn("max_length", reason)
        self.assertIn("image_resolution", reason)
        self.assertIn("trust_remote_code", reason)

    def test_build_pretokenization_metadata_rejects_non_qwen_processors(self):
        class FakeProcessor:
            def save_pretrained(self, path):
                Path(path, "tokenizer.json").write_text("fake", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Qwen processors only"):
            build_pretokenization_metadata(
                processor=FakeProcessor(),
                processor_name_or_path="processor-a",
                max_length=None,
                image_resolution=512,
                trust_remote_code=False,
            )

    def test_build_pretokenization_metadata_hashes_saved_qwen_processor(self):
        class FakeQwenProcessor:
            def save_pretrained(self, path):
                Path(path, "tokenizer.json").write_text(
                    "fake-tokenizer", encoding="utf-8"
                )
                Path(path, "chat_template.jinja").write_text(
                    "template", encoding="utf-8"
                )

        metadata = build_pretokenization_metadata(
            processor=FakeQwenProcessor(),
            processor_name_or_path="processor-a",
            max_length=4096,
            image_resolution=512,
            trust_remote_code=False,
        )

        self.assertEqual(metadata["processor_family"], "qwen")
        self.assertEqual(metadata["processor_class"], "FakeQwenProcessor")
        self.assertEqual(metadata["processor_name_or_path"], "processor-a")
        self.assertEqual(metadata["max_length"], 4096)
        self.assertRegex(metadata["processor_snapshot_hash"], r"^[0-9a-f]{64}$")

    def test_cached_pretokenization_metadata_avoids_rehashing_processor(self):
        class FakeQwenProcessor:
            save_calls = 0

            def save_pretrained(self, path):
                type(self).save_calls += 1
                Path(path, "tokenizer.json").write_text(
                    "fake-tokenizer", encoding="utf-8"
                )

        temp_root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        cache_path = temp_root / "pretokenization_metadata.json"
        processor = FakeQwenProcessor()

        first_metadata = build_or_load_cached_pretokenization_metadata(
            processor=processor,
            processor_name_or_path="processor-a",
            max_length=4096,
            image_resolution=256,
            trust_remote_code=False,
            cache_path=cache_path,
        )
        second_metadata = build_or_load_cached_pretokenization_metadata(
            processor=processor,
            processor_name_or_path="processor-a",
            max_length=4096,
            image_resolution=256,
            trust_remote_code=False,
            cache_path=cache_path,
        )

        self.assertEqual(first_metadata, second_metadata)
        self.assertEqual(FakeQwenProcessor.save_calls, 1)
        self.assertRegex(second_metadata["processor_snapshot_hash"], r"^[0-9a-f]{64}$")

    def test_resolve_hf_dataset_reference_parses_revision_suffix(self):
        self.assertEqual(
            resolve_hf_dataset_reference("user/repo:branch-name"),
            ("user/repo", "branch-name"),
        )
        self.assertEqual(
            resolve_hf_dataset_reference(
                "user/repo:branch-name",
                explicit_revision="explicit-branch",
            ),
            ("user/repo:branch-name", "explicit-branch"),
        )

    def test_build_preprocessed_dataset_dict_falls_back_to_parallel_copy_when_hardlinks_fail(
        self,
    ):
        dataset_root = self._build_single_trajectory_dataset_root()
        train_examples = build_decentralized_examples(
            dataset_root=dataset_root,
            task_names=["hot_dog_setup"],
        )[:1]
        validation_examples: list = []
        output_dir = dataset_root / "artifact-copy-fallback"
        progress_messages: list[str] = []

        def _raise_cross_device_error(
            src,
            dst,
            *,
            src_dir_fd=None,
            dst_dir_fd=None,
            follow_symlinks=True,
        ):
            raise OSError("simulated hardlink failure")

        with mock.patch("os.link", side_effect=_raise_cross_device_error):
            dataset_dict = build_preprocessed_dataset_dict(
                train_examples=train_examples,
                validation_examples=validation_examples,
                output_dir=output_dir,
                progress_callback=progress_messages.append,
            )

        self.assertEqual(len(dataset_dict["train"]), len(train_examples))
        self.assertTrue((output_dir / "images").is_dir())
        self.assertTrue(
            any(
                "Copying" in message and "artifact images" in message
                for message in progress_messages
            )
        )
        original_image_path = Path(train_examples[0].image_paths[0])
        artifact_image_path = next((output_dir / "images").iterdir())
        self.assertNotEqual(
            os.stat(original_image_path).st_ino,
            os.stat(artifact_image_path).st_ino,
        )


if __name__ == "__main__":
    unittest.main()
