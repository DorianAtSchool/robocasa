#!/usr/bin/env python3
"""Test loading the robocasa trajectories dataset from HuggingFace Hub.

Usage:
    python scripts/test_hf_dataset.py --repo-id DorianAtSchool/robocasa-trajectories
"""

import argparse
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Test HF dataset loading")
    parser.add_argument("--repo-id", type=str, required=True, help="HuggingFace repo id")
    args = parser.parse_args()

    print(f"Loading dataset from {args.repo_id}...")
    ds = load_dataset(args.repo_id, split="train")
    print(f"\n=== Dataset Summary ===")
    print(f"Rows:    {len(ds)}")
    print(f"Columns: {ds.column_names}")
    print(f"Features:")
    for name, feat in ds.features.items():
        print(f"  {name}: {feat}")

    tasks = set(ds["task"])
    print(f"\nTasks ({len(tasks)}): {sorted(tasks)}")
    episodes = set(ds["episode_id"])
    print(f"Episodes: {len(episodes)}")

    # Check sidecar path coverage
    adapted_paths = [p for p in ds["adapted_trajectory_path"] if p]
    original_paths = [p for p in ds["original_trajectory_path"] if p]
    execution_paths = [p for p in ds["execution_metadata_path"] if p]
    print(f"\nAdapted trajectory refs: {len(set(adapted_paths))} unique")
    print(f"Original trajectory refs: {len(set(original_paths))} unique")
    print(f"Execution metadata refs: {len(set(execution_paths))} unique")

    # Size info
    if ds.dataset_size:
        print(f"\nDataset size on disk: {ds.dataset_size / 1e6:.1f} MB")
    if ds.info.download_size:
        print(f"Download size: {ds.info.download_size / 1e6:.1f} MB")

    # Tool distribution
    tool_counts = {}
    for tool in ds["tool_name"]:
        tool_counts[tool] = tool_counts.get(tool, 0) + 1
    print(f"\nTool distribution:")
    for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        print(f"  {tool}: {count}")

    episode_counts = {}
    for episode_id in ds["episode_id"]:
        episode_counts[episode_id] = episode_counts.get(episode_id, 0) + 1
    counts = list(episode_counts.values())
    print(f"\nSteps per episode: min={min(counts)}, avg={sum(counts)/len(counts):.1f}, max={max(counts)}")

    # Image columns
    image_cols = ["room_view", "top_view", "map", "agentview_center",
                  "agentview_left", "agentview_right", "wrist"]
    print(f"\nImage columns (non-null counts):")
    for col in image_cols:
        non_null = sum(1 for image in ds[col] if image is not None)
        print(f"  {col}: {non_null}")

    # Show first step row
    print(f"\n=== First Step Row ===")
    row = ds[0]
    for k, v in row.items():
        if k in image_cols:
            print(f"  {k}: {'PIL Image ' + str(v.size) if v is not None else 'None'}")
        else:
            val_str = str(v)
            if len(val_str) > 100:
                val_str = val_str[:100] + "..."
            print(f"  {k}: {val_str}")

    print("\nAll checks passed!")


if __name__ == "__main__":
    main()
