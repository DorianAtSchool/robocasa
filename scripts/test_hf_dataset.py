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

    row_granularity = "trajectory" if "adapted_trajectory" in ds.column_names else "step"
    print(f"\nRow granularity: {row_granularity}")

    tasks = set(ds["task"])
    print(f"\nTasks ({len(tasks)}): {sorted(tasks)}")
    episodes = set(ds["episode_id"])
    print(f"Episodes: {len(episodes)}")

    if row_granularity == "step":
        adapted_paths = [p for p in ds["adapted_trajectory_path"] if p]
        original_paths = [p for p in ds["original_trajectory_path"] if p]
        execution_paths = [p for p in ds["execution_metadata_path"] if p]
        print(f"\nAdapted trajectory refs: {len(set(adapted_paths))} unique")
        print(f"Original trajectory refs: {len(set(original_paths))} unique")
        print(f"Execution metadata refs: {len(set(execution_paths))} unique")
    else:
        adapted_sizes = [len(t.encode()) for t in ds["adapted_trajectory"] if t]
        original_sizes = [len(t.encode()) for t in ds["original_trajectory"] if t]
        execution_sizes = [len(t.encode()) for t in ds["execution_metadata"] if t]
        print(f"\nAdapted trajectory JSON: {len(adapted_sizes)} non-empty, avg {sum(adapted_sizes)//max(len(adapted_sizes),1)//1024} KB")
        print(f"Original trajectory JSON: {len(original_sizes)} non-empty, avg {sum(original_sizes)//max(len(original_sizes),1)//1024} KB")
        print(f"Execution metadata JSON: {len(execution_sizes)} non-empty, avg {sum(execution_sizes)//max(len(execution_sizes),1)//1024} KB")

    # Size info
    if ds.dataset_size:
        print(f"\nDataset size on disk: {ds.dataset_size / 1e6:.1f} MB")
    if ds.info.download_size:
        print(f"Download size: {ds.info.download_size / 1e6:.1f} MB")

    # Tool distribution
    tool_counts = {}
    if row_granularity == "step":
        for tool in ds["tool_name"]:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
    else:
        for tool_list in ds["tool_name"]:
            for tool in tool_list:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
    print(f"\nTool distribution:")
    for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        print(f"  {tool}: {count}")

    if row_granularity == "step":
        episode_counts = {}
        for episode_id in ds["episode_id"]:
            episode_counts[episode_id] = episode_counts.get(episode_id, 0) + 1
        counts = list(episode_counts.values())
    else:
        counts = ds["num_steps"]
    print(f"\nSteps per episode: min={min(counts)}, avg={sum(counts)/len(counts):.1f}, max={max(counts)}")

    # Image columns
    image_cols = ["room_view", "top_view", "map", "agentview_center",
                  "agentview_left", "agentview_right", "wrist"]
    print(f"\nImage columns (non-null counts):")
    for col in image_cols:
        if row_granularity == "step":
            non_null = sum(1 for image in ds[col] if image is not None)
        else:
            non_null = sum(sum(1 for image in image_list if image is not None) for image_list in ds[col])
        print(f"  {col}: {non_null}")

    # Show first row
    print(f"\n=== First Row ===")
    row = ds[0]
    for k, v in row.items():
        if k in image_cols:
            if row_granularity == "step":
                print(f"  {k}: {'PIL Image ' + str(v.size) if v is not None else 'None'}")
            else:
                sample = next((img for img in v if img is not None), None)
                print(f"  {k}: {len(v)} entries, first non-null={'PIL Image ' + str(sample.size) if sample is not None else 'None'}")
        else:
            val_str = str(v)
            if len(val_str) > 100:
                val_str = val_str[:100] + "..."
            print(f"  {k}: {val_str}")

    print("\nAll checks passed!")


if __name__ == "__main__":
    main()
