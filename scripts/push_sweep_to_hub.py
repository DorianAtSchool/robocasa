#!/usr/bin/env python3
"""Push existing sweep output to HuggingFace Hub.

Usage:
    python scripts/push_sweep_to_hub.py \
        --sweep-dir tmp/sweep_all_tasks_all_trajectories_L11_L42 \
        --repo-id MasonN808/robocasa-trajectories
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so we can import sweep_trajectories
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_trajectories import (
    sweep_output_to_dataset,
    upload_dataset_card,
    upload_sweep_sidecars,
)


def main():
    parser = argparse.ArgumentParser(description="Push sweep output to HuggingFace Hub")
    parser.add_argument("--sweep-dir", type=str, required=True, help="Path to sweep output directory")
    parser.add_argument("--repo-id", type=str, required=True, help="HuggingFace repo id (e.g. username/dataset-name)")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not (sweep_dir / "sweep_summary.json").exists():
        print(f"Error: No sweep_summary.json found in {sweep_dir}")
        sys.exit(1)

    print(f"Converting {sweep_dir} to HuggingFace dataset...")
    ds = sweep_output_to_dataset(sweep_dir)
    print(f"\nDataset info:")
    print(f"  rows: {len(ds)}")
    print(f"  columns: {ds.column_names}")
    print(f"\nPushing to {args.repo_id}...")
    ds.push_to_hub(args.repo_id)
    print("Uploading sweep metadata sidecars...")
    upload_sweep_sidecars(args.repo_id, sweep_dir)
    print("Uploading dataset card...")
    upload_dataset_card(args.repo_id, ds)
    print(f"Done! Dataset at https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
