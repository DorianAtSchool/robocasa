# Pretokenization Speedup Plan

## Goal

Make task-VLM pretokenization finish in minutes instead of hours by removing
serial resume work and distributing pretokenization across Slurm array shards.

The motivating run is:

```text
slurm_logs/bc-task-vlm-preprocessed-20260425T184149Z-17869508.err
```

That run spent almost seven hours between finishing example loading and logging
`Reusing 278131 existing staged artifact images`, then started a serial
pretokenization loop that processed only `5000 / 106083` train examples in about
20 minutes before hitting the 8 hour Slurm limit.

## Current Bottlenecks

- Existing artifact image resume validates every staged image with a serial
  filesystem check.
- Pretokenization runs in one Python process over one example at a time.
- `--example-build-workers` only helps raw example construction. It does not
  parallelize pretokenization.
- A single 8 CPU Slurm job cannot provide 100x wall-clock speedup, even if local
  multiprocessing is added.

## Target Architecture

Use a two-phase artifact build:

1. Build or reuse examples and artifact images once.
2. Pretokenize examples in many independent Slurm array shards.
3. Merge shard outputs into the final Hugging Face `DatasetDict` artifact.

Target launch shape:

```bash
pretok_job_id=$(
  sbatch --parsable --array=0-127 training/scripts/pretokenize_shard.sh
)

sbatch \
  --dependency=afterok:${pretok_job_id} \
  training/scripts/merge_pretok_shards.sh
```

This is the realistic path to approximately 100x wall-clock improvement. Local
multiprocessing is still useful inside each shard, but the major speedup comes
from using many nodes/jobs at once.

## Phase 1: Fast Resume For Staged Images

Add a mode that reconstructs artifact image relative paths without checking all
files.

Proposed flag:

```bash
--resume-existing-artifact-images unchecked
```

or:

```bash
--skip-existing-artifact-image-validation
```

Implementation notes:

- Keep the current validating resume path as the safe default.
- Add an unchecked path for known-good partial artifacts.
- The artifact image relpath is deterministic from the source image path, so the
  source-to-artifact mapping can be rebuilt without touching every output file.
- Optionally add a parallel validation mode later, but unchecked resume is the
  highest-leverage fix for the observed seven hour resume gap.

Expected impact:

- Removes the resume-time scan of `278131` staged images.
- Converts the observed multi-hour resume step into a metadata-only pass.

## Phase 2: Shard Pretokenization

Add shard controls to the preprocessing entrypoint.

Proposed flags:

```bash
--pretokenize-shard-index N
--pretokenize-num-shards M
--pretokenize-shard-output-dir PATH
```

Behavior:

- Load the same cached train and validation examples as the full preprocess job.
- Select examples where `stable_sample_rank(sample_id) % M == N`, or use an
  index-range split if deterministic ordering is already guaranteed.
- Load the Qwen processor once per shard process.
- Pretokenize only the shard's examples.
- Write shard output files such as:

```text
pretokenized_shards/train-000042.arrow
pretokenized_shards/validation-000042.arrow
pretokenized_shards/manifest-000042.json
```

Important details:

- Use stable sample ids for assignment so resubmitting one shard produces the
  same sample set.
- Include pretokenization metadata in every shard and reject merges when shard
  metadata disagrees.
- Write shard outputs atomically: write to a temp file, then rename into place.
- Make shards idempotent so failed array tasks can be resubmitted safely.

## Phase 3: Batch Within Each Shard

Add a batch size for pretokenization.

Proposed flag:

```bash
--pretokenize-batch-size 4
```

Start with small batches, for example `4` or `8`, because each sample may carry
multiple images and Qwen vision tensors can be large.

Implementation notes:

- Reuse `build_lazy_vision_sft_batch(...)`, which already accepts multiple
  features.
- Split batched sequence fields by row:
  - `input_ids`
  - `attention_mask`
  - `labels`
  - `mm_token_type_ids`
- Split Qwen concatenated image fields using per-sample image counts and
  `image_grid_thw`:
  - `pixel_values`
  - `image_grid_thw`
  - `pixel_values_videos`
  - `video_grid_thw`
  - `second_per_grid_ts`
- Add tests that batched pretokenization produces the same serialized tensors as
  the current single-sample path on a small fixture.

Expected impact:

- Reduces Python overhead.
- Lets tokenizer and image processor calls amortize work across multiple
  examples.
- Improves each shard's runtime before Slurm-level parallelism is applied.

## Phase 4: Optional Local Workers Per Shard

After sharding works, add local worker processes inside each shard if CPU usage
is still low.

Proposed flags:

```bash
--pretokenize-workers 4
--pretokenize-batch-size 4
```

Worker behavior:

- Each worker loads its own processor once.
- Each worker receives batches or chunks from that shard only.
- Each worker returns `(sample_id, pretokenized_blob)` pairs.

Set these environment variables in workers to avoid oversubscription:

```bash
TOKENIZERS_PARALLELISM=false
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
```

Start conservatively on an 8 CPU node:

```bash
--pretokenize-workers 4 --pretokenize-batch-size 4
```

## Phase 5: Merge Shards

Add a merge command or script that builds the final artifact from shard outputs.

Proposed command:

```bash
python -m training.bc_task_vlm.merge_pretokenized_shards \
  --dataset-root data_generation/task_level/data/image/20260413T205634Z \
  --output-dir training/bc_task_vlm/preprocessed/task_vlm_same_task_qwen35_08b_pretok \
  --shard-dir training/bc_task_vlm/preprocessed/task_vlm_same_task_qwen35_08b_pretok/pretokenized_shards \
  --num-shards 128
```

Merge responsibilities:

- Verify every expected shard exists.
- Verify shard pretokenization metadata is identical.
- Verify sample ids are unique and cover the expected train and validation
  sample sets.
- Build `train_pretokenized_blobs_by_sample_id` and
  `validation_pretokenized_blobs_by_sample_id`.
- Call the existing dataset serialization and artifact save path.

## Phase 6: Slurm Launchers

Add scripts:

```text
training/scripts/pretokenize_shard.sh
training/scripts/merge_pretok_shards.sh
training/scripts/launch_pretokenize_array.sh
```

`launch_pretokenize_array.sh` should:

- Submit the array job.
- Pass dataset root, output dir, processor, max length, artifact image size,
  shard count, batch size, and optional local worker count.
- Submit the merge job with `--dependency=afterok:<array_job_id>`.
- Print the array job id, merge job id, output dir, and log paths.

Useful initial defaults:

```bash
--array=0-127
--cpus-per-task=4
--mem=32g
--time=01:00:00
--pretokenize-batch-size=4
```

Tune after measuring filesystem pressure and per-shard memory.

## Verification Plan

Unit tests:

- Unchecked image resume returns the same relpath mapping as validating resume.
- Shard assignment is deterministic and covers every sample exactly once.
- Batched pretokenization matches single-sample pretokenization for supported
  tensor fields.
- Merge rejects missing shards, duplicate sample ids, and metadata mismatches.
- Merge creates an artifact that `load_preprocessed_artifact_from_disk(...)`
  can load.

Small integration test:

```bash
python -m training.bc_task_vlm.preprocess ... \
  --pretokenize \
  --pretokenize-num-shards 4 \
  --pretokenize-shard-index 0 \
  --pretokenize-shard-output-dir /tmp/pretok-shards
```

Then run all four shards locally on a tiny dataset subset and merge them.

Cluster benchmark:

- Run `8`, `32`, and `128` shards on the same artifact image directory.
- Record examples per second per shard.
- Record total wall-clock time including merge.
- Check CPU utilization, memory, and metadata server load.

## Rollout Order

1. Implement unchecked staged-image resume.
2. Implement deterministic shard selection and shard output writing.
3. Implement merge command.
4. Add Slurm array launcher.
5. Add batched pretokenization inside each shard.
6. Add optional local pretokenization workers if needed.
7. Benchmark and tune shard count, batch size, and CPUs per task.

## Operational Notes

- Keep the old single-process preprocessing path as a fallback.
- Do not overwrite completed artifacts silently.
- Prefer writing shard files under the final output directory so cleanup and
  provenance are obvious.
- Consider keeping failed shard logs permanently until the merged artifact is
  verified.
- For known-good `images/` directories, use unchecked resume. For new or
  uncertain artifacts, use validating resume once.
