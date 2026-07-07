#!/usr/bin/env bash
# Convert the ViFailback dataset to LeRobot v3.
#
# - Re-running the SAME command resumes an interrupted conversion (per-shard sentinel +
#   progress file); finished shards are skipped, partial shards continue where they stopped.
# - Add --save-depth to also store the uint16 depth streams (roughly doubles size/time).
# - QC warnings (idle-arm stale-qpos episodes, FK-vs-action_eef residual > 5 mm) end up in
#   <output>/vifailback_lerobot/meta/qc_warnings.jsonl.
set -euo pipefail

PYTHON=/localscratch/cgao304/dev/envs/miniconda3/envs/lerobot/bin/python
RAW_DIR=/localscratch/cgao304/dev/datasets/ViFailback-Dataset/raw_data
LOCAL_DIR=/localscratch/cgao304/dev/datasets/lerobot_v3/ViFailback-Dataset/

export HDF5_USE_FILE_LOCKING=FALSE

"$PYTHON" "$(dirname "$0")/convert.py" \
    --raw-dir "$RAW_DIR" \
    --local-dir "$LOCAL_DIR" \
    --repo-id vifailback \
    --num-proc 8 \
    --overwrite
