#!/usr/bin/env bash
# Convert the SOAR dataset (WidowX 250, RLDS) to LeRobot v3 -- one dataset per split.
#
# - Re-running the SAME command resumes an interrupted conversion (per-shard sentinel +
#   progress file); finished shards are skipped, partial shards continue mid-shard.
# - The success split may still be downloading: workers stop gracefully at the first
#   missing/truncated tfrecord and the run reports which shards are incomplete -- just
#   re-run this script once more data has arrived. Merging happens only when a split is
#   fully converted.
# - Per-worker logs: <local-dir>/_shards_<split>/logs/shardNNN.log
set -euo pipefail

PYTHON=/localscratch/cgao304/dev/envs/miniconda3/envs/lerobot/bin/python
RAW_DIR=/localscratch/cgao304/dev/datasets/soar/rlds
LOCAL_DIR=/localscratch/cgao304/dev/datasets/lerobot_v3/soar

export CUDA_VISIBLE_DEVICES=""
# Memory-leak fix (same rationale as openx2lerobot/convert.sh): cap glibc malloc arenas so
# TF worker threads don't each pin a 64 MiB arena.
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=131072
export OMP_NUM_THREADS=2
export TF_CPP_MIN_LOG_LEVEL=2

for SPLIT in failure success; do
    "$PYTHON" "$(dirname "$0")/convert.py" \
        --raw-dir "$RAW_DIR" \
        --local-dir "$LOCAL_DIR" \
        --split "$SPLIT" \
        --repo-id "soar_$SPLIT" \
        --num-proc 8 \
        --image-writer-process 1 \
        --image-writer-threads 4 \
        || echo "[convert.sh] split=$SPLIT incomplete (download still in progress?); re-run later to resume."
done
