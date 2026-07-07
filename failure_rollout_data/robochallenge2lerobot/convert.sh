#!/usr/bin/env bash
# Convert RoboChallenge Table30-v2 crawled rollouts (.rrd) to LeRobot v3, one dataset per
# embodiment. Re-running the same command resumes an interrupted conversion (per-shard
# sentinels + mid-list progress). Set DATA_DIR / OUT_DIR / NUM_PROC to taste.
set -euo pipefail

PY="${PY:-/localscratch/cgao304/dev/envs/miniconda3/envs/lerobot/bin/python}"
DATA_DIR="${DATA_DIR:-/localscratch/cgao304/dev/datasets/robochallenge/data}"
OUT_DIR="${OUT_DIR:-/localscratch/cgao304/dev/datasets/robochallenge/lerobot}"
NUM_PROC="${NUM_PROC:-8}"
ROBOTS="${ROBOTS:-arx5 ur5 aloha dos_w1}"

export HDF5_USE_FILE_LOCKING=FALSE

for robot in $ROBOTS; do
  echo "==== converting $robot ===="
  "$PY" convert.py \
    --data-dir "$DATA_DIR" \
    --local-dir "$OUT_DIR" \
    --robot "$robot" \
    --num-proc "$NUM_PROC" \
    --skip-bad-episodes
done
