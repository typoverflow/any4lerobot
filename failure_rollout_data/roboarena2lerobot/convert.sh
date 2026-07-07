#!/usr/bin/env bash
# Convert both RoboArena data dumps into two separate LeRobot v3 datasets.
#
# Each dump becomes <OUTPUT_ROOT>/<name>_lerobot (name derived from the dump date), e.g.
#   roboarena_2025_08_05_lerobot   and   roboarena_2026_02_03_lerobot
#
# Re-running the same command resumes: finished shards are skipped, partial shards continue.
set -euo pipefail

PYTHON="${PYTHON:-/localscratch/cgao304/dev/envs/miniconda3/envs/lerobot/bin/python}"
RAW_ROOT="${RAW_ROOT:-/localscratch/cgao304/dev/datasets/roboarena}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/localscratch/cgao304/dev/datasets/lerobot_v3/roboarena_dump}"
NUM_PROC="${NUM_PROC:-8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

for DUMP in DataDump_02-03-2026; do
    RAW_DIR="$RAW_ROOT/$DUMP"
    if [[ ! -d "$RAW_DIR/evaluation_sessions" ]]; then
        echo "[convert.sh] skip missing dump: $RAW_DIR"
        continue
    fi
    echo "[convert.sh] converting $DUMP"
    "$PYTHON" convert.py \
        --raw-dir "$RAW_DIR" \
        --local-dir "$OUTPUT_ROOT" \
        --num-proc "$NUM_PROC" \
        --skip-bad-episodes \
        "$@"
done

echo "[convert.sh] done -> $OUTPUT_ROOT"
