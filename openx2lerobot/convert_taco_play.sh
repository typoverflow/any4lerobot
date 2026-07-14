#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/localscratch/cgao304/dev/envs/miniconda3/envs/lerobot/bin/python}"
RAW_ROOT="${RAW_ROOT:-/localscratch/cgao304/dev/datasets/rlds}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/localscratch/cgao304/dev/datasets/lerobot_v3}"
NUM_PROC="${NUM_PROC:-8}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-131072}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-4}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"

args=(
    --raw-dir "$RAW_ROOT/taco_play/0.1.0"
    --local-dir "$OUTPUT_ROOT/taco_play"
    --use-videos
    --num-proc "$NUM_PROC"
    --image-writer-process "${IMAGE_WRITER_PROCESS:-1}"
    --image-writer-threads "${IMAGE_WRITER_THREADS:-4}"
    --prefetch-buffer "${PREFETCH_BUFFER:-2}"
    --repo-id "${REPO_ID:-typoverflow/taco_play}"
)
[[ "${PUSH_TO_HUB:-0}" == "1" ]] && args+=(--push-to-hub)

exec "$PYTHON" "$SCRIPT_DIR/openx_rlds.py" "${args[@]}" "$@"
