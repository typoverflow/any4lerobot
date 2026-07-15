#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/scratch/cgao304/dev/envs/miniconda3/envs/fastwam/bin/python3.10}"
SOURCE_BASE="${SOURCE_BASE:-/scratch/cgao304/dev/FastWAM/data/libero_mujoco3.3.2}"
OUTPUT_BASE="${OUTPUT_BASE:-/scratch/cgao304/dev/FastWAM/data/datasets/lerobot_v3}"
PARTITION="${1:-libero_10_no_noops_lerobot}"
shift || true

case "$PARTITION" in
  libero_10_no_noops_lerobot) OUTPUT_NAME="libero_10_mj332" ;;
  libero_goal_no_noops_lerobot) OUTPUT_NAME="libero_goal_mj332" ;;
  libero_object_no_noops_lerobot) OUTPUT_NAME="libero_object_mj332" ;;
  libero_spatial_no_noops_lerobot) OUTPUT_NAME="libero_spatial_mj332" ;;
  *)
    echo "unknown partition: $PARTITION" >&2
    exit 2
    ;;
esac

exec "$PYTHON" "$SCRIPT_DIR/convert.py" \
  --source-dir "$SOURCE_BASE/$PARTITION" \
  --output-dir "$OUTPUT_BASE/$OUTPUT_NAME" \
  "$@"
