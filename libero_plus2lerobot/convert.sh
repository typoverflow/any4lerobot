#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/scratch/cgao304/dev/envs/miniconda3/envs/fastwam/bin/python3.10}"
SOURCE_BASE="${SOURCE_BASE:-/scratch/cgao304/dev/FastWAM/data/datasets/libero_plus}"
OUTPUT_BASE="${OUTPUT_BASE:-/scratch/cgao304/dev/FastWAM/data/datasets/lerobot_v3}"
PARTITION="${1:-libero_plus_10}"
shift || true

case "$PARTITION" in
  libero_plus_10|libero_plus_object|libero_plus_goal|libero_plus_spatial) ;;
  *)
    echo "unknown partition: $PARTITION" >&2
    exit 2
    ;;
esac

exec "$PYTHON" "$SCRIPT_DIR/convert.py" \
  --source-dir "$SOURCE_BASE/$PARTITION" \
  --output-dir "$OUTPUT_BASE/$PARTITION" \
  "$@"
