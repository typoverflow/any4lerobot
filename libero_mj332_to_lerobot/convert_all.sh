#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARTITIONS=(
  libero_10_no_noops_lerobot
  libero_goal_no_noops_lerobot
  libero_object_no_noops_lerobot
  libero_spatial_no_noops_lerobot
)

for partition in "${PARTITIONS[@]}"; do
  "$SCRIPT_DIR/convert.sh" "$partition" "$@"
done
