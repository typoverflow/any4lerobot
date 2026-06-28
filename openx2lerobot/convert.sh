#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=""

# Memory-leak fix: default glibc gives each of the worker's ~200 threads its own 64 MiB malloc arena
# and never returns it to the OS -> anonymous RSS crept ~9.7 MiB/episode and OOM'd the box. Capping
# arenas (and trimming) flattens it to a stable plateau. The orchestrator also sets these per worker;
# exporting here covers the orchestrator process and any --num-proc 1 run.
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=131072
export OMP_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=2

# Re-running this exact command resumes: completed shards are skipped and a shard that died midway
# continues from where it stopped (nothing already converted is redone). Per-worker logs are written
# to <local-dir>/_shards/logs/shardNNN.log -- tail them to watch progress / see a crash cause.
# If memory-bound, lower --num-proc (fewer concurrent workers) and/or --prefetch-buffer.
# Add --skip-bad-episodes to tolerate individual undecodable episodes, --overwrite to force a rebuild.
PYTHON=/localscratch/cgao304/dev/envs/miniconda3/envs/lerobot/bin/python

"$PYTHON" openx_rlds.py \
    --raw-dir /localscratch/cgao304/dev/datasets/rlds/droid/1.0.1 \
    --local-dir /localscratch/cgao304/dev/datasets/lerobot_v3/droid/ \
    --use-videos \
    --num-proc 8 \
    --image-writer-process 1 \
    --image-writer-threads 4 \
    --prefetch-buffer 2 \
    --repo-id typoverflow/droid \
    --push-to-hub
