# LIBERO MuJoCo 3.3.2 to detailed LeRobot v3

This converter reads the four local LIBERO MuJoCo 3.3.2 LeRobot v2.1 partitions and writes
detailed LeRobot v3 datasets following
[`failure_rollout_data/dataset.md`](../failure_rollout_data/dataset.md).

The exact source vectors are retained as `raw_state.ref_state` (8D) and
`raw_action.ref_action` (7D). Native joint, EEF, and gripper fields; absolute controller targets;
canonical rot9d poses; normalized gripper states; and GT-next `debug.gripper_eef_*` fields are
also stored.

The source release already filtered historical no-op transitions during simulator replay and kept
successful episodes. The converter audits that predicate but does not filter any additional
frames. Reports are written to `meta/noop_audit.json` and
`meta/noop_audit_episodes.jsonl`.

The source action gripper has already been normalized to `0=closed, 1=open`, so
`target.gripper_state` preserves it directly. This differs from LIBERO Plus, whose native
`{-1, +1}` gripper action requires remapping.

Every entry in `meta/stats.json` includes `q01` and `q99`. Low-dimensional quantiles are exact
over all frames; video quantiles are per-channel values computed from a deterministic uniform
sample of up to 10,000 stored frames per view. Conversion assumptions and the quantile strategy
are retained in `meta/conversion_config.json`.

## Required camera transform for canonical training

The converter preserves `observation.images.image` and `observation.images.wrist_image` exactly
as stored. **For canonical training, horizontally flip both camera views at load time.**

```python
# NumPy (..., height, width, channels)
image = np.flip(image, axis=-2)

# PyTorch (..., channels, height, width)
image = torch.flip(image, dims=(-1,))
```

This is an image-only transform. Do not reflect low-dimensional fields.

## Conversion

The default input is `/scratch/cgao304/dev/FastWAM/data/libero_mujoco3.3.2`. Converted partitions
are written to `/scratch/cgao304/dev/FastWAM/data/datasets/lerobot_v3`, using the same directory
names as their Hugging Face repositories.

Convert one partition:

```bash
./libero_mj332_to_lerobot/convert.sh libero_10_no_noops_lerobot --overwrite
```

Convert all four partitions:

```bash
./libero_mj332_to_lerobot/convert_all.sh --overwrite
```

Run a short smoke conversion:

```bash
OUTPUT_BASE=/tmp/libero_mj332_v3 \
  ./libero_mj332_to_lerobot/convert.sh \
  libero_10_no_noops_lerobot --max-episodes 3 --overwrite
```

Inspect provenance and debug fields:

```bash
/scratch/cgao304/dev/envs/miniconda3/envs/fastwam/bin/python3.10 \
  libero_mj332_to_lerobot/inspect_debug.py \
  --dataset-dir /tmp/libero_mj332_v3/libero_10_mj332 \
  --episode-index 0 --rows 8
```

After all four conversions complete, upload each partition to its own repository using the cached
Hugging Face login or `HF_TOKEN` environment variable:

```bash
/scratch/cgao304/dev/envs/miniconda3/envs/fastwam/bin/python3.10 \
  libero_mj332_to_lerobot/upload.py
```

The default destinations are:

- `typoverflow/libero_10_mj332`
- `typoverflow/libero_goal_mj332`
- `typoverflow/libero_object_mj332`
- `typoverflow/libero_spatial_mj332`
