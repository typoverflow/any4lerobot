# LIBERO Plus to detailed LeRobot v3

This converter reads the four local LIBERO Plus LeRobot v2.1 partitions and writes detailed
LeRobot v3 datasets following [`failure_rollout_data/dataset.md`](../failure_rollout_data/dataset.md).

The exact source vectors are retained as `raw_state.ref_state` (8D) and
`raw_action.ref_action` (7D). Native split fields, absolute controller targets, canonical rot9d
poses, normalized grippers, and GT-next `debug.gripper_eef_*` fields are also stored.

No frames are filtered. The historical LIBERO no-op predicate is evaluated for reporting only,
because dropping rows without simulator replay can compress controller/physics dynamics. Reports
are written to `meta/noop_audit.json` and `meta/noop_audit_episodes.jsonl`.

Every entry in `meta/stats.json` includes `q01` and `q99`. Low-dimensional quantiles are exact
over all frames; video quantiles are per-channel values computed from a deterministic uniform
sample of up to 10,000 stored frames per view. Conversion assumptions and the quantile strategy
are retained in `meta/conversion_config.json`. No private `.conversion_complete` sentinel is
written to the dataset.

## Required camera transform for canonical training

The converter preserves `observation.images.front` and `observation.images.wrist` exactly as they
appear in the source datasets; it does not decode, flip, or re-encode them. **For canonical
training, horizontally flip both camera views at load time.** For example:

The source-pipeline root cause is a composition of two image transforms: robosuite returns a
vertically inverted render, and the original dataset writer then rotates it by 180 degrees (flips
both image axes). The vertical flips cancel, leaving the stored image horizontally mirrored. This
is also tracked in [LeRobot issue #3830](https://github.com/huggingface/lerobot/issues/3830).

```python
# NumPy (..., height, width, channels)
image = np.flip(image, axis=-2)

# PyTorch (..., channels, height, width)
image = torch.flip(image, dims=(-1,))
```

This is an image-only transform. Do not negate or reflect any `raw_state.*`, `raw_target.*`,
`state.*`, `target.*`, or `debug.*` field: their rotations and relative motions remain proper,
right-handed coordinate representations. Every generated dataset includes the same warning in its
top-level `README.md` and records the requirement in `meta/conversion_config.json`.

## Conversion

```bash
./libero_plus2lerobot/convert.sh libero_plus_10 --overwrite
```

The default output is:

```text
/scratch/cgao304/dev/FastWAM/data/datasets/lerobot_v3/libero_plus_10
```

Run a short smoke conversion by overriding the output and limiting episodes:

```bash
OUTPUT_BASE=/tmp/libero_plus_v3 \
  ./libero_plus2lerobot/convert.sh libero_plus_10 --max-episodes 3 --overwrite
```

Inspect provenance and debug entries:

```bash
/scratch/cgao304/dev/envs/miniconda3/envs/fastwam/bin/python3.10 \
  libero_plus2lerobot/inspect_debug.py \
  --dataset-dir /tmp/libero_plus_v3/libero_plus_10 \
  --episode-index 0 --rows 8
```

Use the same runner for `libero_plus_object`, `libero_plus_goal`, and `libero_plus_spatial`.

Upload all completed partitions using the cached Hugging Face login or `HF_TOKEN` environment
variable (never put a token in this repository):

```bash
/scratch/cgao304/dev/envs/miniconda3/envs/fastwam/bin/python3.10 \
  libero_plus2lerobot/upload.py
```
