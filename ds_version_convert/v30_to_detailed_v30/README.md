# DROID: LeRobot v3.0 -> "detailed" v3.0

Converts the official DROID LeRobot v3.0 dataset into a v3.0 dataset whose entries implement
`openx2lerobot/design_of_state_and_action_space.md` -- i.e. exactly the entries that
`openx2lerobot/openx_rlds.py` (with `droid_baseact_transform`) produces from the raw RLDS dataset,
but **without** going through RLDS/TF or re-encoding any video:

- per-frame `state.*` / `action.*` features are recomputed directly from the source parquet columns
  (`observation.state.{cartesian,joint,gripper}_position`, `action.gripper_position`)
  with numpy ports of `oxe_utils/transform_utils.py` (verified to match the TF implementation to
  float32 precision). All rotation features use DROID's native **extrinsic (fixed-axis) XYZ** euler
  convention (`R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`);
- videos are reused via hardlinks (default), only the keys are renamed to the RLDS names
  (`exterior_1_left -> exterior_image_1_left`, `exterior_2_left -> exterior_image_2_left`,
  `wrist_left -> wrist_image_left`);
- `meta/episodes` is rebuilt with per-episode stats for the new features (computed with lerobot's
  `get_feature_stats`, so the format/numerics match lerobot's own pipeline), and `meta/stats.json`
  is produced by aggregating them with lerobot's `aggregate_feature_stats` (video and
  index-column stats are carried over from the source).

Output reference vectors (per `OXE_DATASET_CONFIGS["droid"]`; single-arm -> **gripper-frame** actions):

- `observation.state` (20) = `[eef_xyz (3), eef_rpy (3), eef_rot6d (6), joint_position (7), gripper_state (1)]`
- `action` (10) = `[gripper_eef_xyz (3), gripper_eef_rot6d (6), gripper_state (1)]`

Per-frame `action.*` also includes the world-frame variants (`world_eef_xyz`, `world_eef_rpy`,
`world_eef_rot6d`), the gripper-frame fields, the realized `joint_position` delta, and `gripper_state`.

Gripper convention: DROID stores `gripper_position` as 0 = fully open / 1 = fully closed;
`state.gripper_state` and `action.gripper_state` are the inverted values
(`1 - gripper_position`), i.e. **1 = fully open / 0 = fully closed** (OpenVLA convention).
`action.gripper_state` is the commanded absolute gripper target, not a delta.

Note: the converter only processes the data/video files actually referenced by `meta/episodes`;
the official DROID v3 dataset ships 70 orphan data parquets (duplicating ~42k episodes, likely
leftovers of a partial re-conversion) which LeRobot's reader ignores, and so does this converter.

## Usage

```bash
conda activate lerobot
bash convert.sh
# or
python convert_droid_v30_to_detailed_v30.py \
    --source-dir /path/to/droid_lerobot_v3 \
    --output-dir /path/to/droid_detailed_v3 \
    --num-workers 16 \
    --video-mode hardlink   # hardlink | symlink | copy
```

`--max-data-files N` converts only the first N data files (debug; skips meta/videos).

## Known deviation from `openx_rlds.py`

`droid_baseact_transform` randomly swaps the two exterior cameras per episode
(`rand_swap_exterior_images`, a train-time augmentation that leaked into conversion). In v3 many
episodes share one video file, so a per-episode swap is impossible without re-encoding; this
converter maps the cameras deterministically instead. No information is lost -- both views are
kept, only the random name assignment differs.

Also note the source stores float32 (the raw RLDS data is float64), so recomputed rotation
features can differ from an RLDS-derived conversion by ~1e-6 (float32 rounding).
