---
name: standardize-oxe-dataset
description: Standardize an Open-X-Embodiment / RLDS robot dataset into this repo's unified state/action space (eef_xyz, eef_rpy, eef_rot6d, world/body/diff eef deltas, gripper). Use when adding a new dataset to openx2lerobot, writing or fixing a per-dataset transform in oxe_utils/transforms.py, or when asked to "standardize", "add", or "convert" an OXE/RLDS dataset. Covers what to verify about the raw schema (rotation convention, frames, command-vs-ground-truth, gripper, camera extrinsics) and how to implement + empirically validate the transform.
---

# Standardizing an OXE/RLDS dataset

Goal: map one raw RLDS dataset onto the unified space defined in
`design_of_state_and_action_space.md`, using the backend-agnostic math in `alignment/`,
and **prove the mapping is correct against real data** before committing.

The unified output (per dataset, subset selected by config):

- **state** (world frame): `eef_xyz` (3), `eef_rpy` (3), `eef_rot6d` (6), `joint_pos` (7, if any), `gripper_state` (1, `1=open`)
- **action** (per-step delta `e → e*`): `body_eef_xyz`/`body_eef_rot6d` (canonical, body frame), `world_eef_xyz`/`world_eef_rot6d`, `diff_eef_xyz`, `diff_eef_rpy`, `diff_joint_pos`, `gripper_state`

`eef_rot6d` is Zhou et al. 6D (first two rows of R). Deltas are stored in the **body frame** as the
canonical form; any other frame is recovered at load time from `R_e^c`. See the design doc for the SE(3) math.

**LeRobot output layout.** Each named field above is written as its own feature: `state.<key>`
(e.g. `state.eef_xyz`) and `action.<key>` (e.g. `action.body_eef_xyz`). The **out-of-box ready-to-train
vectors** are the per-dataset concatenations selected by `state_encoding` / `action_encoding`, written as
**`observation.state`** and **`action`**. `eef_rpy` (and any field not in the encoding) remains available as
a standalone `state.*` column but is not part of the `observation.state` vector.

> The reference implementations to copy from are `rt1_dataset_transform` (fractal, **mode 1 / command**)
> and `bridge_orig_dataset_transform` / `_droid_state_and_action` (**mode 2 / ground-truth next**) in
> `openx2lerobot/oxe_utils/transforms.py`.

---

## Step 0 — Get the schema and a few real episodes

You cannot standardize correctly from the README alone; **every** convention below must be checked against bytes.

1. Locate/download the data. OXE datasets live under `gs://gresearch/robotics/<name>/<version>/`.
   List first (`gsutil ls`), the version is a subdir (e.g. `0.1.0`, `0.0.1`) — don't assume `1.0.0`.
   `gsutil -m cp -r` the shards; also fetch `features.json` and `dataset_info.json` (small) so TFDS can load.
2. Read `features.json` — this is the ground-truth schema: every observation/action field, its shape and dtype.
3. Load 3–4 episodes with `tfds.builder_from_directory(dir).as_dataset(split="train").take(4)` for the probe.

`.take(N)` reads shards in order, so it works even before a large download finishes.

---

## Step 1 — The checklist (what to verify)

Run a **probe script** (template: `probe_template.py` in this skill folder) and answer each item from data, not assumption:

### 1. Camera extrinsics / pose
- Search `features.json` for any camera-pose field (e.g. `*extrinsics*`, `camera/T_*`, `*_pose` that is a camera, not the eef).
- If a dataset **does** expose per-frame extrinsics, carry them through (the design doc's frame-`c` math needs `R_c^w`).
- Most OXE/RLDS datasets have **none** → there is nothing to carry; note it explicitly. Verified-absent so far:
  fractal, bridge V2, and **DROID** (the `gs://gresearch/robotics/droid` RLDS, 1.0.0/1.0.1, has no
  camera_extrinsics/intrinsics — only eef `cartesian_position`, joints, gripper, and three image streams; its
  state/action are all base/world frame). Note: the *full* DROID release (droid-dataset.github.io, ~1.7 TB) does
  ship per-camera calibration, but that is NOT the subset converted here — do not assume it from the README.
- Do not mistake `base_pose_tool_reached`, `cartesian_position`, `workspace_bounds`, etc. for a camera pose — those are eef/workspace.

### 2. Orientation representation & exact ordering
- What is the raw eef orientation? quaternion / euler-rpy / rotation matrix / axis-angle.
- **Quaternion order is not documented reliably — verify empirically.** Test both `xyzw` (scalar-last) and `wxyz`
  (scalar-first): build R both ways and check which makes the world-frame relative rotation `R_{t+1} R_t^T`
  consistent with any commanded `rotation_delta`, or which round-trips sanely. (fractal = **xyzw**.)

### 3. Euler: extrinsic vs intrinsic, and axis order
- `alignment.rpy_to_matrix(rpy, extrinsic=False)` = **intrinsic XYZ** `Rx(r)Ry(p)Rz(y)` (pytorch3d / DROID `*_rot_6d`).
- `extrinsic=True` = **fixed-axis (extrinsic) XYZ** `Rz(y)Ry(p)Rx(r)` (= scipy lowercase `'xyz'`, transforms3d `'sxyz'`).
- Decide empirically: build R from the rpy, compare to `scipy Rotation.from_euler('xyz', rpy)` (extrinsic) vs
  `'XYZ'` (intrinsic). The one matching to ~1e-7 is the dataset's convention. (DROID & bridge = **extrinsic**.)
- Getting this wrong silently corrupts every downstream rotation. Always guard with a `rpy→R→rpy` round-trip.

### 4. Frame of the **state**
- The unified state is **world/base frame**. Confirm the raw eef pose is world/base-relative (it usually is for OXE),
  not camera-relative. If camera-relative, transform to world using the extrinsics first.
- Refer to the dataset's official documentation first, do not trust the name of the data entry

### 5. Frame of the **action**
- In which frame is the raw action expressed: world/base, body/gripper, or camera?
- fractal's `world_vector`/`rotation_delta` are **base/world-frame** commands (verified: world `R_{t+1}R_t^T` ≈
  `rotation_delta`, ~3× the per-step achieved motion). Build `e*` in that frame, then derive body/world/diff via `alignment`.
- If the action is already a body-frame delta, it maps to `body_eef_*` more directly — still verify.
- Refer to the dataset's official documentation first, do not trust the name of the data entry


### 6. Command (mode 1) vs ground-truth-next (mode 2) — choosing `e*`
- **Mode 1**: the raw action is a *meaningful command* (the target sent to the controller). Use it as `e*`.
  Keep all steps (no discard). fractal uses this — its command is genuinely different from (≈3× larger than) the
  achieved motion.
- **Mode 2**: no meaningful command → use the **ground-truth next pose** as `e*`. **Discard the last step** (it has no
  successor) — **do not pad**. DROID and bridge use this.
- Decide by **comparing raw action to the achieved state difference**: if `raw_action ≈ state[t+1]−state[t]` it's just a
  finite difference (mode 2 is honest); if it differs materially it's a real command (mode 1 is available). bridge's raw
  action *differs* from the state delta but the repo's canonical `relabel_bridge_actions` still uses the state delta, so
  we picked mode 2 there for consistency with the established pipeline.
- Refer to the dataset's official documentation first to check which mode is more reliable, do not trust the name of the data entry


### 7. Gripper convention
- Open/close polarity: is `1=open` or `1=closed`? Unified convention is **`1=open`, `0=closed`**.
  Use `invert_gripper_actions` (`1-x`) if the raw is `0=open`. bridge is already `1=open` → no inversion.
- Continuous vs binary: binarize the *action* gripper with `binarize_gripper_actions`.
- Absolute vs relative: if the action gripper is relative (`+1 close / −1 open`), map with `rel2abs_gripper_actions` (fractal).
- These helpers live in `oxe_utils/transform_utils.py`.

### 8. Units, ranges, sentinels
- Check ranges in the probe: rpy in radians (~[−π, π]) not degrees; xyz in meters; sane magnitudes.
- Watch for a **first-step all-zero action sentinel** or reset frames (bridge's first raw action is a sentinel; mode 2
  from state differences sidesteps it).

### 9. Joints, images, metadata
- If joint angles exist, add `joint_pos` (state) and `diff_joint_pos` (action). Otherwise omit them.
- Map image views to `image_obs_keys` `{primary, secondary, wrist}` in the config.
- Note `control_frequency` and `robot_type` for the config.

---

## Step 2 — Implement the transform

Add `def <name>_dataset_transform(trajectory)` to `oxe_utils/transforms.py`. The repo root is already on
`sys.path` there; use `from alignment import transforms_tf as align_tf` (already imported at the top — RLDS runs in
**TF graph mode**, so use the `_tf` backend and index tensors, never `tf.unstack` on unknown static dims).

Skeleton **template** — replace every `<...>` with the choice you established in Step 1 for *this*
dataset's schema (key names, slice indices, orientation source, gripper handling, and `e*` mode). The
control flow (build `R` → state dict → world/body deltas → assemble action → length bookkeeping) is the
same for every dataset; only the inputs differ.

```python
obs = trajectory["observation"]

# --- raw eef pose: adapt slices/keys to THIS schema (Step 0 features.json) ---
eef_xyz = obs["<pos_key>"][..., :3]

# Orientation: pick ONE source per Step 1.2/1.3, all roads lead to a rotation matrix R.
eef_rpy = obs["<rpy_key>"]                              # (a) raw is euler/rpy
R = align_tf.rpy_to_matrix(eef_rpy, extrinsic=<bool>)  #     <bool> = Step 1.3 result
# R = align_tf.quaternion_to_matrix(obs["<quat_key>"]) # (b) raw is quaternion (xyzw; reorder if wxyz)
# R = obs["<matrix_key>"]                               # (c) raw is already a 3x3 matrix
# eef_rpy = align_tf.matrix_to_rpy(R, extrinsic=<bool>)#     derive rpy from R for (b)/(c)

state = {
    "eef_xyz": eef_xyz,
    "eef_rpy": eef_rpy,
    "eef_rot6d": align_tf.matrix_to_rotation_6d(R),
    "gripper_state": <gripper_state>,                  # 1=open; invert_gripper_actions(...) if raw is 0=open
    # "joint_pos": obs["<joint_key>"],                 # ONLY if the dataset has joints (Step 1.9)
}

# --- choose e* (current pose e -> target pose e*), per Step 1.6 ---
# Mode 2 (ground-truth next): e=pose[t], e*=pose[t+1]; discard the last step (no successor).
R_cur, p_cur, R_tgt, p_tgt = R[:-1], eef_xyz[:-1], R[1:], eef_xyz[1:]
# Mode 1 (command): build the target from the raw command and KEEP all steps (no discard), e.g.
#   p_tgt = eef_xyz + <world_vector_cmd>;  R_tgt = <dR_world_cmd> @ R;  R_cur, p_cur = R, eef_xyz

world_R, world_p = align_tf.world_delta(R_cur, p_cur, R_tgt, p_tgt)
body_R,  body_p  = align_tf.relative_pose(R_cur, p_cur, R_tgt, p_tgt)
action = {
    "diff_eef_xyz":    world_p,
    "diff_eef_rpy":    align_tf.matrix_to_rpy(R_tgt, extrinsic=<bool>)
                       - align_tf.matrix_to_rpy(R_cur, extrinsic=<bool>),   # = rpy(e*) - rpy(e)
    "world_eef_xyz":   world_p,
    "world_eef_rot6d": align_tf.matrix_to_rotation_6d(world_R),
    "body_eef_xyz":    body_p,
    "body_eef_rot6d":  align_tf.matrix_to_rotation_6d(body_R),
    "gripper_state":   <gripper_action>,               # commanded gripper, per Step 1.7
    # "diff_joint_pos": <joint_tgt - joint_cur>,       # ONLY if joints
}

# Length bookkeeping. Mode 2: drop the last step everywhere so all fields are length T-1 and action[t]
# pairs with obs[t]. Mode 1: keep all steps (length T) -- delete the `[:-1]` slices below.
trajectory["state"] = {k: v[:-1] for k, v in state.items()}
trajectory["observation"] = {k: v[:-1] for k, v in trajectory["observation"].items()}
trajectory["action"] = action
trajectory["language_instruction"] = trajectory["language_instruction"][:-1]
return trajectory
```

> Concrete instantiations of this template: `bridge_orig_dataset_transform` (rpy source, mode 2) and
> `rt1_dataset_transform` (quaternion source, mode 1) in `oxe_utils/transforms.py`.

Key `alignment` functions (identical API across `transforms_numpy/_torch/_tf`):

- `rpy_to_matrix(rpy, extrinsic=)` / `matrix_to_rpy(matrix, extrinsic=)` — inverse pair
- `quaternion_to_matrix(q)` (xyzw) / `matrix_to_quaternion`
- `matrix_to_rotation_6d` / `rotation_6d_to_matrix`
- `relative_pose(R_cur, p_cur, R_tgt, p_tgt) -> (R_body, p_body)` — body-frame delta `R_cur^T R_tgt`, `R_cur^T (p_tgt-p_cur)`
- `world_delta(R_cur, p_cur, R_tgt, p_tgt) -> (R_world, p_world)` — `R_tgt R_cur^T`, `p_tgt-p_cur`
- `change_delta_frame`, `world_pose_from_model_delta`, `body_delta_from_model_delta` — for the deployment side

### Wire up the registries
- `oxe_utils/transforms.py`: register `"<dataset>": <name>_dataset_transform` in the transform dict at the bottom.
- `oxe_utils/configs.py`: add a **dict-form** config (not the legacy `StateEncoding`/`ActionEncoding` IntEnum):
  ```python
  "<dataset>": {
      "image_obs_keys": {"primary": ..., "secondary": ..., "wrist": ...},
      "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
      "state_encoding":  {"eef_xyz": 3, "eef_rot6d": 6, "gripper_state": 1},   # -> observation.state (eef_rpy stays a standalone column)
      "action_encoding": {"body_eef_xyz": 3, "body_eef_rot6d": 6, "gripper_state": 1},   # -> action
      "control_frequency": ...,
      "robot_type": ...,
  }
  ```
  `state_encoding`/`action_encoding` define the concatenation order/length of the out-of-box
  `observation.state` / `action` vectors. `openx_rlds.py` (`generate_features_from_raw` +
  `save_as_lerobot_dataset`) builds those two vectors from these dicts — no edit needed there per dataset.
- `oxe_utils/constants.py`: every key the transform emits must exist in `STATE_NAMES` / `ACTION_NAMES`
  (`generate_features_from_raw` looks each one up). Add any new key.

---

## Step 3 — Validate against real data (required)

Adapt `probe_template.py` into a test that runs the transform on 3–4 real episodes and asserts (target ~1e-7):

- **Lengths**: mode 2 → every state/action/observation/language field is `T-1`; mode 1 → all `T`.
- **State passthrough**: `eef_xyz`/`eef_rpy` equal the raw slices; `eef_rot6d` is orthonormal and reconstructs the same R.
- **Action consistency**: `world_eef_xyz == diff_eef_xyz`; `world_eef_rot6d == 6D(R_{t+1} R_t^T)` (mode 2) or `== 6D(dR_command)` (mode 1).
- **Body delta reconstructs `e*`**: `R_e @ body_R == R_{e*}` and `p_e + R_e @ body_p == p_{e*}`.
- **Schema plumbing**: every emitted key is in `STATE_NAMES`/`ACTION_NAMES`; `state_encoding`/`action_encoding` keys exist
  in the output; the concatenated `observation.state` / `action` dims sum as expected.

Run with the Python env that has TF + tfds installed (this project uses its `lerobot` conda env — find its
interpreter with `which python` after activating, or check the project setup; do not assume a fixed path).

Only report "done" once the probe passes on real episodes — convention bugs (quaternion order, extrinsic/intrinsic)
produce plausible-looking-but-wrong numbers and are caught **only** by these reconstruction checks.

---

## Reference

- Spec & SE(3) math: `design_of_state_and_action_space.md`
- Math utils: `alignment/transforms_{numpy,torch,tf}.py`
- Transforms & helpers: `openx2lerobot/oxe_utils/{transforms,transform_utils,configs,constants}.py`
- Worked examples: fractal (`rt1_dataset_transform`, mode 1), bridge/DROID (mode 2)
- Probe/validate template: `probe_template.py` (in this skill folder)
