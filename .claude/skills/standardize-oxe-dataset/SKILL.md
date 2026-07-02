---
name: standardize-oxe-dataset
description: Standardize an Open-X-Embodiment / RLDS robot dataset into this repo's unified state/action space (eef_xyz, eef_rpy, eef_rot6d, world/gripper/diff eef deltas, gripper). Use when adding a new dataset to openx2lerobot, writing or fixing a per-dataset transform in oxe_utils/transforms.py, or when asked to "standardize", "add", or "convert" an OXE/RLDS dataset. Covers what to verify about the raw schema (rotation convention, frames, command-vs-ground-truth, gripper, camera extrinsics) and how to implement + empirically validate the transform.
---

# Standardizing an OXE/RLDS dataset

Goal: map one raw RLDS dataset onto the unified space defined in
`design_of_state_and_action_space.md`, using the backend-agnostic math in `alignment/`,
and **prove the mapping is correct against real data** before committing.

The unified output (per dataset, subset selected by config):

- **state** (canonical world frame): `eef_xyz` (3), `eef_rpy` (3), `eef_rot6d` (6), `joint_pos` (7, if any), `gripper_state` (1, `1=open`)
- **action** (per-step delta `e → e*`): `gripper_eef_xyz`/`gripper_eef_rot6d` (canonical, gripper frame), `world_eef_xyz`/`world_eef_rot6d`, `diff_eef_xyz`, `diff_eef_rpy`, `diff_joint_pos`, `gripper_state`. These default fields always target the **ground-truth next pose**. When the raw data carries a real command, the same six eef fields are emitted **again** with a `_command` suffix (e.g. `action.world_eef_rot6d_command`) — *in addition to*, not instead of, the defaults. The trajectory keeps **all `T` steps**: the final step (no successor) gets a **no-op** default action (identity rotation, zero translation), it is **not discarded**.

`eef_rot6d` is Zhou et al. 6D (first two rows of R). Deltas are stored in the **gripper frame** (the design doc's
"gripper frame") as the canonical form; any other frame is recovered at load time from `R_e^c`. See the design doc
for the SE(3) math and the deployment / revert-to-native steps.

**Canonical frames — the alignment target (design doc §Canonical Frames + §Preprocessing).** Everything is expressed
in one fixed *right-handed* convention so the same dimension means the same motion across datasets:
- **World**: x forward, y left, z up — Forward-Left-Up (the common ROS robot-base convention).
- **Gripper**: OpenCV — z = approach (forward), x right, y down (camera-aligned).

A dataset whose **native axes differ** must be mapped in by a **constant proper rotation** (signed permutation, `det = +1`)
applied to `eef_xyz` and `R` *before* any rot6d/delta is computed — a `det = −1` map means you mis-identified handedness.
Use the `alignment` helpers: build the rotation with `axis_alignment_matrix(x_to, y_to, z_to)` (each arg says where the
native x/y/z axis points in canonical coords, e.g. native Forward-Right-Down world → `axis_alignment_matrix("x","-y","-z")`),
then `align_axis(R, p, R_world_align, R_gripper_align=None)` re-bases every pose into the canonical frames. **World
align is identity for every dataset so far** (fractal / bridge / DROID share the FLU base frame, so `R_world_align = I`),
but the **gripper** frame is usually *not* canonical OpenCV and needs a per-robot `R_gripper_align` (Franka
`_GRIPPER_ALIGN_FRANKA`, WidowX `_GRIPPER_ALIGN_WIDOWX`, Google `_GRIPPER_ALIGN_GOOGLE = I`). The transforms apply both
via the local **`_to_canonical(R_native, p_native, R_gripper_align)`** wrapper (= `align_axis` with world identity). For a
new dataset, identify both alignments in Step 1, apply `_to_canonical` right after building `R_native`/`eef_xyz` in Step 2,
then verify the round-trip (Step 3).

**LeRobot output layout.** Each named field above is written as its own feature: `state.<key>`
(e.g. `state.eef_xyz`) and `action.<key>` (e.g. `action.gripper_eef_xyz`). The **out-of-box ready-to-train
vectors** are the per-dataset concatenations selected by `state_encoding` / `action_encoding`, written as
**`observation.state`** and **`action`**. `eef_rpy` (and any field not in the encoding) remains available as
a standalone `state.*` column but is not part of the `observation.state` vector.

> The transforms share helpers in `openx2lerobot/oxe_utils/transforms.py`: **`_next_step_delta_fields(R, p)`**
> builds the default ground-truth-next action (all `T` steps, no-op last); **`_eef_delta_fields(R, p, R_cmd, p_cmd,
> suffix="_command")`** adds the parallel command set; **`_diff_with_dummy_last(x)`** is the forward difference with a
> zero last step; **`_to_canonical(R_native, p, R_gripper_align)`** maps native → canonical frames. Reference
> implementations: `bridge_orig_dataset_transform` (GT-next only), `rt1_dataset_transform` (fractal, GT-next +
> `_command` from `world_vector`/`rotation_delta`), `droid_transform` (GT-next + `_command` from `action_dict`).

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
- The unified state is the **canonical world frame** (Forward-Left-Up). Confirm the raw eef pose is world/base-relative
  (it usually is for OXE), not camera-relative. If camera-relative, transform to world using the extrinsics first.
- Also check the **axis directions** of that native frame against canonical FLU (x-fwd / y-left / z-up). If they differ
  (e.g. z-down, or x-right), build the constant signed-permutation rotation `R_{w'→w}` (`det = +1`) and apply it to
  `eef_xyz` and `R` before computing rot6d/deltas (see the canonical-frames note above). FLU-compatible base frames
  (fractal/bridge/DROID) need no rotation.
- Refer to the dataset's official documentation first, do not trust the name of the data entry

### 5. Frame of the **action**
- In which frame is the raw action expressed: world/base, gripper, or camera?
- fractal's `world_vector`/`rotation_delta` are **base/world-frame** commands (verified: world `R_{t+1}R_t^T` ≈
  `rotation_delta`, ~3× the per-step achieved motion). Build `e*` in that frame, then derive gripper/world/diff via `alignment`.
- If the action is already a gripper-frame delta, it maps to `gripper_eef_*` more directly — still verify.
- Refer to the dataset's official documentation first, do not trust the name of the data entry


### 6. Target `e*`: default ground-truth-next (always) + `*_command` extras (when available)
- **Default — every dataset:** `e* =` the **ground-truth next pose**, i.e. the per-step delta `obs[t] → obs[t+1]`.
  Always emitted as the unsuffixed `world_/gripper_/diff_*` fields, built by `_next_step_delta_fields(R, p)`.
- **End of trajectory — append a no-op, do NOT discard.** The last step has no successor, so its default action is a
  **dummy no-op**: identity rotation (`_IDENTITY_ROT6D`) and zero translation (`diff_joint_pos`'s zero last comes from
  `_diff_with_dummy_last`). Every field — state, observation, action, language — stays length `T`, and `action[t]` pairs
  with `obs[t]`. (This is the current behavior; it replaces the old "drop the last step → length `T-1`" approach, and as a
  bonus makes 1-step episodes valid instead of producing an empty buffer.)
- **`*_command` extras — only when a real command exists.** If the raw data carries a *meaningful* desired pose (the
  target sent to the controller), emit the SAME six eef fields **again** with a `_command` suffix via
  `_eef_delta_fields(R, p, R_cmd, p_cmd, suffix="_command")`, *in addition to* the defaults. The command exists at every
  step (including the last), so the `_command` fields are naturally length `T` with **no** dummy — only the GT-next default
  needs the no-op last.
- **Is the raw action a real command?** Compare it to the achieved state difference: if `raw_action ≈ state[t+1]−state[t]`
  it's just a finite difference → the default already captures it, add no `_command` (bridge is here — its raw action is the
  unreliable one the repo relabels to the state delta, so it emits GT-next only). If it differs materially → add the
  `_command` set (fractal's `world_vector`/`rotation_delta` is ≈3× the achieved motion; DROID's `action_dict` is the teleop target).
- Refer to the dataset's official documentation first to decide whether a command field is meaningful; do not trust the name of the data entry.


### 7. Gripper convention
- Open/close polarity: is `1=open` or `1=closed`? Unified convention is **`1=open`, `0=closed`**.
  Use `invert_gripper_actions` (`1-x`) if the raw is `0=open`. bridge is already `1=open` → no inversion.
- Continuous vs binary: binarize the *action* gripper with `binarize_gripper_actions`.
- Absolute vs relative: if the action gripper is relative (`+1 close / −1 open`), map with `rel2abs_gripper_actions` (fractal).
- These helpers live in `oxe_utils/transform_utils.py`.

### 8. Units, ranges, sentinels
- Check ranges in the probe: rpy in radians (~[−π, π]) not degrees; xyz in meters; sane magnitudes.
- Watch for a **first-step all-zero action sentinel** or reset frames (bridge's first raw action is a sentinel; the
  GT-next default, built from state differences rather than the raw action, sidesteps it).

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
dataset's schema (key names, slice indices, orientation source, gripper align, gripper handling, and whether a
`*_command` set applies). The control flow (build `R_native` → `_to_canonical` → state dict → default GT-next action
via `_next_step_delta_fields` → optional `*_command` set → **no truncation**) is the same for every dataset; only the
inputs differ.

```python
obs = trajectory["observation"]

# --- raw eef pose: adapt slices/keys to THIS schema (Step 0 features.json) ---
eef_xyz = obs["<pos_key>"][..., :3]

# Orientation: pick ONE source per Step 1.2/1.3 -> a NATIVE rotation matrix R_native.
R_native = align_tf.rpy_to_matrix(obs["<rpy_key>"], extrinsic=<bool>)  # (a) euler/rpy; <bool> = Step 1.3
# R_native = align_tf.quaternion_to_matrix(obs["<quat_key>"])          # (b) quaternion (xyzw; reorder if wxyz)
# R_native = obs["<matrix_key>"]                                       # (c) already a 3x3 matrix

# --- map native -> canonical frames (Step 1.4/1.5): world align identity, gripper align per-robot ---
R, p = _to_canonical(R_native, eef_xyz, <_GRIPPER_ALIGN_ROBOT>)        # e.g. _GRIPPER_ALIGN_FRANKA (Google = I)

state = {
    "eef_xyz": p,
    "eef_rpy": align_tf.matrix_to_rpy(R, extrinsic=True),
    "eef_rot6d": align_tf.matrix_to_rotation_6d(R),
    "gripper_state": <gripper_state>,                  # 1=open; invert_gripper_actions(...) if raw is 0=open
    # "joint_pos": obs["<joint_key>"],                 # ONLY if the dataset has joints (Step 1.9)
}

# --- DEFAULT action: ground-truth next pose over ALL T steps, with a no-op LAST step (Step 1.6) ---
action = _next_step_delta_fields(R, p)                 # world_/gripper_/diff_eef_* ; length T; final step = no-op
# action["diff_joint_pos"] = _diff_with_dummy_last(joint)        # ONLY if joints

# --- OPTIONAL *_command set: include ONLY if the raw data has a meaningful command (Step 1.6) ---
# Build the commanded target e* and map it through _to_canonical the SAME way, then emit the parallel set:
#   R_cmd, p_cmd = _to_canonical(<dR_world_cmd> @ R_native, eef_xyz + <world_vector_cmd>, <_GRIPPER_ALIGN_ROBOT>)
#   action.update(_eef_delta_fields(R, p, R_cmd, p_cmd, suffix="_command"))   # naturally length T (cmd at every step)
#   action["diff_joint_pos_command"] = <joint_cmd - joint>      # ONLY if joints + command

action["gripper_state"] = <gripper_action>             # commanded gripper, 1=open, per Step 1.7

# No truncation: state / observation / action / language all stay length T -- the no-op last step (above)
# already handles the successor-less final frame. Do NOT slice [:-1].
trajectory["state"], trajectory["action"] = state, action
trajectory["language_instruction"] = <language length-T>   # e.g. trajectory["language_instruction"] (unchanged)
return trajectory
```

> Concrete instantiations: `bridge_orig_dataset_transform` (rpy source, GT-next only),
> `rt1_dataset_transform` (quaternion source, GT-next + `_command`), and `droid_transform`
> (rpy + joints, GT-next + `_command`) in `oxe_utils/transforms.py`.

Key `alignment` functions (identical API across `transforms_numpy/_torch/_tf`):

- `rpy_to_matrix(rpy, extrinsic=)` / `matrix_to_rpy(matrix, extrinsic=)` — inverse pair
- `quaternion_to_matrix(q)` (xyzw) / `matrix_to_quaternion`
- `matrix_to_rotation_6d` / `rotation_6d_to_matrix`
- `axis_alignment_matrix(x_to, y_to, z_to) -> R` — constant proper rotation mapping native axes → canonical (raises on `det≠+1`)
- `align_axis(R, p, R_world_align, R_gripper_align=None) -> (R, p)` — re-base native poses into canonical frames (preprocessing)
- `gripper_delta_pose(R_cur, p_cur, R_tgt, p_tgt) -> (R_gripper, p_gripper)` — gripper-frame delta `R_cur^T R_tgt`, `R_cur^T (p_tgt-p_cur)`
- `world_delta_pose(R_cur, p_cur, R_tgt, p_tgt) -> (R_world, p_world)` — `R_tgt R_cur^T`, `p_tgt-p_cur`
- `change_delta_pose_frame`, `world_pose_from_model_delta` (deploy Case 1, absolute world target),
  `gripper_delta_from_model_delta` (deploy Case 2, gripper relative) — the deployment side (design doc §Deploying).
  Note these take no axis-alignment rotations: the **revert-to-native** step (canonical → the controller's native
  frames) has no helper yet and must be applied separately if the controller's frame ≠ canonical.

Higher-level helpers in `oxe_utils/transforms.py` (build on the above — **prefer these** in a new transform):

- `_to_canonical(R_native, p_native, R_gripper_align)` — `align_axis` with world identity; native → canonical frames
- `_next_step_delta_fields(R, p)` — the default GT-next action: all six eef fields, length `T`, **no-op last step**
- `_eef_delta_fields(R_cur, p_cur, R_tgt, p_tgt, suffix="")` — the six eef fields for one `e→e*` pair; pass `suffix="_command"` for the command set
- `_diff_with_dummy_last(x)` — forward difference `x[t+1]-x[t]` with a zero last step (for `diff_joint_pos`)

### Wire up the registries
- `oxe_utils/transforms.py`: register `"<dataset>": <name>_dataset_transform` in the transform dict at the bottom.
- `oxe_utils/configs.py`: add a **dict-form** config (not the legacy `StateEncoding`/`ActionEncoding` IntEnum):
  ```python
  "<dataset>": {
      "image_obs_keys": {"primary": ..., "secondary": ..., "wrist": ...},
      "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
      "state_encoding":  {"eef_xyz": 3, "eef_rot6d": 6, "gripper_state": 1},   # -> observation.state (eef_rpy stays a standalone column)
      "action_encoding": {"gripper_eef_xyz": 3, "gripper_eef_rot6d": 6, "gripper_state": 1},   # -> action
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

- **Lengths**: EVERY field — state / action / observation / language — is length `T` (no discard); `action[t]` pairs with `obs[t]`.
- **No-op last step** (default action): final step is identity rotation (`world_eef_rot6d[-1] == gripper_eef_rot6d[-1] == [1,0,0,0,1,0]`)
  and zero translation (`*_eef_xyz[-1] == 0`, `diff_eef_rpy[-1] == 0`, and `diff_joint_pos[-1] == 0` if joints).
- **State passthrough**: `eef_xyz`/`eef_rpy` equal the raw slices (after `_to_canonical`); `eef_rot6d` is orthonormal and reconstructs the same R.
- **Default action consistency (steps `0..T-2`)**: `world_eef_xyz == diff_eef_xyz`; `world_eef_rot6d == 6D(R_{t+1} R_t^T)`.
- **`_command` set (only if emitted)**: present for the same six eef keys with `_command` suffix, length `T` (no dummy),
  and `world_eef_rot6d_command == 6D(dR_command)`; spot-check it **differs** from the default where the command is real.
- **Gripper-frame delta reconstructs `e*`** (steps `0..T-2`): `R_e @ gripper_R == R_{e*}` and `p_e + R_e @ gripper_p == p_{e*}`.
- **Schema plumbing**: every emitted key is in `STATE_NAMES`/`ACTION_NAMES`; `state_encoding`/`action_encoding` keys exist
  in the output; the concatenated `observation.state` / `action` dims sum as expected.

Run with the Python env that has TF + tfds installed (this project uses its `lerobot` conda env — find its
interpreter with `which python` after activating, or check the project setup; do not assume a fixed path).

Only report "done" once the probe passes on real episodes — convention bugs (quaternion order, extrinsic/intrinsic)
produce plausible-looking-but-wrong numbers and are caught **only** by these reconstruction checks.

---

## Reference

- Spec & SE(3) math: `design_of_state_and_action_space.md` (§Canonical Frames → §Preprocessing → §State/Action → §Deploying)
- Math utils: `alignment/transforms_{numpy,torch,tf}.py`
- Transforms & helpers: `openx2lerobot/oxe_utils/{transforms,transform_utils,configs,constants}.py`
- Worked examples: fractal (`rt1_dataset_transform`, GT-next + `_command`), bridge (`bridge_orig_dataset_transform`, GT-next only), DROID (`droid_transform`, GT-next + `_command`)
- Probe/validate template: `probe_template.py` (in this skill folder)
