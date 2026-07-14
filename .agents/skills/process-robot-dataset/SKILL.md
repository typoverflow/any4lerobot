---
name: process-robot-dataset
description: Convert a raw single-arm, dual-arm, or mobile-manipulator dataset into this repository's canonical raw_state, raw_target, state, and target feature groups. Use when inspecting a new robot dataset, implementing or reviewing a failure_rollout_data converter, aligning world and end-effector frames, normalizing grippers, defining LeRobot features, or validating converted episodes against failure_rollout_data/dataset.md.
---

# Process a Robot Dataset

Convert data without changing its physical meaning. The same canonical field must describe the same motion for every robot.

## Read the contract

Read `failure_rollout_data/dataset.md` before editing a converter. Use `alignment/transforms_numpy.py` or its Torch/TensorFlow sibling for rotation and frame math. Treat dataset-specific READMEs and source metadata as evidence, not as replacements for the contract.

## Inspect the source

Determine these facts before writing output:

- sampling timestamps and desired output FPS;
- single arm, dual arm, and mobile-base components;
- state fields, command fields, images, units, and missing values;
- controller mode for each component: joint position, joint velocity, absolute EEF pose, or relative EEF pose;
- quaternion order, Euler convention, and whether EEF poses come from measurements or FK;
- native world and EEF axis directions;
- gripper units, range, polarity, and whether commands differ from observations;
- whether a source action is an absolute target or a delta from the current state.

Do not guess silently. Record uncertain conventions in the dataset README and block full conversion when they affect physical meaning.

## Build native groups

Split source vectors by meaning. Preserve source values in `raw_state.*` and `raw_target.*`; do not align, normalize, invert, smooth, or change units in these groups.

Use fields such as `raw_state.joint_pos`, `raw_state.joint_vel`, `raw_state.eef_xyz`, one native EEF rotation representation, and `raw_state.gripper_state`. Use matching `raw_target.*` fields for commands actually sent to the controller.

If the source command is a delta, compose it with the native current state and store the resulting absolute target in `raw_target.*`. Keep joint-velocity targets absolute; do not integrate them.

Omit unavailable fields. If the dataset has no recorded command, omit both `raw_target.*` and `target.*`. Use the next canonical state as a training target only at load time.

## Convert canonical poses

Use right-handed canonical frames:

- world: FLU, with x forward, y left, z up;
- EEF: OpenCV, with x right, y down, z along the approach direction.

Define constant proper rotations `R_world_align = R_{w'}^w` and `R_gripper_align = R_{e'}^e`. Require both determinants to be `+1`. For a native absolute pose `(R_raw, p_raw)`, compute:

```text
p = R_world_align @ p_raw
R = R_world_align @ R_raw @ R_gripper_align.T
```

Use `align_axis`; do not reproduce the frame math ad hoc. Joint values are frame-independent.

Store the canonical orientation as `eef_rot9d`: flatten the complete `3 x 3` matrix in row-major order:

```text
[R11, R12, R13, R21, R22, R23, R31, R32, R33]
```

When a raw field, debug field, or model interface uses rot6d, use the Cosmos convention: concatenate the first two columns:

```text
[R11, R21, R31, R12, R22, R32]
```

## Build canonical groups

Create `state.*` from `raw_state.*` and `target.*` from `raw_target.*` using identical pose conversion rules.

- Copy joint position and velocity without frame alignment.
- Store canonical position in `eef_xyz` and orientation in `eef_rot9d`.
- Map gripper state to `[0, 1]`, where `0` is closed and `1` is open. Clip only when the adapter's physical range justifies clipping.
- Keep canonical xyz values in meters and angles in radians unless the dataset contract explicitly says otherwise.

Preserve the commanded target rather than replacing it with the achieved next state. Do not store a monolithic training action; derive it during loading.

## Name robot components

For a single arm, use direct names such as `state.eef_xyz`.

For two arms, apply `left_` and `right_` inside each group:

```text
state.left_joint_pos
state.left_eef_xyz
state.left_eef_rot9d
state.right_joint_pos
state.right_eef_xyz
state.right_eef_rot9d
```

Calibrate and validate each arm independently. Share an alignment only when the kinematics and camera/gripper mounting justify it.

Use `base_` for mobile-base fields, for example `target.base_vel`. State the base control mode explicitly. Do not apply arm EEF alignment to base commands.

## Preserve timing and provenance

Choose one episode timeline. Resample each stream according to its semantics: images and discrete commands normally use nearest-neighbor selection; continuous state may use interpolation. Never interpolate categorical values. Keep state, target, and images synchronized.

Store task, success/failure, source episode identity, control mode, camera availability, and quality-control decisions in features or metadata as appropriate.

## Validate before full conversion

Run conversion on several short episodes and verify:

- feature keys, shapes, dtypes, names, and episode lengths match emitted arrays;
- timestamps are monotonic and all streams have equal output length;
- rotation matrices satisfy `R.T @ R ~= I` and `det(R) ~= +1`;
- rot9d reshapes back to the intended matrix;
- gripper endpoints and polarity match video or source documentation;
- native-to-canonical alignment matches known robot geometry;
- target fields represent commands, not observations;
- joint and EEF targets agree through FK when both exist;
- dual-arm fields never swap sides and base fields use their own control mode;
- missing or corrupt streams are handled explicitly and logged.

Document source conventions, alignment matrices, normalization constants, rejected episodes, and unresolved risks in the converter README. Convert the full dataset only after these checks pass.
