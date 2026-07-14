---
name: encode-decode-robot-actions
description: Encode canonical processed robot states and targets into training actions, then decode model outputs into native controller commands for single-arm, dual-arm, or mobile robots. Use when integrating failure_rollout_data datasets into a model loader, defining action-vector layouts and normalization, implementing policy inference, changing action reference frames, or adapting predicted joint, EEF, gripper, and base actions to a real robot.
---

# Encode and Decode Robot Actions

Keep geometry, vector layout, normalization, and robot adaptation as separate reversible steps.

## Read the contract

Read `failure_rollout_data/dataset.md` and the matching converter README. Use `alignment/transforms_numpy.py` or its Torch/TensorFlow sibling instead of rewriting frame math.

Before assembling an action vector, declare:

- controller mode for every arm and the base;
- exact dimension order and size;
- whether the target is a recorded command or the next observed state;
- model action frame: canonical gripper, world, or a calibrated camera;
- rotation representation;
- normalization method and statistics;
- native axis alignments, units, limits, and gripper inverse mapping needed at deployment.

Reject data or inference when this metadata is missing or inconsistent.

## Reconstruct canonical poses

Read `eef_xyz` directly. Reshape row-major `eef_rot9d` from `[..., 9]` to `[..., 3, 3]`. Validate orthogonality and determinant before using the matrix.

For recorded commands, pair `state.*[t]` with `target.*[t]`. When commands are absent and ground-truth-next supervision is intended, pair `state.*[t]` with `state.*[t+1]`; drop or mask the final step instead of inventing a target.

## Encode one arm

Follow its controller mode.

### Joint-position control

```text
delta_q = target.joint_pos - state.joint_pos
```

Append `target.gripper_state` as the gripper action.

### Joint-velocity control

Use `target.joint_vel` directly. Do not difference or integrate it during loading. Append `target.gripper_state`.

### EEF-pose control

Given canonical current pose `(R, p)` and target pose `(R_target, p_target)`, encode motion in the current canonical gripper frame:

```text
R_delta = R.T @ R_target
p_delta = R.T @ (p_target - p)
```

Use `gripper_delta_pose`. Encode `R_delta` as Cosmos rot6d by default:

```text
[R11, R21, R31, R12, R22, R32]
```

Append translation, rotation, and `target.gripper_state` in the layout declared by the pipeline. If another rotation representation is selected, record it and use the same representation during decoding.

To train in another frame `c`, re-express both parts:

```text
R_delta_c = R_e_to_c @ R_delta @ R_e_to_c.T
p_delta_c = R_e_to_c @ p_delta
```

Use `change_delta_pose_frame` and calibrated frame orientations.

## Assemble multi-component actions

Encode each dual-arm side independently using its own state, target, calibration, and control mode. Use a fixed declared order; default to:

```text
left arm motion, left gripper, right arm motion, right gripper, base
```

Do not mirror one arm or share normalization statistics accidentally. A mobile-base velocity target is normally used directly. If the base uses position control, encode its delta according to the declared base frame and controller contract.

Keep mixed control modes in separate typed layouts or include an unambiguous mode selector. Never interpret a velocity block as a position delta.

## Normalize for the model

First compute the physical canonical action. Then apply the training pipeline's normalization per declared dimension. Fit statistics only on training data and save them with the checkpoint or policy configuration.

Do not normalize raw targets before geometric differencing. Do not recompute or reorder statistics at inference. Mask padded action-chunk steps so they do not affect statistics or loss.

## Decode a model output

Reverse the training path in strict order:

1. Select the valid predicted step or action chunk.
2. Undo action normalization.
3. Split the vector using the saved layout.
4. Decode rot6d or the configured rotation representation to a valid matrix.
5. Convert the predicted delta from the model frame to the frame required by the controller.
6. Reconstruct the controller command.
7. Undo canonical-to-native mappings, restore units, and enforce safety limits.

Use the same robot state snapshot that conditioned the prediction. Stale state changes the meaning of relative commands.

## Reconstruct controller commands

### Joint controllers

For a native absolute joint-position controller:

```text
q_command = q_current + delta_q
```

For a joint-velocity controller, send the decoded desired velocity directly after restoring native units and limits.

### Cartesian controllers

If the controller wants an absolute canonical world target and the model emitted a delta in frame `c`, use `world_pose_from_model_delta`. It changes the delta to world coordinates and computes:

```text
R_target = R_delta_world @ R_current
p_target = p_current + p_delta_world
```

If the controller wants a relative canonical gripper-frame command, use `gripper_delta_from_model_delta`.

Convert an absolute canonical target back to native frames using the converter's saved alignments:

```text
p_raw_target = R_world_align.T @ p_target
R_raw_target = R_world_align.T @ R_target @ R_gripper_align
```

For a relative command expected in the native gripper frame, let `A = R_gripper_align` and compute:

```text
p_delta_raw = A.T @ p_delta
R_delta_raw = A.T @ R_delta @ A
```

Serialize the result in the controller's quaternion order, Euler convention, matrix layout, or delta convention only after frame conversion.

### Gripper and base

Apply the exact inverse of the dataset adapter's gripper mapping. Restore native width, range, polarity, binary threshold, and units; a generic multiplication is not sufficient for every robot.

Decode each base block with its declared mode and frame. Do not rotate base velocity using an EEF alignment.

## Guard inference

Before sending a command, verify:

- dimensions, ordering, mode, and normalization metadata match the checkpoint;
- all values are finite;
- decoded rotations are proper and sufficiently orthogonal;
- joint, velocity, workspace, rotation-step, gripper, and base limits are satisfied;
- the state timestamp is recent enough;
- dual-arm commands retain their correct side and simultaneous timing;
- clipping policy is explicit and logged.

Test each adapter with round trips: canonical state/target to action, action back to target, canonical target to native target, and native target through conversion back to canonical. Include identity, single-axis translation, single-axis rotation, near-180-degree rotation, gripper endpoints, both arms, and base commands.
