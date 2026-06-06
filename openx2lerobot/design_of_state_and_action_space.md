# Design of State and Action Space

All poses are absolute end-effector poses in the **world (base) frame**:
`T_t = [[R_t, p_t], [0, 1]]`.

## State Space

Per-step state fields:

| field | meaning |
|---|---|
| `eef_xyz` | EEF position `p_t` (3) |
| `eef_rpy` | EEF orientation as euler "XYZ" (3) |
| `eef_quat` | `eef_rpy` as quaternion `(x, y, z, w)` (4) |
| `eef_rot6d` | `eef_rpy` as 6D rotation (Zhou et al. 2019) (6) |
| `joint_position` | joint positions |
| `gripper_state` | gripper open/close |

Typically the data gives only one of `eef_rpy` / `eef_quat`; convert between
representations with `{euler, quaternion, rotation_6d}` converters in `transform_utils.py`.

A reference vector `state` concatenates a per-dataset subset of these fields for out-of-box use.

## Action Space

Per-step delta paired with the observation at step `t`. Frames are **mixed by design**:

| field | definition | frame |
|---|---|---|
| `eef_xyz` | `p_{t+1} - p_t` (literal difference) | world |
| `eef_rpy` | `rpy_{t+1} - rpy_t` (componentwise) | world |
| `eef_quat` | `quat_{t+1} - quat_t` (componentwise) | world |
| `eef_rot6d` | 6D of `R_t^T R_{t+1} = T_t^{-1} T_{t+1}` | body (gripper) |
| `joint_position` | `joint_{t+1} - joint_t` (literal difference) | — |

Notes:
- `eef_rpy` / `eef_quat` are **componentwise differences of coordinates**, not rotational deltas
  (not additive; kept only for reference). `eef_rot6d` is the true relative rotation.
- The last step has no successor: zero translation, identity rotation.

For every field above we also record the **commanded** delta (current state → commanded target),
prefixed `command_`. These differ from the realized deltas because the commanded target is not
necessarily reached.

A reference vector `action` concatenates a per-dataset subset of these fields for out-of-box use.
