# Design of State and Action Space

All poses are absolute end-effector poses in the **world (base) frame**:
`T_t = [[R_t, p_t], [0, 1]]`.

For single arm dataset, we provide states in the world frame, and provide actions both in the world frame and in the gripper frame. 

For ego-centric dataset, all poses and movements will be converted to the ego camera frame. 

Positions of the cameras (in the world frame for single arm and in the ego camera frame for ego-centric dataset) will be supplied if possible.  

## State Space

Per-step state fields:

| field | meaning |
|---|---|
| `eef_xyz` | EEF position `p_t` (3) |
| `eef_rpy` | EEF orientation as euler "XYZ" (3) |
<!-- | `eef_rot6d` | EEF orientation as 6D rotation (Zhou et al. 2019) (6). | -->
| `joint_position` | joint positions |
| `gripper_state` | gripper open/close |

A reference vector `observation.state` concatenates a per-dataset subset of these fields for out-of-box use.

## Action Space

Per-step delta paired with the observation at step `t`. 

| field | meaning |
|---|---|---|
| `world_eef_xyz` | `p_{t+1} - p_t` in the world frame |
| `world_eef_rpy` | `rpy_{t+1} - rpy_t` (componentwise) in the world frame |
| `world_eef_rot6d` | 6D of `R_{t+1} R_t^T` in the world frame |
| `body_eef_xyz` | `p_{t+1} - p_t` in the body frame: `R_t^T (p_{t+1} - p_t)` |
| `body_eef_rot6d` | 6D of `R_t^T R_{t+1}` in the body frame |
| `joint_position` | `joint_{t+1} - joint_t` |
| `gripper_state` | commanded gripper state |

Notes: The last step has no successor: zero translation, identity rotation.

A reference vector `action` concatenates a per-dataset subset of these fields for out-of-box use. Again, for single arm dataset, we will use body frame actions; for ego-centric, we will use ego-camera frame actions. 
