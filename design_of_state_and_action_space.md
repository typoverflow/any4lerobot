# Design of State and Action Space

Notations:
+ $T_t = [R_t, p_t; 0, 1] \in SE(3)$ is the end-effector pose at time step $t$, with rotation $R_t \in SO(3)$ and translation $p_t \in \mathbb{R}^3$.
+ $T_{e,t}^w$ denotes the pose of frame $e$ expressed in frame $w$ at time step $t$ (likewise $R_e^w$ for rotation).
+ We assume the camera pose is known at every frame.

Every state and action field below is defined on poses that have **already** been mapped into the canonical frames. We therefore describe the per-dataset preprocessing (rotation recovery + axis alignment) first, then the spaces built on top of it.

## Canonical Frames

One right-handed convention is fixed for all datasets, so that the same dimension means the same motion everywhere. **The ultimate goal is to make similar actions have visually similar motions across datasets.**
+ World frame: x forward, y left, z up — Forward-Left-Up (ROS body convention).
+ Gripper frame: OpenCV convention — z is the approaching direction (forward), x right, y down. Aligned with the camera.

Both must be right-handed, $\det[\hat x\,\hat y\,\hat z]=+1$. With x forward and y left this forces z **up**; pairing y-left with z-down gives $\det=-1$, a left-handed frame that is not a rotation and cannot be encoded by rot6d.

## Preprocessing (per raw dataset)

**Rotation recovery.** Raw orientation is usually roll/pitch/yaw. Reconstruct $R$ respecting the dataset's extrinsic-vs-intrinsic convention and axis order; guard it with a round-trip test (`rpy → R → rot6d → R`) against known poses.

**Axis alignment.** Let the dataset's native frames be $w'$ (world) and $e'$ (gripper). Matching axis directions yields **constant** proper rotations (fixed signed-permutation matrices)
$$
R_{w'}^w,\; R_{e'}^e \in SO(3), \qquad \det R_{w'}^w = \det R_{e'}^e = +1 .
$$
Every pose is then mapped to canonical frames:
$$
p_t^w = R_{w'}^w\, p_t^{w'}, \qquad
R_{e,t}^w = R_{w'}^w\, R_{e',t}^{w'}\, R_e^{e'}, \qquad R_e^{e'} = (R_{e'}^e)^{-1} ,
$$
re-basing the world reference on the left and relabeling the gripper axes on the right; joint angles are unchanged. Because rot6d and any $SO(3)$ delta cannot encode a reflection, a forced $\det=-1$ map means the canonical frame is left-handed and must be redefined.

## State Space

All state fields are expressed in the (aligned) world frame.

| field           | meaning                         |
| --------------- | ------------------------------- |
| `joint_pos`     | (7) joint angles                |
| `eef_xyz`       | (3) EEF translation $p_t$       |
| `eef_rot6d`     | (6) EEF orientation $R_{e,t}^w$ |
| `gripper_state` | (1) gripper open/close          |

`observation.state` concatenates a per-dataset subset of these fields — selected by the dataset's `state_encoding` — into the out-of-box, ready-to-train state vector.

## Action Space

Every action field is a **per-step delta**. The current EEF orientation is $R_e^w$ and the target is $R_{e*}^w$; we store deltas in the **gripper frame** and convert to any other frame at load time.

**Rotation.** The gripper-frame delta rotation is
$$
R_{e\to e*}^e = (R_e^w)^{-1} R_{e*}^w = R_w^e R_{e*}^w = R_{e*}^e ,
$$
which right-multiplies onto the current pose ($R_{e*}^w = R_e^w\,R_{e\to e*}^e$). The world-frame delta is the conjugate, and the same conjugation maps the delta into any frame $c$ given $R_e^c$:
$$
R_{e\to e*}^w = R_{e*}^w (R_e^w)^{-1} = R_e^w R_{e\to e*}^e R_w^e , \qquad
R_{e\to e*}^c = R_e^c\, R_{e\to e*}^e\, R_c^e .
$$

**Translation.** With the world-frame difference $p_{e\to e*}^w = p_{e*}^w - p_e^w$,
$$
p_{e\to e*}^e = R_w^e\, p_{e\to e*}^w = (R_e^w)^{-1} p_{e\to e*}^w , \qquad
p_{e\to e*}^c = R_e^c\, p_{e\to e*}^e .
$$

Storing the gripper-frame pair $(R_{e\to e*}^e,\, p_{e\to e*}^e)$ is sufficient — it is exactly the rotation/translation of the relative transform $T_{e*}^e = (T_e^w)^{-1} T_{e*}^w$, and any frame is recovered from $R_e^c$ at load time. We also keep the world-frame delta and the raw finite differences for out-of-box convenience.

| field               | meaning                                                                                              |
| ------------------- | ---------------------------------------------------------------------------------------------------- |
| `world_eef_xyz`     | delta translation in world frame, $p_{e\to e*}^w$                                                    |
| `world_eef_rot6d`   | delta rotation in world frame, $R_{e\to e*}^w$                                                       |
| `gripper_eef_xyz`   | delta translation in gripper frame, $p_{e\to e*}^e$                                                  |
| `gripper_eef_rot6d` | delta rotation in gripper frame, $R_{e\to e*}^e$                                                     |
| `diff_eef_xyz`      | raw world-frame translation difference $p_{e*}^w - p_e^w$ (numerically identical to `world_eef_xyz`) |
| `diff_joint_pos`    | joint-space delta, target minus current $q_{e*} - q_e$                                               |
| `gripper_state`     | commanded gripper state, 0 closed / 1 open                                                           |

The target $e^*$ has two modes. For every dataset, we can calculate the use the next ground truth state as the target, that's the default mode. If the raw data carries a meaningful desired pose (e.g. the teleop command sent to the controller), add another set of entries (e.g. `world_eef_rot6d_command`) with the `*_command` suffix. 

**End of trajectory.** The default ground-truth-next target is undefined at the final step (it has no successor). Rather than discarding that step, we append a **no-op** action there — identity rotation ($R_{e\to e*}^e = I$, encoded `*_eef_rot6d = [1,0,0,0,1,0]`) and zero translation (`*_eef_xyz`, `diff_eef_rpy`, `diff_joint_pos` all zero) — so every per-frame field keeps its full length $T$ and `action[t]` stays aligned with the state/observation at step $t$. A `*_command` target, when present, is defined at every step (including the last) and needs no such padding.

`action` (the top-level LeRobot field) concatenates a per-dataset subset of these fields — selected by the dataset's `action_encoding` — into the out-of-box, ready-to-train action vector.

## Deploying the Model Output to a Controller

The model is trained on deltas in some frame $c$ and outputs $(R_{e\to e*}^c,\, p_{e\to e*}^c)$ plus the gripper command. At inference we know the current EEF pose $(R_e^w,\, p_e^w)$ and the frame-$c$ orientation $R_c^w$ (e.g. from camera extrinsics), so
$$
R_e^c = R_w^c R_e^w = (R_c^w)^{-1} R_e^w , \qquad R_c^e = (R_e^c)^{-1} .
$$

**Map to canonical.** First recover the gripper-frame relative transform — the single sufficient object (see Action Space) — in the canonical frames:
$$
R_{e\to e*}^e = R_c^e\, R_{e\to e*}^c\, R_e^c , \qquad
p_{e\to e*}^e = R_c^e\, p_{e\to e*}^c .
$$
(When the model is trained directly in the gripper frame, $c = e$ and $R_e^c = I$, so this step is the identity.)

**Revert to native.** The controller speaks its **native** frames $w',e'$, which differ from canonical only by the constant alignment rotations; we undo them with their inverses $R_w^{w'} = (R_{w'}^w)^{-1}$, $R_e^{e'} = (R_{e'}^e)^{-1}$. The gripper command passes through unchanged.

- *Case 1 — controller wants the absolute target pose* $(R_{e'*}^{w'},\, p_{e'*}^{w'})$. Compose the relative transform onto the current pose (gripper-frame delta right-multiplies), then revert:
$$
R_{e*}^w = R_e^w\, R_{e\to e*}^e , \qquad
p_{e*}^w = p_e^w + R_e^w\, p_{e\to e*}^e ,
$$
$$
R_{e'*}^{w'} = R_w^{w'}\, R_{e*}^w\, R_{e'}^e , \qquad
p_{e'*}^{w'} = R_w^{w'}\, p_{e*}^w .
$$

- *Case 2 — controller wants the relative motion* $(R_{e'\to e'*}^{e'},\, p_{e'\to e'*}^{e'})$. Only the gripper relabel applies:
$$
R_{e'\to e'*}^{e'} = R_e^{e'}\, R_{e\to e*}^e\, R_{e'}^e , \qquad
p_{e'\to e'*}^{e'} = R_e^{e'}\, p_{e\to e*}^e .
$$

Since the revert is the exact inverse of alignment, running a dataset's own native poses through with $c=e$ round-trips to the identity.
