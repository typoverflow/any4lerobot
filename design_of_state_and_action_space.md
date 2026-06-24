# Design of State and Action Space

Notations:
+ $T_t = [R_t, p_t; 0, 1] \in SE(3)$ is the end-effector pose at time step $t$, with rotation $R_t \in SO(3)$ and translation $p_t \in \mathbb{R}^3$.
+ $T_{e,t}^w$ denotes the pose of frame $e$ expressed in frame $w$ at time step $t$ (likewise $R_e^w$ for rotation).
+ We assume the camera pose is known at every frame.

## State Space

All state fields are expressed in the world frame.

| field           | meaning                         |
| --------------- | ------------------------------- |
| `joint_pos`     | (7) joint angles                |
| `eef_xyz`       | (3) EEF translation $p_t$       |
| `eef_rot6d`     | (6) EEF orientation $R_{e,t}^w$ |
| `gripper_state` | (1) gripper open/close          |

`eef_rot6d` is converted from the original dataset's roll/pitch/yaw. The conversion must respect the dataset's extrinsic-vs-intrinsic convention and axis order; guard it with a round-trip test (`rpy → R → rot6d → R`) against known poses.

`state.ref_state` concatenates a per-dataset subset of these fields for out-of-box use.

## Action Space

Every action field is a **per-step delta**. The current EEF orientation is $R_e^w$ and the target is $R_{e*}^w$; we store deltas in the **body frame** and convert to any other frame at load time.

**Rotation.** The body-frame delta rotation is
$$
R_{e\to e*}^e = (R_e^w)^{-1} R_{e*}^w = R_w^e R_{e*}^w = R_{e*}^e ,
$$
which right-multiplies onto the current pose ($R_{e*}^w = R_e^w\,R_{e\to e*}^e$). The world-frame delta is the conjugate
$$
R_{e\to e*}^w = R_{e*}^w (R_e^w)^{-1} = R_e^w R_{e\to e*}^e R_w^e ,
$$
and the same conjugation maps the delta into any frame $c$ given $R_e^c$:
$$
R_{e\to e*}^c = R_e^c\, R_{e\to e*}^e\, R_c^e .
$$

**Translation.** With the world-frame difference $p_{e\to e*}^w = p_{e*}^w - p_e^w$, the body-frame delta is
$$
p_{e\to e*}^e = R_w^e\, p_{e\to e*}^w = (R_e^w)^{-1} p_{e\to e*}^w ,
$$
and conversion to frame $c$ is
$$
p_{e\to e*}^c = R_e^c\, p_{e\to e*}^e .
$$

Storing the body-frame pair $(R_{e\to e*}^e,\, p_{e\to e*}^e)$ is sufficient — it is exactly the rotation/translation of the relative transform $T_{e*}^e = (T_e^w)^{-1} T_{e*}^w$, and any frame is recovered from $R_e^c$ during data loading. We also keep the world-frame delta and the raw finite differences below for out-of-box convenience.

| field             | meaning                                                                              |
| ----------------- | ------------------------------------------------------------------------------------ |
| `world_eef_xyz`   | delta translation in world frame, $p_{e\to e*}^w$                                    |
| `world_eef_rot6d` | delta rotation in world frame, $R_{e\to e*}^w$                                       |
| `body_eef_xyz`    | delta translation in body frame, $p_{e\to e*}^e$                                     |
| `body_eef_rot6d`  | delta rotation in body frame, $R_{e\to e*}^e$                                        |
| `diff_eef_xyz`    | un-rotated world-frame translation difference $p_{e*}^w - p_e^w$ (= `world_eef_xyz`) |
| `diff_joint_pos`  | difference between current and target joint positions                                |
| `gripper_state`   | commanded gripper state, 0 closed / 1 open                                           |

The target $e^*$ is chosen per dataset:
1. If the raw data carries a meaningful desired pose (e.g. the teleop command sent to the controller), use it as $e^*$.
2. Otherwise, use the ground-truth next pose as $e^*$.

`action.ref_action` concatenates a per-dataset subset of these fields for out-of-box use.

## Deploying the Model Output to a Controller

Assume the model is trained on deltas in some frame $c$ and outputs $(R_{e\to e*}^c,\, p_{e\to e*}^c)$ plus the gripper command. At inference we know the current EEF pose $(R_e^w,\, p_e^w)$ and the frame-$c$ orientation $R_c^w$ (e.g. from camera extrinsics), so
$$
R_e^c = R_w^c R_e^w = (R_c^w)^{-1} R_e^w , \qquad R_c^e = (R_e^c)^{-1} .
$$
The gripper command passes through unchanged in every case.

**Case 1 — controller wants the absolute target pose in the world frame** $(R_{e*}^w,\, p_{e*}^w)$.
First lift the delta from frame $c$ to the world frame:
$$
R_{e\to e*}^w = R_c^w\, R_{e\to e*}^c\, R_w^c , \qquad
p_{e\to e*}^w = R_c^w\, p_{e\to e*}^c ,
$$
then compose with the current pose (world delta left-multiplies the rotation):
$$
R_{e*}^w = R_{e\to e*}^w\, R_e^w , \qquad
p_{e*}^w = p_e^w + p_{e\to e*}^w .
$$

**Case 2 — controller wants the relative motion in the gripper frame** $e$, i.e. $(R_{e\to e*}^e,\, p_{e\to e*}^e)$.
Only a frame change is needed (conjugate the rotation, rotate the translation):
$$
R_{e\to e*}^e = R_c^e\, R_{e\to e*}^c\, R_e^c , \qquad
p_{e\to e*}^e = R_c^e\, p_{e\to e*}^c .
$$

When the model is trained directly in the body frame ($c = e$, so $R_e^c = I$) Case 2 is the identity and Case 1 reduces to $R_{e*}^w = R_e^w R_{e\to e*}^e$, $p_{e*}^w = p_e^w + R_e^w p_{e\to e*}^e$.
