# Dataset Convention

This document defines the dataset convention and the design of the unified state and action space. The guiding principle is that **the same vector dimension means the same physical motion across every dataset**, so a model sees consistent geometry no matter which robot produced the data.


## 1. Preliminaries

### 1.1 Terminologies

**Canonical frames.** One right-handed convention is fixed for *all* datasets.

+ **Base (world) frame** — FLU: x forward, y left, z up (ROS body convention).
+ **EEF frame** — OpenCV camera convention: z is the approaching direction (forward),
  x right, y down. This aligns the EEF frame with a forward-looking camera.

Both frames are right-handed, $\det[\hat x\;\hat y\;\hat z]=+1$.

**Raw frame.** The `raw frame` is the native axis orientation shipped by a dataset — both its
raw base frame and its raw EEF frame. Because these axis directions differ from dataset to
dataset, every pose is first mapped from its raw frames into the canonical frames (see
[Axis Alignment](#12-axis-alignment)) before any state or action field is computed.

**Notation.**

+ $T = \begin{bmatrix} R & p \\ 0 & 1 \end{bmatrix} \in SE(3)$ is a pose with rotation
  $R \in SO(3)$ and translation $p \in \mathbb{R}^3$.
+ A primed symbol ($T'_t$, $R'_t$, $p'_t$) is expressed in the **raw** frames; an unprimed
  symbol ($T_t$, $R_t$, $p_t$) is expressed in the **canonical** frames.
+ $R_a^b$ (" $R_a$ to $b$ ") is the orientation of frame $a$ expressed in frame $b$; a vector
  converts as $v^b = R_a^b\  v^a$. We write $w'/e'$ for the raw base/EEF frames and
  $w/e$ for the canonical base/EEF frames.
+ A star ($\cdot^{\ast}$) marks the **target** (commanded / next) quantity that drives an action.

### 1.2 Axis Alignment

Because the canonical axes only relabel/reorient the raw axes, the maps between them are
**constant** proper rotations — fixed signed-permutation matrices, one pair per dataset:

$$
R_{w'}^w,\; R_{e'}^e \in SO(3), \qquad \det R_{w'}^w = \det R_{e'}^e = +1 .
$$

A raw pose $T'_t=(R'_t, p'_t)$ (the EEF $e'$ expressed in the raw base $w'$) is mapped to
the canonical pose $T_t=(R_t, p_t)$ by re-basing the world reference on the left and
relabeling the EEF axes on the right:

$$
p_t = R_{w'}^w\ p'_t , \qquad
R_t = R_{w'}^w\ R'_t\ R_e^{e'} , \qquad R_e^{e'} = (R_{e'}^e)^{-1} .
$$

Joint angles are frame-independent and are left unchanged.

> **Handedness guard.** rot6d and any $SO(3)$ delta cannot encode a reflection. If aligning
> the axes ever forces $\det = -1$, the target canonical frame was defined left-handed and
> must be redefined. Always verify $\det R_{w'}^w = \det R_{e'}^e = +1$.

### 1.3 Relative Motion and Reference Frames

Absolute poses give *where* the EEF is; control and learning usually need the **relative
motion** from the current pose $T_e$ to a target pose $T_{e^*}$ (both already aligned to
canonical). The EEF-frame relative transform is

$$
T_{e\to e^\ast}^e = T_{e}^{-1}\ T_{e^*} , \qquad
R_{e\to e^\ast}^e = R_{e}^\top R_{e^*} , \qquad
p_{e\to e^\ast}^e = R_e^\top\ (p_{e^\ast} - p_e) .
$$

This EEF-frame pair is *sufficient*: the same motion in any other frame $c$ (world,
camera, …) is recovered from the known $R_e^c$ by conjugation,

$$
R_{e\to e^\ast}^c = R_e^c\ R_{e\to e^\ast}^e\ R_c^e , \qquad
p_{e\to e^\ast}^c = R_e^c\ p_{e\to e^\ast}^e .
$$

Two common cases: the **world** frame ($c = w$, $R_e^w = R_e$) gives
$R_{e\to e^\ast}^w = R_e\ R_{e\to e^\ast}^e\ R_e^\top$ and $p_{e\to e^\ast}^w = p_{e^\ast} - p_e$; a **camera**
frame uses that camera's extrinsics for $R_e^c$.

## 2. What will be included in the dataset

There will be multiple entry groups in the final converted dataset. We use the single-arm dataset as examples; for dual-arm datasets, the entry names will be prefixed with `left_` and `right_`. Also, for base movements, it will have prefix `base_`. 

### 2.1 `raw_state.*`
The `raw_state.*` contains a detailed group of metrics present in the original dataset. No transformation or alignment is done; we only split the vector based on the meaning of its entries.

Missing entries from the raw dataset will be omitted.

| entry name                                                     | meaning                                                                                                                                        |
| :------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------- |
| `raw_state.joint_pos`                                          | joint position (in rad)                                                                                                                        |
| `raw_state.joint_vel`                                          | joint velocity                                                                                                                                 |
| `raw_state.eef_xyz`                                            | translation $p'_t$ of the EEF in the raw base frame                                                                                            |
| `raw_state.eef_rot6d`/`raw_state.eef_rpy`/`raw_state.eef_quat` | rotation $R'_t$ of the EEF (raw EEF frame in raw base frame), in a specific representation. Only keep the one shipped by the original dataset. |
| `raw_state.gripper_state`                                      | raw gripper state from the dataset                                                                                                             |

### 2.2 `raw_target.*`
The `raw_target.*` contains the desired target state sent to the low-level controller. No transformation or alignment is done. If the raw dataset contains action commands that express the delta transformation, it will be applied to the raw state to get the absolute raw target. The entries are the same as `raw_state`; otherwise, for example the dataset does not ship commands and wants you to use the ground-truth next state as the target, this group will be omitted. 

For joint velocity control, `raw_target.joint_vel` will be the absolute desired velocity.



### 2.3 `state.*`
We apply a series of transformations and alignments to the original dataset to get the canonical states:
1. Compute the EEF orientation as a rotation matrix $R_t$ and store it in **6D representation**: rot6d is the flattened first two rows of $R_t$, $\mathrm{rot6d}(R_t) = [\ R_{11}, R_{12}, R_{13},\; R_{21}, R_{22}, R_{23}\ ] \in \mathbb{R}^6$. The third row (and thus the full matrix) is recovered by Gram–Schmidt at load time.
2. Apply axis alignment to map the pose into the canonical base and EEF frames (see [Axis Alignment](#12-axis-alignment)).
3. Normalize the gripper state to $[0, 1]$ by dividing by its maximum value, where 0 is fully closed and 1 is fully open. If the raw gripper state is binary, keep the binary and verify the sign convention.

| entry name            | meaning                                                                                 |
| :-------------------- | :-------------------------------------------------------------------------------------- |
| `state.joint_pos`     | joint position (in rad); frame-independent, copied from `raw_state`                     |
| `state.joint_vel`     | joint velocity; frame-independent, copied from `raw_state`                              |
| `state.eef_xyz`       | canonical translation $p_t$ of the EEF in the canonical base frame                      |
| `state.eef_rot6d`     | canonical EEF rotation $R_t$ (canonical EEF frame in canonical base frame), as rot6d    |
| `state.gripper_state` | gripper state normalized to $[0, 1]$ (0 = fully closed, 1 = fully open), or kept binary |

### 2.4 `target.*`
Same fields and transformations as `state.*`, applied to the canonical `raw_target.*`. The target $\cdot^{\ast}$ is always the command actually sent to the controller during collection; if the dataset ships no such command, this group is omitted (mirroring `raw_target.*`). When a ground-truth-next target ($T_{e^*} = T_{t+1}$) is desired instead, derive it from the next `state.*` at load time.

## 3. How to calculate the action at load time?

Actions are computed at load time from `state.*` (current, step $t$) and `target.*` (command, step $t$; or the ground-truth next `state.*`, step $t+1$). The action space follows the dataset's control mode:

1. **Joint-position control** — the per-step joint delta $\Delta q = q^{\ast} - q$, where $q$ is `state.joint_pos` and $q^{\ast}$ is `target.joint_pos`.
2. **Joint-velocity control** — the desired velocity itself, `target.joint_vel` (already absolute; no differencing).
3. **EEF-pose control** — the EEF-frame relative motion $(R_{e\to e^\ast}^e, p_{e\to e^\ast}^e)$ of §[1.3](#13-relative-motion-and-reference-frames), from the canonical current pose $T_t$ (`state.*`) to the target pose $T_t^{\ast}$ (`target.*`). To express the same motion in another frame $c$ (world, camera, …), apply the conjugation in §[1.3](#13-relative-motion-and-reference-frames).

In every mode the gripper action is the target gripper state, `target.gripper_state` (normalized as in §[2.3](#23-state)).