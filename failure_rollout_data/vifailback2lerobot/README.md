---
license: mit
modalities:
- image
- video
- tabular
- text
task_categories:
- visual-question-answering
- robotics
language:
- en
tags:
- hdf5
---

# **ViFailback Dataset: Real-World Robotic Manipulation Failure Dataset with Visual Symbol Guidance**

<p align="center">
  <a href="https://x1nyuzhou.github.io/vifailback.github.io/">
    <img src="https://img.shields.io/badge/Project-Page-4285F4?logo=googlechrome&logoColor=white" alt="Project Page">
  </a>
  <a href="https://arxiv.org/abs/2512.02787">
    <img src="https://img.shields.io/badge/arXiv-2512.02787-b31b1b?logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  <a href="https://github.com/x1nyuzhou/ViFailback">
    <img src="https://img.shields.io/badge/GitHub-Code-181717?logo=github&logoColor=white" alt="GitHub">
  </a>
</p>

A real-world dataset (CVPR 2026) for diagnosing, correcting, and learning from robotic manipulation
failures via visual symbols. **5,202** trajectories (657 success / 4,545 failure) across **100**
tasks on an **ALOHA dual-arm + mobile-base** platform (AgileX PiPER arms).

### Failure Taxonomy

| Type | % |
| :--- | :--- |
| **Gripper 6D-Pose** (fails to reach correct position/orientation) | 53.27% |
| **Gripper State** (fails to close/open properly) | 18.99% |
| **Task Planning** (high-level plan errors) | 12.40% |
| **Human Intervention** (external disruptions) | 2.71% |

## **📂 Raw Data Format (HDF5)**

Each `episode_X.hdf5`:

```text
├── action        # target joint positions (≈ qpos[t+1]), 14d
├── action_eef    # target EEF pose (16d)
├── action_leader # master-arm joint positions
├── base_action   # mobile-base (linear, angular) velocity command, 2d
└── observations
    ├── qpos      # current joint states, 14d
    ├── qvel, effort
    ├── images        # compressed RGB (cam_high, cam_left_wrist, cam_right_wrist)
    └── images_depth  # raw uint16 depth, 480×640
```

- **Joint space (14D)**: left arm `[0:7]` (joint 1–6 + gripper) | right arm `[7:14]`.
- **EEF space (16D)**: left `[0:8]` (`x,y,z,qx,qy,qz,qw,gripper`, xyzw) | right `[8:16]`.
- ⚠️ **Dabai cameras**: RGB and depth are **not spatially aligned** — align before RGB-D fusion.

## **🔄 Conversion to LeRobot v3 (`convert.py`)**

```bash
export HDF5_USE_FILE_LOCKING=FALSE
python convert.py --raw-dir .../raw_data --local-dir .../output --num-proc 8   # or ./convert.sh
```

fps = 25; `robot_type = aloha_agilex_piper`; task = folder name (`_epN` stripped, `_` → space).
Re-running resumes an interrupted run; `--save-depth` adds `observation.images.*_depth` (uint16 PNG).

Output follows [`../dataset.md`](../dataset.md): each pose is stored natively (`raw_state.*` /
`raw_target.*`, no transform) and canonically axis-aligned (`state.*` / `target.*`); the per-step
**action is not stored** — it is derived at load time (§3). `debug.*` holds a precomputed
ground-truth-next reference. Dual-arm + mobile base → `left_` / `right_` / `base_` prefixes.

**EEF is FK-solved** (the raw data ships no observed EEF pose). With `assets/piper_description.urdf`
([agilexrobotics/piper_ros](https://github.com/agilexrobotics/piper_ros) @ `humble`, chain
`base_link → link6`, joints `qpos[0:6]`/`qpos[7:13]`): `raw_state`/`state` EEF = `FK(qpos)`,
`raw_target`/`target` EEF = `FK(action)`. Control mode is **joint-position** (`action` = the real
target-joint command). `action_eef` is a relabeled achieved-next-pose used **only** to QC the FK
solution (`FK(qpos[t+1]) == action_eef[t]` to ~0.1–0.4 mm on moving arms).

### Schema

Raw EEF uses **quat (xyzw)**, canonical EEF **rot6d**; joints are frame-independent so their
canonical copies equal the raw ones. `side ∈ {left, right}`.

| feature | dim | source |
| --- | --- | --- |
| `observation.images.cam_high\|cam_left_wrist\|cam_right_wrist` | 480×640×3 | RGB video (`*_depth` uint16 with `--save-depth`) |
| `raw_state.{side}_joint_pos` / `_joint_vel` | 6 / 6 | `qpos` / `qvel` joints |
| `raw_state.{side}_eef_xyz` / `_eef_quat` | 3 / 4 | `FK(qpos)`, **native** link6 frame |
| `raw_state.{side}_gripper_state` | 1 | `qpos` gripper, **raw width (m)** |
| `raw_target.{side}_joint_pos` | 6 | `action` target joints |
| `raw_target.{side}_eef_xyz` / `_eef_quat` | 3 / 4 | `FK(action)`, **native** link6 frame |
| `raw_target.{side}_gripper_state` | 1 | `action` gripper, **raw width (m)** |
| `raw_target.base_vel` | 2 | `base_action` (linear, angular) |
| `state.{side}_joint_pos` / `_joint_vel` | 6 / 6 | = `raw_state` (frame-independent) |
| `state.{side}_eef_xyz` / `_eef_rot6d` | 3 / 6 | canonical `FK(qpos)` (world → I, gripper → OpenCV) |
| `state.{side}_gripper_state` | 1 | `clip(width/0.095, 0, 1)`, **0 = closed, 1 = open** |
| `target.{side}_joint_pos` | 6 | = `raw_target` |
| `target.{side}_eef_xyz` / `_eef_rot6d` | 3 / 6 | canonical `FK(action)` |
| `target.{side}_gripper_state` | 1 | normalized as above |
| `target.base_vel` | 2 | `base_action` (used directly) |
| `debug.{side}_gripper_eef_xyz` / `_eef_rot6d` | 3 / 6 | GT-next `state[t]→state[t+1]` Δ, canonical gripper frame; last step no-op |

### Load-time action (`dataset.md` §3)

- **Arm** (joint-position): `Δq = target.{side}_joint_pos − state.{side}_joint_pos`.
- **Gripper**: `target.{side}_gripper_state`.
- **Base** (velocity): `target.base_vel` directly.

An EEF-pose action can instead be derived from `state.{side}_eef_*` → `target.{side}_eef_*`; the
`debug.*` fields precompute the GT-next variant.

### Gripper

The raw gripper channel is a **continuous width in meters** (≈ −0.008 … 0.099, p99 = 0.095), polarity
**un-inverted (larger = more open)**. `raw_*.gripper_state` stores the width verbatim;
`state.*`/`target.*` normalize to `clip(width / 0.095, 0, 1)` → **0 = closed, 1 = open** (`0.095` =
`GRIPPER_MAX` in `convert.py`).

### QC drop

Each episode is validated against `action_eef` per arm — a state check `‖FK(qpos[t+1]) − action_eef‖`
and a target check `‖FK(action) − action_eef‖` (mean over moving steps) — and **dropped** when the max
exceeds `--qc-drop-threshold` (default 0.02 m). This removes the **488/5202 (9.4%)** stale-idle-arm
episodes (healthy ≤ ~10 mm, artifact ≥ 100 mm; two tasks dropped entirely). Set `0` to keep them;
drops/warnings are logged to `<output>/vifailback_lerobot/meta/qc_warnings.jsonl`.

### Frames

World = each arm's `base_link` (ROS FLU) = the canonical world frame → alignment is the identity,
positions pass through. `raw_*` EEF is the **native** `link6` frame; canonical `state.*`/`target.*`
re-base the gripper to **OpenCV** (z = approach, x = right, y = down) via
`axis_alignment_matrix("y","-x","z")` (empirically: approach = native +z, native x → cam +y, native
y → cam −x). To reproduce from a raw pose:

```python
from alignment import transforms_numpy as tn
R_native = tn.quaternion_to_matrix(raw_state_left_eef_quat)   # xyzw
R_align  = tn.axis_alignment_matrix("y", "-x", "z")           # R_{e'}^e, det = +1 (both arms)
R_canon, p_canon = tn.align_axis(R_native, raw_state_left_eef_xyz, np.eye(3), R_align)
# R_canon == rotation_6d_to_matrix(state_left_eef_rot6d), p_canon == state_left_eef_xyz
```

## **📜 Citation**

```bibtex
@inproceedings{zeng2026diagnose,
  title={Diagnose, correct, and learn from manipulation failures via visual symbols},
  author={Zeng, Xianchao and Zhou, Xinyu and Li, Youcheng and Shi, Jiayou and Li, Tianle and Chen, Liangming and Ren, Lei and Li, Yong-Lu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={42386--42395},
  year={2026}
}
```
