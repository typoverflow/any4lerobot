# Failure-Rollout Dataset Summaries

Short reference cards for the datasets converted under `failure_rollout_data/`. Each entry is
converted to LeRobot v3 following the state/action convention in [`dataset.md`](dataset.md).

---

## SOAR (`soar2lerobot`)

Berkeley RAIL, [*SOAR: Autonomous Improvement of Instruction Following Skills via Foundation
Models*](https://auto-improvement.github.io/). Data collected **autonomously** by policies
(iql / calql / gcbc / mixed) on the BridgeData setup, with **VLM-judged** success labels.

**Robot.** WidowX 250 (`robot_type="widowx_250"`), single arm, control rate **fps = 5**.
6-DoF EEF pose + 1-DoF gripper; **no joint data**. Single RGB camera (`image_0`, 256×256).

**Size & split.** RLDS ships two splits, each converted into its **own** LeRobot dataset:

| Split     | Episodes | Success label     | Dataset        |
| --------- | -------- | ----------------- | -------------- |
| `success` | 10,018   | all success (VLM) | `soar_success` |
| `failure` | 20,562   | all failure (VLM) | `soar_failure` |

Episode length is variable (~100–110 steps typical). The success/failure label is stored
per-step (constant within an episode) in the `success` feature and in `meta/soar_metadata.jsonl`.

**Caveats.**
- **Corrupt-gripper episodes are dropped.** In ~1 % (success) / ~4 % (failure) of episodes the
  raw gripper channel is overwritten by a monotonic step-counter (11–210) instead of the
  reading; the whole episode is dropped and logged to `meta/qc_warnings.jsonl` (threshold
  configurable via `--gripper-qc-threshold`). Reported counts above are pre-drop.
- **No joint positions/velocities** anywhere → no `*.joint_pos` / `*.joint_vel` features.
- **No camera extrinsics/intrinsics.** The SuSIE `goal` subgoal image is not converted.
- **Autonomous / noisy labels.** Trajectories come from imperfect policies, not teleop; success
  is VLM-judged (not ground-truth). Provenance fields `object_list` / `task_list` / `robot_id` /
  `time` are often empty or junk — trust `file_path`-derived fields and `success`.

---

## RoboArena (`roboarena2lerobot`)

[*RoboArena: Distributed Real-World Evaluation of Generalist Robot Policies*](https://arxiv.org/pdf/2506.18123)
([project](https://robo-arena.github.io/)). A distributed real-world **evaluation** benchmark: a human
sets a scene + language instruction, several generalist policies each attempt the task **autonomously**,
and every attempt is scored; each session also carries a pairwise policy **preference** (A / B / TIE).

**Robot.** DROID **Franka Emika Panda** (`robot_type="franka"`), 7-DoF arm + parallel gripper, control
rate **fps = 15**. Joint-space control; **no commanded cartesian target**. Three RGB cameras — two
exterior (`left`, `right`) + `wrist` (Zed-Mini), 288×512.

**Size.** Two dated dumps, each converted into its **own** LeRobot dataset (one episode = one policy
rollout). Episodes are short (median 400 steps ≈ 27 s; range 17–800).

| Dump                  | Sessions | Episodes | Frames    | ≈ dur @15fps | Policies           | Dataset                    |
| --------------------- | -------- | -------- | --------- | ------------ | ------------------ | -------------------------- |
| `DataDump_02-03-2026` | 3,284    | 9,589    | 3,428,488 | 63.5 h       | 15 (8 vel / 7 pos) | `roboarena_2026_02_03`     |
| `DataDump_08-05-2025` | 796      | 4,613    | 1,461,519 | 27.1 h       | 8 (all vel)        | `roboarena_2025_08_05`     |
| **total**             | 4,080    | 14,202   | 4,890,007 | ≈90.6 h      | —                  | —                          |

**Success / failure.** A **failure-heavy** eval set — most rollouts do not complete the task.
Per-episode `binary_success` ∈ {0,1} and `partial_success` ∈ [0,1] (broadcast per-step; also in
`meta/roboarena_metadata.jsonl`); no labels missing.

| Dump                  | Binary success | # success | # failure | Partial (mean) |
| --------------------- | -------------- | --------- | --------- | -------------- |
| `DataDump_02-03-2026` | 11.1 %         | 1,060     | 8,529     | 0.377          |
| `DataDump_08-05-2025` | 13.4 %         | 617       | 3,996     | 0.368          |

**Caveats.**
- **Mixed control mode.** Velocity and position policies coexist (only 210 / 9,589 episodes are
  `joint_position` in the 02-03 dump). The joint command goes to `raw_target.joint_vel` **or**
  `raw_target.joint_pos` (inactive one **0-filled**); select by the per-frame `control_is_position`
  flag. **Do not** mix the two in one normalized action vector. Velocity target is used directly;
  position target is differenced (`Δq`) at load time.
- **Gripper is normalized, not metric.** `raw_state.gripper_state` is a continuous openness fraction
  ≈ [0,1] (not a jaw distance in meters), DROID polarity **larger = closed**, with small negative
  overshoot (~7 % of steps just below 0). Canonical `state.gripper_state = 1 − raw` (0 = closed,
  1 = open). Commanded gripper is **binary {0,1}**.
- **Wrist camera not flipped.** Stored as shipped; the Zed-Mini wrist is mounted rolled 180°, so
  co-training with other datasets needs a 180° image rotation on the wrist view (see README).
- **Missing `left` camera → black frames.** Absent in a sizable minority of sessions (~1.7k / 9.6k in
  02-03); those views are black-padded. `cameras_present` records which were real.
- **No joint velocities in state**, and **no top-level state/action vector** — fields stay split
  (`raw_state.*` / `raw_target.*` / `state.*` / `target.*` / `debug.*`), assembled at load time.
- xyz in **meters**, rpy in **radians** (standard DROID); world frame already canonical FLU (only the
  gripper frame is re-based).

---

## ViFailback (`vifailback2lerobot`)

[*Diagnose, Correct, and Learn from Manipulation Failures via Visual Symbols*](https://arxiv.org/abs/2512.02787)
(CVPR 2026). Real-world manipulation **failure** dataset for failure diagnosis/recovery via visual
symbols; **teleoperated** (not autonomous).

**Robot.** ALOHA **dual-arm + mobile base**, AgileX **PiPER** arms (`robot_type="aloha_agilex_piper"`),
control rate **fps = 25**. **Joint-position** arm control + **velocity** base control. Three RGB cameras
(`cam_high`, `cam_left_wrist`, `cam_right_wrist`, 480×640) + optional uint16 depth (`--save-depth`).

**Size.** Single merged dataset, **5,202** trajectories (657 success / 4,545 failure) across **100**
tasks (task = folder name). Failure taxonomy: gripper 6D-pose 53 %, gripper state 19 %, task
planning 12 %, human intervention 3 %.

**Caveats.**
- **EEF is FK-solved** — the raw data ships no observed EEF pose. `raw_state`/`state` EEF = `FK(qpos)`,
  `raw_target`/`target` EEF = `FK(action)` (PiPER URDF, `base_link→link6`). `action_eef` is a relabeled
  achieved-next-pose used **only** for QC.
- **Stale-idle-arm episodes dropped.** In 488/5202 (9.4 %) an idle arm's joint stream is frozen while
  `action_eef` stays live; per-arm FK-vs-`action_eef` residual (state + target checks) > `--qc-drop-threshold`
  (default 0.02 m) drops them (two tasks entirely). Logged to `meta/qc_warnings.jsonl`.
- **Dual-arm → `left_`/`right_` prefixes; base → `base_`.** Joints frame-independent; base has target
  only (no state). Action derived at load: arm `Δq`, gripper `target.gripper_state`, base `target.base_vel`.
- xyz in **meters**; raw gripper is a **jaw width in meters** (0 = closed, ~0.07 open, up to ~0.099 with
  overtravel), un-inverted; `state.gripper_state = clip(width/0.095, 0, 1)`. World = per-arm `base_link`
  already canonical FLU (identity); gripper `link6` re-based to OpenCV.
- ⚠️ **Dabai RGB/depth not spatially aligned** — align before RGB-D fusion.
