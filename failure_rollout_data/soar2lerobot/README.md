# soar2lerobot

Convert the **SOAR dataset** (Berkeley RAIL, [SOAR: Autonomous Improvement of Instruction
Following Skills via Foundation Models](https://auto-improvement.github.io/)) from RLDS to
**LeRobot v3**. SOAR is a WidowX 250 dataset collected *autonomously* by policies
(iql / calql / gcbc / mixed) on the BridgeData setup, with **VLM-judged success labels**.
The RLDS export ships two splits — `success` (10,018 episodes) and `failure` (20,562
episodes), episode length variable (~100–110 steps typical) — and each split is converted into its **own
LeRobot dataset** (`soar_success`, `soar_failure`).

```bash
bash convert.sh          # converts both splits (resumable; see below)
```

## Raw schema (verified empirically on real episodes)

| RLDS field | shape | meaning |
|---|---|---|
| `observation.state` | (7,) | `[x, y, z (m), roll, pitch, yaw (rad), gripper]`; euler is **extrinsic-XYZ** (Bridge/widowx_envs convention); gripper continuous ~[0, 1], **1=open** |
| `action` | (7,) | `[Δxyz, Δrpy, gripper_absolute]`; gripper binary {0, 1}, 1=open |
| `observation.image_0` | 256×256×3 | main camera (converted) |
| `goal` | 256×256×3 | SuSIE-generated subgoal image (**not converted**) |
| `episode_metadata` | — | `file_path` (encodes robot/scene/policy/date/traj), `success` (VLM), `object_list`, `task_list`, `robot_id`, `time`, `has_language` |

Facts checked against bytes (see `convert.py` docstring):

- **The raw `action` is a real command, not a relabeled state difference**: its residual
  vs the achieved motion `s[t+1]−s[t]` is 25–50% of its own magnitude (controller tracking
  error). The WidowX controller applies it **elementwise**, so the commanded absolute
  target (stored as `raw_target.*`) is `state[:6] + action[:6]`.
- **No joint positions** exist anywhere in the data → no `*.joint_pos` features.
- No camera extrinsics/intrinsics.

## Output features (`../dataset.md` convention)

Following [`../dataset.md`](../dataset.md), **raw** (native, untransformed) and **canonical**
(axis-aligned) groups are stored separately, and **the action is not stored** — it is derived
at **load time** from `state.*` and `target.*` (`dataset.md` §3). SOAR ships a real controller
command, so both `raw_target.*` and `target.*` are present.

| feature | shape | content |
|---|---|---|
| `observation.images.image_0` | 256×256×3 video | main camera |
| `raw_state.eef_xyz` | (3,) | native base-frame eef position, **meters** |
| `raw_state.eef_rpy` | (3,) | native euler (**extrinsic XYZ**) — the only rotation rep SOAR ships |
| `raw_state.gripper_state` | (1,) | native gripper reading (continuous ~[0, 1], 1=open) |
| `raw_target.eef_xyz` | (3,) | absolute raw target = `raw_state.eef_xyz + Δxyz`, **meters** |
| `raw_target.eef_rpy` | (3,) | absolute raw target = `raw_state.eef_rpy + Δrpy` (extrinsic XYZ) |
| `raw_target.gripper_state` | (1,) | commanded gripper, **binary {0, 1}**, 1=open |
| `state.eef_xyz` | (3,) | **canonical** eef position (world align = identity ⇒ == raw), **meters** |
| `state.eef_rot6d` | (6,) | **canonical** eef rotation as Zhou-6D (gripper → OpenCV) |
| `state.gripper_state` | (1,) | gripper normalized to [0, 1] (already ~[0, 1], 1=open; kept as-is) |
| `target.eef_xyz` | (3,) | **canonical** target position (world align = identity), **meters** |
| `target.eef_rot6d` | (6,) | **canonical** target rotation as Zhou-6D |
| `target.gripper_state` | (1,) | commanded gripper, {0, 1}, 1=open |
| `debug.gripper_eef_xyz` | (3,) | GT-next delta in the **canonical gripper frame** (debug only, below) |
| `debug.gripper_eef_rot6d` | (6,) | GT-next relative rotation, canonical gripper frame (debug only) |
| `success` | (1,) | episode success flag (VLM), constant over the episode |
| `task` | string | `language_instruction` |

`fps=5` (Bridge WidowX control rate), `robot_type="widowx_250"`.

`eef_xyz` is in **meters** (raw `observation.state[:3]`): the workspace spans x ≈ 0.19–0.45 m
in front of the base and grasps close at z ≈ 0.03–0.06 m above the table.

### Computing the action at load time

Per `dataset.md` §3, this is **EEF-pose control**: the action is the EEF-frame relative motion
from the canonical current pose (`state.*`) to the canonical target pose (`target.*`), plus the
target gripper. Either the shipped command (`target.*`) or a ground-truth-next target
(next-step `state.*`) can drive it — see `alignment.transforms_numpy.gripper_delta_pose` /
`change_delta_pose_frame` to express the delta in any frame. The `debug.*` fields precompute
exactly the GT-next variant (below).

### Gripper state processing

Every gripper channel is already on its natural scale, so **no width field is emitted and no
rescaling is applied** — each is stored as a single `*.gripper_state`:

- `raw_state.gripper_state` / `state.gripper_state` — from `observation.state[6]`, a sensor
  reading that is **already continuous in ~[0, 1]** (observed ≈ 0.05 … 1.1; a few readings
  slightly overrange). `dataset.md` §2.3 asks for a [0, 1] normalization (0 = closed, 1 = open);
  since this channel is already normalized (not a width in meters), the canonical
  `state.gripper_state` is kept identical to the raw value.
- `raw_target.gripper_state` / `target.gripper_state` — from `action[6]`, which is **already
  binary `{0, 1}`**. It is stored as-is and **not binarized by the converter** (no thresholding
  of our own).

**Polarity is raw / un-inverted: 1 = open** everywhere (no polarity flip applied).

> **Corrupt-gripper QC (episodes dropped).** In ~1 % (success) / ~4 % (failure) of episodes the
> raw `observation.state` gripper slot is overwritten by a **monotonic step-counter** (values
> 11–210) rather than the real gripper reading — a data artifact affecting *only* that channel
> (pose and the binary command gripper stay valid). Since `raw_state.gripper_state` is
> unrecoverable there, the **whole episode is dropped**: detected as
> `max(raw_state.gripper_state) > --gripper-qc-threshold` (default `2.0`; clean grippers are
> ≤ ~1.1 and the artifact ≥ 11, so the cutoff is insensitive), and each dropped episode is
> logged to `meta/qc_warnings.jsonl`. Pass `--gripper-qc-threshold 0` to keep them.

Per-episode provenance is written to `meta/soar_metadata.jsonl` (one JSON line per
`episode_index`): `file_path, robot, scene, policy, collect_date, orig_traj_id, success,
object_list, task_list, robot_id, time, has_language, language_instruction, split`.
(`object_list`/`task_list`/`robot_id`/`time` are stored as-is and are empty or junk for
many episodes — trust `file_path`-derived fields and `success`.)

## Axis-orientation alignment (applied by the converter)

`state.*` and `target.*` are **already** in the canonical frames of [`../dataset.md`](../dataset.md)
(world = FLU x-fwd/y-left/z-up; gripper = OpenCV z-approach/x-right/y-down); `raw_state.*` /
`raw_target.*` keep the native frames. Both conventions were validated on SOAR data itself:

- **World frame: already canonical FLU** → world alignment is the **identity** (positions
  and world-frame quantities need no change, so `state.eef_xyz` == `raw_state.eef_xyz`).
  Verified: z is up (grasps close at z ≈ 0.03–0.06 m above the table, descent-to-grasp),
  workspace x ≈ 0.19–0.45 m in front of the base, y spans both signs.
- **Stored gripper frame is NOT the URDF `ee_gripper_link` (+x approach).** It is the
  widowx_envs `DEFAULT_ROTATION`-composed frame that reads ≈ identity when the gripper
  points straight down: stored **−z = approach**, finger axis along stored **y** — exactly
  BridgeData V2's convention (same software stack). Verified on SOAR: mean |pitch| =
  0.13 rad during near-table manipulation (the URDF frame would require ~π/2), and the
  stored −z axis dotted with world −ẑ is 0.975 at grasp steps. The **in-plane sign** was
  fixed against video: at the neutral gripper-down pose, canonical x must read
  world-RIGHT (= stored −y) and canonical y world-backward / image-down (= stored −x).

The canonical gripper alignment is the same constant as `_GRIPPER_ALIGN_WIDOWX` in
`openx2lerobot/oxe_utils/transforms.py`; the converter applies it as:

```python
import numpy as np
from alignment import transforms_numpy as tn   # repo-root package

R_ALIGN_WIDOWX = tn.axis_alignment_matrix("-y", "-x", "-z")   # stored gripper -> OpenCV

# native rpy (extrinsic XYZ) -> canonical rot6d, as stored in state.eef_rot6d:
R_native = tn.rpy_to_matrix(raw_state_eef_rpy, extrinsic=True)
R_canon, p_canon = tn.align_axis(R_native, raw_state_eef_xyz, np.eye(3), R_ALIGN_WIDOWX)
# world align = identity => p_canon == the raw xyz; only orientations are re-based.
# state.eef_rot6d = tn.matrix_to_rotation_6d(R_canon);  reload with tn.rotation_6d_to_matrix.
```

### `debug.*` fields

`debug.gripper_eef_xyz` / `debug.gripper_eef_rot6d` precompute the **GT-next** EEF-pose action
(`dataset.md` §3) directly from the stored `state.*`: the canonical poses `(R_c, p_c)` at `t`
and `t+1` (i.e. `state.eef_rot6d`/`state.eef_xyz` at consecutive steps) expressed as the
gripper-frame delta
`R_g = R_c[t]ᵀ R_c[t+1]`, `p_g = R_c[t]ᵀ (p_c[t+1] − p_c[t])`
(`tn.gripper_delta_pose`). This is the load-time action when the **next state** is used as the
target instead of the shipped `target.*`. The last step has no successor and holds a **no-op**
(rot6d `[1,0,0,0,1,0]`, zero xyz). Verified on real episodes:
`R_c[t] @ R_g[t] == R_c[t+1]` and `p_c[t] + R_c[t] @ p_g[t] == p_c[t+1]` to ~2e-7.

## Resume / partial download

The RLDS download can be in progress while converting:

- Re-running the **same command resumes**: finished shards are skipped (sentinel
  `meta/.conversion_complete`), partial shards continue **mid-shard** via absolute
  example-range slicing (`split[start+consumed:stop]`) — nothing is re-read or written twice.
- A worker that hits a **missing or truncated tfrecord** (download frontier) finalizes
  what it has, records progress, and exits *without* its sentinel; the orchestrator then
  refuses to merge and tells you to re-run later. Merging into the final dataset (+
  concatenated `soar_metadata.jsonl` with re-offset `episode_index`) happens only once
  every shard is complete.

Note: the benign `ResourceTracker ... '_recursion_count'` traceback at interpreter exit is
multiprocess teardown noise, not a failure.
