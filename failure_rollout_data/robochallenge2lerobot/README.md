# robochallenge2lerobot

Convert crawled **RoboChallenge Table30-v2** rollouts (Rerun `.rrd`) to **LeRobot v3**, one
dataset per embodiment: **ARX5**, **UR5** (single-arm), **ALOHA**, **DOS-W1** (dual-arm).

```bash
export HDF5_USE_FILE_LOCKING=FALSE
python convert.py --data-dir .../robochallenge/data --local-dir .../out --robot arx5 --num-proc 8
# or convert all four:  ./convert.sh
```
Re-running the same command resumes an interrupted conversion (per-shard sentinels +
mid-list progress). fps = 30. Outputs `robochallenge_{arx5,ur5,aloha,dos_w1}_lerobot`.

## What the crawled data is (and is not)

The crawled files under `table30_v2/<task>_<hash>/<run_id>/rollouts/<rollout_id>.rrd` are
RoboChallenge **leaderboard-evaluation rollouts** (policy rollouts by submitted models), stored
as Rerun *video-preview* recordings. Each `.rrd` contains **only**, per arm:

- `/<arm>/cur_joint/joint_1..6` — joint angles (rad), logged ~138 Hz on the `log_time` timeline;
- `/<arm>/cur_gripper` — gripper finger width (m), logged on its own staggered rows;
- up to three H.264 (`avc1`) video streams, ~28 fps (per-camera frame counts differ). Resolution is
  per-robot: **480×640** for ARX5/ALOHA/DOS-W1, **720×1280** for UR5 (which ships only 2 streams).

The seven scalars are each logged on **separate** `log_time` rows (never co-valid), so they are
read as independent series and resampled. There is **no end-effector pose and no commanded
action** in the file. (The released HF *training* set `RoboChallenge/Table30v2` stores
`ee_positions` directly, but that is a **different** set of trajectories — human demos, not these
eval rollouts — so it cannot be matched per episode. It *is* used to validate FK below.)

Consequences:
- EEF pose is recovered by **forward kinematics** from the joint angles (per-robot URDF).
- `raw_target.*` / `target.*` are **omitted** entirely (no commanded target exists in the data);
  the ground-truth-next target is derived at load time (see [`../dataset.md`](../dataset.md) §2.4/§3).

## Forward kinematics (validated against HF `ee_positions`)

`rc_fk.SerialChainFK` is a self-contained NumPy URDF FK (generalized from
`vifailback2lerobot`'s `PiperFK`). `validate_fk.py` checks `FK(joint_positions) ≈ ee_positions`
(quaternion xyzw, arm-base frame) on real HF episodes:

| robot | urdf | chain | tool offset (m, tip frame) | residual |
| --- | --- | --- | --- | --- |
| ARX5 | `assets/arx5.urdf` | `base_link → eef_link` | — | 0.00 mm / 0.003° |
| ALOHA | `../vifailback2lerobot/assets/piper_description.urdf` (AgileX PiPER) | `base_link → link6` | — | 0.20 mm / 0.027° |
| UR5 | `assets/ur5.urdf` | `base_link → wrist_3_link` | — | 1.77 mm / 0.033° |
| DOS-W1 | `assets/dos_w1.urdf` (Dexmal, per-arm) | `base_link → end_link` | `[0.0992, 0.0004, -0.00197]` | 1.17 mm / 0.098° |

DOS-W1's real TCP sits ~99 mm along the wrist x-axis beyond `end_link`; the constant offset was
calibrated from HF `ee_positions` (std < 1 mm). Dual-arm robots use the same single-arm URDF for
both arms; each arm's pose is in that arm's own base frame (two unrelated base frames).

## Converted schema

Fields follow [`../dataset.md`](../dataset.md): each pose is stored twice — native `raw_state.*`
(no transform) and canonically axis-aligned `state.*`. Joints are frame-independent, so their
canonical copies equal the raw ones. Single-arm (`arm`) shown; dual-arm mirrors every
`raw_state.*`/`state.*`/`debug.*` field with `left_`/`right_` prefixes.

| feature | dim | source |
| --- | --- | --- |
| `observation.images.<cam>` | H×W×3 | H.264 → video (single: `cam_1/2/3`; dual: `cam_high/cam_left_wrist/cam_right_wrist`). 480×640 for ARX5/ALOHA/DOS-W1; **720×1280 for UR5** (see caveats) |
| `raw_state.joint_pos` | 6 | `cur_joint`, resampled to 30 Hz (native) |
| `raw_state.eef_xyz`, `raw_state.eef_rot6d` | 3, 6 | FK(joints): EEF pose in **native** arm-base frame (no alignment) |
| `raw_state.gripper_state` | 1 | raw `cur_gripper` width (m) |
| `state.joint_pos` | 6 | = `raw_state.joint_pos` (frame-independent) |
| `state.eef_xyz`, `state.eef_rot6d` | 3, 6 | canonical FK pose (world → I; gripper → OpenCV per robot) |
| `state.gripper_state` | 1 | `clip(cur_gripper / per-robot max, 0, 1)`, **0 = closed, 1 = open** |
| `debug.gripper_eef_xyz`, `debug.gripper_eef_rot6d` | 3, 6 | GT-next delta in the canonical gripper frame (last step no-op) |
| `success` | 1 | rollout `completion` (constant per episode; NaN if unknown) |
| `score` | 1 | rollout `score` (constant per episode; NaN if unknown) |

`raw_target.*` / `target.*` are **omitted** (no commanded action; derive the GT-next target from
the next `state.*` at load time, per `../dataset.md` §2.4/§3).

`task` (language instruction) = the run's `prompt`. Per-episode provenance (task, run/rollout ids,
model, user, arena, run+rollout score/completion, ranked flag, tags, …) is written to
`meta/robochallenge_metadata.jsonl`.

### Notes / caveats

- **No `raw_target.*` / `target.*`** — the preview `.rrd` carries no commanded target, so those
  groups are omitted (per `../dataset.md`); the load-time action uses the GT-next `state.*`.
- **Frames.** World alignment is the **identity** for all robots (each arm's `base_link` is taken
  as the canonical FLU world frame → `state.eef_xyz == raw_state.eef_xyz`). The gripper relabel
  (native → canonical OpenCV) drives both `state.eef_rot6d` and `debug.*` and is per-robot
  (`gripper_align` in `convert.py::ROBOTS`). Current status **after a sample-video review** (see
  [Conversion status](#conversion-status--resume-here) — one episode per robot pushed to HF):

  | robot | `gripper_align` | status |
  | --- | --- | --- |
  | **ALOHA** | `('y','-x','z')` | ✅ **confirmed** — vifailback-validated PiPER→OpenCV |
  | **UR5** | `None` (native) | ✅ **confirmed correct by reviewer** — native gripper frame already reads as canonical OpenCV |
  | **ARX5** | `None` (native) | ⚠️ **unverified** — reviewer could not judge from the sample video; `state.eef_rot6d == raw_state.eef_rot6d` (placeholder identity relabel) |
  | **DOS-W1** | `None` (native) | ⚠️ **unverified** — same as ARX5 |

  For the two unverified robots, `state.eef_rot6d` is a **placeholder** equal to the native
  `raw_state.eef_rot6d` until a relabel is validated. Tune their `gripper_align` before a full run.
- **Gripper→OpenCV inference (attempted, not trusted).** `infer_gripper_axes.py` runs vifailback's
  wrist-camera optical-flow probe (correlate tip-frame EEF velocity with mean image flow → signed
  axis permutation). Validated against the **known** PiPER answer it recovered only 1 of 3 axes
  (`opencv-x = native -y` ✓; `y`/`z` wrong; R² ≤ 0.12), so its ARX5/UR5/DOS-W1 outputs are **not**
  baked in. Reliable geometric facts for a future calibration: the **approach axis** (tool
  direction) is native **+x** for ARX5 (`gripper_fixed_joint` origin `[0.145,0,0]`) and DOS-W1 (TCP
  offset `[0.099,…]`). Determining the in-plane x/y (and signs) needs a better method (background-
  segmented flow, or a wrist-camera extrinsic — absent from the crawled `.rrd`).
- **Cameras.** Dual-arm camera names are semantic (`videos_front/left/right` → `cam_high` /
  `cam_left_wrist` / `cam_right_wrist`). Single-arm `videos_1/2/3` → `cam_1/2/3`; the index→view
  (wrist/global/side) mapping is not documented upstream. A missing camera stream is black-padded.
- **Image resolution is per-robot.** ARX5 / ALOHA / DOS-W1 stream **480×640**; **UR5** streams
  **720×1280** (and ships only `cam_1`/`cam_2` — `cam_3` is absent → black-padded). The feature
  shape is set per robot via `ROBOTS[...]["img_shape"]` (default `(480,640,3)`; UR5 `(720,1280,3)`);
  frames are stored at native resolution (no resize).
- **Resampling.** Uniform 30 Hz grid over the joint-timeline span; joints/gripper and each camera
  are nearest-neighbour sampled onto it. Video is decoded one frame at a time (O(1) memory).
- ⚠️ **Trajectories are bumpy / not smoothed.** These are raw *recordings* (per-channel scalars
  logged at ~138 Hz, then nearest-neighbour resampled to 30 Hz). No filtering/smoothing is applied,
  so `state.joint_pos` and the FK-derived `state.eef_*` are jittery. A future pass should smooth the
  joint (and/or EEF) signals before the full conversion — see [Conversion status](#conversion-status--resume-here).
- HF ground-truth `episode_meta.json` also carries camera intrinsics + per-arm extrinsics; these
  are **not** present in the crawled `.rrd` and are not emitted.

## Conversion status / resume here

**Conversion is PAUSED — do not run the full `convert.sh` yet.** The converter is finished and
follows `../dataset.md`; two things must be resolved first (below). Summary of where we are:

**Done**
- Fields restructured to the `../dataset.md` convention (`raw_state.*` / `state.*` / `debug.*`;
  `raw_target.*` / `target.*` omitted — no commanded action). Schema validated on one episode/robot.
- Fixed two pre-existing bugs surfaced while sampling: (1) `_REPO_ROOT` needed three `dirname`
  levels to import the repo-root `alignment` package; (2) per-robot `img_shape` — **UR5 is 720×1280**
  (and ships only `cam_1`/`cam_2`), all others 480×640.
- **Sample datasets pushed to HF** (one episode each, except ALOHA has two) for the video review:
  - `typoverflow/robochallenge_dump_arx5`
  - `typoverflow/robochallenge_dump_ur5`
  - `typoverflow/robochallenge_dump_aloha` (2 episodes)
  - `typoverflow/robochallenge_dump_dos_w1`
- **Reviewer verdict on gripper alignment** (see [Frames](#notes--caveats)): **ALOHA** and **UR5**
  are correct as-is; **ARX5** and **DOS-W1** could not be judged from the video (left `None`).

**Blocking / open before a full run**
1. ⚠️ **Smoothing.** Trajectories are bumpy (raw recording, NN-resampled, unfiltered). Add a
   smoothing pass over `state.joint_pos` (and re-derive FK `state.eef_*` from the smoothed joints;
   keep `raw_state.*` un-smoothed as the native record). Decide filter + cutoff, then re-sample.
2. ⚠️ **ARX5 / DOS-W1 gripper→OpenCV relabel** still unverified. Options: a better wrist-camera probe
   than `infer_gripper_axes.py`, or a manual call from a clearer view. Known constraint: approach axis
   is native **+x** for both (ARX5 `gripper_fixed_joint` origin `[0.145,0,0]`; DOS-W1 TCP offset
   `[0.099,…]`). Set each robot's `gripper_align` in `convert.py::ROBOTS` once determined.

**How to resume**
- Regenerate a review sample for one robot: `python convert.py --data-dir <DATA> --local-dir <OUT>
  --robot <r> --max-episodes 1 --overwrite` (env python:
  `/localscratch/cgao304/dev/envs/miniconda3/envs/lerobot/bin/python`; `DATA` =
  `/localscratch/cgao304/dev/datasets/robochallenge/data`). Push via `LeRobotDataset(repo, root=…).push_to_hub(...)`.
- After (1) and (2) are resolved, run the full conversion with `./convert.sh` (all four robots,
  parallel shards + resume).

## Files

- `convert.py` — main converter (per-robot datasets, parallel shards + resume + aggregate).
- `rc_fk.py` — `SerialChainFK` URDF forward kinematics.
- `rc_rrd.py` — `.rrd` reader: staggered-scalar alignment, H.264 streaming decode, 30 Hz resample.
- `validate_fk.py` — FK-vs-HF-`ee_positions` check (needs a few HF episodes extracted locally).
- `assets/` — vendored URDFs (`arx5.urdf`, `ur5.urdf`, `dos_w1.urdf`).
- `crawler/` — the downloader that produced the raw `.rrd` tree.
