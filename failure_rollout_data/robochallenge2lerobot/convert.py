#!/usr/bin/env python
"""Convert RoboChallenge Table30-v2 crawled rollouts (.rrd) to LeRobot v3, per embodiment.

The crawled data under ``<data>/table30_v2/<task>_<hash>/<run_id>/`` are RoboChallenge
*leaderboard-evaluation* rollouts stored as Rerun ``.rrd`` files. Each ``.rrd`` carries ONLY,
per arm, ``cur_joint/joint_1..6`` + ``cur_gripper`` (scalars, ~138 Hz) and up to three H.264 video
streams (~28 fps; 480x640 for ARX5/ALOHA/DOS-W1, 720x1280 for UR5). There is NO end-effector pose
and NO commanded action in the file
(the released HF training set stores ``ee_positions`` directly, but that is a different set of
trajectories). We therefore:

  * recover the EEF pose by forward kinematics from the joint angles (per-robot URDF, validated
    against HF ``ee_positions`` to <=2 mm / <=0.1 deg -- see ``validate_fk.py``);
  * omit ``raw_target.*`` / ``target.*`` entirely (no commanded target exists -- the GT-next
    target is derived at load time; see ``../dataset.md`` §2.4/§3);
  * resample joints + each camera onto a uniform 30 fps grid, decoding video one frame at a time so
    memory stays O(1) in the frame count. ``raw_state.*`` uses a nearest-neighbour resample (raw
    record); ``state.*`` (and thus the differenced ``debug.*`` action) uses joints linearly
    interpolated onto the grid and zero-phase Butterworth low-pass filtered before FK, so the action
    is not jittered by joint recording noise / NN quantization (``--smooth-cutoff-hz``, default 5 Hz;
    ``--no-smooth`` to disable).

One LeRobot dataset is produced per embodiment: ARX5, UR5 (single-arm), ALOHA, DOS-W1 (dual-arm).

Fields follow ``../dataset.md``: each pose is stored twice -- native ``raw_state.*`` (no transform)
and canonically axis-aligned ``state.*``. Single-arm shown; dual-arm mirrors with left_/right_:
    observation.images.<cam>      video HxWx3 (single: cam_1/2/3; dual: cam_high/left_wrist/right_wrist)
    raw_state.joint_pos (6)       joint angles (rad), native
    raw_state.eef_xyz (3)         FK EEF translation, native arm-base frame (no alignment)
    raw_state.eef_rot6d (6)       FK EEF orientation, native, rot6d
    raw_state.gripper_state (1)   raw gripper width (m)
    state.joint_pos (6)           = raw_state.joint_pos (joints are frame-independent)
    state.eef_xyz (3)             canonical FK EEF translation (world -> I)
    state.eef_rot6d (6)           canonical FK EEF orientation (gripper -> OpenCV per robot), rot6d
    state.gripper_state (1)       gripper width / per-robot max, clip [0,1], 0=closed 1=open
    debug.gripper_eef_xyz (3)     GT-next delta translation in the canonical gripper frame (last step no-op)
    debug.gripper_eef_rot6d(6)    GT-next delta rotation in the canonical gripper frame (last step identity)
    success (1)                   rollout completion flag (constant per episode; NaN if unknown)
    score (1)                     rollout score (constant per episode; NaN if unknown)

Per-episode provenance is written to ``meta/robochallenge_metadata.jsonl``.

Example:
    python convert.py --data-dir /path/to/robochallenge/data --local-dir /path/out \
        --robot arx5 --num-proc 8
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from alignment import transforms_numpy as tn  # noqa: E402
import rc_rrd  # noqa: E402
from rc_fk import SerialChainFK  # noqa: E402

IMG_SHAPE = (480, 640, 3)  # default; per-robot override via ROBOTS[...]["img_shape"] (e.g. UR5)
XYZ = ["x", "y", "z"]
ROT6D = ["rot1", "rot2", "rot3", "rot4", "rot5", "rot6"]
JOINTS = [f"joint_{i}" for i in range(1, 7)]
_IDENTITY_ROT6D = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32)

# Camera entity (in the .rrd) -> LeRobot image key. Dual-arm names are semantic; single-arm
# index->view mapping is not documented, so the numeric names are kept as-is.
CAMS_SINGLE = {"/videos_1": "cam_1", "/videos_2": "cam_2", "/videos_3": "cam_3"}
CAMS_DUAL = {"/videos_front": "cam_high", "/videos_left": "cam_left_wrist", "/videos_right": "cam_right_wrist"}

# Per-robot config. tip/tool_offset validated in validate_fk.py against HF ee_positions.
# gripper_max = physical max opening (m) used to normalize the gripper width to [0, 1].
# gripper_align = native gripper -> canonical OpenCV relabel; drives both state.eef_* and
# debug.* (world align is identity for all robots). Status after the sample-video review:
#   ALOHA  = ("y","-x","z") both arms (FK validated vs HF left-arm ee to 0.027deg -> frame is correct
#            & right-handed). Same PiPER relabel as vifailback; a 30-episode / 10-task sample confirmed
#            the per-arm action matches the wrist-camera motion for BOTH arms (see README) -- there is
#            NO left/right mirror flip (the base mounting cancels in the gripper-frame action).
#   UR5    = None            -- native frame confirmed correct by reviewer
#   ARX5   = ("z","-x","-y") -- confirmed from action viz (native x=approach,y=left,z=up -> OpenCV)
#   DOS-W1 = None            -- UNVERIFIED placeholder (reviewer could not judge from video)
# For the unverified DOS-W1, state.eef_rot6d == raw_state.eef_rot6d until a relabel is set. The
# optical-flow probe in infer_gripper_axes.py failed its PiPER validation (1/3 axes, R^2<=0.12),
# so nothing is baked in for them. See README "Conversion status".
ROBOTS = {
    "arx5": dict(
        robot_type="arx5", dual=False, urdf="assets/arx5.urdf",
        base="base_link", tip="eef_link", tool_offset=(0.0, 0.0, 0.0),
        # Native gripper frame (confirmed from the action viz): x=approach, y=left, z=up. Relabel to
        # OpenCV (z=fwd, x=right, y=down): native x->z, y->-x (left=-right), z->-y (up=-down).
        gripper_max=0.088, gripper_align=("z", "-x", "-y"),
    ),
    "ur5": dict(
        robot_type="ur5", dual=False, urdf="assets/ur5.urdf",
        base="base_link", tip="wrist_3_link", tool_offset=(0.0, 0.0, 0.0),
        gripper_max=0.086, gripper_align=None,
        img_shape=(720, 1280, 3),  # UR5 streams are natively 720x1280 (others 480x640)
    ),
    "aloha": dict(
        robot_type="aloha_agilex_piper", dual=True,
        urdf="../vifailback2lerobot/assets/piper_description.urdf",
        base="base_link", tip="link6", tool_offset=(0.0, 0.0, 0.0),
        # Both arms use the SAME PiPER native gripper -> OpenCV relabel (vifailback). FK is validated
        # against the HF LEFT-arm ee_positions to 0.027deg (validate_fk.py), so each arm's computed
        # gripper frame is physically correct & right-handed -> this align gives both arms canonical
        # OpenCV in their OWN frame. The per-step action is expressed in each arm's own gripper body
        # frame (T_t^-1 T_{t+1}), so the arm's base mounting cancels: identical actions produce
        # identical wrist-camera motion on BOTH arms -- confirmed on a 30-episode / 10-task sample,
        # same as vifailback. (There is NO left/right mirror/reflection; an earlier note claiming one
        # was wrong.) Per-arm gripper_align IS supported (dict) if a genuine per-arm need ever arises.
        gripper_max=0.10, gripper_align=("y", "-x", "z"),
    ),
    "dos_w1": dict(
        robot_type="dexmal_dos_w1", dual=True, urdf="assets/dos_w1.urdf",
        base="base_link", tip="end_link", tool_offset=(0.0992, 0.0004, -0.00197),
        gripper_max=0.066, gripper_align=None,
    ),
}


# --------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------
def build_features(cfg: dict, use_videos: bool = True) -> dict:
    cams = (CAMS_DUAL if cfg["dual"] else CAMS_SINGLE).values()
    img_shape = cfg.get("img_shape", IMG_SHAPE)
    features = {
        f"observation.images.{cam}": {
            "dtype": "video" if use_videos else "image",
            "shape": img_shape,
            "names": ["height", "width", "rgb"],
        }
        for cam in cams
    }
    sides = ["left", "right"] if cfg["dual"] else [""]
    for s in sides:
        pre = f"{s}_" if s else ""
        # raw_state.* -- native (no axis alignment); FK EEF pose in the native arm-base frame.
        features[f"raw_state.{pre}joint_pos"] = {"dtype": "float32", "shape": (6,), "names": JOINTS}
        features[f"raw_state.{pre}eef_xyz"] = {"dtype": "float32", "shape": (3,), "names": XYZ}
        features[f"raw_state.{pre}eef_rot6d"] = {"dtype": "float32", "shape": (6,), "names": ROT6D}
        features[f"raw_state.{pre}gripper_state"] = {"dtype": "float32", "shape": (1,), "names": ["width_m"]}
        # state.* -- canonically axis-aligned (world -> I, gripper -> OpenCV per robot).
        features[f"state.{pre}joint_pos"] = {"dtype": "float32", "shape": (6,), "names": JOINTS}
        features[f"state.{pre}eef_xyz"] = {"dtype": "float32", "shape": (3,), "names": XYZ}
        features[f"state.{pre}eef_rot6d"] = {"dtype": "float32", "shape": (6,), "names": ROT6D}
        features[f"state.{pre}gripper_state"] = {"dtype": "float32", "shape": (1,), "names": ["gripper"]}
    for s in sides:
        pre = f"{s}_" if s else ""
        features[f"debug.{pre}gripper_eef_xyz"] = {"dtype": "float32", "shape": (3,), "names": XYZ}
        features[f"debug.{pre}gripper_eef_rot6d"] = {"dtype": "float32", "shape": (6,), "names": ROT6D}
    features["success"] = {"dtype": "float32", "shape": (1,), "names": ["success"]}
    features["score"] = {"dtype": "float32", "shape": (1,), "names": ["score"]}
    return features


# --------------------------------------------------------------------------------------
# Rollout discovery
# --------------------------------------------------------------------------------------
def robot_key(meta: dict) -> str | None:
    """Normalize a run's robot to one of the ROBOTS keys."""
    r = meta.get("hardware", {}).get("robot", "")
    if not r:
        for t in meta.get("hardware", {}).get("task_tags", []):
            if t in ("DOS-W1", "ALOHA", "ARX5", "UR5"):
                r = t
                break
    return {"ARX5": "arx5", "UR5": "ur5", "ALOHA": "aloha", "DOS-W1": "dos_w1"}.get(r)


def discover_rollouts(data_dir: Path, robot: str) -> list[dict]:
    """All downloaded rollouts for one robot: (rrd path, run meta, rollout meta), sorted stably."""
    root = data_dir / "table30_v2"
    jobs = []
    for meta_path in sorted(root.glob("*/*/metadata.json")):
        run = json.loads(meta_path.read_text())
        if robot_key(run) != robot:
            continue
        by_id = {r.get("rollout_id"): r for r in run.get("rollouts", [])}
        for rrd in sorted((meta_path.parent / "rollouts").glob("*.rrd")):
            rid = rrd.stem
            jobs.append({"rrd": rrd, "run": run, "rollout": by_id.get(rid, {"rollout_id": rid})})
    return jobs


# --------------------------------------------------------------------------------------
# Per-rollout processing
# --------------------------------------------------------------------------------------
def resolve_gripper_align(cfg, side):
    """This arm's native-gripper -> OpenCV relabel. ``gripper_align`` may be a single spec (both
    arms / single-arm) or a per-side dict ``{'left': ..., 'right': ...}`` for a dual robot whose two
    grippers genuinely differ. ALOHA uses a single shared spec (both arms are identical PiPER and,
    as verified, need no per-arm distinction)."""
    ga = cfg["gripper_align"]
    return ga[side] if isinstance(ga, dict) else ga


def _arm_state(arm, grid, cfg, fk, fps, cutoff_hz, order, smooth, gripper_align):
    """Resample one arm's (times, joints, gripper) onto ``grid`` -> raw_state + state + debug.

    ``raw_state.*`` is the un-smoothed native record (nearest-neighbour resample + FK). ``state.*``
    (and the differenced ``debug.*`` action) use joints linearly interpolated onto the grid and
    zero-phase Butterworth low-pass filtered before FK, so the action is not jittered by joint
    recording noise or NN-quantization. ``smooth=False`` keeps the linear-interp joints unfiltered.
    ``gripper_align`` is this arm's native-gripper -> OpenCV relabel (per-arm for dual robots).
    """
    # raw_state: nearest-neighbour resample of the native recording (kept un-smoothed).
    idx = rc_rrd.nearest_indices(arm["times"], grid)
    raw_joints = arm["joints"][idx].astype(np.float32)      # (T, 6)
    width = arm["gripper"][idx][:, None].astype(np.float32)  # (T, 1) raw meters
    R_raw, p_raw = fk(raw_joints)                            # native arm-base frame

    # state: linear-interp joints onto the grid, low-pass, then FK (smooth pose -> smooth action).
    sm_joints = rc_rrd.resample_linear(arm["times"], arm["joints"], grid)
    if smooth:
        sm_joints = rc_rrd.smooth_butter(sm_joints, fps, cutoff_hz, order)
    sm_joints = sm_joints.astype(np.float32)
    R, p = fk(sm_joints)
    grip_norm = np.clip(width / cfg["gripper_max"], 0.0, 1.0).astype(np.float32)

    # Canonical pose: world -> identity, gripper -> OpenCV relabel (per arm; None = native).
    R_align = None if gripper_align is None else tn.axis_alignment_matrix(*gripper_align)
    R_c, p_c = tn.align_axis(R, p, np.eye(3, dtype=np.float32), R_align)

    # Debug: GT-next delta in the canonical gripper frame (last step no-op).
    R_g, p_g = tn.gripper_delta_pose(R_c[:-1], p_c[:-1], R_c[1:], p_c[1:])
    dbg_xyz = np.concatenate([p_g, np.zeros((1, 3), np.float32)]).astype(np.float32)
    dbg_rot = np.concatenate([tn.matrix_to_rotation_6d(R_g), _IDENTITY_ROT6D]).astype(np.float32)
    return {
        "raw_joint_pos": raw_joints,
        "raw_eef_xyz": p_raw.astype(np.float32),
        "raw_eef_rot6d": tn.matrix_to_rotation_6d(R_raw).astype(np.float32),
        "raw_gripper_state": width,
        "joint_pos": sm_joints,  # smoothed (state.* is the smoothed record)
        "eef_xyz": p_c.astype(np.float32),
        "eef_rot6d": tn.matrix_to_rotation_6d(R_c).astype(np.float32),
        "gripper_state": grip_norm,
        "debug_gripper_eef_xyz": dbg_xyz,
        "debug_gripper_eef_rot6d": dbg_rot,
    }


def process_rollout(job: dict, cfg: dict, fps: int, fk, cutoff_hz: float, order: int, smooth: bool):
    """Return (T, state_data dict, camera-stream dict, instruction, provenance).

    ``state_data`` holds small (T, d) arrays. Camera frames are NOT materialized here; the
    returned streams yield frames on demand (O(1) memory) during the write loop.
    """
    import rerun.dataframe as rd

    rec = rd.load_recording(str(job["rrd"]))
    prefixes = (["left_arm", "right_arm"] if cfg["dual"] else ["arm"])
    sides = (["left", "right"] if cfg["dual"] else [""])

    arms = {pfx: rc_rrd.read_arm(rec, pfx) for pfx in prefixes}
    # Grid spans the union of the arm(s) joint timelines.
    t_start = min(a["times"][0] for a in arms.values())
    t_end = max(a["times"][-1] for a in arms.values())
    grid = rc_rrd.grid_times(t_start, t_end, fps)
    T = len(grid)

    state_data = {}
    for pfx, side in zip(prefixes, sides):
        arm = _arm_state(arms[pfx], grid, cfg, fk, fps, cutoff_hz, order, smooth,
                         resolve_gripper_align(cfg, side))
        pre = f"{side}_" if side else ""
        state_data[f"raw_state.{pre}joint_pos"] = arm["raw_joint_pos"]
        state_data[f"raw_state.{pre}eef_xyz"] = arm["raw_eef_xyz"]
        state_data[f"raw_state.{pre}eef_rot6d"] = arm["raw_eef_rot6d"]
        state_data[f"raw_state.{pre}gripper_state"] = arm["raw_gripper_state"]
        state_data[f"state.{pre}joint_pos"] = arm["joint_pos"]
        state_data[f"state.{pre}eef_xyz"] = arm["eef_xyz"]
        state_data[f"state.{pre}eef_rot6d"] = arm["eef_rot6d"]
        state_data[f"state.{pre}gripper_state"] = arm["gripper_state"]
        state_data[f"debug.{pre}gripper_eef_xyz"] = arm["debug_gripper_eef_xyz"]
        state_data[f"debug.{pre}gripper_eef_rot6d"] = arm["debug_gripper_eef_rot6d"]

    rollout = job["rollout"]
    success = rollout.get("completion")
    score = rollout.get("score")
    state_data["success"] = np.full((T, 1), np.nan if success is None else float(success), np.float32)
    state_data["score"] = np.full((T, 1), np.nan if score is None else float(score), np.float32)

    cam_map = CAMS_DUAL if cfg["dual"] else CAMS_SINGLE
    present = set(rc_rrd.camera_entities(rec, cfg["dual"]))
    img_shape = cfg.get("img_shape", IMG_SHAPE)
    streams = {}
    for entity, key in cam_map.items():
        if entity in present:
            streams[key] = rc_rrd.NearestFrameStream(
                rc_rrd.decode_video_frames(rec, entity), img_shape
            )
        else:
            streams[key] = None  # missing camera -> black-pad

    run = job["run"]
    instruction = run.get("prompt") or run.get("task_description") or run.get("task_name", "")
    provenance = {
        "task_name": run.get("task_name"), "task_id": run.get("task_id"),
        "run_id": run.get("run_id"), "rollout_id": rollout.get("rollout_id"),
        "robot": run.get("hardware", {}).get("robot") or "DOS-W1",
        "model_name": run.get("model_name"), "display_name": run.get("display_name"),
        "user_name": run.get("user_name"), "is_ranked": run.get("is_ranked"),
        "is_multi_task_model": run.get("is_multi_task_model"),
        "arenas": run.get("hardware", {}).get("arenas"),
        "task_tags": run.get("hardware", {}).get("task_tags"),
        "run_score": run.get("score"), "run_success_rate": run.get("success_rate"),
        "rollout_score": score, "rollout_completion": success,
        "rollout_status": rollout.get("status"), "comments": rollout.get("comments"),
        "execution_time_utc": run.get("execution_time_utc"),
        "prompt": run.get("prompt"), "task_description": run.get("task_description"),
        "n_frames": T, "fps": fps,
    }
    return T, state_data, streams, grid, instruction, provenance


def add_rollout_frames(dataset, T, state_data, streams, grid, instruction, black):
    """Stream frames into the dataset one at a time (cameras pulled lazily per grid step).

    ``black`` is the per-robot black frame (``img_shape``) used to pad a missing camera.
    """
    img_keys = {k: f"observation.images.{k}" for k in streams}
    for i in range(T):
        frame = {k: v[i] for k, v in state_data.items()}
        gi = grid[i]
        for cam, stream in streams.items():
            frame[img_keys[cam]] = stream.at(gi) if stream is not None else black
        frame["task"] = instruction
        dataset.add_frame(frame)


# --------------------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------------------
def convert_worker(args, jobs, local_dir: Path, repo_id: str, desc: str, position: int = 0):
    cfg = ROBOTS[args.robot]
    fk = SerialChainFK(
        os.path.join(_HERE, cfg["urdf"]) if not os.path.isabs(cfg["urdf"]) else cfg["urdf"],
        base_link=cfg["base"], tip_link=cfg["tip"], tool_offset=cfg["tool_offset"],
    )
    black = np.zeros(cfg.get("img_shape", IMG_SHAPE), dtype=np.uint8)  # missing-camera pad
    sentinel = local_dir / "meta" / ".conversion_complete"
    progress_file = local_dir / "meta" / "_progress.json"
    metadata_file = local_dir / "meta" / "robochallenge_metadata.jsonl"
    if args.overwrite and local_dir.exists():
        shutil.rmtree(local_dir)
    if sentinel.exists():
        print(f"[convert] {local_dir.name}: already complete; skipping.")
        return None

    dataset, consumed = None, 0
    if local_dir.exists():
        try:
            dataset = LeRobotDataset.resume(
                repo_id=repo_id, root=local_dir,
                image_writer_processes=args.image_writer_process,
                image_writer_threads=args.image_writer_threads,
            )
            prog = 0
            if progress_file.exists():
                prog = int(json.loads(progress_file.read_text()).get("consumed", 0))
            consumed = max(dataset.meta.total_episodes, prog)
            print(f"[convert] {local_dir.name}: resuming after {consumed} rollouts.")
        except Exception as e:
            print(f"[convert] {local_dir.name}: cannot resume ({type(e).__name__}: {e}); rebuilding.")
            shutil.rmtree(local_dir)
            dataset, consumed = None, 0

    if dataset is None:
        dataset = LeRobotDataset.create(
            repo_id=repo_id, robot_type=cfg["robot_type"], root=local_dir, fps=args.fps,
            use_videos=args.use_videos, features=build_features(cfg, args.use_videos),
            image_writer_processes=args.image_writer_process,
            image_writer_threads=args.image_writer_threads,
        )

    todo = jobs[consumed:]
    if args.max_episodes is not None:
        todo = todo[: max(0, args.max_episodes - consumed)]
    if not todo:
        dataset.finalize()
        _write_sentinel(sentinel, dataset)
        return dataset

    ep_index = dataset.meta.total_episodes
    pbar = tqdm(total=len(todo), desc=desc, unit="ep", position=position, dynamic_ncols=True)
    for job in todo:
        try:
            T, state_data, streams, grid, instruction, prov = process_rollout(
                job, cfg, args.fps, fk, args.smooth_cutoff_hz, args.smooth_order, args.smooth
            )
            add_rollout_frames(dataset, T, state_data, streams, grid, instruction, black)
            dataset.save_episode()
            metadata_file.parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_file, "a") as mf:
                mf.write(json.dumps({"episode_index": ep_index, **prov}) + "\n")
            ep_index += 1
        except Exception as e:
            if not args.skip_bad_episodes:
                raise
            if dataset.has_pending_frames():
                dataset.clear_episode_buffer()
            tqdm.write(f"[{desc}] skip {job['rrd'].name}: {type(e).__name__}: {e}")
        consumed += 1
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        progress_file.write_text(json.dumps({"consumed": consumed}))
        pbar.update(1)
    pbar.close()

    dataset.finalize()
    if args.max_episodes is None:
        _write_sentinel(sentinel, dataset)
    return dataset


def _write_sentinel(sentinel: Path, dataset) -> None:
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(json.dumps({"total_episodes": dataset.meta.total_episodes}))


# --------------------------------------------------------------------------------------
# Parallel orchestration (contiguous rollout shards + merge) -- mirrors soar2lerobot
# --------------------------------------------------------------------------------------
def run_parallel(args):
    n = args.num_proc
    jobs = discover_rollouts(args.data_dir, args.robot)
    print(f"[parallel] {args.robot}: {len(jobs)} rollouts across {n} workers")
    bounds = np.linspace(0, len(jobs), n + 1).astype(int)
    shards_root = args.local_dir / f"_shards_{args.robot}"
    tags = [f"shard{i:03d}" for i in range(n)]
    worker_roots = [shards_root / f"robochallenge_{args.robot}_lerobot_{t}" for t in tags]
    worker_repo_ids = [f"{args.repo_id}_{t}" for t in tags]
    log_dir = shards_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    sentinel_of = lambda i: worker_roots[i] / "meta" / ".conversion_complete"

    worker_env = {**os.environ, "MALLOC_ARENA_MAX": "2", "OMP_NUM_THREADS": "2"}
    procs, handles = [], []
    for i in range(n):
        if bounds[i] == bounds[i + 1]:
            procs.append(None); handles.append(None); continue
        if sentinel_of(i).exists() and not args.overwrite:
            print(f"[parallel] {tags[i]} complete; skipping.")
            procs.append(None); handles.append(None); continue
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--data-dir", str(args.data_dir), "--local-dir", str(shards_root),
            "--robot", args.robot, "--repo-id", worker_repo_ids[i],
            "--shard-index", str(i), "--num-shards", str(n), "--fps", str(args.fps),
            "--image-writer-process", str(args.image_writer_process),
            "--image-writer-threads", str(args.image_writer_threads),
            "--smooth-cutoff-hz", str(args.smooth_cutoff_hz),
            "--smooth-order", str(args.smooth_order),
        ]
        if not args.smooth:
            cmd.append("--no-smooth")
        if args.use_videos:
            cmd.append("--use-videos")
        if args.skip_bad_episodes:
            cmd.append("--skip-bad-episodes")
        if args.overwrite:
            cmd.append("--overwrite")
        lf = open(log_dir / f"{tags[i]}.log", "a")
        handles.append(lf)
        print(f"[parallel] {tags[i]} [{bounds[i]}:{bounds[i+1]}] -> {log_dir/(tags[i]+'.log')}")
        procs.append(subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=worker_env))

    rcs = [(p.wait() if p is not None else 0) for p in procs]
    for lf in handles:
        if lf is not None:
            lf.close()
    incomplete = [i for i in range(n) if bounds[i] != bounds[i + 1] and not sentinel_of(i).exists()]
    if incomplete:
        raise RuntimeError(f"[parallel] shards {incomplete} incomplete (rc={rcs}); inspect {log_dir}.")

    aggr_root = args.local_dir / f"robochallenge_{args.robot}_lerobot"
    if aggr_root.exists():
        shutil.rmtree(aggr_root)
    active = [i for i in range(n) if bounds[i] != bounds[i + 1]]
    print(f"[parallel] merging {len(active)} shards -> {aggr_root}")
    aggregate_datasets(
        repo_ids=[worker_repo_ids[i] for i in active],
        aggr_repo_id=args.repo_id,
        roots=[worker_roots[i] for i in active],
        aggr_root=aggr_root,
    )
    offset = 0
    with open(aggr_root / "meta" / "robochallenge_metadata.jsonl", "w") as out:
        for i in active:
            n_eps = int(json.loads(sentinel_of(i).read_text())["total_episodes"])
            mf = worker_roots[i] / "meta" / "robochallenge_metadata.jsonl"
            if mf.exists():
                for line in mf.read_text().splitlines():
                    rec = json.loads(line)
                    rec["episode_index"] += offset
                    out.write(json.dumps(rec) + "\n")
            offset += n_eps
    shutil.rmtree(shards_root, ignore_errors=True)
    print(f"[parallel] done -> {aggr_root}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True, help="RoboChallenge crawled data dir (has table30_v2/).")
    p.add_argument("--local-dir", type=Path, required=True, help="Output dir for the LeRobot dataset(s).")
    p.add_argument("--robot", required=True, choices=list(ROBOTS), help="Embodiment to convert.")
    p.add_argument("--repo-id", default=None)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--use-videos", action="store_true", default=True)
    p.add_argument("--image-writer-process", type=int, default=4)
    p.add_argument("--image-writer-threads", type=int, default=8)
    p.add_argument("--num-proc", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=None, help="Internal: worker shard id.")
    p.add_argument("--num-shards", type=int, default=None, help="Internal: total shard count.")
    p.add_argument("--max-episodes", type=int, default=None, help="Debug: convert at most this many rollouts.")
    p.add_argument("--skip-bad-episodes", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    # state.* joint smoothing (zero-phase Butterworth low-pass before FK); raw_state.* stays un-smoothed.
    p.add_argument("--smooth-cutoff-hz", type=float, default=5.0, help="Butterworth low-pass cutoff (Hz) for state.* joints.")
    p.add_argument("--smooth-order", type=int, default=2, help="Butterworth filter order for state.* joints.")
    p.add_argument("--no-smooth", dest="smooth", action="store_false", help="Disable state.* joint smoothing (linear-interp only).")
    p.set_defaults(smooth=True)
    args = p.parse_args()
    if args.repo_id is None:
        args.repo_id = f"robochallenge_{args.robot}"

    if args.num_proc > 1 and args.shard_index is None:
        run_parallel(args)
        return

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    jobs = discover_rollouts(args.data_dir, args.robot)
    if args.shard_index is not None:
        bounds = np.linspace(0, len(jobs), args.num_shards + 1).astype(int)
        jobs = jobs[bounds[args.shard_index]:bounds[args.shard_index + 1]]
        local_dir = args.local_dir / f"robochallenge_{args.robot}_lerobot_shard{args.shard_index:03d}"
        desc, position = f"shard{args.shard_index:03d}", args.shard_index
    else:
        local_dir = args.local_dir / f"robochallenge_{args.robot}_lerobot"
        desc, position = f"robochallenge_{args.robot}", 0
    convert_worker(args, jobs, local_dir, args.repo_id, desc, position)


if __name__ == "__main__":
    main()
