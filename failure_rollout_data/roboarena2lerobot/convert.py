#!/usr/bin/env python
"""Convert the RoboArena dataset (Franka / DROID autonomous policy rollouts) to LeRobot v3.

RoboArena (https://robo-arena.github.io/) is a distributed real-world evaluation benchmark: for
each evaluation *session* a human sets up a scene, gives a language instruction, and several
generalist policies each attempt the task; every attempt is scored (binary + partial success) and
the session carries a pairwise policy *preference*. Robots are the DROID Franka Panda setup.

Raw layout (two dated dumps, each converted into its own LeRobot dataset)::

    <raw_dir>/                                     # e.g. .../roboarena/DataDump_08-05-2025
    ├── global_metadata.yaml                       # policy_index -> {open_source, action_space}
    └── evaluation_sessions/
        └── <session_uuid>/
            ├── metadata.yaml                      # instruction, preference, per-policy scores
            └── <LETTER>_<policy_name>/            # ONE episode (one policy's rollout)
                ├── *_npz_file.npz                 # proprio + actions
                └── *_video_{left,right,wrist}.mp4 # some sessions omit `left`

Each ``*_npz_file.npz`` holds a single object array ``data`` of length T; each step is a dict::

    cartesian_position (6)  eef [x, y, z, roll, pitch, yaw], base frame, extrinsic-XYZ euler (DROID)
    joint_position     (7)  arm joint angles
    gripper_position   (1)  CONTINUOUS gripper width, raw DROID polarity (larger = more closed)
    action             (8)  action[:7] = 7 arm joint-space command values, action[7] = gripper. The
                            joint command's UNIT is per-policy: rad/s for a ``joint_velocity`` policy,
                            rad (absolute next-step target) for a ``joint_position`` policy -- BOTH
                            occur in the 02-03 dump (8 velocity / 7 position policies), so it is routed
                            to ``raw_target.joint_vel`` / ``raw_target.joint_pos`` (inactive one
                            0-filled) and tagged by ``control_is_position``. There is NO commanded
                            cartesian target. The gripper channel is binary {0, 1}.

Videos have T+1 frames (one more than proprio); we keep the first T. All views are stored AS SHIPPED
(un-flipped). The wrist camera is the DROID Zed-Mini, physically mounted rolled 180 deg, so a consumer
co-training with other datasets should apply a 180 deg image rotation to the wrist (a proper rotation;
see decode_video / README.md) to align its image axes with the canonical gripper frame.

Output schema follows dataset.md: raw_state.* / raw_target.* (native, no transform) and state.* /
target.* (canonically axis-aligned). The per-step ACTION is computed at load time from state.*/target.*
(dataset.md 3); ``debug.*`` holds a precomputed GT-next reference in the canonical gripper frame::

    observation.images.{left,right,wrist}  video 288x512x3 (missing camera -> black frames)
    raw_state.joint_pos (7)       raw joint_position
    raw_state.eef_xyz (3)         raw cartesian_position[:3], base frame
    raw_state.eef_rpy (3)         raw extrinsic-XYZ rpy passthrough (shipped rotation rep)
    raw_state.gripper_state (1)   RAW continuous gripper_position (larger = more closed; NOT inverted)
    raw_target.joint_pos (7)      action[:7] for a joint_position policy, else 0.0 (absolute rad target)
    raw_target.joint_vel (7)      action[:7] for a joint_velocity policy, else 0.0 (absolute rad/s)
    raw_target.gripper_state (1)  raw commanded gripper, binary {0, 1} (NOT inverted; DROID polarity)
    state.joint_pos (7)           canonical joint_pos (frame-independent, = raw_state.joint_pos)
    state.eef_xyz (3)             canonical eef translation (base already FLU -> identity)
    state.eef_rot9d (9)           canonical eef rotation (Franka hand -> OpenCV gripper), full row-major matrix
    state.gripper_state (1)       1 - gripper_position (canonical: 0=closed, 1=open)
    target.joint_pos (7)          canonical joint_pos target (= raw_target.joint_pos)
    target.joint_vel (7)          canonical joint_vel target (= raw_target.joint_vel)
    target.gripper_state (1)      1 - raw binary gripper command (canonical: 0=closed, 1=open)
    control_is_position (1)       1.0 if joint_position (target.joint_pos active, differenced at load),
                                  0.0 if joint_velocity (target.joint_vel used directly); const/episode
    debug.gripper_eef_xyz (3)     GT-next delta translation in the CANONICAL gripper frame (debug only)
    debug.gripper_eef_rot6d (6)   GT-next relative rotation, canonical gripper frame; last step no-op
    binary_success (1)            episode binary success label (constant per episode)
    partial_success (1)           episode partial-success score in [0, 1] (constant per episode)

Per-episode provenance is written to ``meta/roboarena_metadata.jsonl`` (session id, policy, scores,
preference, evaluator, timestamps, which cameras were real vs padded, ...).

Example::

    python convert.py \
        --raw-dir /path/to/roboarena/DataDump_08-05-2025 \
        --local-dir /path/to/output \
        --dataset-name roboarena_2025_08_05 \
        --num-proc 8
"""

import argparse
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Shared backend-agnostic rotation/frame math lives in the repo-root ``alignment`` package. This
# script sits at <repo>/failure_rollout_data/roboarena2lerobot/, so the repo root is three levels up.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from alignment import transforms_numpy as tn  # noqa: E402

CAMERAS = ("left", "right", "wrist")
IMG_SHAPE = (288, 512, 3)  # (H, W, C)
ROT6D_NAMES = ["r11", "r21", "r31", "r12", "r22", "r32"]
ROT9D_NAMES = [f"r{row}{col}" for row in range(1, 4) for col in range(1, 4)]
XYZ_NAMES = ["x", "y", "z"]
RPY_NAMES = ["roll", "pitch", "yaw"]
JOINT_NAMES = [f"joint_{i}" for i in range(7)]
ROT6D_IDENTITY = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32)

# Franka panda_hand native EEF -> canonical OpenCV gripper (z=approach, x=finger-open "right",
# y=down). Identical to openx2lerobot's ``_GRIPPER_ALIGN_FRANKA``. World is already canonical FLU
# (base frame), so the world alignment is the identity. Used ONLY for the debug.* delta fields.
R_GRIPPER_ALIGN = tn.axis_alignment_matrix("-y", "x", "z")

# Frame-count tolerance between decoded video and proprio; larger mismatches drop the episode.
FRAME_MISMATCH_TOL = 2

np.set_printoptions(precision=4, suppress=True)


def build_features(use_videos: bool = True) -> dict:
    features = {
        f"observation.images.{cam}": {
            "dtype": "video" if use_videos else "image",
            "shape": IMG_SHAPE,
            "names": ["height", "width", "rgb"],
        }
        for cam in CAMERAS
    }
    # raw_state.*: native metrics, no transform / no alignment (dataset.md 2.1). rpy is the shipped
    # rotation representation (DROID cartesian_position is extrinsic-XYZ euler).
    features["raw_state.joint_pos"] = {"dtype": "float32", "shape": (7,), "names": JOINT_NAMES}
    features["raw_state.eef_xyz"] = {"dtype": "float32", "shape": (3,), "names": XYZ_NAMES}
    features["raw_state.eef_rpy"] = {"dtype": "float32", "shape": (3,), "names": RPY_NAMES}
    features["raw_state.gripper_state"] = {"dtype": "float32", "shape": (1,), "names": ["gripper"]}
    # raw_target.*: controller command, no transform (dataset.md 2.2). The joint command's UNIT is
    # per-policy, so it lives in the matching field (joint_pos = absolute rad target for a
    # joint_position policy, joint_vel = absolute rad/s for a joint_velocity policy); the inactive
    # column is 0-filled and disambiguated per-frame by `control_is_position`.
    features["raw_target.joint_pos"] = {"dtype": "float32", "shape": (7,), "names": JOINT_NAMES}
    features["raw_target.joint_vel"] = {"dtype": "float32", "shape": (7,), "names": JOINT_NAMES}
    features["raw_target.gripper_state"] = {"dtype": "float32", "shape": (1,), "names": ["gripper"]}
    # state.*: canonical, axis-aligned (dataset.md 2.3). Joints are frame-independent (copied);
    # eef is re-based onto the canonical base (identity) + OpenCV gripper frame; gripper normalized
    # to 0=closed / 1=open. No canonical eef_rpy -- the canonical orientation is stored as rot9d.
    features["state.joint_pos"] = {"dtype": "float32", "shape": (7,), "names": JOINT_NAMES}
    features["state.eef_xyz"] = {"dtype": "float32", "shape": (3,), "names": XYZ_NAMES}
    features["state.eef_rot9d"] = {"dtype": "float32", "shape": (9,), "names": ROT9D_NAMES}
    features["state.gripper_state"] = {"dtype": "float32", "shape": (1,), "names": ["gripper"]}
    # target.*: canonical target (dataset.md 2.4). Joints frame-independent (copied from raw_target);
    # gripper inverted to canonical 0=closed / 1=open.
    features["target.joint_pos"] = {"dtype": "float32", "shape": (7,), "names": JOINT_NAMES}
    features["target.joint_vel"] = {"dtype": "float32", "shape": (7,), "names": JOINT_NAMES}
    features["target.gripper_state"] = {"dtype": "float32", "shape": (1,), "names": ["gripper"]}
    # Control-mode flag: 1.0 if the policy is joint_position (target.joint_pos active, differenced
    # at load time), 0.0 if joint_velocity (target.joint_vel used directly). Constant per episode.
    features["control_is_position"] = {"dtype": "float32", "shape": (1,), "names": ["is_position"]}
    features["debug.gripper_eef_xyz"] = {"dtype": "float32", "shape": (3,), "names": XYZ_NAMES}
    features["debug.gripper_eef_rot6d"] = {"dtype": "float32", "shape": (6,), "names": ROT6D_NAMES}
    features["binary_success"] = {"dtype": "float32", "shape": (1,), "names": ["binary_success"]}
    features["partial_success"] = {"dtype": "float32", "shape": (1,), "names": ["partial_success"]}
    return features


# --------------------------------------------------------------------------------------
# Video / npz loading
# --------------------------------------------------------------------------------------
def decode_video(path: str, num_frames: int, flip180: bool) -> np.ndarray:
    """Decode an mp4 to (num_frames, H, W, 3) uint8 RGB via pyav.

    The RoboArena videos carry T+1 frames (one more than proprio); we return exactly the first
    ``num_frames`` so frame[t] aligns with proprio step t. ``flip180`` rotates each frame 180 deg
    (reverse H and W -- a proper rotation, not a mirror); it is OFF by default here (all views stored
    as shipped). The DROID Zed-Mini wrist is physically mounted rolled 180 deg, so a consumer
    co-training with other datasets should pass ``flip180=True`` for the wrist to align its image axes
    with the canonical gripper frame. Raises if the decoded count differs from ``num_frames`` by more
    than ``FRAME_MISMATCH_TOL``.
    """
    import av

    frames = []
    with av.open(path) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= num_frames + FRAME_MISMATCH_TOL + 1:
                break
    if abs(len(frames) - num_frames) > FRAME_MISMATCH_TOL:
        raise ValueError(f"{os.path.basename(path)}: {len(frames)} frames vs proprio {num_frames}")
    if len(frames) < num_frames:
        raise ValueError(f"{os.path.basename(path)}: only {len(frames)} frames, need {num_frames}")
    arr = np.stack(frames[:num_frames])
    if flip180:
        arr = arr[:, ::-1, ::-1, :]
    return np.ascontiguousarray(arr, dtype=np.uint8)


def load_npz_steps(npz_path: str):
    """Load one episode's npz -> stacked per-step arrays (cartesian, joint, gripper, action)."""
    data = np.load(npz_path, allow_pickle=True)["data"]
    cartesian = np.stack([s["cartesian_position"] for s in data]).astype(np.float32)  # (T, 6)
    joint = np.stack([s["joint_position"] for s in data]).astype(np.float32)          # (T, 7)
    gripper = np.stack([s["gripper_position"] for s in data]).astype(np.float32)      # (T, 1)
    action = np.stack([s["action"] for s in data]).astype(np.float32)                 # (T, 8)
    return cartesian, joint, gripper, action


def process_episode(ep: "Episode") -> dict:
    """Build the per-step LeRobot feature arrays for one policy rollout."""
    cartesian, joint, gripper, action = load_npz_steps(ep.npz_path)
    T = len(cartesian)

    eef_xyz, rpy = cartesian[:, :3], cartesian[:, 3:6]
    R = tn.rpy_to_matrix(rpy, extrinsic=True)  # DROID cartesian rpy is fixed-axis (extrinsic) XYZ

    # Canonical alignment (dataset.md 2.3): world already FLU -> identity; Franka panda_hand -> OpenCV
    # gripper. R_c/p_c are the canonical state pose; the debug delta is GT-next (t -> t+1) in the
    # canonical gripper frame, with a no-op last step so every field stays length T.
    R_c, p_c = tn.align_axis(R, eef_xyz, np.eye(3, dtype=np.float32), R_GRIPPER_ALIGN)
    dR, dp = tn.gripper_delta_pose(R_c[:-1], p_c[:-1], R_c[1:], p_c[1:])

    # Joint command's UNIT is per-policy: route action[:, :7] to the matching raw_target column,
    # 0-fill the inactive one. Joints are frame-independent, so target.* == raw_target.* (copied).
    is_pos = ep.action_space == "joint_position"
    joint_cmd = action[:, :7]
    zeros7 = np.zeros((T, 7), dtype=np.float32)
    raw_target_joint_pos = joint_cmd if is_pos else zeros7
    raw_target_joint_vel = zeros7 if is_pos else joint_cmd
    raw_target_gripper = action[:, 7:8]  # raw commanded gripper, binary {0, 1} (NOT inverted)

    data = {
        # raw_state.*: native metrics, no transform (dataset.md 2.1)
        "raw_state.joint_pos": joint,
        "raw_state.eef_xyz": eef_xyz,
        "raw_state.eef_rpy": rpy,
        "raw_state.gripper_state": gripper,  # RAW continuous, NOT inverted (DROID polarity)
        # raw_target.*: controller command, no transform (dataset.md 2.2)
        "raw_target.joint_pos": raw_target_joint_pos,
        "raw_target.joint_vel": raw_target_joint_vel,
        "raw_target.gripper_state": raw_target_gripper,
        # state.*: canonical, axis-aligned (dataset.md 2.3)
        "state.joint_pos": joint,  # frame-independent, copied from raw_state
        "state.eef_xyz": p_c,
        "state.eef_rot9d": R_c.reshape(-1, 9).astype(np.float32),
        "state.gripper_state": (1.0 - gripper).astype(np.float32),  # canonical: 0=closed, 1=open
        # target.*: canonical target (dataset.md 2.4)
        "target.joint_pos": raw_target_joint_pos,  # frame-independent, copied from raw_target
        "target.joint_vel": raw_target_joint_vel,
        "target.gripper_state": (1.0 - raw_target_gripper).astype(np.float32),  # inverted binary
        "control_is_position": np.full((T, 1), 1.0 if is_pos else 0.0, dtype=np.float32),
        "debug.gripper_eef_xyz": np.concatenate([dp, np.zeros((1, 3), np.float32)]).astype(np.float32),
        "debug.gripper_eef_rot6d": np.concatenate(
            [tn.matrix_to_rotation_6d(dR), ROT6D_IDENTITY]
        ).astype(np.float32),
        "binary_success": np.full((T, 1), float(ep.binary_success), dtype=np.float32),
        "partial_success": np.full((T, 1), float(ep.partial_success), dtype=np.float32),
    }

    # Cameras: real frames where the file exists, black padding otherwise. The wrist view is stored
    # AS SHIPPED (un-flipped); the DROID Zed-Mini wrist is mounted rolled 180 deg, so consumers
    # co-training with other datasets should apply a 180 deg image rotation (see README / decode_video).
    for cam in CAMERAS:
        vid = ep.videos.get(cam)
        if vid is not None:
            data[f"observation.images.{cam}"] = decode_video(vid, T, flip180=False)
        else:
            data[f"observation.images.{cam}"] = np.zeros((T, *IMG_SHAPE), dtype=np.uint8)
    return data


# --------------------------------------------------------------------------------------
# Episode discovery (session x policy) + metadata
# --------------------------------------------------------------------------------------
class Episode:
    """One policy rollout: paths + the metadata needed to convert and to write the sidecar."""

    __slots__ = ("npz_path", "videos", "session_id", "letter", "policy_name", "instruction",
                 "binary_success", "partial_success", "duration", "preference", "evaluator_name",
                 "evaluation_location", "session_creation", "session_completion", "longform_feedback",
                 "open_source", "action_space", "cameras_present")

    def to_record(self, episode_index: int, num_frames: int) -> dict:
        return {
            "episode_index": episode_index,
            "session_id": self.session_id,
            "policy_letter": self.letter,
            "policy_name": self.policy_name,
            "language_instruction": self.instruction,
            "binary_success": self.binary_success,
            "partial_success": self.partial_success,
            "duration": self.duration,
            "preference": self.preference,
            "evaluator_name": self.evaluator_name,
            "evaluation_location": self.evaluation_location,
            "session_creation_timestamp": self.session_creation,
            "session_completion_timestamp": self.session_completion,
            "longform_feedback": self.longform_feedback,
            "open_source": self.open_source,
            "action_space": self.action_space,
            "cameras_present": self.cameras_present,
            "num_frames": num_frames,
        }


_POLICY_DIR_RE = re.compile(r"^([A-Z])_(.+)$")


def discover_episodes(raw_dir: Path) -> list["Episode"]:
    """Deterministic (sorted) list of every policy rollout across all sessions.

    Sorted by (session_uuid, policy_letter) so shard slicing and resume are stable across runs and
    workers. Policy dirs without an npz are skipped.
    """
    global_meta = {}
    gpath = raw_dir / "global_metadata.yaml"
    if gpath.exists():
        global_meta = yaml.safe_load(gpath.read_text()) or {}
    policy_index = global_meta.get("policy_index", {}) or {}

    sessions_root = raw_dir / "evaluation_sessions"
    episodes: list[Episode] = []
    for session_id in sorted(os.listdir(sessions_root)):
        sdir = sessions_root / session_id
        if not sdir.is_dir():
            continue
        smeta_path = sdir / "metadata.yaml"
        smeta = yaml.safe_load(smeta_path.read_text()) if smeta_path.exists() else {}
        smeta = smeta or {}
        policies = smeta.get("policies", {}) or {}

        for entry in sorted(os.listdir(sdir)):
            pdir = sdir / entry
            m = _POLICY_DIR_RE.match(entry)
            if not pdir.is_dir() or not m:
                continue
            letter, policy_name = m.group(1), m.group(2)
            npzs = glob.glob(str(pdir / "*.npz"))
            if not npzs:
                continue
            videos = {}
            for cam in CAMERAS:
                hits = glob.glob(str(pdir / f"*_video_{cam}.mp4"))
                if hits:
                    videos[cam] = hits[0]

            pol = policies.get(letter, {}) or {}
            pidx = policy_index.get(policy_name, {}) or {}
            ep = Episode()
            ep.npz_path = npzs[0]
            ep.videos = videos
            ep.session_id = session_id
            ep.letter = letter
            ep.policy_name = policy_name
            ep.instruction = str(smeta.get("language_instruction", "") or "")
            ep.binary_success = pol.get("binary_success")
            ep.partial_success = pol.get("partial_success")
            ep.duration = pol.get("duration")
            ep.preference = smeta.get("preference")
            ep.evaluator_name = smeta.get("evaluator_name")
            ep.evaluation_location = smeta.get("evaluation_location")
            ep.session_creation = smeta.get("session_creation_timestamp")
            ep.session_completion = smeta.get("session_completion_timestamp")
            ep.longform_feedback = smeta.get("longform_feedback")
            ep.open_source = pidx.get("open_source")
            ep.action_space = pidx.get("action_space")
            ep.cameras_present = sorted(videos.keys())
            episodes.append(ep)
    return episodes


def _success_floats(ep: "Episode") -> None:
    """Coerce success labels to floats, defaulting missing to 0.0 (recorded verbatim in the sidecar)."""
    ep.binary_success = 0.0 if ep.binary_success is None else float(ep.binary_success)
    ep.partial_success = 0.0 if ep.partial_success is None else float(ep.partial_success)


# --------------------------------------------------------------------------------------
# Conversion (single worker)
# --------------------------------------------------------------------------------------
def convert_worker(args, episodes: list["Episode"], local_dir: Path, repo_id: str, desc: str, position: int = 0):
    sentinel = local_dir / "meta" / ".conversion_complete"
    progress_file = local_dir / "meta" / "_progress.json"
    metadata_file = local_dir / "meta" / "roboarena_metadata.jsonl"
    if args.overwrite and local_dir.exists():
        shutil.rmtree(local_dir)
    if sentinel.exists():
        print(f"[convert] {local_dir.name}: already complete (sentinel present); skipping.")
        return

    # Resume dispatch: a partial dir with loadable metadata is appended to; anything else is rebuilt.
    dataset = None
    start_offset = 0
    if local_dir.exists():
        try:
            dataset = LeRobotDataset.resume(
                repo_id=repo_id,
                root=local_dir,
                image_writer_processes=args.image_writer_process,
                image_writer_threads=args.image_writer_threads,
            )
            prog_done = 0
            if progress_file.exists():
                try:
                    prog_done = int(json.loads(progress_file.read_text()).get("consumed", 0))
                except Exception:
                    prog_done = 0
            # meta counts durably-saved episodes; progress also counts deliberately-skipped bad
            # ones. Take the max so no episode is re-read or written twice.
            start_offset = max(dataset.meta.total_episodes, prog_done)
            print(f"[convert] {local_dir.name}: resuming after {start_offset} episodes.")
        except Exception as e:
            print(f"[convert] {local_dir.name}: cannot resume ({type(e).__name__}: {e}); rebuilding.")
            shutil.rmtree(local_dir)
            dataset = None
            start_offset = 0

    if dataset is None:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            robot_type="franka",
            root=local_dir,
            fps=args.fps,
            use_videos=args.use_videos,
            features=build_features(args.use_videos),
            image_writer_processes=args.image_writer_process,
            image_writer_threads=args.image_writer_threads,
        )

    todo = episodes[start_offset:]
    if args.max_episodes is not None:
        todo = todo[: args.max_episodes]

    consumed = start_offset
    for ep in tqdm(todo, desc=desc, unit="ep", position=position, dynamic_ncols=True):
        try:
            _success_floats(ep)
            data = process_episode(ep)
            num_frames = len(data["state.joint_pos"])
            for i in range(num_frames):
                dataset.add_frame(
                    {key: value[i] for key, value in data.items()} | {"task": ep.instruction}
                )
            dataset.save_episode()
            metadata_file.parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_file, "a") as mf:
                mf.write(json.dumps(ep.to_record(dataset.meta.total_episodes - 1, num_frames)) + "\n")
        except Exception as e:
            if not args.skip_bad_episodes:
                raise
            if dataset.has_pending_frames():
                dataset.clear_episode_buffer()
            tqdm.write(f"[{desc}] skip bad episode {ep.session_id}/{ep.letter}: {type(e).__name__}: {e}")
        # Persist progress AFTER the episode is durably saved (or deliberately skipped).
        consumed += 1
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        progress_file.write_text(json.dumps({"consumed": consumed}))

    # Flush parquet footers; without this the last data/meta files are unreadable.
    dataset.finalize()
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(json.dumps({"total_episodes": dataset.meta.total_episodes}))
    return dataset


# --------------------------------------------------------------------------------------
# Orchestrator (parallel shards + merge), mirroring vifailback2lerobot/convert.py
# --------------------------------------------------------------------------------------
def run_parallel_conversion(args):
    n = args.num_proc
    name = args.dataset_name
    base_repo = args.repo_id or name
    shards_root = args.local_dir / f"_shards_{name}"
    tags = [f"shard{i:03d}" for i in range(n)]
    worker_roots = [shards_root / f"{name}_lerobot_{t}" for t in tags]
    worker_repo_ids = [f"{base_repo}_{t}" for t in tags]
    log_dir = shards_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    sentinel_of = lambda i: worker_roots[i] / "meta" / ".conversion_complete"

    # Cap per-worker allocator arenas and thread pools so n workers don't blow up RSS / cores.
    worker_env = {
        **os.environ,
        "MALLOC_ARENA_MAX": "2",
        "MALLOC_TRIM_THRESHOLD_": "131072",
        "OMP_NUM_THREADS": "2",
        "OPENCV_FFMPEG_THREADS": "2",
    }

    print(f"[parallel] {n} workers; per-worker logs in {log_dir}")
    procs, log_handles = [], []
    for i in range(n):
        if sentinel_of(i).exists() and not args.overwrite:
            print(f"[parallel] {tags[i]} already complete; skipping launch.")
            procs.append(None)
            log_handles.append(None)
            continue
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--raw-dir", str(args.raw_dir),
            "--local-dir", str(shards_root),
            "--dataset-name", name,
            "--repo-id", worker_repo_ids[i],
            "--shard-tag", tags[i],
            "--num-shards", str(n),
            "--fps", str(args.fps),
            "--image-writer-process", str(args.image_writer_process),
            "--image-writer-threads", str(args.image_writer_threads),
        ]
        if args.use_videos:
            cmd.append("--use-videos")
        if args.max_episodes is not None:
            cmd += ["--max-episodes", str(args.max_episodes)]
        if args.skip_bad_episodes:
            cmd.append("--skip-bad-episodes")
        if args.overwrite:
            cmd.append("--overwrite")
        lf = open(log_dir / f"{tags[i]}.log", "a")
        log_handles.append(lf)
        print(f"[parallel] {tags[i]} started -> {log_dir / (tags[i] + '.log')}")
        procs.append(subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=worker_env))

    rcs = [(p.wait() if p is not None else 0) for p in procs]
    for lf in log_handles:
        if lf is not None:
            lf.close()
    for i, rc in enumerate(rcs):
        if procs[i] is not None:
            print(f"[parallel] {tags[i]} exited rc={rc}")

    # aggregate_datasets does not verify completeness -- gate on the per-shard sentinels.
    incomplete = [i for i in range(n) if not sentinel_of(i).exists()]
    if incomplete:
        raise RuntimeError(
            f"[parallel] shards {incomplete} incomplete; not merging. Inspect {log_dir}/shardNNN.log, "
            f"then re-run the SAME command to resume (finished shards are skipped)."
        )

    aggr_root = args.local_dir / f"{name}_lerobot"
    if aggr_root.exists():
        shutil.rmtree(aggr_root)
    print(f"[parallel] merging {n} shards -> {aggr_root}")
    aggregate_datasets(
        repo_ids=worker_repo_ids,
        aggr_repo_id=base_repo,
        roots=worker_roots,
        aggr_root=aggr_root,
    )
    # Concatenate the per-shard metadata sidecars, re-offsetting episode_index by the number of
    # episodes in the preceding shards (aggregate preserves shard order).
    offset = 0
    with open(aggr_root / "meta" / "roboarena_metadata.jsonl", "w") as out:
        for i, root in enumerate(worker_roots):
            mf = root / "meta" / "roboarena_metadata.jsonl"
            n_eps = int(json.loads(sentinel_of(i).read_text())["total_episodes"])
            if mf.exists():
                for line in mf.read_text().splitlines():
                    rec = json.loads(line)
                    rec["episode_index"] += offset
                    out.write(json.dumps(rec) + "\n")
            offset += n_eps
    shutil.rmtree(shards_root, ignore_errors=True)

    if args.push_to_hub:
        LeRobotDataset(base_repo, root=aggr_root).push_to_hub(
            tags=["LeRobot", "roboarena", "franka", "droid"], private=False, push_videos=True, license="mit"
        )
    print(f"[parallel] done -> {aggr_root}")
    return aggr_root


def _default_dataset_name(raw_dir: Path) -> str:
    """DataDump_08-05-2025 -> roboarena_2025_08_05 (fallback: sanitized basename)."""
    base = raw_dir.name
    m = re.search(r"(\d{2})-(\d{2})-(\d{4})", base)
    if m:
        mm, dd, yyyy = m.groups()
        return f"roboarena_{yyyy}_{mm}_{dd}"
    return "roboarena_" + re.sub(r"[^0-9A-Za-z]+", "_", base).strip("_").lower()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, required=True, help="One RoboArena DataDump dir (contains evaluation_sessions/).")
    parser.add_argument("--local-dir", type=Path, required=True, help="Output directory for the LeRobot dataset.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Output name (default derived from the dump date).")
    parser.add_argument("--repo-id", type=str, default=None, help="Repository id (default: dataset name).")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--fps", type=int, default=15, help="Control/record frequency (DROID default: 15).")
    parser.add_argument("--use-videos", action="store_true", default=True, help="Encode cameras as mp4 (default on).")
    parser.add_argument("--image-writer-process", type=int, default=5)
    parser.add_argument("--image-writer-threads", type=int, default=10)
    parser.add_argument("--num-proc", type=int, default=1, help=">1 converts disjoint episode shards in parallel and merges them.")
    parser.add_argument("--shard-tag", type=str, default=None, help="Internal: marks this process as a shard worker.")
    parser.add_argument("--num-shards", type=int, default=None, help="Internal: total shard count for a worker.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Debug: convert at most this many episodes (per worker; no sentinel gating on partial).")
    parser.add_argument("--skip-bad-episodes", action="store_true", help="Log and skip episodes that raise instead of aborting.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild from scratch even if output exists (disables resume).")
    args = parser.parse_args()

    if args.dataset_name is None:
        args.dataset_name = _default_dataset_name(args.raw_dir)

    if args.num_proc > 1 and args.shard_tag is None:
        run_parallel_conversion(args)
        return

    # SIGTERM -> SystemExit so parquet writers close their footers on the way out, keeping the
    # partial dataset resumable.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    episodes = discover_episodes(args.raw_dir)
    name = args.dataset_name
    if args.shard_tag is not None:
        # Worker mode: take this shard's contiguous slice of the (deterministic) episode list.
        shard_id = int(args.shard_tag[len("shard"):])
        indices = np.array_split(np.arange(len(episodes)), args.num_shards)[shard_id]
        episodes = [episodes[i] for i in indices]
        local_dir = args.local_dir / f"{name}_lerobot_{args.shard_tag}"
        desc, position = args.shard_tag, shard_id
    else:
        local_dir = args.local_dir / f"{name}_lerobot"
        desc, position = name, 0

    repo_id = args.repo_id or name
    dataset = convert_worker(args, episodes, local_dir, repo_id, desc, position)
    if args.push_to_hub and args.shard_tag is None and dataset is not None:
        dataset.push_to_hub(
            tags=["LeRobot", "roboarena", "franka", "droid"], private=False, push_videos=True, license="mit"
        )


if __name__ == "__main__":
    main()
