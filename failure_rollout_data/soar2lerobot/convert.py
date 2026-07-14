#!/usr/bin/env python
"""Convert the SOAR dataset (WidowX 250, autonomous collection, RLDS) to LeRobot v3.

SOAR (Berkeley RAIL) was collected autonomously by policies (iql / calql / gcbc / mixed)
on the BridgeData WidowX 250 setup, with VLM-judged success labels. The RLDS export has
two splits -- ``success`` (10,018 eps) and ``failure`` (20,562 eps); every episode is
of variable length (~100-110 steps typical). Each split is converted into its own
LeRobot dataset.

Raw schema (verified empirically on real episodes, both splits):
    observation.state  (7)  [x, y, z (m), roll, pitch, yaw (rad), gripper]
                            euler is extrinsic-XYZ (Bridge/widowx_envs convention);
                            gripper continuous ~[0, 1], 1=open (no inversion)
    action             (7)  [dxyz, drpy, gripper_absolute]; a REAL command (residual vs
                            achieved motion is 25-50% of its magnitude), applied
                            elementwise by the controller: target = state[:6] + action[:6];
                            gripper action is binary {0, 1}, 1=open
    observation.image_0     256x256x3 jpeg (main camera); the SuSIE ``goal`` image is
                            intentionally NOT converted
    episode_metadata        file_path (encodes <robot>/<scene>/<policy>/<date>/<split>/trajN),
                            success (bool), object_list / task_list (sometimes junk),
                            robot_id / time (often empty), has_language

Output features follow ../dataset.md: raw (native, untransformed) vs canonical (axis-aligned)
are split, and the action is NOT stored -- it is derived at load time from state.* and target.*
(dataset.md sec 3). SOAR ships a real command, so raw_target.*/target.* are present.
    observation.images.image_0   video 256x256x3
    raw_state.eef_xyz (3)        native base-frame eef position, meters
    raw_state.eef_rpy (3)        native euler (extrinsic XYZ) -- the only rep SOAR ships
    raw_state.gripper_state (1)  native gripper reading (continuous ~[0, 1], 1=open)
    raw_target.eef_xyz (3)       absolute raw target = raw_state xyz + dxyz, meters
    raw_target.eef_rpy (3)       absolute raw target = raw_state rpy + drpy (extrinsic XYZ)
    raw_target.gripper_state (1) commanded gripper, raw binary {0, 1}, 1=open (not binarized)
    state.eef_xyz (3)            canonical eef position (world align = identity), meters
    state.eef_rot9d (9)          canonical eef rotation as full row-major matrix (gripper -> OpenCV)
    state.gripper_state (1)      normalized [0, 1] gripper (already ~[0, 1], 1=open; kept as-is);
                                 episodes whose raw gripper is a corrupt step-counter (below) are dropped
    target.eef_xyz (3)           canonical target position (world align = identity), meters
    target.eef_rot9d (9)         canonical target rotation as full row-major matrix
    target.gripper_state (1)     commanded gripper, {0, 1}, 1=open
    debug.gripper_eef_xyz (3)    GT-next delta in the CANONICAL gripper frame (debug only)
    debug.gripper_eef_rot6d (6)  GT-next relative rotation, canonical gripper frame;
                                 last step is a no-op ([1,0,0,0,1,0] / zeros)
    success (1)                  episode success flag (VLM), constant per episode

No joint positions exist in SOAR, so no *.joint_pos fields. Per-episode provenance is
written to ``meta/soar_metadata.jsonl`` (one JSON line per episode_index).

Corrupt-gripper QC: in ~1% (success) / ~4% (failure) of episodes the ``observation.state``
gripper slot [6] is overwritten by a monotonic step-counter (values 11-210) instead of the
real gripper reading; the pose and the binary ``action`` gripper are unaffected. These are
detected (max state gripper > ``--gripper-qc-threshold``, default 2.0; clean grippers are
<=~1.1) and the whole episode is DROPPED, one JSON line each in ``meta/qc_warnings.jsonl``.
Pass ``--gripper-qc-threshold 0`` to keep them.

The success split download may still be in progress: a worker that hits a missing or
truncated shard finalizes what it has, records progress, and exits WITHOUT the completion
sentinel -- re-running the same command later resumes where it stopped.

Example:
    python convert.py \
        --raw-dir /path/to/soar/rlds \
        --local-dir /path/to/output \
        --split failure \
        --num-proc 8
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

# Shared backend-agnostic rotation/frame math lives in the repo-root ``alignment`` package;
# this script runs with CWD=soar2lerobot, so put the repo root on sys.path first.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from alignment import transforms_numpy as tn  # noqa: E402

IMG_SHAPE = (256, 256, 3)
ROT6D_NAMES = ["r11", "r21", "r31", "r12", "r22", "r32"]
ROT9D_NAMES = [f"r{row}{col}" for row in range(1, 4) for col in range(1, 4)]
XYZ_NAMES = ["x", "y", "z"]
RPY_NAMES = ["roll", "pitch", "yaw"]
_IDENTITY_ROT6D = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32)

# Stored gripper frame -> canonical OpenCV gripper (z=approach, x=finger-open, y=down).
# SOAR's stored rpy is the widowx_envs DEFAULT_ROTATION-composed frame (reads ~identity at
# gripper-down: stored -z = approach, stored -y = canonical x). The in-plane sign was fixed
# against video (2026-07-06): at neutral pose canonical x must read world-RIGHT (stored -y)
# and canonical y world-backward (stored -x); the earlier ("y","x","-z") was 180deg off
# about the approach axis. Used ONLY for debug.*.
R_ALIGN_WIDOWX = tn.axis_alignment_matrix("-y", "-x", "-z")

np.set_printoptions(precision=4, suppress=True)


def build_features(use_videos: bool = True) -> dict:
    features = {
        "observation.images.image_0": {
            "dtype": "video" if use_videos else "image",
            "shape": IMG_SHAPE,
            "names": ["height", "width", "rgb"],
        },
        "raw_state.eef_xyz": {"dtype": "float32", "shape": (3,), "names": XYZ_NAMES},
        "raw_state.eef_rpy": {"dtype": "float32", "shape": (3,), "names": RPY_NAMES},
        "raw_state.gripper_state": {"dtype": "float32", "shape": (1,), "names": ["gripper"]},
        "raw_target.eef_xyz": {"dtype": "float32", "shape": (3,), "names": XYZ_NAMES},
        "raw_target.eef_rpy": {"dtype": "float32", "shape": (3,), "names": RPY_NAMES},
        "raw_target.gripper_state": {"dtype": "float32", "shape": (1,), "names": ["gripper"]},
        "state.eef_xyz": {"dtype": "float32", "shape": (3,), "names": XYZ_NAMES},
        "state.eef_rot9d": {"dtype": "float32", "shape": (9,), "names": ROT9D_NAMES},
        "state.gripper_state": {"dtype": "float32", "shape": (1,), "names": ["gripper"]},
        "target.eef_xyz": {"dtype": "float32", "shape": (3,), "names": XYZ_NAMES},
        "target.eef_rot9d": {"dtype": "float32", "shape": (9,), "names": ROT9D_NAMES},
        "target.gripper_state": {"dtype": "float32", "shape": (1,), "names": ["gripper"]},
        "debug.gripper_eef_xyz": {"dtype": "float32", "shape": (3,), "names": XYZ_NAMES},
        "debug.gripper_eef_rot6d": {"dtype": "float32", "shape": (6,), "names": ROT6D_NAMES},
        "success": {"dtype": "float32", "shape": (1,), "names": ["success"]},
    }
    return features


def _text(v) -> str:
    v = v.numpy() if hasattr(v, "numpy") else v
    return v.decode() if isinstance(v, bytes) else str(v)


def parse_episode_metadata(md, split: str) -> dict:
    """episode_metadata -> one JSON-able provenance record.

    file_path looks like .../soar-dataset-local/<robot>/<scene>/<policy>/<date>/<split>/traj<N>
    (some use mixed_policies / mixed_dates / undated). Parsed defensively: unparseable paths
    keep the raw string with empty structured fields.
    """
    file_path = _text(md["file_path"])
    parts = Path(file_path).parts
    robot = scene = policy = collect_date = orig_traj_id = ""
    if len(parts) >= 6 and parts[-2] in ("success", "failure"):
        robot, scene, policy, collect_date, _, orig_traj_id = parts[-6:]
    return {
        "file_path": file_path,
        "robot": robot,
        "scene": scene,
        "policy": policy,
        "collect_date": collect_date,
        "orig_traj_id": orig_traj_id,
        "success": bool(md["success"].numpy()),
        "object_list": _text(md["object_list"]),
        "task_list": _text(md["task_list"]),
        "robot_id": _text(md["robot_id"]),
        "time": _text(md["time"]),
        "has_language": bool(md["has_language"].numpy()),
        "split": split,
    }


def process_episode(ep, split: str) -> tuple[dict, dict, str]:
    """One RLDS episode -> (per-step feature arrays, metadata record, instruction)."""
    steps = list(ep["steps"])
    state = np.stack([s["observation"]["state"].numpy() for s in steps]).astype(np.float32)
    action = np.stack([s["action"].numpy() for s in steps]).astype(np.float32)
    images = np.stack([s["observation"]["image_0"].numpy() for s in steps])
    instruction = _text(steps[0]["language_instruction"])
    meta = parse_episode_metadata(ep["episode_metadata"], split)

    # --- raw (native, untransformed) state and target (dataset.md sec 2.1 / 2.2) ---------------
    xyz, rpy, grip = state[:, :3], state[:, 3:6], state[:, 6:7]
    R = tn.rpy_to_matrix(rpy, extrinsic=True)
    # SOAR ships a real command: the controller applies the 6D delta elementwise on [xyz, rpy]
    # to reach the absolute raw target; action[6] is the absolute (binary) gripper command.
    t_xyz = xyz + action[:, :3]
    t_rpy = rpy + action[:, 3:6]
    t_grip = action[:, 6:7]
    R_t = tn.rpy_to_matrix(t_rpy, extrinsic=True)

    # --- canonical state and target (dataset.md sec 2.3 / 2.4): align world (already FLU ->
    # identity) and gripper (stored widowx_envs frame -> OpenCV). rot9d is the canonical stored representation.
    R_c, p_c = tn.align_axis(R, xyz, np.eye(3, dtype=np.float32), R_ALIGN_WIDOWX)
    R_tc, p_tc = tn.align_axis(R_t, t_xyz, np.eye(3, dtype=np.float32), R_ALIGN_WIDOWX)

    # Debug-only canonical gripper-frame delta t -> t+1 (GT-next target); no-op last step.
    R_g, p_g = tn.gripper_delta_pose(R_c[:-1], p_c[:-1], R_c[1:], p_c[1:])

    data = {
        "observation.images.image_0": images,
        "raw_state.eef_xyz": xyz,
        "raw_state.eef_rpy": rpy,
        "raw_state.gripper_state": grip,
        "raw_target.eef_xyz": t_xyz,
        "raw_target.eef_rpy": t_rpy,
        "raw_target.gripper_state": t_grip,
        "state.eef_xyz": p_c.astype(np.float32),
        "state.eef_rot9d": R_c.reshape(-1, 9).astype(np.float32),
        "state.gripper_state": grip,
        "target.eef_xyz": p_tc.astype(np.float32),
        "target.eef_rot9d": R_tc.reshape(-1, 9).astype(np.float32),
        "target.gripper_state": t_grip,
        "debug.gripper_eef_xyz": np.concatenate([p_g, np.zeros((1, 3), np.float32)]).astype(np.float32),
        "debug.gripper_eef_rot6d": np.concatenate(
            [tn.matrix_to_rotation_6d(R_g), _IDENTITY_ROT6D]
        ).astype(np.float32),
        "success": np.full((len(steps), 1), float(meta["success"]), dtype=np.float32),
    }
    return data, meta, instruction


# --------------------------------------------------------------------------------------
# Conversion (single worker)
# --------------------------------------------------------------------------------------
def convert_worker(args, start: int, stop: int, local_dir: Path, repo_id: str, desc: str, position: int = 0):
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")

    sentinel = local_dir / "meta" / ".conversion_complete"
    progress_file = local_dir / "meta" / "_progress.json"
    metadata_file = local_dir / "meta" / "soar_metadata.jsonl"
    qc_file = local_dir / "meta" / "qc_warnings.jsonl"
    if args.overwrite and local_dir.exists():
        shutil.rmtree(local_dir)
    if sentinel.exists():
        print(f"[convert] {local_dir.name}: already complete (sentinel present); skipping.")
        return

    # Resume dispatch: a partial dir with loadable metadata is appended to; anything else is rebuilt.
    dataset = None
    consumed = 0
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
            consumed = max(dataset.meta.total_episodes, prog_done)
            print(f"[convert] {local_dir.name}: resuming after {consumed} episodes.")
        except Exception as e:
            print(f"[convert] {local_dir.name}: cannot resume ({type(e).__name__}: {e}); rebuilding.")
            shutil.rmtree(local_dir)
            dataset = None
            consumed = 0

    if dataset is None:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            robot_type="widowx_250",
            root=local_dir,
            fps=args.fps,
            use_videos=args.use_videos,
            features=build_features(args.use_videos),
            image_writer_processes=args.image_writer_process,
            image_writer_threads=args.image_writer_threads,
        )

    n_total = stop - start - consumed
    if args.max_episodes is not None:
        n_total = min(n_total, args.max_episodes)
    if n_total <= 0:
        dataset.finalize()
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text(json.dumps({"total_episodes": dataset.meta.total_episodes}))
        return dataset

    builder = tfds.builder_from_directory(str(args.raw_dir))
    # Absolute example-range slicing resumes mid-shard without re-reading consumed episodes.
    ds = builder.as_dataset(split=f"{args.split}[{start + consumed}:{stop}]", shuffle_files=False)

    def save_progress():
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        progress_file.write_text(json.dumps({"consumed": consumed}))

    ep_index = dataset.meta.total_episodes
    complete = True
    pbar = tqdm(total=n_total, desc=desc, unit="ep", position=position, dynamic_ncols=True)
    it = iter(ds)
    while consumed < stop - start and pbar.n < n_total:
        try:
            ep = next(it)
        except StopIteration:
            break
        except (tf.errors.NotFoundError, tf.errors.DataLossError) as e:
            # Shard missing or truncated: the download is still in progress. Keep what we
            # have (resumable), but do NOT mark this worker complete.
            tqdm.write(
                f"[{desc}] stopping at episode {start + consumed}: shard unavailable "
                f"({type(e).__name__}); re-run the same command once the download progressed."
            )
            complete = False
            break
        try:
            data, meta, instruction = process_episode(ep, args.split)
            # QC: ~1% (success) / ~4% (failure) of episodes have observation.state[6] overwritten by a
            # monotonic step-counter instead of the gripper reading (clean grippers are <=~1.1, the
            # artifact reaches 11-210). Drop them -- the pose and the binary target gripper are fine, but
            # the raw_state gripper channel is unrecoverable. Detection is clean-cut (no episode maxes in
            # the (1.5, 11) gap). Set --gripper-qc-threshold 0 to keep them.
            grip_max = float(np.max(data["raw_state.gripper_state"]))
            if args.gripper_qc_threshold > 0 and grip_max > args.gripper_qc_threshold:
                qc_file.parent.mkdir(parents=True, exist_ok=True)
                with open(qc_file, "a") as qf:
                    qf.write(json.dumps({"episode": start + consumed, "file_path": meta["file_path"],
                                         "state_gripper_max": round(grip_max, 2),
                                         "reason": "corrupt_gripper_counter", "dropped": True}) + "\n")
                tqdm.write(f"[{desc}] drop corrupt-gripper episode {start + consumed}: state gripper max={grip_max:.1f}")
                consumed += 1
                save_progress()
                pbar.update(1)
                continue
            num_frames = len(data["state.eef_xyz"])
            for i in range(num_frames):
                dataset.add_frame({key: value[i] for key, value in data.items()} | {"task": instruction})
            dataset.save_episode()
            metadata_file.parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_file, "a") as mf:
                mf.write(json.dumps({"episode_index": ep_index, "language_instruction": instruction} | meta) + "\n")
            ep_index += 1
        except Exception as e:
            if not args.skip_bad_episodes:
                raise
            if dataset.has_pending_frames():
                dataset.clear_episode_buffer()
            tqdm.write(f"[{desc}] skip bad episode {start + consumed}: {type(e).__name__}: {e}")
        # Persist progress AFTER the episode is durably saved (or deliberately skipped).
        consumed += 1
        save_progress()
        pbar.update(1)
    pbar.close()

    # Flush parquet footers; without this the last data/meta files are unreadable.
    dataset.finalize()
    if complete and consumed >= stop - start and args.max_episodes is None:
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text(json.dumps({"total_episodes": dataset.meta.total_episodes}))
    return dataset


# --------------------------------------------------------------------------------------
# Orchestrator (parallel shards + merge), mirroring vifailback2lerobot/convert.py
# --------------------------------------------------------------------------------------
def run_parallel_conversion(args):
    n = args.num_proc
    base_repo = args.repo_id
    shards_root = args.local_dir / f"_shards_{args.split}"
    tags = [f"shard{i:03d}" for i in range(n)]
    worker_roots = [shards_root / f"soar_{args.split}_lerobot_{t}" for t in tags]
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
        "TF_CPP_MIN_LOG_LEVEL": "2",
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
            "--split", args.split,
            "--repo-id", worker_repo_ids[i],
            "--shard-tag", tags[i],
            "--num-shards", str(n),
            "--fps", str(args.fps),
            "--image-writer-process", str(args.image_writer_process),
            "--image-writer-threads", str(args.image_writer_threads),
            "--gripper-qc-threshold", str(args.gripper_qc_threshold),
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
    # A worker that stopped at a missing shard (download in progress) has no sentinel.
    incomplete = [i for i in range(n) if not sentinel_of(i).exists()]
    if incomplete:
        raise RuntimeError(
            f"[parallel] shards {incomplete} incomplete; not merging. Inspect {log_dir}/shardNNN.log; "
            f"if the split is still downloading, re-run the SAME command later to resume "
            f"(finished shards are skipped)."
        )

    aggr_root = args.local_dir / f"soar_{args.split}_lerobot"
    if aggr_root.exists():
        shutil.rmtree(aggr_root)
    print(f"[parallel] merging {n} shards -> {aggr_root}")
    aggregate_datasets(
        repo_ids=worker_repo_ids,
        aggr_repo_id=base_repo,
        roots=worker_roots,
        aggr_root=aggr_root,
    )
    # Concatenate the per-shard metadata sidecars, re-offsetting episode_index by the
    # number of episodes in the preceding shards (aggregate preserves shard order).
    offset = 0
    with open(aggr_root / "meta" / "soar_metadata.jsonl", "w") as out:
        for i, root in enumerate(worker_roots):
            mf = root / "meta" / "soar_metadata.jsonl"
            n_eps = int(json.loads(sentinel_of(i).read_text())["total_episodes"])
            if mf.exists():
                for line in mf.read_text().splitlines():
                    rec = json.loads(line)
                    rec["episode_index"] += offset
                    out.write(json.dumps(rec) + "\n")
            offset += n_eps
    # Carry the per-shard QC warnings (dropped corrupt-gripper episodes) into the merged output.
    with open(aggr_root / "meta" / "qc_warnings.jsonl", "a") as out:
        for root in worker_roots:
            qc = root / "meta" / "qc_warnings.jsonl"
            if qc.exists():
                out.write(qc.read_text())
    shutil.rmtree(shards_root, ignore_errors=True)

    if args.push_to_hub:
        LeRobotDataset(base_repo, root=aggr_root).push_to_hub(
            tags=["LeRobot", "soar", "widowx"], private=False, push_videos=True, license="mit"
        )
    print(f"[parallel] done -> {aggr_root}")
    return aggr_root


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, required=True, help="SOAR RLDS directory (features.json + tfrecords).")
    parser.add_argument("--local-dir", type=Path, required=True, help="Output directory for the LeRobot dataset.")
    parser.add_argument("--split", type=str, required=True, choices=["success", "failure"],
                        help="Which RLDS split to convert (each becomes its own dataset).")
    parser.add_argument("--repo-id", type=str, default=None, help="Repository id (default: soar_<split>).")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--fps", type=int, default=5, help="Control frequency (Bridge WidowX default: 5).")
    parser.add_argument("--use-videos", action="store_true", default=True, help="Encode camera as mp4 (default on).")
    parser.add_argument("--image-writer-process", type=int, default=5)
    parser.add_argument("--image-writer-threads", type=int, default=10)
    parser.add_argument("--num-proc", type=int, default=1, help=">1 converts disjoint episode ranges in parallel and merges them.")
    parser.add_argument("--shard-tag", type=str, default=None, help="Internal: marks this process as a shard worker.")
    parser.add_argument("--num-shards", type=int, default=None, help="Internal: total shard count for a worker.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Debug: convert at most this many episodes (per worker; no sentinel).")
    parser.add_argument(
        "--gripper-qc-threshold",
        type=float,
        default=2.0,
        help="Drop episodes whose max observation.state gripper exceeds this (catches the step-counter "
        "artifact: clean grippers <=~1.1, corrupt reach 11-210). 0 disables the drop.",
    )
    parser.add_argument("--skip-bad-episodes", action="store_true", help="Log and skip episodes that raise instead of aborting.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild from scratch even if output exists (disables resume).")
    args = parser.parse_args()
    if args.repo_id is None:
        args.repo_id = f"soar_{args.split}"

    if args.num_proc > 1 and args.shard_tag is None:
        run_parallel_conversion(args)
        return

    # SIGTERM -> SystemExit so parquet writers close their footers on the way out, keeping the
    # partial dataset resumable.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    import tensorflow_datasets as tfds

    num_examples = tfds.builder_from_directory(str(args.raw_dir)).info.splits[args.split].num_examples
    if args.shard_tag is not None:
        # Worker mode: take this shard's contiguous example range of the split.
        shard_id = int(args.shard_tag[len("shard"):])
        bounds = np.linspace(0, num_examples, args.num_shards + 1).astype(int)
        start, stop = int(bounds[shard_id]), int(bounds[shard_id + 1])
        local_dir = args.local_dir / f"soar_{args.split}_lerobot_{args.shard_tag}"
        desc, position = args.shard_tag, shard_id
    else:
        start, stop = 0, num_examples
        local_dir = args.local_dir / f"soar_{args.split}_lerobot"
        desc, position = f"soar_{args.split}", 0

    dataset = convert_worker(args, start, stop, local_dir, args.repo_id, desc, position)
    if args.push_to_hub and args.shard_tag is None and dataset is not None:
        dataset.push_to_hub(
            tags=["LeRobot", "soar", "widowx"], private=False, push_videos=True, license="mit"
        )


if __name__ == "__main__":
    main()
