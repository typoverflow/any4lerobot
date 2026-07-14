#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
For all datasets in the RLDS format.
For https://github.com/google-deepmind/open_x_embodiment (OPENX) datasets.

NOTE: You need to install tensorflow and tensorflow_datsets before running this script.

Example:
    python openx_rlds.py \
        --raw-dir /path/to/bridge_orig/1.0.0 \
        --local-dir /path/to/local_dir \
        --repo-id your_id \
        --use-videos \
        --push-to-hub
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from functools import partial
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME
from oxe_utils.configs import OXE_DATASET_CONFIGS
from oxe_utils.transforms import OXE_STANDARDIZATION_TRANSFORMS
from oxe_utils.constants import STATE_NAMES, ACTION_NAMES

np.set_printoptions(precision=2)

CONTRACT_GROUPS = ("raw_state", "raw_target", "state", "target", "debug")
ROT6D_NAMES = ["r11", "r21", "r31", "r12", "r22", "r32"]
ROT9D_NAMES = [f"r{row}{col}" for row in range(1, 4) for col in range(1, 4)]


def _contract_feature_names(key: str, shape: tuple[int, ...]) -> list[str]:
    """Dimension names for fields in failure_rollout_data/dataset.md."""
    size = int(np.prod(shape)) if shape else 1
    if key == "eef_xyz" or key == "gripper_eef_xyz":
        return ["x", "y", "z"]
    if key == "eef_rpy":
        return ["roll", "pitch", "yaw"]
    if key == "eef_quat":
        return ["x", "y", "z", "w"]
    if key == "eef_axis_angle":
        return ["axis_angle_x", "axis_angle_y", "axis_angle_z"]
    if key == "eef_rot6d" or key == "gripper_eef_rot6d":
        return ROT6D_NAMES
    if key == "eef_rot9d":
        return ROT9D_NAMES
    if key in ("joint_pos", "joint_vel"):
        return [f"joint_{i}" for i in range(size)]
    if key == "gripper_state":
        return ["gripper"]
    return [f"{key}_{i}" for i in range(size)]


def _decode_bc_z_image(image_bytes):
    """Decode BC-Z's variable-resolution JPEG and restore its declared 640x512 frame size."""
    image = tf.io.decode_image(
        image_bytes, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [640, 512], method="bilinear", antialias=True)
    return tf.cast(
        tf.clip_by_value(tf.round(image), 0.0, 255.0), tf.uint8
    )


def transform_raw_dataset(episode, dataset_name):
    steps = episode["steps"]
    traj = next(iter(steps.batch(steps.cardinality())))
    if dataset_name == "bc_z":
        # The TFDS metadata declares 640x512, but the records contain lower-resolution JPEGs with
        # the same aspect ratio. SkipDecoding lets the scalar byte strings batch cleanly; decode and
        # restore every image only after the episode has been batched.
        traj["observation"]["image"] = tf.map_fn(
            _decode_bc_z_image,
            traj["observation"]["image"],
            fn_output_signature=tf.TensorSpec([640, 512, 3], tf.uint8),
        )

    if dataset_name in OXE_STANDARDIZATION_TRANSFORMS:
        traj = OXE_STANDARDIZATION_TRANSFORMS[dataset_name](traj)

    if "raw_state" in traj:
        # Contract path: preserve the five named groups and do not manufacture a training action.
        # raw_target/target are optional by design for datasets without recorded controller commands.
        for group in CONTRACT_GROUPS:
            if group in traj:
                traj[group] = {
                    key: tf.cast(value, tf.float32) for key, value in traj[group].items()
                }
    else:
        # Legacy Open-X path retained for datasets that have not moved to dataset.md yet.
        traj["state"] = {key: tf.cast(value, tf.float32) for key, value in traj["state"].items()}
        traj["action"] = {key: tf.cast(value, tf.float32) for key, value in traj["action"].items()}

    traj["task"] = traj.pop("language_instruction")

    episode["steps"] = traj
    return episode


def generate_features_from_raw(episode, builder: tfds.core.DatasetBuilder, use_videos: bool = True):
    dataset_name = Path(builder.data_dir).parent.name

    # Image specs come from the raw builder info (their shapes are unchanged by the transform).
    obs = builder.info.features["steps"]["observation"]
    obs_features = {
        f"observation.images.{key}": {
            "dtype": "video" if use_videos else "image",
            "shape": value.shape,
            "names": ["height", "width", "rgb"],
        }
        for key, value in obs.items()
        if "depth" not in key and any(x in key for x in ["image", "rgb"])
    }

    steps = episode["steps"]
    if "raw_state" in steps:
        # dataset.md path: every semantic field is stored independently. In particular, no
        # observation.state or monolithic action vector is materialized during conversion.
        contract_features = {}
        for group in CONTRACT_GROUPS:
            for key, value in steps.get(group, {}).items():
                shape = tuple(value.shape[1:])
                contract_features[f"{group}.{key}"] = {
                    "dtype": "float32",
                    "shape": shape,
                    "names": _contract_feature_names(key, shape),
                }
        return {**obs_features, **contract_features}

    # Legacy state/action path. These groups are produced by the standardization transform and do
    # not exist on the raw builder, so derive their specs from one transformed episode.
    state_features = {
        f"state.{key}": {
            "dtype": "float32",
            "shape": tuple(value.shape[1:]),
            "names": STATE_NAMES[key],
        }
        for key, value in steps["state"].items()
    }
    state_encoding = OXE_DATASET_CONFIGS[dataset_name]["state_encoding"]
    state_names = []
    for key, value in state_encoding.items():
        if key == "pad":
            state_names.extend(["pad"] * value)
        else:
            state_names.extend(STATE_NAMES[key])
    state_features["observation.state"] = {
        "dtype": "float32",
        "shape": (sum(state_encoding.values()),),
        "names": state_names,
    }

    action_features = {
        f"action.{key}": {
            "dtype": "float32",
            "shape": tuple(value.shape[1:]),
            "names": ACTION_NAMES[key],
        }
        for key, value in steps["action"].items()
    }
    action_encoding = OXE_DATASET_CONFIGS[dataset_name]["action_encoding"]
    action_names = []
    for key, value in action_encoding.items():
        if key == "pad":
            action_names.extend(["pad"] * value)
        else:
            action_names.extend(ACTION_NAMES[key])
    action_features["action"] = {
        "dtype": "float32",
        "shape": (sum(action_encoding.values()),),
        "names": action_names,
    }

    return {**obs_features, **state_features, **action_features}


def save_as_lerobot_dataset(
    lerobot_dataset: LeRobotDataset,
    raw_dataset: tf.data.Dataset,
    dataset_name: str,
    total_episodes: int = None,
    desc: str = "convert",
    position: int = 0,
    start_offset: int = 0,
    progress_file: Path = None,
    skip_bad_episodes: bool = False,
    **kwargs,
):
    # dataset_name is passed in explicitly: inferring it from the output path breaks when the
    # dataset is written to a per-worker subdirectory (e.g. ``_shards/...``) during parallel runs.
    # tqdm gives a live episodes/s rate + ETA; ``total_episodes`` comes from dataset_info so the ETA
    # is available from the first episode. In parallel runs each worker draws its own bar at its own
    # ``position`` (the shard index) so they stack instead of clobbering each other.
    progress = tqdm(
        raw_dataset.as_numpy_iterator(),
        total=total_episodes,
        desc=desc,
        unit="ep",
        position=position,
        dynamic_ncols=True,
    )
    # ``consumed`` is the absolute number of episodes drawn from this slice across all runs
    # (``start_offset`` already done in earlier runs + the ones processed here). It is persisted after
    # every episode so an interrupted run can resume by skipping exactly this many (see
    # create_lerobot_dataset). It counts deliberately-skipped bad episodes too, so resume never
    # re-reads or double-skips them.
    consumed = start_offset
    for episode in progress:
        try:
            traj = episode["steps"]
            contract_format = "raw_state" in traj
            timeline_group = traj["state"] if contract_format else traj["action"]
            num_frames = next(iter(timeline_group.values())).shape[0]
            for i in range(num_frames):
                image_dict = {
                    f"observation.images.{key}": value[i]
                    for key, value in traj["observation"].items()
                    if "depth" not in key and any(x in key for x in ["image", "rgb"])
                }
                if contract_format:
                    grouped_features = {
                        f"{group}.{key}": value[i]
                        for group in CONTRACT_GROUPS
                        for key, value in traj.get(group, {}).items()
                    }
                    lerobot_dataset.add_frame(
                        {**image_dict, **grouped_features, "task": traj["task"][0].decode()}
                    )
                else:
                    state_dict = {
                        f"state.{key}": value[i]
                        for key, value in traj["state"].items()
                    }
                    action_dict = {
                        f"action.{key}": value[i]
                        for key, value in traj["action"].items()
                    }
                    state_vec = []
                    state_encoding = OXE_DATASET_CONFIGS[dataset_name]["state_encoding"]
                    for key, value in state_encoding.items():
                        if key == "pad":
                            state_vec.append(np.zeros(value, dtype=np.float32))
                        else:
                            state_vec.append(traj["state"][key][i])
                    state_vec = np.concatenate(state_vec, axis=0)
                    action_vec = []
                    action_encoding = OXE_DATASET_CONFIGS[dataset_name]["action_encoding"]
                    for key, value in action_encoding.items():
                        if key == "pad":
                            action_vec.append(np.zeros(value, dtype=np.float32))
                        else:
                            action_vec.append(traj["action"][key][i])
                    action_vec = np.concatenate(action_vec, axis=0)
                    lerobot_dataset.add_frame(
                        {
                            **image_dict,
                            **state_dict,
                            **action_dict,
                            "observation.state": state_vec,
                            "action": action_vec,
                            "task": traj["task"][0].decode(),
                        },
                    )
            if num_frames > 0:
                lerobot_dataset.save_episode()
            else:
                # Defensive fallback; empty RLDS episodes are normally filtered before transformation.
                tqdm.write(f"[{desc}] skipping empty episode (0 frames) at index {consumed}")
        except Exception as e:
            # One malformed/undecodable episode shouldn't kill the whole worker when opted in:
            # discard the partial episode buffer (and its temp images) and move on.
            if not skip_bad_episodes:
                raise
            if lerobot_dataset.has_pending_frames():
                lerobot_dataset.clear_episode_buffer()
            tqdm.write(f"[{desc}] skip bad episode at index {consumed}: {type(e).__name__}: {e}")
        # Persist progress AFTER the episode is durably saved (or deliberately skipped). On crash this
        # may lag the true count by at most one; resume reconciles with meta.total_episodes.
        consumed += 1
        if progress_file is not None:
            try:
                progress_file.parent.mkdir(parents=True, exist_ok=True)
                progress_file.write_text(json.dumps({"consumed": consumed}))
            except Exception:
                pass


def _parse_raw_dir(raw_dir: Path):
    """Split a raw dir into (dataset_name, version, data_dir), matching TFDS' layout.

    ``.../bridge_orig/0.0.1`` -> ("bridge_orig", "0.0.1", ".../"); a trailing non-version
    component is treated as the dataset name with an empty version.
    """
    last_part = raw_dir.name
    if re.match(r"^\d+\.\d+\.\d+$", last_part):
        return raw_dir.parent.name, last_part, raw_dir.parent.parent
    return last_part, "", raw_dir.parent


def create_lerobot_dataset(
    raw_dir: Path,
    repo_id: str = None,
    local_dir: Path = None,
    push_to_hub: bool = False,
    fps: int = None,
    robot_type: str = None,
    use_videos: bool = True,
    image_writer_process: int = 5,
    image_writer_threads: int = 10,
    keep_images: bool = True,
    split: str = "train",
    shard_tag: str = None,
    max_episodes: int = None,
    prefetch_buffer: int = 2,
    skip_bad_episodes: bool = False,
    overwrite: bool = False,
):
    dataset_name, version, data_dir = _parse_raw_dir(raw_dir)

    if local_dir is None:
        local_dir = Path(HF_LEROBOT_HOME)
    # ``shard_tag`` keeps each parallel worker's partial dataset in its own directory so the
    # orchestrator can merge them afterwards (see run_parallel_conversion).
    tag_suffix = f"_{shard_tag}" if shard_tag else ""
    local_dir /= f"{dataset_name}_{version}_lerobot{tag_suffix}"

    # Resume/skip dispatch. ``.conversion_complete`` is written only after a successful finalize(),
    # so it marks a fully-built shard; ``_progress.json`` tracks how many episodes have been consumed
    # from this slice so an interrupted shard can pick up exactly where it stopped instead of
    # restarting (or being silently wiped, as the old unconditional rmtree did).
    sentinel = local_dir / "meta" / ".conversion_complete"
    progress_file = local_dir / "meta" / "_progress.json"
    if overwrite and local_dir.exists():
        shutil.rmtree(local_dir)
    if sentinel.exists():
        print(f"[convert] {local_dir.name}: already complete (sentinel present); skipping.")
        return local_dir

    builder = tfds.builder(dataset_name, data_dir=data_dir, version=version)

    # Decide create-vs-resume. A partial dir with loadable metadata is resumed via
    # LeRobotDataset.resume() (appends new episodes to the existing data/video files); a missing or
    # unreadable dir is (re)built from scratch.
    lerobot_dataset = None
    start_offset = 0
    if local_dir.exists():
        try:
            lerobot_dataset = LeRobotDataset.resume(
                repo_id=repo_id,
                root=local_dir,
                image_writer_processes=image_writer_process,
                image_writer_threads=image_writer_threads,
            )
            meta_done = lerobot_dataset.meta.total_episodes
            prog_done = 0
            if progress_file.exists():
                try:
                    prog_done = int(json.loads(progress_file.read_text()).get("consumed", 0))
                except Exception:
                    prog_done = 0
            # meta counts durably-saved episodes; progress also counts deliberately-skipped bad ones.
            # Take the max so we never re-read an already-consumed episode nor write a duplicate.
            start_offset = max(meta_done, prog_done)
            print(
                f"[convert] {local_dir.name}: resuming after {start_offset} episodes "
                f"(meta={meta_done}, progress={prog_done})."
            )
        except Exception as e:
            print(
                f"[convert] {local_dir.name}: cannot resume ({type(e).__name__}: {e}); "
                f"rebuilding from scratch."
            )
            shutil.rmtree(local_dir)
            lerobot_dataset = None
            start_offset = 0

    # ``split`` may be a TFDS slice like "train[0%:25%]" (one worker's disjoint episode range).
    # Slice membership is derived from dataset_info shard lengths, so it is exact and independent
    # of read order -- safe even though RLDS read order is only deterministic with shuffle_files=False.
    # Apply ``skip(start_offset)`` BEFORE the expensive transform so already-converted episodes are
    # not re-decoded on resume.
    def filter_fn(episode):
        # A transform cannot infer a feature structure from an episode with no steps. DROID contains
        # a small number of these malformed episodes, so discard them before batching/mapping.
        keep = episode["steps"].cardinality() > 0
        if dataset_name == "kuka":
            keep = tf.logical_and(keep, episode["success"])
        return keep

    decoders = None
    if dataset_name == "bc_z":
        decoders = {"steps": {"observation": {"image": tfds.decode.SkipDecoding()}}}
    base_dataset = builder.as_dataset(split=split, decoders=decoders).filter(filter_fn)
    if start_offset:
        base_dataset = base_dataset.skip(start_offset)
    raw_dataset = base_dataset.map(partial(transform_raw_dataset, dataset_name=dataset_name))
    if max_episodes is not None:  # debug/smoke-test escape hatch (limits episodes converted this run)
        raw_dataset = raw_dataset.take(max_episodes)
    # Overlap tf.data production (image decode + transform) with the Python writer loop below. A small
    # bounded buffer (not tf.data.AUTOTUNE) caps how many fully-decoded multi-camera episodes are held
    # in RAM at once -- AUTOTUNE could grow this unboundedly and OOM-kill parallel workers.
    raw_dataset = raw_dataset.prefetch(prefetch_buffer)

    # Episode count for the tqdm ETA, read from dataset_info (no data scan). For kuka the success
    # filter trims it, so this is an upper bound; resume offset / --max-episodes cap it.
    try:
        total_episodes = builder.info.splits[split].num_examples
    except Exception:
        total_episodes = None
    if total_episodes is not None:
        total_episodes = max(total_episodes - start_offset, 0)
    if max_episodes is not None:
        total_episodes = min(total_episodes, max_episodes) if total_episodes else max_episodes
    # Per-worker bar identity: label by shard tag and stack bars at distinct positions in parallel runs.
    progress_desc = shard_tag or f"{dataset_name}_{version}"
    progress_position = int(shard_tag[len("shard"):]) if (shard_tag or "").startswith("shard") else 0

    if lerobot_dataset is None:
        # Fresh create: peek one transformed episode so the feature schema reflects the keys this
        # dataset's transform actually emits (datasets may produce only a subset of the names).
        sample_episode = next(iter(raw_dataset.as_numpy_iterator()))
        features = generate_features_from_raw(sample_episode, builder, use_videos)

        if fps is None:
            if dataset_name in OXE_DATASET_CONFIGS:
                fps = OXE_DATASET_CONFIGS[dataset_name]["control_frequency"]
            else:
                fps = 10

        if robot_type is None:
            if dataset_name in OXE_DATASET_CONFIGS:
                robot_type = OXE_DATASET_CONFIGS[dataset_name]["robot_type"]
                robot_type = robot_type.lower().replace(" ", "_").replace("-", "_")
            else:
                robot_type = "unknown"

        lerobot_dataset = LeRobotDataset.create(
            repo_id=repo_id,
            robot_type=robot_type,
            root=local_dir,
            fps=int(fps),
            use_videos=use_videos,
            features=features,
            image_writer_threads=image_writer_threads,
            image_writer_processes=image_writer_process,
        )

    save_as_lerobot_dataset(
        lerobot_dataset,
        raw_dataset,
        dataset_name,
        total_episodes=total_episodes,
        desc=progress_desc,
        position=progress_position,
        keep_images=keep_images,
        start_offset=start_offset,
        progress_file=progress_file,
        skip_bad_episodes=skip_bad_episodes,
    )

    # Close the parquet writers so footer metadata is flushed to disk. Without this the last
    # data/meta parquet files are left without their footer ("PAR1" magic bytes), which makes the
    # dataset unreadable (e.g. HF viewer: "Parquet magic bytes not found in footer").
    lerobot_dataset.finalize()

    # Mark the shard fully built. The orchestrator and resume dispatch key off this sentinel: a shard
    # without it is considered incomplete and is resumed/rebuilt rather than merged.
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(json.dumps({"total_episodes": lerobot_dataset.meta.total_episodes}))

    if push_to_hub:
        assert repo_id is not None
        # On the resume path robot_type is never recomputed (it lives in the existing meta), so fall
        # back to the dataset's own metadata to keep the tag list well-formed.
        if robot_type is None:
            robot_type = getattr(lerobot_dataset.meta, "robot_type", "unknown")
        tags = ["LeRobot", dataset_name, "rlds"]
        if dataset_name in OXE_DATASET_CONFIGS:
            tags.append("openx")
        if robot_type and robot_type != "unknown":
            tags.append(robot_type)
        lerobot_dataset.push_to_hub(
            tags=tags,
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

    return local_dir


def _push_aggregated(repo_id, root, dataset_name, robot_type):
    """Push an already-built (merged) LeRobotDataset directory to the hub."""
    assert repo_id is not None
    tags = ["LeRobot", dataset_name, "rlds"]
    if dataset_name in OXE_DATASET_CONFIGS:
        tags.append("openx")
    if robot_type and robot_type != "unknown":
        tags.append(robot_type)
    LeRobotDataset(repo_id, root=root).push_to_hub(
        tags=tags, private=False, push_videos=True, license="apache-2.0"
    )


def run_parallel_conversion(args):
    """Convert one RLDS dataset with ``args.num_proc`` worker processes, then merge.

    Each worker converts a disjoint, exact TFDS split slice (``train[a%:b%]``) into its own
    ``_shards/`` sub-directory; the slices tile the dataset exactly (membership comes from
    dataset_info shard lengths, independent of read order). When all workers finish, the partial
    datasets are merged with ``aggregate_datasets`` (videos are copied, not re-encoded; episode and
    frame indices are renumbered). This is CPU data-parallelism over the decode + video-encode work
    that dominates single-process conversion.
    """
    n = args.num_proc
    dataset_name, version, _ = _parse_raw_dir(args.raw_dir)
    base_repo = args.repo_id or dataset_name
    shards_root = args.local_dir / "_shards"

    slices = [f"train[{i * 100 // n}%:{(i + 1) * 100 // n}%]" for i in range(n)]
    tags = [f"shard{i:03d}" for i in range(n)]
    worker_roots = [shards_root / f"{dataset_name}_{version}_lerobot_{tags[i]}" for i in range(n)]
    worker_repo_ids = [f"{base_repo}_{tags[i]}" for i in range(n)]

    # Per-worker log files: bare Popen would interleave 8 workers' output on one terminal and lose it
    # on scroll, so a crash leaves no trace. One log per shard keeps the real traceback / "Killed"
    # (OOM) line. Tail them live with e.g. ``tail -f <local_dir>/_shards/logs/shard000.log``.
    log_dir = shards_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    sentinel_of = lambda i: worker_roots[i] / "meta" / ".conversion_complete"

    # Worker environment: cap glibc malloc arenas and thread pools. Each worker otherwise spawns ~200
    # threads (TF intra/inter-op pools sized to 40 cores + image writers + per-episode encoder forks),
    # and default glibc hands every thread its own 64 MiB arena that it never returns to the OS -> the
    # ~9.7 MiB/episode anonymous-RSS creep that OOMs the box. MALLOC_ARENA_MAX caps that regardless of
    # thread count; MALLOC_TRIM_THRESHOLD_ makes glibc release freed memory; the *_THREADS caps both
    # shrink the thread explosion and stop 8 workers oversubscribing 40 cores.
    worker_env = {
        **os.environ,
        "MALLOC_ARENA_MAX": "2",
        "MALLOC_TRIM_THRESHOLD_": "131072",
        "OMP_NUM_THREADS": "2",
        "TF_NUM_INTRAOP_THREADS": "4",
        "TF_NUM_INTEROP_THREADS": "2",
    }

    print(f"[parallel] {dataset_name} {version}: {n} workers over slices {slices}")
    print(f"[parallel] per-worker logs in {log_dir}")
    print("[parallel] worker env: MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=131072 "
          "OMP_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2")
    procs = []
    log_handles = []
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
            "--repo-id", worker_repo_ids[i],
            "--split", slices[i],
            "--shard-tag", tags[i],
            "--image-writer-process", str(args.image_writer_process),
            "--image-writer-threads", str(args.image_writer_threads),
            "--prefetch-buffer", str(args.prefetch_buffer),
        ]
        if args.use_videos:
            cmd.append("--use-videos")
        if args.fps is not None:
            cmd += ["--fps", str(args.fps)]
        if args.robot_type is not None:
            cmd += ["--robot-type", args.robot_type]
        if args.max_episodes is not None:
            cmd += ["--max-episodes", str(args.max_episodes)]
        if args.skip_bad_episodes:
            cmd.append("--skip-bad-episodes")
        if args.overwrite:
            cmd.append("--overwrite")
        # Append (not truncate) so a resumed run keeps the prior attempt's log alongside the new one.
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

    # Completeness gate: aggregate_datasets does NOT check that inputs are complete -- it would
    # silently merge a truncated shard and produce a short dataset. Gate on the per-shard sentinel
    # instead, and tell the user that simply re-running this same command resumes the rest.
    incomplete = [i for i in range(n) if not sentinel_of(i).exists()]
    if incomplete:
        raise RuntimeError(
            f"[parallel] shards {incomplete} are incomplete (no completion sentinel); not merging. "
            f"Inspect {log_dir}/shard{{NNN}}.log for the cause, then re-run the SAME command to "
            f"resume them (finished shards are skipped, partial shards continue where they stopped)."
        )

    aggr_root = args.local_dir / f"{dataset_name}_{version}_lerobot"
    if aggr_root.exists():
        shutil.rmtree(aggr_root)
    print(f"[parallel] merging {n} shards -> {aggr_root}")
    aggregate_datasets(
        repo_ids=worker_repo_ids,
        aggr_repo_id=base_repo,
        roots=worker_roots,
        aggr_root=aggr_root,
    )
    shutil.rmtree(shards_root, ignore_errors=True)

    if args.push_to_hub:
        robot_type = args.robot_type
        if robot_type is None and dataset_name in OXE_DATASET_CONFIGS:
            robot_type = OXE_DATASET_CONFIGS[dataset_name]["robot_type"].lower().replace(" ", "_").replace("-", "_")
        _push_aggregated(base_repo, aggr_root, dataset_name, robot_type)
    print(f"[parallel] done -> {aggr_root}")
    return aggr_root


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing input raw datasets (e.g. `path/to/dataset` or `path/to/dataset/version).",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        required=True,
        help="When provided, writes the dataset converted to LeRobotDataset format in this directory  (e.g. `data/lerobot/aloha_mobile_chair`).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help="Repositery identifier on Hugging Face: a community or a user name `/` the name of the dataset, required when push-to-hub is True",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload to hub.",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default=None,
        help="Robot type of this dataset.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Frame rate used to collect videos. Default fps equals to the control frequency of the robot.",
    )
    parser.add_argument(
        "--use-videos",
        action="store_true",
        help="Convert each episode of the raw dataset to an mp4 video. This option allows 60 times lower disk space consumption and 25 faster loading time during training.",
    )
    parser.add_argument(
        "--image-writer-process",
        type=int,
        default=5,
        help="Number of processes of image writer for saving images.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=10,
        help="Number of threads per process of image writer for saving images.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=1,
        help="Number of parallel worker processes. >1 splits the dataset into that many disjoint "
        "TFDS slices, converts them concurrently, and merges the results. Set near your CPU core "
        "count (but mind that each worker also spawns --image-writer-process image writers).",
    )
    # Internal/advanced args used by the parallel orchestrator (and handy for smoke tests).
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="TFDS split or slice to convert, e.g. 'train' or 'train[0%%:25%%]'. Set automatically "
        "for each worker when --num-proc > 1.",
    )
    parser.add_argument(
        "--shard-tag",
        type=str,
        default=None,
        help="Internal: per-worker output-dir suffix. Presence marks this process as a worker.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Debug: convert at most this many episodes (after slicing).",
    )
    parser.add_argument(
        "--prefetch-buffer",
        type=int,
        default=2,
        help="Number of episodes tf.data prefetches ahead of the writer loop. Small bounded value "
        "(default 2) instead of tf.data.AUTOTUNE, which can grow unboundedly and OOM-kill workers "
        "when several run in parallel. Raise for throughput if you have RAM headroom.",
    )
    parser.add_argument(
        "--skip-bad-episodes",
        action="store_true",
        help="Log and skip an episode that raises during conversion instead of aborting the worker. "
        "Off by default so data is never silently dropped unless you ask for it.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild shards from scratch even if a partial/complete output exists (disables resume).",
    )

    args = parser.parse_args()

    # Orchestrator mode: spawn workers + merge. A process that already carries --shard-tag is a
    # worker, so it falls through to the single-conversion path regardless of --num-proc.
    if args.num_proc > 1 and args.shard_tag is None:
        run_parallel_conversion(args)
    else:
        create_lerobot_dataset(
            raw_dir=args.raw_dir,
            repo_id=args.repo_id,
            local_dir=args.local_dir,
            push_to_hub=args.push_to_hub,
            fps=args.fps,
            robot_type=args.robot_type,
            use_videos=args.use_videos,
            image_writer_process=args.image_writer_process,
            image_writer_threads=args.image_writer_threads,
            split=args.split,
            shard_tag=args.shard_tag,
            max_episodes=args.max_episodes,
            prefetch_buffer=args.prefetch_buffer,
            skip_bad_episodes=args.skip_bad_episodes,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
