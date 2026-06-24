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
from oxe_utils.configs import OXE_DATASET_CONFIGS, ActionEncoding, StateEncoding
from oxe_utils.transforms import OXE_STANDARDIZATION_TRANSFORMS
from oxe_utils.constants import STATE_NAMES, ACTION_NAMES

np.set_printoptions(precision=2)


def transform_raw_dataset(episode, dataset_name):
    traj = next(iter(episode["steps"].batch(episode["steps"].cardinality())))

    if dataset_name in OXE_STANDARDIZATION_TRANSFORMS:
        traj = OXE_STANDARDIZATION_TRANSFORMS[dataset_name](traj)

    # The standardization transform populates "state" and "action" as dicts of named sub-features
    # (e.g. state.eef_xyz, action.eef_rpy, ...). Cast every sub-feature to float32 for LeRobot.
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

    # "state"/"action" are produced by the standardization transform and do not exist on the raw
    # builder, so derive their specs from one already-transformed episode. Each sub-feature is
    # batched as (T, D); the per-frame shape is therefore value.shape[1:].
    steps = episode["steps"]
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
    for episode in progress:
        traj = episode["steps"]
        num_frames = next(iter(traj["action"].values())).shape[0]
        for i in range(num_frames):
            image_dict = {
                f"observation.images.{key}": value[i]
                for key, value in traj["observation"].items()
                if "depth" not in key and any(x in key for x in ["image", "rgb"])
            }
            state_dict = {
                f"state.{key}": value[i]
                for key, value in traj["state"].items()
            }
            action_dict = {
                f"action.{key}": value[i]
                for key, value in traj["action"].items()
            }
            # append the state and action
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
        lerobot_dataset.save_episode()


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
):
    dataset_name, version, data_dir = _parse_raw_dir(raw_dir)

    if local_dir is None:
        local_dir = Path(HF_LEROBOT_HOME)
    # ``shard_tag`` keeps each parallel worker's partial dataset in its own directory so the
    # orchestrator can merge them afterwards (see run_parallel_conversion).
    tag_suffix = f"_{shard_tag}" if shard_tag else ""
    local_dir /= f"{dataset_name}_{version}_lerobot{tag_suffix}"
    if local_dir.exists():
        shutil.rmtree(local_dir)

    builder = tfds.builder(dataset_name, data_dir=data_dir, version=version)
    # features = generate_features_from_raw(builder, use_videos)
    # ``split`` may be a TFDS slice like "train[0%:25%]" (one worker's disjoint episode range).
    # Slice membership is derived from dataset_info shard lengths, so it is exact and independent
    # of read order -- safe even though RLDS read order is only deterministic with shuffle_files=False.
    filter_fn = lambda e: e["success"] if dataset_name == "kuka" else True
    raw_dataset = (
        builder.as_dataset(split=split)
        .filter(filter_fn)
        .map(partial(transform_raw_dataset, dataset_name=dataset_name))
    )
    if max_episodes is not None:  # debug/smoke-test escape hatch
        raw_dataset = raw_dataset.take(max_episodes)
    # Overlap tf.data production (image decode + transform) with the Python writer loop below.
    raw_dataset = raw_dataset.prefetch(tf.data.AUTOTUNE)

    # Episode count for the tqdm ETA, read from dataset_info (no data scan). For kuka the success
    # filter trims it, so this is an upper bound; --max-episodes caps it.
    try:
        total_episodes = builder.info.splits[split].num_examples
    except Exception:
        total_episodes = None
    if max_episodes is not None:
        total_episodes = min(total_episodes, max_episodes) if total_episodes else max_episodes
    # Per-worker bar identity: label by shard tag and stack bars at distinct positions in parallel runs.
    progress_desc = shard_tag or f"{dataset_name}_{version}"
    progress_position = int(shard_tag[len("shard"):]) if (shard_tag or "").startswith("shard") else 0

    # Peek one transformed episode so the feature schema reflects the keys this dataset's transform
    # actually emits (datasets may produce only a subset of STATE_NAMES / ACTION_NAMES).
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
    )

    # Close the parquet writers so footer metadata is flushed to disk. Without this the last
    # data/meta parquet files are left without their footer ("PAR1" magic bytes), which makes the
    # dataset unreadable (e.g. HF viewer: "Parquet magic bytes not found in footer").
    lerobot_dataset.finalize()

    if push_to_hub:
        assert repo_id is not None
        tags = ["LeRobot", dataset_name, "rlds"]
        if dataset_name in OXE_DATASET_CONFIGS:
            tags.append("openx")
        if robot_type != "unknown":
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

    print(f"[parallel] {dataset_name} {version}: {n} workers over slices {slices}")
    procs = []
    for i in range(n):
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--raw-dir", str(args.raw_dir),
            "--local-dir", str(shards_root),
            "--repo-id", worker_repo_ids[i],
            "--split", slices[i],
            "--shard-tag", tags[i],
            "--image-writer-process", str(args.image_writer_process),
            "--image-writer-threads", str(args.image_writer_threads),
        ]
        if args.use_videos:
            cmd.append("--use-videos")
        if args.fps is not None:
            cmd += ["--fps", str(args.fps)]
        if args.robot_type is not None:
            cmd += ["--robot-type", args.robot_type]
        if args.max_episodes is not None:
            cmd += ["--max-episodes", str(args.max_episodes)]
        procs.append(subprocess.Popen(cmd))

    failed = [i for i, p in enumerate(procs) if p.wait() != 0]
    if failed:
        raise RuntimeError(f"[parallel] worker(s) {failed} failed; aborting before merge")

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
        )


if __name__ == "__main__":
    main()
