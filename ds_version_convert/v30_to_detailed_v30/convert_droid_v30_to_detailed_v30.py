#!/usr/bin/env python
"""
Convert the official DROID LeRobot v3.0 dataset into a "detailed" v3.0 dataset implementing
openx2lerobot/design_of_state_and_action_space.md, producing exactly the same entries as
openx2lerobot/openx_rlds.py (droid_baseact_transform) would produce from the raw RLDS dataset --
but orders of magnitude faster, because it works directly on the v3 parquet files and reuses the
already-encoded videos via hardlinks instead of re-encoding 27M frames.

Per-frame entries of the output dataset:
    observation.images.exterior_image_1_left   <- observation.images.exterior_1_left  (video, linked)
    observation.images.exterior_image_2_left   <- observation.images.exterior_2_left  (video, linked)
    observation.images.wrist_image_left        <- observation.images.wrist_left       (video, linked)
    state.eef_xyz            = observation.state.cartesian_position[:3]
    state.eef_rpy            = observation.state.cartesian_position[3:6]
    state.joint_position     = observation.state.joint_position
    state.gripper_state      = 1 - observation.state.gripper_position
                               # DROID stores gripper_position as 0 = fully open, 1 = fully closed;
                               # gripper_state is the INVERTED value: 1 = fully open, 0 = fully
                               # closed (OpenVLA convention, via invert_gripper_actions)
    state.eef_rot6d          = rpy (extrinsic XYZ) -> 6D rotation (Zhou et al. 2019)
    observation.state        = [eef_xyz, eef_rpy, eef_rot6d, joint_position, gripper_state]   (20)
    action.world_eef_xyz     = p_{t+1} - p_t                  (world frame, last step = 0)
    action.world_eef_rpy     = rpy_{t+1} - rpy_t              (componentwise, last step = 0)
    action.world_eef_rot6d   = 6D of R_{t+1} R_t^T            (world frame, last step = identity)
    action.gripper_eef_xyz      = R_t^T (p_{t+1} - p_t)          (gripper frame, last step = 0)
    action.gripper_eef_rot6d    = 6D of R_t^T R_{t+1}            (gripper frame, last step = identity)
    action.joint_position    = joint_{t+1} - joint_t          (realized delta, last step = 0)
    action.gripper_state     = 1 - action.gripper_position
                               # commanded ABSOLUTE gripper target (not a delta), inverted like
                               # gripper_state: 1 = command fully open, 0 = command fully closed
    action                   = [gripper_eef_xyz, gripper_eef_rot6d, gripper_state]              (10)

All rotation features use DROID's native euler convention -- extrinsic (fixed-axis) XYZ,
R = Rz(yaw) @ Ry(pitch) @ Rx(roll) -- matching the updated droid_baseact_transform.

NOTE: openx_rlds.py randomly swaps the two exterior cameras per episode (a train-time augmentation
that leaked into conversion). Episodes share video files in v3, so a per-episode swap is impossible
without re-encoding; this converter maps the cameras deterministically instead. No information is
lost -- both views are kept, only the (random) name assignment differs.

Example:
    python convert_droid_v30_to_detailed_v30.py \
        --source-dir /path/to/droid_lerobot_v3 \
        --output-dir /path/to/droid_detailed_v3 \
        --num-workers 16
"""

import argparse
import json
import logging
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.compute_stats import aggregate_feature_stats, get_feature_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -------------------------------------------------------------------------------------------------
# Output schema (mirrors openx_rlds.py + oxe_utils for the "droid" dataset)
# -------------------------------------------------------------------------------------------------
VIDEO_KEY_MAP = {  # source key -> output (RLDS) key
    "observation.images.exterior_1_left": "observation.images.exterior_image_1_left",
    "observation.images.exterior_2_left": "observation.images.exterior_image_2_left",
    "observation.images.wrist_left": "observation.images.wrist_image_left",
}

STATE_NAMES = {
    "eef_xyz": ["x", "y", "z"],
    "eef_rpy": ["roll", "pitch", "yaw"],
    "eef_rot6d": ["rot1", "rot2", "rot3", "rot4", "rot5", "rot6"],
    "joint_position": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    "gripper_state": ["gripper"],
}
ACTION_NAMES = {
    "world_eef_xyz": ["x", "y", "z"],
    "world_eef_rpy": ["roll", "pitch", "yaw"],
    "world_eef_rot6d": ["rot1", "rot2", "rot3", "rot4", "rot5", "rot6"],
    "gripper_eef_xyz": ["x", "y", "z"],
    "gripper_eef_rot6d": ["rot1", "rot2", "rot3", "rot4", "rot5", "rot6"],
    "joint_position": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    "gripper_state": ["gripper"],
}
# key order matches the dict insertion order of droid_baseact_transform
STATE_KEYS = ["eef_xyz", "eef_rpy", "joint_position", "gripper_state", "eef_rot6d"]
ACTION_KEYS = [
    "world_eef_xyz", "world_eef_rpy", "world_eef_rot6d", "gripper_eef_xyz", "gripper_eef_rot6d",
    "joint_position", "gripper_state",
]
# OXE_DATASET_CONFIGS["droid"]: observation.state / action reference vectors (single-arm -> gripper frame)
STATE_ENCODING = [("eef_xyz", 3), ("eef_rpy", 3), ("eef_rot6d", 6), ("joint_position", 7), ("gripper_state", 1)]
ACTION_ENCODING = [("gripper_eef_xyz", 3), ("gripper_eef_rot6d", 6), ("gripper_state", 1)]

# columns copied verbatim from the source data parquets
PASSTHROUGH_COLUMNS = ["timestamp", "frame_index", "episode_index", "index", "task_index"]

DIMS = {f"state.{k}": len(STATE_NAMES[k]) for k in STATE_KEYS}
DIMS["observation.state"] = sum(d for _, d in STATE_ENCODING)
DIMS.update({f"action.{k}": len(ACTION_NAMES[k]) for k in ACTION_KEYS})
DIMS["action"] = sum(d for _, d in ACTION_ENCODING)
COMPUTED_COLUMNS = list(DIMS.keys())

STAT_KEYS = ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]


# -------------------------------------------------------------------------------------------------
# numpy ports of oxe_utils/transform_utils.py: rotation_6d = first two matrix rows (Zhou et al.
# 2019). DROID's native euler is extrinsic (fixed-axis) XYZ, so rpy_to_matrix is called with
# extrinsic=True (R = Rz @ Ry @ Rx), matching the updated droid_baseact_transform.
# -------------------------------------------------------------------------------------------------
def _axis_rotation(axis: str, angle: np.ndarray) -> np.ndarray:
    cos, sin = np.cos(angle), np.sin(angle)
    one, zero = np.ones_like(angle), np.zeros_like(angle)
    if axis == "X":
        flat = [one, zero, zero, zero, cos, -sin, zero, sin, cos]
    elif axis == "Y":
        flat = [cos, zero, sin, zero, one, zero, -sin, zero, cos]
    else:
        flat = [cos, -sin, zero, sin, cos, zero, zero, zero, one]
    rows = [np.stack(flat[i : i + 3], axis=-1) for i in (0, 3, 6)]
    return np.stack(rows, axis=-2)


def rpy_to_matrix(rpy: np.ndarray, extrinsic: bool = False) -> np.ndarray:
    """Build rotation matrices from RPY = (roll about X, pitch about Y, yaw about Z).

    extrinsic=False -> intrinsic XYZ, R = Rx(roll) @ Ry(pitch) @ Rz(yaw)  (pytorch3d/DROID-published).
    extrinsic=True  -> extrinsic (fixed-axis) XYZ, R = Rz(yaw) @ Ry(pitch) @ Rx(roll)  (DROID native).
    """
    x = _axis_rotation("X", rpy[..., 0])
    y = _axis_rotation("Y", rpy[..., 1])
    z = _axis_rotation("Z", rpy[..., 2])
    return (z @ y @ x) if extrinsic else (x @ y @ z)


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate((matrix[..., 0, :], matrix[..., 1, :]), axis=-1)


def world_gripper_eef_motion(xyz: np.ndarray, rpy: np.ndarray, extrinsic: bool = False) -> dict[str, np.ndarray]:
    """See transform_utils.world_gripper_eef_motion: per-step forward relative eef motion in the world
    and gripper frames; action[t] pairs with observation[t]; the last step has no successor, so its
    delta is zero translation / identity rotation."""
    R = rpy_to_matrix(rpy, extrinsic=extrinsic)  # [T, 3, 3]
    R_curr_T = np.swapaxes(R[:-1], -1, -2)        # R_t^T
    dp = xyz[1:] - xyz[:-1]                        # world-frame translation delta
    world_rot6d = matrix_to_rotation_6d(R[1:] @ R_curr_T)  # 6D of R_{t+1} R_t^T
    gripper_rot6d = matrix_to_rotation_6d(R_curr_T @ R[1:])   # 6D of R_t^T R_{t+1}
    gripper_xyz = (R_curr_T @ dp[..., None])[..., 0]          # R_t^T (p_{t+1} - p_t)
    world_rpy = rpy[1:] - rpy[:-1]
    eye6 = matrix_to_rotation_6d(np.eye(3)[None])          # 6D of identity
    zero3 = np.zeros((1, 3))
    return {
        "world_eef_xyz": np.concatenate([dp, zero3], axis=0),
        "world_eef_rpy": np.concatenate([world_rpy, zero3], axis=0),
        "world_eef_rot6d": np.concatenate([world_rot6d, eye6], axis=0),
        "gripper_eef_xyz": np.concatenate([gripper_xyz, zero3], axis=0),
        "gripper_eef_rot6d": np.concatenate([gripper_rot6d, eye6], axis=0),
    }


# -------------------------------------------------------------------------------------------------
# per-episode transform (numpy port of droid_baseact_transform)
# -------------------------------------------------------------------------------------------------
def transform_episode(cart, joint, grip, act_grip) -> dict[str, np.ndarray]:
    state = {
        "eef_xyz": cart[:, :3],
        "eef_rpy": cart[:, 3:6],
        "joint_position": joint,
        # invert_gripper_actions: DROID gripper_position is 0 = open / 1 = closed; the stored
        # gripper_state is 1 = fully open / 0 = fully closed (OpenVLA convention)
        "gripper_state": 1.0 - grip,
    }
    # DROID's native euler is extrinsic (fixed-axis) XYZ, so build rotations with extrinsic=True.
    state["eef_rot6d"] = matrix_to_rotation_6d(rpy_to_matrix(state["eef_rpy"], extrinsic=True))

    motion = world_gripper_eef_motion(state["eef_xyz"], state["eef_rpy"], extrinsic=True)
    joint_delta = np.concatenate([joint[1:] - joint[:-1], np.zeros((1, joint.shape[1]))], axis=0)
    action = {
        **motion,
        # joint_{t+1} - joint_t (realized delta from state; last step has no successor -> 0)
        "joint_position": joint_delta,
        # commanded absolute gripper target, inverted like gripper_state (1 = open / 0 = closed)
        "gripper_state": 1.0 - act_grip,
    }

    out = {f"state.{k}": v.astype(np.float32) for k, v in state.items()}
    out.update({f"action.{k}": v.astype(np.float32) for k, v in action.items()})
    out["observation.state"] = np.concatenate(
        [out[f"state.{k}"] for k, _ in STATE_ENCODING], axis=1
    )
    num_frames = cart.shape[0]
    out["action"] = np.concatenate(
        [
            np.zeros((num_frames, dim), dtype=np.float32) if k == "pad" else out[f"action.{k}"]
            for k, dim in ACTION_ENCODING
        ],
        axis=1,
    )
    return out


# -------------------------------------------------------------------------------------------------
# parquet helpers
# -------------------------------------------------------------------------------------------------
def list_column_to_numpy(table: pa.Table, name: str, dim: int) -> np.ndarray:
    col = table.column(name).combine_chunks()
    values = np.asarray(col.flatten(), dtype=np.float64)
    if values.size != table.num_rows * dim:
        raise ValueError(f"column {name}: expected fixed length {dim}, got ragged data")
    return values.reshape(-1, dim)


def scalar_column_to_numpy(table: pa.Table, name: str) -> np.ndarray:
    return table.column(name).combine_chunks().to_numpy(zero_copy_only=False)


def numpy_to_list_array(arr: np.ndarray, value_type: pa.DataType) -> pa.ListArray:
    n, d = arr.shape
    values = pa.array(np.ascontiguousarray(arr).reshape(-1), type=value_type)
    offsets = pa.array(np.arange(0, (n + 1) * d, d, dtype=np.int32), type=pa.int32())
    return pa.ListArray.from_arrays(offsets, values)


def episode_slices(episode_index: np.ndarray):
    """Yield (episode_id, start, stop) for contiguous runs; asserts episodes are contiguous."""
    if len(episode_index) == 0:
        return
    boundaries = np.flatnonzero(np.diff(episode_index)) + 1
    starts = np.concatenate([[0], boundaries])
    stops = np.concatenate([boundaries, [len(episode_index)]])
    seen = set()
    for start, stop in zip(starts, stops):
        ep = int(episode_index[start])
        if ep in seen:
            raise ValueError(f"episode {ep} is not contiguous within its data file")
        seen.add(ep)
        yield ep, int(start), int(stop)


# -------------------------------------------------------------------------------------------------
# data-file worker
# -------------------------------------------------------------------------------------------------
SRC_DATA_COLUMNS = [
    "observation.state.cartesian_position",
    "observation.state.joint_position",
    "observation.state.gripper_position",
    "action.gripper_position",
    *PASSTHROUGH_COLUMNS,
]


def process_data_file(src_path: str, dst_path: str):
    """Convert one data parquet file. Returns (episode_ids, per-episode stats arrays)."""
    table = pq.read_table(src_path, columns=SRC_DATA_COLUMNS)
    cart = list_column_to_numpy(table, "observation.state.cartesian_position", 6)
    joint = list_column_to_numpy(table, "observation.state.joint_position", 7)
    grip = scalar_column_to_numpy(table, "observation.state.gripper_position").astype(np.float64)[:, None]
    act_grip = scalar_column_to_numpy(table, "action.gripper_position").astype(np.float64)[:, None]
    episode_index = scalar_column_to_numpy(table, "episode_index")

    columns = {name: [] for name in COMPUTED_COLUMNS}
    episode_ids = []
    episode_stats = {name: {stat: [] for stat in STAT_KEYS} for name in COMPUTED_COLUMNS}
    for ep, start, stop in episode_slices(episode_index):
        s = slice(start, stop)
        out = transform_episode(cart[s], joint[s], grip[s], act_grip[s])
        episode_ids.append(ep)
        for name in COMPUTED_COLUMNS:
            columns[name].append(out[name])
            stats = get_feature_stats(out[name], axis=0, keepdims=False)
            for stat in STAT_KEYS:
                episode_stats[name][stat].append(stats[stat])

    arrays, fields = [], []
    for name in COMPUTED_COLUMNS:
        data = np.concatenate(columns[name], axis=0)
        if DIMS[name] == 1:  # shape-(1,) features are stored as scalar columns (lerobot convention)
            arrays.append(pa.array(data[:, 0], type=pa.float32()))
            fields.append(pa.field(name, pa.float32()))
        else:
            arrays.append(numpy_to_list_array(data, pa.float32()))
            fields.append(pa.field(name, pa.list_(pa.float32())))
    for name in PASSTHROUGH_COLUMNS:
        col = table.column(name).combine_chunks()
        arrays.append(col)
        fields.append(pa.field(name, col.type))

    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(arrays, schema=pa.schema(fields)), dst_path)

    stacked = {
        name: {stat: np.stack(values, axis=0) for stat, values in feature_stats.items()}
        for name, feature_stats in episode_stats.items()
    }
    return np.asarray(episode_ids, dtype=np.int64), stacked


# -------------------------------------------------------------------------------------------------
# meta writers
# -------------------------------------------------------------------------------------------------
def build_features(src_features: dict) -> dict:
    features = {}
    for old_key, new_key in VIDEO_KEY_MAP.items():
        features[new_key] = {
            "dtype": "video",
            "shape": src_features[old_key]["shape"],
            "names": ["height", "width", "rgb"],
            "info": src_features[old_key]["info"],
        }
    for key in STATE_KEYS:
        features[f"state.{key}"] = {
            "dtype": "float32",
            "shape": [len(STATE_NAMES[key])],
            "names": STATE_NAMES[key],
        }
    features["observation.state"] = {
        "dtype": "float32",
        "shape": [DIMS["observation.state"]],
        "names": [n for k, _ in STATE_ENCODING for n in STATE_NAMES[k]],
    }
    for key in ACTION_KEYS:
        features[f"action.{key}"] = {
            "dtype": "float32",
            "shape": [len(ACTION_NAMES[key])],
            "names": ACTION_NAMES[key],
        }
    features["action"] = {
        "dtype": "float32",
        "shape": [DIMS["action"]],
        "names": [
            n
            for k, dim in ACTION_ENCODING
            for n in (["pad"] * dim if k == "pad" else ACTION_NAMES[k])
        ],
    }
    for key in PASSTHROUGH_COLUMNS:
        features[key] = src_features[key]
    return features


def write_episodes_meta(src_root: Path, dst_root: Path, global_stats: dict):
    """Rewrite meta/episodes parquets: copy structure/video pointers (renamed), keep passthrough
    stats, attach the freshly computed per-episode stats for the new features."""
    src_files = sorted((src_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    for src_file in src_files:
        table = pq.read_table(src_file)
        ep_idx = table.column("episode_index").combine_chunks().to_numpy(zero_copy_only=False)

        arrays, fields = [], []

        def copy_col(src_name, dst_name=None):
            col = table.column(src_name).combine_chunks()
            arrays.append(col)
            fields.append(pa.field(dst_name or src_name, col.type))

        for name in ["episode_index", "tasks", "length", "data/chunk_index", "data/file_index",
                     "dataset_from_index", "dataset_to_index"]:
            copy_col(name)
        for old_key, new_key in VIDEO_KEY_MAP.items():
            for sub in ["chunk_index", "file_index", "from_timestamp", "to_timestamp"]:
                copy_col(f"videos/{old_key}/{sub}", f"videos/{new_key}/{sub}")
        # stats columns, in output-feature order
        for old_key, new_key in VIDEO_KEY_MAP.items():
            for stat in STAT_KEYS:
                copy_col(f"stats/{old_key}/{stat}", f"stats/{new_key}/{stat}")
        for name in COMPUTED_COLUMNS:
            for stat in STAT_KEYS:
                data = global_stats[name][stat][ep_idx]
                value_type = pa.int64() if stat == "count" else pa.float64()
                arr = numpy_to_list_array(data.astype(np.int64 if stat == "count" else np.float64), value_type)
                arrays.append(arr)
                fields.append(pa.field(f"stats/{name}/{stat}", pa.list_(value_type)))
        for name in PASSTHROUGH_COLUMNS:
            for stat in STAT_KEYS:
                copy_col(f"stats/{name}/{stat}")
        for name in ["meta/episodes/chunk_index", "meta/episodes/file_index"]:
            copy_col(name)

        dst_file = dst_root / src_file.relative_to(src_root)
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(arrays, schema=pa.schema(fields)), dst_file)
        logging.info(f"wrote {dst_file}")


def write_stats_json(src_root: Path, dst_root: Path, global_stats: dict, total_episodes: int):
    with open(src_root / "meta" / "stats.json") as f:
        src_stats = json.load(f)

    stats = {}
    for old_key, new_key in VIDEO_KEY_MAP.items():  # unchanged pixels: copy aggregated stats
        stats[new_key] = src_stats[old_key]
    for name in COMPUTED_COLUMNS:
        per_ep = global_stats[name]
        stats_list = [{stat: per_ep[stat][i] for stat in STAT_KEYS} for i in range(total_episodes)]
        agg = aggregate_feature_stats(stats_list)
        stats[name] = {stat: np.asarray(value).tolist() for stat, value in agg.items()}
    for name in PASSTHROUGH_COLUMNS:  # unchanged columns: copy aggregated stats
        stats[name] = src_stats[name]

    with open(dst_root / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    logging.info(f"wrote {dst_root / 'meta' / 'stats.json'}")


def read_episode_refs(source_dir: Path, total_episodes: int):
    """Collect, from meta/episodes, the per-episode lengths and the (chunk, file) pairs actually
    referenced for data and for each video key. The data/videos directories may contain orphan
    files not referenced by any episode (e.g. leftovers of a partial re-conversion; the official
    DROID v3 dataset ships 70 such data files duplicating 42k episodes) -- LeRobot's reader ignores
    them, and so must we."""
    lengths = np.full(total_episodes, -1, dtype=np.int64)
    data_refs = set()
    video_refs = {old_key: set() for old_key in VIDEO_KEY_MAP}
    columns = ["episode_index", "length", "data/chunk_index", "data/file_index"] + [
        f"videos/{key}/{sub}" for key in VIDEO_KEY_MAP for sub in ["chunk_index", "file_index"]
    ]
    for src_file in sorted((source_dir / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(src_file, columns=columns)
        cols = {name: table.column(name).to_numpy() for name in columns}
        lengths[cols["episode_index"]] = cols["length"]
        data_refs.update(zip(cols["data/chunk_index"].tolist(), cols["data/file_index"].tolist()))
        for key in VIDEO_KEY_MAP:
            video_refs[key].update(
                zip(cols[f"videos/{key}/chunk_index"].tolist(), cols[f"videos/{key}/file_index"].tolist())
            )
    if (lengths < 0).any():
        raise RuntimeError("meta/episodes does not cover all episodes")
    return lengths, sorted(data_refs), video_refs


def link_videos(src_root: Path, dst_root: Path, mode: str, video_refs: dict):
    for old_key, new_key in VIDEO_KEY_MAP.items():
        src_dir = src_root / "videos" / old_key
        n = 0
        for chunk_idx, file_idx in sorted(video_refs[old_key]):
            src_file = src_dir / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
            dst_file = dst_root / "videos" / new_key / src_file.relative_to(src_dir)
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            if dst_file.exists():
                dst_file.unlink()
            if mode == "hardlink":
                os.link(src_file, dst_file)
            elif mode == "symlink":
                os.symlink(src_file, dst_file)
            else:
                shutil.copy2(src_file, dst_file)
            n += 1
        logging.info(f"{mode}ed {n} video files: {old_key} -> {new_key}")


# -------------------------------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------------------------------
def convert(source_dir: Path, output_dir: Path, num_workers: int, video_mode: str, max_data_files: int | None):
    with open(source_dir / "meta" / "info.json") as f:
        src_info = json.load(f)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "meta").mkdir(parents=True)

    total_episodes = src_info["total_episodes"]
    # Only convert the data/video files actually referenced by meta/episodes: the source dirs may
    # contain orphan files (duplicated episodes from a partial re-conversion) that LeRobot ignores.
    ep_lengths, data_refs, video_refs = read_episode_refs(source_dir, total_episodes)
    data_files = [
        source_dir / "data" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet"
        for chunk_idx, file_idx in data_refs
    ]
    n_on_disk = len(list((source_dir / "data").glob("chunk-*/file-*.parquet")))
    if n_on_disk != len(data_files):
        logging.warning(f"{n_on_disk - len(data_files)} orphan data files on disk are not referenced by meta; skipping them")
    if max_data_files is not None:
        data_files = data_files[:max_data_files]
    logging.info(f"converting {len(data_files)} data files with {num_workers} workers")
    global_stats = {
        name: {
            stat: np.full(
                (total_episodes, 1 if stat == "count" else DIMS[name]),
                np.nan if stat != "count" else -1,
                dtype=np.int64 if stat == "count" else np.float64,
            )
            for stat in STAT_KEYS
        }
        for name in COMPUTED_COLUMNS
    }
    covered = np.zeros(total_episodes, dtype=bool)

    start_time = time.time()
    jobs = {}
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for src_file in data_files:
            dst_file = output_dir / src_file.relative_to(source_dir)
            jobs[pool.submit(process_data_file, str(src_file), str(dst_file))] = src_file
        done = 0
        for future in as_completed(jobs):
            episode_ids, stacked = future.result()
            covered[episode_ids] = True
            for name in COMPUTED_COLUMNS:
                for stat in STAT_KEYS:
                    global_stats[name][stat][episode_ids] = stacked[name][stat]
            done += 1
            if done % 10 == 0 or done == len(data_files):
                elapsed = time.time() - start_time
                logging.info(f"data files: {done}/{len(data_files)} ({elapsed:.0f}s elapsed)")

    if max_data_files is not None:
        logging.warning("--max-data-files set: data parquets written, skipping meta/videos (debug run)")
        return
    if not covered.all():
        raise RuntimeError(f"{(~covered).sum()} episodes missing from data files")
    counts = global_stats["observation.state"]["count"][:, 0]
    if not np.array_equal(counts, ep_lengths):
        bad = int((counts != ep_lengths).sum())
        raise RuntimeError(f"{bad} episodes have a frame count inconsistent with meta/episodes lengths")

    write_episodes_meta(source_dir, output_dir, global_stats)
    write_stats_json(source_dir, output_dir, global_stats, total_episodes)
    shutil.copy2(source_dir / "meta" / "tasks.parquet", output_dir / "meta" / "tasks.parquet")

    info = dict(src_info)
    info["robot_type"] = "franka"
    info["features"] = build_features(src_info["features"])
    with open(output_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)
    logging.info(f"wrote {output_dir / 'meta' / 'info.json'}")

    link_videos(source_dir, output_dir, video_mode, video_refs)
    logging.info(f"done in {time.time() - start_time:.0f}s -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, required=True, help="DROID LeRobot v3.0 dataset root")
    parser.add_argument("--output-dir", type=Path, required=True, help="output dataset root (wiped if exists)")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--video-mode",
        choices=["hardlink", "symlink", "copy"],
        default="hardlink",
        help="how to materialize videos under the renamed keys (hardlink requires same filesystem)",
    )
    parser.add_argument("--max-data-files", type=int, default=None, help="debug: only convert the first N data files")
    args = parser.parse_args()
    convert(args.source_dir, args.output_dir, args.num_workers, args.video_mode, args.max_data_files)


if __name__ == "__main__":
    main()
