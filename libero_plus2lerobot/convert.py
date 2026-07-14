#!/usr/bin/env python3
"""Convert a LIBERO Plus LeRobot v2.1 partition to detailed LeRobot v3.0.

The converter preserves every source frame.  It audits the historical LIBERO
no-op predicate, but deliberately does not filter: without simulator replay a
zero command may still have a non-zero state transition due to controller and
physics dynamics.

The source columns are retained exactly as ``raw_state.ref_state`` and
``raw_action.ref_action``.  Split native state/absolute-target fields and
canonical state/target/debug fields follow ``failure_rollout_data/dataset.md``.

Because frames are not removed, videos are repacked without decoding or
re-encoding via the repository's v2.1 -> v3.0 conversion helpers.

The source camera streams are also preserved pixel-for-pixel. Consumers using
the canonical training convention must horizontally flip both camera views at
load time; the converter documents this requirement but does not alter videos.
The source pipeline receives vertically inverted robosuite renders and then
rotates them by 180 degrees, leaving a horizontal mirror after the vertical
components cancel.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# <repo>/libero_plus2lerobot/convert.py -> <repo>
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alignment import transforms_numpy as tn  # noqa: E402
from ds_version_convert.v21_to_v30.convert_dataset_v21_to_v30 import (  # noqa: E402
    convert_data,
    convert_episodes_metadata,
    convert_info,
    convert_tasks,
    convert_videos,
)

FPS = 20
POSITION_SCALE = 0.05
ROTATION_SCALE = 0.5
FINGER_LIMIT_M = 0.04
NOOP_THRESHOLD = 1e-4
VIDEO_QUANTILE_SAMPLE_FRAMES = 10_000

XYZ_NAMES = ["x", "y", "z"]
AXIS_ANGLE_NAMES = ["ax", "ay", "az"]
ROT6D_NAMES = ["r11", "r21", "r31", "r12", "r22", "r32"]
ROT9D_NAMES = [f"r{row}{col}" for row in range(1, 4) for col in range(1, 4)]
FINGER_NAMES = ["finger_0", "finger_1"]
REF_STATE_NAMES = ["x", "y", "z", "ax", "ay", "az", "finger_0", "finger_1"]
REF_ACTION_NAMES = ["dx", "dy", "dz", "dax", "day", "daz", "gripper"]

R_WORLD_ALIGN = np.eye(3, dtype=np.float32)
R_GRIPPER_ALIGN = tn.axis_alignment_matrix("-y", "x", "z")
IDENTITY_ROT6D = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32)


def dataset_card(partition: str) -> str:
    """Return the Hugging Face dataset card installed in each converted tree."""
    return f"""---
tags:
- lerobot
- robotics
- libero
- libero-plus
pretty_name: LIBERO Plus {partition} (detailed LeRobot v3.0)
---

# {partition}: detailed LeRobot v3.0

This dataset was converted from the LIBERO Plus LeRobot v2.1 `{partition}` partition.
The original 8D state and 7D action vectors are preserved exactly as
`raw_state.ref_state` and `raw_action.ref_action`. Canonical low-dimensional fields follow
`failure_rollout_data/dataset.md`; `debug.gripper_eef_*` contains the ground-truth next-step
relative EEF motion for inspection.

## Required camera transform for canonical training

The source `observation.images.front` and `observation.images.wrist` videos are preserved
unchanged. **For canonical training, horizontally flip both camera views at load time.** The
videos in this repository are deliberately not rewritten or re-encoded.

The source-pipeline root cause is a composition of two image transforms: robosuite returns a
vertically inverted render, and the original dataset writer then rotates it by 180 degrees (flips
both image axes). The vertical flips cancel, leaving the stored image horizontally mirrored. This
is also tracked in [LeRobot issue #3830](https://github.com/huggingface/lerobot/issues/3830).

For an array whose layout ends in `(height, width, channels)`:

```python
image = np.flip(image, axis=-2)
```

For a tensor whose width is the last dimension, such as `(..., channels, height, width)`:

```python
image = torch.flip(image, dims=(-1,))
```

Apply the image transform only to the camera pixels. Do not flip or negate `raw_state.*`,
`raw_target.*`, `state.*`, `target.*`, or `debug.*`; those fields remain proper right-handed
coordinate representations.

## Conversion notes

- No frames were filtered. The historical LIBERO no-op predicate is audit-only; see
  `meta/noop_audit.json` and `meta/noop_audit_episodes.jsonl`.
- Rotation and reconstruction checks are recorded in `meta/conversion_validation.json`.
- Alignment and controller-scale assumptions are recorded in `meta/conversion_config.json`.
- `meta/stats.json` includes `q01` and `q99` for every numeric and video feature. Numeric
  quantiles use every frame; video quantiles use a deterministic uniform sample of stored frames.
"""


def write_dataset_card(output_dir: Path) -> None:
    (output_dir / "README.md").write_text(dataset_card(output_dir.name))


def _read_json(path: Path):
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _write_jsonl(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for value in values:
            f.write(json.dumps(value) + "\n")


def _feature(dtype: str, shape: tuple[int, ...], names: list[str]) -> dict:
    return {"dtype": dtype, "shape": list(shape), "names": names}


def detailed_features(source_features: dict) -> dict:
    """Return v2.1 feature metadata for the temporary detailed dataset."""
    features = {
        key: value
        for key, value in source_features.items()
        if value["dtype"] == "video"
    }
    features.update(
        {
            "raw_state.ref_state": _feature("float32", (8,), REF_STATE_NAMES),
            "raw_state.eef_xyz": _feature("float32", (3,), XYZ_NAMES),
            "raw_state.eef_axis_angle": _feature("float32", (3,), AXIS_ANGLE_NAMES),
            "raw_state.gripper_state": _feature("float32", (2,), FINGER_NAMES),
            "raw_action.ref_action": _feature("float32", (7,), REF_ACTION_NAMES),
            "raw_target.eef_xyz": _feature("float32", (3,), XYZ_NAMES),
            "raw_target.eef_axis_angle": _feature("float32", (3,), AXIS_ANGLE_NAMES),
            "raw_target.gripper_state": _feature("float32", (1,), ["gripper"]),
            "state.eef_xyz": _feature("float32", (3,), XYZ_NAMES),
            "state.eef_rot9d": _feature("float32", (9,), ROT9D_NAMES),
            "state.gripper_state": _feature("float32", (2,), FINGER_NAMES),
            "target.eef_xyz": _feature("float32", (3,), XYZ_NAMES),
            "target.eef_rot9d": _feature("float32", (9,), ROT9D_NAMES),
            "target.gripper_state": _feature("float32", (1,), ["gripper"]),
            "debug.gripper_eef_xyz": _feature("float32", (3,), XYZ_NAMES),
            "debug.gripper_eef_rot6d": _feature("float32", (6,), ROT6D_NAMES),
        }
    )
    for key, value in source_features.items():
        if key not in {"observation.state", "action"} and value["dtype"] != "video":
            features[key] = value
    return features


def audit_candidate_noops(action: np.ndarray, threshold: float = NOOP_THRESHOLD) -> np.ndarray:
    """Audit the historical LIBERO filter without dropping any rows.

    ``prev_action`` advances only for a would-be-retained action, matching
    ``libero2lerobot/libero_utils/regenerate_libero_dataset.py``.
    """
    candidate = np.zeros(len(action), dtype=bool)
    prev_action = None
    for i, current in enumerate(action):
        arm_is_zero = np.linalg.norm(current[:-1]) < threshold
        is_noop = arm_is_zero and (prev_action is None or current[-1] == prev_action[-1])
        candidate[i] = is_noop
        if not is_noop:
            prev_action = current
    return candidate


def transform_episode(ref_state: np.ndarray, ref_action: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """Split native values and construct absolute targets plus canonical fields."""
    ref_state = np.asarray(ref_state, dtype=np.float32)
    ref_action = np.asarray(ref_action, dtype=np.float32)
    if ref_state.ndim != 2 or ref_state.shape[1] != 8:
        raise ValueError(f"observation.state must be [T,8], got {ref_state.shape}")
    if ref_action.shape != (len(ref_state), 7):
        raise ValueError(f"action must be [T,7], got {ref_action.shape}")

    xyz = ref_state[:, :3]
    axis_angle = ref_state[:, 3:6]
    raw_gripper = ref_state[:, 6:8]
    R_native = tn.axis_angle_to_matrix(axis_angle)

    # robosuite OSC_POSE applies scaled translation in world coordinates and
    # left-multiplies the scaled axis-angle delta onto the current orientation.
    target_xyz = xyz + POSITION_SCALE * ref_action[:, :3]
    R_delta = tn.axis_angle_to_matrix(ROTATION_SCALE * ref_action[:, 3:6])
    R_target_native = R_delta @ R_native
    # The target frequently lies close to pi, where the shared quaternion-based
    # matrix_to_axis_angle inverse loses precision. SciPy handles that branch
    # robustly; the result is still the same native rotation-vector convention.
    target_axis_angle = Rotation.from_matrix(
        np.asarray(R_target_native, dtype=np.float64)
    ).as_rotvec().astype(np.float32)
    raw_target_gripper = ref_action[:, 6:7]

    R_canon, xyz_canon = tn.align_axis(R_native, xyz, R_WORLD_ALIGN, R_GRIPPER_ALIGN)
    R_target_canon, target_xyz_canon = tn.align_axis(
        R_target_native, target_xyz, R_WORLD_ALIGN, R_GRIPPER_ALIGN
    )
    gripper_canon = np.stack(
        (raw_gripper[:, 0] / FINGER_LIMIT_M, -raw_gripper[:, 1] / FINGER_LIMIT_M), axis=-1
    )
    gripper_canon = np.clip(gripper_canon, 0.0, 1.0).astype(np.float32)
    target_gripper_canon = np.clip((1.0 - raw_target_gripper) / 2.0, 0.0, 1.0).astype(np.float32)

    dR, dp = tn.gripper_delta_pose(
        R_canon[:-1], xyz_canon[:-1], R_canon[1:], xyz_canon[1:]
    )
    debug_xyz = np.concatenate((dp, np.zeros((1, 3), dtype=np.float32)), axis=0)
    debug_rot6d = np.concatenate((tn.matrix_to_rotation_6d(dR), IDENTITY_ROT6D), axis=0)

    arrays = {
        "raw_state.ref_state": ref_state.copy(),
        "raw_state.eef_xyz": xyz.copy(),
        "raw_state.eef_axis_angle": axis_angle.copy(),
        "raw_state.gripper_state": raw_gripper.copy(),
        "raw_action.ref_action": ref_action.copy(),
        "raw_target.eef_xyz": target_xyz.astype(np.float32),
        "raw_target.eef_axis_angle": target_axis_angle.astype(np.float32),
        "raw_target.gripper_state": raw_target_gripper.copy(),
        "state.eef_xyz": xyz_canon.astype(np.float32),
        "state.eef_rot9d": R_canon.reshape(-1, 9).astype(np.float32),
        "state.gripper_state": gripper_canon,
        "target.eef_xyz": target_xyz_canon.astype(np.float32),
        "target.eef_rot9d": R_target_canon.reshape(-1, 9).astype(np.float32),
        "target.gripper_state": target_gripper_canon,
        "debug.gripper_eef_xyz": debug_xyz.astype(np.float32),
        "debug.gripper_eef_rot6d": debug_rot6d.astype(np.float32),
    }

    eye = np.eye(3, dtype=np.float32)
    ortho_error = float(np.max(np.abs(np.swapaxes(R_canon, -1, -2) @ R_canon - eye)))
    det_error = float(np.max(np.abs(np.linalg.det(R_canon) - 1.0)))
    if len(ref_state) > 1:
        R_recon = R_canon[:-1] @ dR
        p_recon = xyz_canon[:-1] + (R_canon[:-1] @ dp[..., None])[..., 0]
        debug_rot_error = float(np.max(np.abs(R_recon - R_canon[1:])))
        debug_xyz_error = float(np.max(np.abs(p_recon - xyz_canon[1:])))
    else:
        debug_rot_error = debug_xyz_error = 0.0
    target_roundtrip_error = float(
        np.max(np.abs(tn.axis_angle_to_matrix(target_axis_angle) - R_target_native))
    )
    validation = {
        "max_rotation_orthogonality_error": ortho_error,
        "max_rotation_determinant_error": det_error,
        "max_debug_rotation_reconstruction_error": debug_rot_error,
        "max_debug_xyz_reconstruction_error": debug_xyz_error,
        "max_target_axis_angle_roundtrip_error": target_roundtrip_error,
    }
    return arrays, validation


def _stats(array: np.ndarray) -> dict:
    array = np.asarray(array)
    if array.ndim == 1:
        array = array[:, None]
    quantiles = np.quantile(array, [0.01, 0.99], axis=0)
    return {
        "min": np.min(array, axis=0).tolist(),
        "max": np.max(array, axis=0).tolist(),
        "mean": np.mean(array, axis=0, dtype=np.float64).tolist(),
        "std": np.std(array, axis=0, dtype=np.float64).tolist(),
        "count": [int(len(array))],
        "q01": quantiles[0].tolist(),
        "q99": quantiles[1].tolist(),
    }


def _arrow_numeric_to_numpy(column: pa.ChunkedArray) -> np.ndarray:
    """Convert a scalar or fixed-width list parquet column to a dense 2D array."""
    array = column.combine_chunks()
    if array.null_count:
        raise ValueError("numeric feature contains null values")
    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        offsets = np.asarray(array.offsets)
        widths = np.diff(offsets)
        if len(widths) and not np.all(widths == widths[0]):
            raise ValueError("numeric feature contains ragged vectors")
        width = int(widths[0]) if len(widths) else 0
        values = np.asarray(array.values).reshape(len(array), width)
    elif pa.types.is_fixed_size_list(array.type):
        values = np.asarray(array.values).reshape(len(array), array.type.list_size)
    else:
        values = np.asarray(array).reshape(-1, 1)
    return values


def _histogram_quantile(histogram: np.ndarray, quantile: float) -> float:
    """Match NumPy's linear quantile interpolation for an integer histogram."""
    count = int(histogram.sum())
    if count == 0:
        raise ValueError("cannot compute a quantile from an empty histogram")
    rank = quantile * (count - 1)
    lower_rank = int(np.floor(rank))
    upper_rank = int(np.ceil(rank))
    cumulative = np.cumsum(histogram)
    lower = int(np.searchsorted(cumulative, lower_rank + 1))
    upper = int(np.searchsorted(cumulative, upper_rank + 1))
    return lower + (rank - lower_rank) * (upper - lower)


def _video_quantiles(
    output_dir: Path, video_key: str, total_frames: int
) -> tuple[list[float], list[float], int]:
    """Compute RGB quantiles from deterministic uniformly sampled stored video frames."""
    paths = sorted((output_dir / "videos" / video_key).glob("*/*.mp4"))
    if not paths:
        raise FileNotFoundError(f"no packed videos found for {video_key}")

    sample_count = min(VIDEO_QUANTILE_SAMPLE_FRAMES, total_frames)
    sample_indices = np.unique(
        np.round(np.linspace(0, total_frames - 1, sample_count)).astype(np.int64)
    )

    histograms = np.zeros((3, 256), dtype=np.int64)
    sampled = 0
    sample_cursor = 0
    global_frame = 0
    for path in paths:
        container = av.open(str(path))
        try:
            for frame in container.decode(video=0):
                if sample_cursor < len(sample_indices) and global_frame == sample_indices[sample_cursor]:
                    rgb = frame.to_ndarray(format="rgb24")
                    for channel in range(3):
                        histograms[channel] += np.bincount(
                            rgb[..., channel].reshape(-1), minlength=256
                        )
                    sampled += 1
                    sample_cursor += 1
                global_frame += 1
                if global_frame % 100_000 == 0:
                    print(f"{video_key}: decoded {global_frame}/{total_frames} frames")
        finally:
            container.close()

    if global_frame != total_frames:
        raise RuntimeError(
            f"{video_key}: decoded {global_frame} frames, expected {total_frames}"
        )
    if sampled != len(sample_indices):
        raise RuntimeError(
            f"{video_key}: sampled {sampled} frames, expected {len(sample_indices)}"
        )

    # LeRobot represents image/video channel stats as (channel, 1, 1).
    q01 = [[[_histogram_quantile(hist, 0.01) / 255.0]] for hist in histograms]
    q99 = [[[_histogram_quantile(hist, 0.99) / 255.0]] for hist in histograms]
    return q01, q99, sampled


def add_global_quantiles(output_dir: Path) -> dict:
    """Add exact numeric and sampled-video q01/q99 entries to v3 stats.json."""
    info = _read_json(output_dir / "meta" / "info.json")
    stats_path = output_dir / "meta" / "stats.json"
    stats = _read_json(stats_path)
    numeric_keys = [
        key
        for key, feature in info["features"].items()
        if feature["dtype"] not in {"video", "image", "string"} and key in stats
    ]
    video_keys = [
        key for key, feature in info["features"].items() if feature["dtype"] == "video"
    ]

    chunks = {key: [] for key in numeric_keys}
    data_paths = sorted((output_dir / "data").glob("*/*.parquet"))
    for path in tqdm(data_paths, desc="read global quantiles", unit="file", dynamic_ncols=True):
        table = pq.read_table(path, columns=numeric_keys)
        for key in numeric_keys:
            chunks[key].append(_arrow_numeric_to_numpy(table.column(key)))
    for key in numeric_keys:
        values = np.concatenate(chunks[key], axis=0)
        quantiles = np.quantile(values, [0.01, 0.99], axis=0)
        stats[key]["q01"] = quantiles[0].tolist()
        stats[key]["q99"] = quantiles[1].tolist()

    sampled_video_frames = {}
    with ThreadPoolExecutor(max_workers=len(video_keys)) as pool:
        futures = {
            key: pool.submit(_video_quantiles, output_dir, key, int(info["total_frames"]))
            for key in video_keys
        }
        for key, future in futures.items():
            q01, q99, sampled = future.result()
            stats[key]["q01"] = q01
            stats[key]["q99"] = q99
            sampled_video_frames[key] = sampled

    missing = [key for key, value in stats.items() if "q01" not in value or "q99" not in value]
    if missing:
        raise AssertionError(f"stats missing q01/q99: {missing}")
    _write_json(stats_path, stats)
    return {
        "quantiles": [0.01, 0.99],
        "numeric_strategy": "exact_all_frames",
        "video_strategy": "uniform_sample_of_stored_frames",
        "video_sample_limit_per_view": VIDEO_QUANTILE_SAMPLE_FRAMES,
        "sampled_video_frames": sampled_video_frames,
    }


def _episode_path(source_dir: Path, episode_index: int) -> Path:
    chunk = episode_index // 1000
    return source_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def _video_path(source_dir: Path, video_key: str, episode_index: int) -> Path:
    chunk = episode_index // 1000
    return source_dir / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{episode_index:06d}.mp4"


def prepare_detailed_v21(
    source_dir: Path,
    temp_dir: Path,
    max_episodes: int | None,
) -> tuple[list[dict], dict]:
    """Create a temporary detailed v2.1 tree consumed by the stock v3 packer."""
    source_info = _read_json(source_dir / "meta" / "info.json")
    if source_info.get("codebase_version") != "v2.1":
        raise ValueError(f"expected v2.1 source, got {source_info.get('codebase_version')!r}")
    if int(source_info["fps"]) != FPS:
        raise ValueError(f"expected {FPS} fps, got {source_info['fps']}")

    episodes = _read_jsonl(source_dir / "meta" / "episodes.jsonl")
    episodes = sorted(episodes, key=lambda item: item["episode_index"])
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    if not episodes:
        raise ValueError("no episodes selected")
    expected_indices = list(range(len(episodes)))
    actual_indices = [int(ep["episode_index"]) for ep in episodes]
    if actual_indices != expected_indices:
        raise ValueError("selected episodes must be contiguous from episode 0")

    source_stats = {
        int(item["episode_index"]): item["stats"]
        for item in _read_jsonl(source_dir / "meta" / "episodes_stats.jsonl")
    }
    video_keys = [
        key for key, value in source_info["features"].items() if value["dtype"] == "video"
    ]
    standard_keys = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
    audit_records = []
    stats_records = []
    maxima: dict[str, float] = {}

    for ep in tqdm(episodes, desc="transform parquets", unit="ep", dynamic_ncols=True):
        episode_index = int(ep["episode_index"])
        source_path = _episode_path(source_dir, episode_index)
        frame = pq.read_table(source_path).to_pandas()
        ref_state = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
        ref_action = np.stack(frame["action"].to_numpy()).astype(np.float32)
        arrays, validation = transform_episode(ref_state, ref_action)

        if not np.array_equal(arrays["raw_state.ref_state"], ref_state):
            raise AssertionError(f"episode {episode_index}: raw_state.ref_state changed")
        if not np.array_equal(arrays["raw_action.ref_action"], ref_action):
            raise AssertionError(f"episode {episode_index}: raw_action.ref_action changed")
        for key, value in validation.items():
            maxima[key] = max(maxima.get(key, 0.0), float(value))

        candidates = audit_candidate_noops(ref_action)
        audit_records.append(
            {
                "episode_index": episode_index,
                "total_frames": int(len(ref_action)),
                "candidate_noop_frames": int(candidates.sum()),
                "candidate_noop_proportion": float(candidates.mean()),
                "filtered_frames": 0,
            }
        )

        # LeRobot represents shape-(1,) numeric features as scalar parquet columns;
        # wider vectors are list columns. Match that storage convention so the
        # generated v3 dataset loads through ``LeRobotDataset``.
        output = {
            key: (value[:, 0] if value.ndim == 2 and value.shape[1] == 1 else list(value))
            for key, value in arrays.items()
        }
        for key in standard_keys:
            output[key] = frame[key].to_numpy()
        output_frame = pd.DataFrame(output)
        output_path = _episode_path(temp_dir, episode_index)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_frame.to_parquet(output_path, index=False)

        old_stats = source_stats[episode_index]
        ep_stats = {key: old_stats[key] for key in video_keys}
        ep_stats.update({key: _stats(value) for key, value in arrays.items()})
        ep_stats.update({key: _stats(frame[key].to_numpy()) for key in standard_keys})
        stats_records.append({"episode_index": episode_index, "stats": ep_stats})

        for video_key in video_keys:
            source_video = _video_path(source_dir, video_key, episode_index)
            if not source_video.is_file():
                raise FileNotFoundError(source_video)
            temp_video = _video_path(temp_dir, video_key, episode_index)
            temp_video.parent.mkdir(parents=True, exist_ok=True)
            temp_video.symlink_to(source_video)

    total_frames = sum(int(ep["length"]) for ep in episodes)
    info = dict(source_info)
    info.update(
        {
            "total_episodes": len(episodes),
            "total_frames": total_frames,
            "total_videos": len(episodes) * len(video_keys),
            "total_chunks": math.ceil(len(episodes) / int(source_info["chunks_size"])),
            "splits": {"train": f"0:{len(episodes)}"},
            "features": detailed_features(source_info["features"]),
        }
    )
    _write_json(temp_dir / "meta" / "info.json", info)
    _write_jsonl(temp_dir / "meta" / "episodes.jsonl", episodes)
    _write_jsonl(temp_dir / "meta" / "episodes_stats.jsonl", stats_records)
    shutil.copy2(source_dir / "meta" / "tasks.jsonl", temp_dir / "meta" / "tasks.jsonl")
    return audit_records, maxima


def run_v3_packer(temp_dir: Path, output_dir: Path, data_mb: int, video_mb: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    convert_info(temp_dir, output_dir, data_mb, video_mb)
    convert_tasks(temp_dir, output_dir)
    episodes_metadata = convert_data(temp_dir, output_dir, data_mb)
    videos_metadata = convert_videos(temp_dir, output_dir, video_mb)
    convert_episodes_metadata(temp_dir, output_dir, episodes_metadata, videos_metadata)


def summarize_audit(records: list[dict]) -> dict:
    total = sum(record["total_frames"] for record in records)
    candidates = sum(record["candidate_noop_frames"] for record in records)
    return {
        "definition": {
            "arm_norm_threshold": NOOP_THRESHOLD,
            "requires_unchanged_gripper": True,
            "previous_action_is_previous_would_be_retained_action": True,
        },
        "policy": "audit_only_no_frames_filtered",
        "total_episodes": len(records),
        "total_frames": total,
        "candidate_noop_frames": candidates,
        "candidate_noop_proportion": candidates / total,
        "filtered_frames": 0,
        "filtered_proportion": 0.0,
    }


def self_test() -> None:
    state = np.array(
        [[0.1, -0.2, 0.5, 0.0, 0.0, 0.0, 0.04, -0.04],
         [0.11, -0.2, 0.5, 0.0, 0.0, 0.1, 0.02, -0.02]],
        dtype=np.float32,
    )
    action = np.array(
        [[0.2, 0.0, 0.0, 0.0, 0.0, 0.2, -1.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    arrays, validation = transform_episode(state, action)
    np.testing.assert_array_equal(arrays["raw_state.ref_state"], state)
    np.testing.assert_array_equal(arrays["raw_action.ref_action"], action)
    np.testing.assert_allclose(arrays["raw_target.eef_xyz"][0], [0.11, -0.2, 0.5])
    np.testing.assert_allclose(arrays["state.gripper_state"], [[1.0, 1.0], [0.5, 0.5]])
    np.testing.assert_allclose(arrays["target.gripper_state"], [[1.0], [0.0]])
    np.testing.assert_allclose(arrays["debug.gripper_eef_rot6d"][-1], IDENTITY_ROT6D[0])
    assert max(validation.values()) < 2e-5, validation
    noop_action = np.array(
        [[0, 0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(audit_candidate_noops(noop_action), [True, True, True])
    print("self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="One LIBERO Plus LeRobot v2.1 partition.")
    parser.add_argument("--output-dir", type=Path, help="Exact LeRobot v3 output directory.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Pilot/smoke subset from episode 0.")
    parser.add_argument("--data-file-size-mb", type=int, default=100)
    parser.add_argument("--video-file-size-mb", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-temp", action="store_true", help="Keep the temporary detailed-v2.1 tree.")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.source_dir is None or args.output_dir is None:
        raise SystemExit("--source-dir and --output-dir are required unless --self-test is used")

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    temp_dir = output_dir.parent / f".{output_dir.name}.detailed-v21-tmp"
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {output_dir}; pass --overwrite to rebuild")
        shutil.rmtree(output_dir)
    if temp_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"temporary directory exists: {temp_dir}; pass --overwrite to rebuild")
        shutil.rmtree(temp_dir)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(f"source: {source_dir}")
    print(f"output: {output_dir}")
    print("no-op policy: audit only; no frames will be filtered")

    try:
        audit_records, validation = prepare_detailed_v21(source_dir, temp_dir, args.max_episodes)
        run_v3_packer(
            temp_dir, output_dir, args.data_file_size_mb, args.video_file_size_mb
        )
        quantile_config = add_global_quantiles(output_dir)
        audit_summary = summarize_audit(audit_records)
        _write_json(output_dir / "meta" / "noop_audit.json", audit_summary)
        _write_jsonl(output_dir / "meta" / "noop_audit_episodes.jsonl", audit_records)
        _write_json(output_dir / "meta" / "conversion_validation.json", validation)
        _write_json(
            output_dir / "meta" / "conversion_config.json",
            {
                "source_dir": str(source_dir),
                "source_version": "v2.1",
                "output_version": "v3.0",
                "fps": FPS,
                "position_action_scale": POSITION_SCALE,
                "rotation_action_scale": ROTATION_SCALE,
                "rotation_target_composition": "R_target = R_delta @ R_state",
                "world_alignment": R_WORLD_ALIGN.tolist(),
                "gripper_alignment": R_GRIPPER_ALIGN.tolist(),
                "finger_limit_m": FINGER_LIMIT_M,
                "frames_filtered": 0,
                "camera_videos_modified": False,
                "canonical_training_camera_transform": "horizontal_flip_both_views_at_load_time",
                "stats_quantiles": quantile_config,
            },
        )
        write_dataset_card(output_dir)
        print(json.dumps(audit_summary, indent=2))
        print(json.dumps(validation, indent=2))
    except Exception:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        raise
    finally:
        if temp_dir.exists() and not args.keep_temp:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
