#!/usr/bin/env python3
"""Convert a LIBERO MuJoCo 3.3.2 LeRobot v2.1 partition to detailed v3.0.

The output follows ``failure_rollout_data/dataset.md`` and preserves every
source frame. The source release has already replayed demonstrations, removed
historical no-op actions, retained successful episodes, and mapped the gripper
command to binary ``0=closed, 1=open``.

Camera videos are repacked without decoding or re-encoding. Both views retain
the source pipeline's horizontal mirror and must be flipped at load time for
the repository's canonical training camera convention.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libero_plus2lerobot import convert as base  # noqa: E402

FPS = 20
MUJOCO_VERSION = "3.3.2"
PARTITIONS = (
    "libero_10_no_noops_lerobot",
    "libero_goal_no_noops_lerobot",
    "libero_object_no_noops_lerobot",
    "libero_spatial_no_noops_lerobot",
)

JOINT_NAMES = [f"joint_{i}" for i in range(7)]


def dataset_card(partition: str) -> str:
    return f"""---
license: cc-by-4.0
tags:
- lerobot
- robotics
- libero
- mujoco
pretty_name: LIBERO MuJoCo 3.3.2 {partition} (detailed LeRobot v3.0)
---

# {partition}: detailed LeRobot v3.0

This dataset was converted from the Fast-WAM LIBERO MuJoCo 3.3.2 LeRobot v2.1
`{partition}` partition. It contains successful demonstrations whose historical no-op actions
were removed by simulator replay before this conversion. This converter preserves all remaining
frames and does not apply any additional filtering.

The original 8D state and 7D action vectors are preserved exactly as
`raw_state.ref_state` and `raw_action.ref_action`. Canonical low-dimensional fields follow
`failure_rollout_data/dataset.md`; `debug.gripper_eef_*` contains ground-truth next-step relative
EEF motion for inspection. Source joint positions are exposed as `raw_state.joint_pos` and
`state.joint_pos`.

## Required camera transform for canonical training

The source `observation.images.image` and `observation.images.wrist_image` videos are preserved
unchanged. **For canonical training, horizontally flip both camera views at load time.** The
videos are deliberately not rewritten or re-encoded.

The source replay pipeline vertically flips the raw robosuite render and then rotates it by 180
degrees. Those vertical components cancel, leaving the stored image horizontally mirrored.

```python
# NumPy (..., height, width, channels)
image = np.flip(image, axis=-2)

# PyTorch (..., channels, height, width)
image = torch.flip(image, dims=(-1,))
```

Apply this only to camera pixels. Do not flip or negate any low-dimensional field.

## Conversion notes

- The source action gripper is already binary: `0=closed`, `1=open`.
- No frames were filtered during this conversion; see `meta/noop_audit.json`.
- Rotation and reconstruction checks are in `meta/conversion_validation.json`.
- Controller, alignment, filtering, camera, and source assumptions are in
  `meta/conversion_config.json`.
- Every numeric and video statistic includes `q01` and `q99`.
"""


def detailed_features(source_features: dict) -> dict:
    """Return feature metadata for the temporary detailed v2.1 dataset."""
    features = {
        key: value for key, value in source_features.items() if value["dtype"] == "video"
    }
    features.update(
        {
            "raw_state.ref_state": base._feature("float32", (8,), base.REF_STATE_NAMES),
            "raw_state.joint_pos": base._feature("float32", (7,), JOINT_NAMES),
            "raw_state.eef_xyz": base._feature("float32", (3,), base.XYZ_NAMES),
            "raw_state.eef_axis_angle": base._feature(
                "float32", (3,), base.AXIS_ANGLE_NAMES
            ),
            "raw_state.gripper_state": base._feature(
                "float32", (2,), base.FINGER_NAMES
            ),
            "raw_action.ref_action": base._feature(
                "float32", (7,), base.REF_ACTION_NAMES
            ),
            "raw_target.eef_xyz": base._feature("float32", (3,), base.XYZ_NAMES),
            "raw_target.eef_axis_angle": base._feature(
                "float32", (3,), base.AXIS_ANGLE_NAMES
            ),
            "raw_target.gripper_state": base._feature("float32", (1,), ["gripper"]),
            "state.joint_pos": base._feature("float32", (7,), JOINT_NAMES),
            "state.eef_xyz": base._feature("float32", (3,), base.XYZ_NAMES),
            "state.eef_rot9d": base._feature("float32", (9,), base.ROT9D_NAMES),
            "state.gripper_state": base._feature("float32", (2,), base.FINGER_NAMES),
            "target.eef_xyz": base._feature("float32", (3,), base.XYZ_NAMES),
            "target.eef_rot9d": base._feature("float32", (9,), base.ROT9D_NAMES),
            "target.gripper_state": base._feature("float32", (1,), ["gripper"]),
            "debug.gripper_eef_xyz": base._feature("float32", (3,), base.XYZ_NAMES),
            "debug.gripper_eef_rot6d": base._feature(
                "float32", (6,), base.ROT6D_NAMES
            ),
        }
    )
    for key, value in source_features.items():
        if key not in {
            "observation.state",
            "observation.states.ee_state",
            "observation.states.joint_state",
            "observation.states.gripper_state",
            "action",
        } and value["dtype"] != "video":
            features[key] = value
    return features


def transform_episode(
    ref_state: np.ndarray,
    ref_action: np.ndarray,
    ee_state: np.ndarray,
    joint_pos: np.ndarray,
    gripper_state: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    """Split native values and construct canonical state and absolute targets."""
    ref_state = np.asarray(ref_state, dtype=np.float32)
    ref_action = np.asarray(ref_action, dtype=np.float32)
    ee_state = np.asarray(ee_state, dtype=np.float32)
    joint_pos = np.asarray(joint_pos, dtype=np.float32)
    gripper_state = np.asarray(gripper_state, dtype=np.float32)
    length = len(ref_state)
    if ee_state.shape != (length, 6):
        raise ValueError(f"observation.states.ee_state must be [T,6], got {ee_state.shape}")
    if joint_pos.shape != (length, 7):
        raise ValueError(
            f"observation.states.joint_state must be [T,7], got {joint_pos.shape}"
        )
    if gripper_state.shape != (length, 2):
        raise ValueError(
            f"observation.states.gripper_state must be [T,2], got {gripper_state.shape}"
        )
    np.testing.assert_array_equal(ref_state[:, :6], ee_state)
    np.testing.assert_array_equal(ref_state[:, 6:], gripper_state)
    if not np.all(np.isin(ref_action[:, -1], (0.0, 1.0))):
        values = np.unique(ref_action[:, -1]).tolist()
        raise ValueError(f"expected binary 0=closed, 1=open gripper actions, got {values}")

    arrays, validation = base.transform_episode(ref_state, ref_action)
    arrays["raw_state.joint_pos"] = joint_pos.copy()
    arrays["state.joint_pos"] = joint_pos.copy()
    # Unlike LIBERO Plus, this source has already remapped its gripper action
    # from native {-1=open, +1=closed} to canonical {0=closed, 1=open}.
    arrays["target.gripper_state"] = ref_action[:, 6:7].copy()
    validation["max_duplicate_eef_state_error"] = float(
        np.max(np.abs(ref_state[:, :6] - ee_state))
    )
    validation["max_duplicate_gripper_state_error"] = float(
        np.max(np.abs(ref_state[:, 6:] - gripper_state))
    )
    return arrays, validation


def prepare_detailed_v21(
    source_dir: Path,
    temp_dir: Path,
    max_episodes: int | None,
) -> tuple[list[dict], dict]:
    """Create a temporary detailed v2.1 tree for the repository's v3 packer."""
    source_info = base._read_json(source_dir / "meta" / "info.json")
    if source_info.get("codebase_version") != "v2.1":
        raise ValueError(f"expected v2.1 source, got {source_info.get('codebase_version')!r}")
    if int(source_info["fps"]) != FPS:
        raise ValueError(f"expected {FPS} fps, got {source_info['fps']}")

    episodes = sorted(
        base._read_jsonl(source_dir / "meta" / "episodes.jsonl"),
        key=lambda item: item["episode_index"],
    )
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
        for item in base._read_jsonl(source_dir / "meta" / "episodes_stats.jsonl")
    }
    video_keys = [
        key for key, value in source_info["features"].items() if value["dtype"] == "video"
    ]
    standard_keys = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
    audit_records: list[dict] = []
    stats_records: list[dict] = []
    maxima: dict[str, float] = {}

    for ep in tqdm(episodes, desc="transform parquets", unit="ep", dynamic_ncols=True):
        episode_index = int(ep["episode_index"])
        source_path = base._episode_path(source_dir, episode_index)
        frame = pq.read_table(source_path).to_pandas()
        ref_state = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
        ref_action = np.stack(frame["action"].to_numpy()).astype(np.float32)
        ee_state = np.stack(frame["observation.states.ee_state"].to_numpy()).astype(np.float32)
        joint_pos = np.stack(frame["observation.states.joint_state"].to_numpy()).astype(
            np.float32
        )
        gripper_state = np.stack(
            frame["observation.states.gripper_state"].to_numpy()
        ).astype(np.float32)
        arrays, validation = transform_episode(
            ref_state, ref_action, ee_state, joint_pos, gripper_state
        )

        np.testing.assert_array_equal(arrays["raw_state.ref_state"], ref_state)
        np.testing.assert_array_equal(arrays["raw_action.ref_action"], ref_action)
        np.testing.assert_array_equal(arrays["raw_state.joint_pos"], joint_pos)
        for key, value in validation.items():
            maxima[key] = max(maxima.get(key, 0.0), float(value))

        candidates = base.audit_candidate_noops(ref_action)
        audit_records.append(
            {
                "episode_index": episode_index,
                "total_frames": int(len(ref_action)),
                "candidate_noop_frames": int(candidates.sum()),
                "candidate_noop_proportion": float(candidates.mean()),
                "filtered_frames_by_this_conversion": 0,
            }
        )

        output = {
            key: (value[:, 0] if value.ndim == 2 and value.shape[1] == 1 else list(value))
            for key, value in arrays.items()
        }
        for key in standard_keys:
            output[key] = frame[key].to_numpy()
        output_path = base._episode_path(temp_dir, episode_index)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(output).to_parquet(output_path, index=False)

        old_stats = source_stats[episode_index]
        ep_stats = {key: old_stats[key] for key in video_keys}
        ep_stats.update({key: base._stats(value) for key, value in arrays.items()})
        ep_stats.update({key: base._stats(frame[key].to_numpy()) for key in standard_keys})
        stats_records.append({"episode_index": episode_index, "stats": ep_stats})

        for video_key in video_keys:
            source_video = base._video_path(source_dir, video_key, episode_index)
            if not source_video.is_file():
                raise FileNotFoundError(source_video)
            temp_video = base._video_path(temp_dir, video_key, episode_index)
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
    base._write_json(temp_dir / "meta" / "info.json", info)
    base._write_jsonl(temp_dir / "meta" / "episodes.jsonl", episodes)
    base._write_jsonl(temp_dir / "meta" / "episodes_stats.jsonl", stats_records)
    shutil.copy2(source_dir / "meta" / "tasks.jsonl", temp_dir / "meta" / "tasks.jsonl")
    return audit_records, maxima


def summarize_audit(records: list[dict]) -> dict:
    total = sum(record["total_frames"] for record in records)
    candidates = sum(record["candidate_noop_frames"] for record in records)
    return {
        "source_preprocessing": "historical_noop_filter_applied_during_simulator_replay",
        "definition": {
            "arm_norm_threshold": base.NOOP_THRESHOLD,
            "requires_unchanged_gripper": True,
            "previous_action_is_previous_retained_action": True,
        },
        "policy": "audit_only_no_additional_frames_filtered",
        "total_episodes": len(records),
        "total_frames": total,
        "candidate_noop_frames_remaining": candidates,
        "candidate_noop_proportion_remaining": candidates / total,
        "frames_filtered_by_this_conversion": 0,
        "filtered_proportion_by_this_conversion": 0.0,
    }


def self_test() -> None:
    state = np.array(
        [
            [0.1, -0.2, 0.5, 0.0, 0.0, 0.0, 0.04, -0.04],
            [0.11, -0.2, 0.5, 0.0, 0.0, 0.1, 0.02, -0.02],
        ],
        dtype=np.float32,
    )
    action = np.array(
        [[0.2, 0, 0, 0, 0, 0.2, 1.0], [0, 0, 0, 0, 0, 0, 0.0]],
        dtype=np.float32,
    )
    joint_pos = np.arange(14, dtype=np.float32).reshape(2, 7) / 10
    arrays, validation = transform_episode(
        state, action, state[:, :6], joint_pos, state[:, 6:]
    )
    np.testing.assert_array_equal(arrays["raw_state.ref_state"], state)
    np.testing.assert_array_equal(arrays["raw_action.ref_action"], action)
    np.testing.assert_array_equal(arrays["raw_state.joint_pos"], joint_pos)
    np.testing.assert_array_equal(arrays["state.joint_pos"], joint_pos)
    np.testing.assert_array_equal(arrays["target.gripper_state"], [[1.0], [0.0]])
    np.testing.assert_allclose(arrays["raw_target.eef_xyz"][0], [0.11, -0.2, 0.5])
    assert max(validation.values()) < 2e-5, validation
    print("self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="One MuJoCo 3.3.2 v2.1 partition.")
    parser.add_argument("--output-dir", type=Path, help="Exact LeRobot v3 output directory.")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--data-file-size-mb", type=int, default=100)
    parser.add_argument("--video-file-size-mb", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
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
    if source_dir.name not in PARTITIONS:
        raise ValueError(f"unknown partition directory: {source_dir.name}")
    temp_dir = output_dir.parent / f".{output_dir.name}.detailed-v21-tmp"
    for path, label in ((output_dir, "output"), (temp_dir, "temporary")):
        if path.exists():
            if not args.overwrite:
                raise FileExistsError(f"{label} directory exists: {path}; pass --overwrite")
            shutil.rmtree(path)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(f"source: {source_dir}")
    print(f"output: {output_dir}")
    print("no-op policy: source prefiltered; audit only; no additional filtering")

    try:
        audit_records, validation = prepare_detailed_v21(
            source_dir, temp_dir, args.max_episodes
        )
        base.run_v3_packer(
            temp_dir, output_dir, args.data_file_size_mb, args.video_file_size_mb
        )
        quantile_config = base.add_global_quantiles(output_dir)
        audit_summary = summarize_audit(audit_records)
        base._write_json(output_dir / "meta" / "noop_audit.json", audit_summary)
        base._write_jsonl(
            output_dir / "meta" / "noop_audit_episodes.jsonl", audit_records
        )
        base._write_json(
            output_dir / "meta" / "conversion_validation.json", validation
        )
        base._write_json(
            output_dir / "meta" / "conversion_config.json",
            {
                "source_dir": str(source_dir),
                "source_release": "Fast-WAM LIBERO MuJoCo 3.3.2",
                "source_version": "v2.1",
                "source_license": "CC BY 4.0",
                "output_version": "v3.0",
                "mujoco_version": MUJOCO_VERSION,
                "fps": FPS,
                "controller_mode": "robosuite OSC_POSE delta EEF control",
                "position_action_scale": base.POSITION_SCALE,
                "position_target_composition": "p_target = p_state + scale * action_xyz",
                "rotation_action_scale": base.ROTATION_SCALE,
                "rotation_target_composition": "R_target = R_delta @ R_state",
                "world_alignment": base.R_WORLD_ALIGN.tolist(),
                "gripper_alignment": base.R_GRIPPER_ALIGN.tolist(),
                "finger_limit_m": base.FINGER_LIMIT_M,
                "state_gripper_normalization": {
                    "finger_0": "clip(raw_finger_0 / 0.04, 0, 1)",
                    "finger_1": "clip(-raw_finger_1 / 0.04, 0, 1)",
                },
                "target_gripper_source_range": [0, 1],
                "target_gripper_semantics": "0=closed, 1=open; preserved without remapping",
                "source_preprocessing": {
                    "successful_episodes_only": True,
                    "historical_noop_filter_applied_during_simulator_replay": True,
                },
                "frames_filtered_by_this_conversion": 0,
                "camera_videos_modified": False,
                "canonical_training_camera_transform": "horizontal_flip_both_views_at_load_time",
                "stats_quantiles": quantile_config,
            },
        )
        (output_dir / "README.md").write_text(dataset_card(output_dir.name))
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
