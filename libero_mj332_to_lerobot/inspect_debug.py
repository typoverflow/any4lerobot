#!/usr/bin/env python3
"""Inspect provenance and pose reconstruction in a converted MuJoCo 3.3.2 partition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alignment import transforms_numpy as tn  # noqa: E402

COLUMNS = [
    "frame_index",
    "raw_state.ref_state",
    "raw_state.joint_pos",
    "raw_action.ref_action",
    "state.eef_xyz",
    "state.eef_rot9d",
    "target.gripper_state",
    "debug.gripper_eef_xyz",
    "debug.gripper_eef_rot6d",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--rows", type=int, default=8)
    args = parser.parse_args()

    table = pads.dataset(args.dataset_dir / "data", format="parquet").to_table(
        columns=COLUMNS,
        filter=pads.field("episode_index") == args.episode_index,
    )
    if table.num_rows == 0:
        raise SystemExit(f"episode {args.episode_index} not found")
    values = {key: np.asarray(table[key].to_pylist()) for key in COLUMNS}
    rotations = values["state.eef_rot9d"].reshape(-1, 3, 3)
    positions = values["state.eef_xyz"]
    delta_rotations = tn.rotation_6d_to_matrix(values["debug.gripper_eef_rot6d"])
    delta_positions = values["debug.gripper_eef_xyz"]
    if len(rotations) > 1:
        rotation_reconstruction = rotations[:-1] @ delta_rotations[:-1]
        position_reconstruction = positions[:-1] + (
            rotations[:-1] @ delta_positions[:-1, :, None]
        )[..., 0]
        rotation_error = float(
            np.max(np.abs(rotation_reconstruction - rotations[1:]))
        )
        position_error = float(
            np.max(np.abs(position_reconstruction - positions[1:]))
        )
    else:
        rotation_error = position_error = 0.0
    report = {
        "episode_index": args.episode_index,
        "episode_length": len(rotations),
        "max_rotation_orthogonality_error": float(
            np.max(
                np.abs(
                    rotations.swapaxes(-1, -2) @ rotations - np.eye(3)
                )
            )
        ),
        "max_rotation_determinant_error": float(
            np.max(np.abs(np.linalg.det(rotations) - 1.0))
        ),
        "max_debug_rotation_reconstruction_error": rotation_error,
        "max_debug_xyz_reconstruction_error": position_error,
        "target_gripper_values": np.unique(values["target.gripper_state"]).tolist(),
        "final_debug_xyz": delta_positions[-1].tolist(),
        "final_debug_rot6d": values["debug.gripper_eef_rot6d"][-1].tolist(),
    }
    print(json.dumps(report, indent=2))
    stop = min(args.start + args.rows, len(rotations))
    for i in range(args.start, stop):
        print(
            json.dumps(
                {
                    "frame_index": int(values["frame_index"][i]),
                    "raw_state.ref_state": values["raw_state.ref_state"][i].tolist(),
                    "raw_state.joint_pos": values["raw_state.joint_pos"][i].tolist(),
                    "raw_action.ref_action": values["raw_action.ref_action"][i].tolist(),
                    "target.gripper_state": values["target.gripper_state"][i].tolist(),
                    "debug.gripper_eef_xyz": delta_positions[i].tolist(),
                    "debug.gripper_eef_rot6d": values["debug.gripper_eef_rot6d"][i].tolist(),
                }
            )
        )


if __name__ == "__main__":
    main()
