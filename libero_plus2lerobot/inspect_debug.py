#!/usr/bin/env python3
"""Inspect provenance and debug pose entries in a converted LIBERO Plus v3 dataset."""

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
    "raw_action.ref_action",
    "state.eef_xyz",
    "state.eef_rot9d",
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

    data = pads.dataset(args.dataset_dir / "data", format="parquet")
    table = data.to_table(
        columns=COLUMNS,
        filter=pads.field("episode_index") == args.episode_index,
    )
    if table.num_rows == 0:
        raise SystemExit(f"episode {args.episode_index} not found")
    values = {key: np.asarray(table[key].to_pylist()) for key in COLUMNS}
    R = values["state.eef_rot9d"].reshape(-1, 3, 3)
    p = values["state.eef_xyz"]
    dR = tn.rotation_6d_to_matrix(values["debug.gripper_eef_rot6d"])
    dp = values["debug.gripper_eef_xyz"]
    if len(R) > 1:
        R_recon = R[:-1] @ dR[:-1]
        p_recon = p[:-1] + (R[:-1] @ dp[:-1, :, None])[..., 0]
        rot_error = float(np.max(np.abs(R_recon - R[1:])))
        xyz_error = float(np.max(np.abs(p_recon - p[1:])))
    else:
        rot_error = xyz_error = 0.0
    eye = np.eye(3)
    report = {
        "episode_index": args.episode_index,
        "episode_length": len(R),
        "max_rotation_orthogonality_error": float(np.max(np.abs(R.swapaxes(-1, -2) @ R - eye))),
        "max_rotation_determinant_error": float(np.max(np.abs(np.linalg.det(R) - 1.0))),
        "max_debug_rotation_reconstruction_error": rot_error,
        "max_debug_xyz_reconstruction_error": xyz_error,
        "final_debug_xyz": dp[-1].tolist(),
        "final_debug_rot6d": values["debug.gripper_eef_rot6d"][-1].tolist(),
    }
    print(json.dumps(report, indent=2))
    stop = min(args.start + args.rows, len(R))
    for i in range(args.start, stop):
        print(
            json.dumps(
                {
                    "frame_index": int(values["frame_index"][i]),
                    "raw_state.ref_state": values["raw_state.ref_state"][i].tolist(),
                    "raw_action.ref_action": values["raw_action.ref_action"][i].tolist(),
                    "debug.gripper_eef_xyz": dp[i].tolist(),
                    "debug.gripper_eef_rot6d": values["debug.gripper_eef_rot6d"][i].tolist(),
                }
            )
        )


if __name__ == "__main__":
    main()
