"""Validate SerialChainFK against the released HF ee_positions ground truth.

The crawled .rrd has no EEF pose, so we recover it via FK from joint angles. This script
confirms the URDF chain + joint order reproduce the HF `ee_positions` (xyz + quat xyzw, in
the arm base frame) for each robot, trying candidate tip links and base links.
"""
import glob
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
from alignment import transforms_numpy as tn  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rc_fk import SerialChainFK  # noqa: E402

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
# Directory holding extracted HF episodes as ex_<robot>/*/data/episode_*/states/*states.jsonl
# (each states record has joint_positions + ee_positions). Override via RC_HF_VAL_DIR.
VAL = os.environ.get("RC_HF_VAL_DIR", "./hf_val")

# robot -> (urdf, base candidates, tip candidates, extracted-states glob, states filename)
CFG = {
    "arx5": ("arx5.urdf", ["base_link"], ["link6", "eef_link"], "ex_arx5", "states.jsonl"),
    "ur5": ("ur5.urdf", ["base_link", "base"], ["wrist_3_link", "flange", "tool0"], "ex_ur5", "states.jsonl"),
    "dosw1": ("dos_w1.urdf", ["base_link"], ["link6", "end_link", "eef_connect_base_link"], "ex_dosw1", "left_states.jsonl"),
    "aloha": ("../../vifailback2lerobot/assets/piper_description.urdf", ["base_link"], ["link6", "gripper_base"], "ex_aloha", "left_states.jsonl"),
}


def load_states(robot):
    _, _, _, exdir, fname = CFG[robot]
    files = sorted(glob.glob(f"{VAL}/{exdir}/*/data/episode_*/states/{fname}"))
    joints, ee = [], []
    for sf in files[:3]:
        with open(sf) as f:
            for line in f:
                r = json.loads(line)
                joints.append(r["joint_positions"])
                ee.append(r["ee_positions"])
    return np.array(joints, dtype=np.float64), np.array(ee, dtype=np.float64)


def rot_angle_deg(R_a, quat_xyzw):
    R_b = tn.quaternion_to_matrix(quat_xyzw)
    Rrel = np.matmul(np.swapaxes(R_a, -1, -2), R_b)
    tr = np.clip((Rrel[..., 0, 0] + Rrel[..., 1, 1] + Rrel[..., 2, 2] - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


def main():
    for robot, (urdf, bases, tips, _, _) in CFG.items():
        path = os.path.join(ASSETS, urdf)
        if not os.path.exists(path):
            print(f"[{robot}] URDF missing: {path}")
            continue
        joints, ee = load_states(robot)
        if len(joints) == 0:
            print(f"[{robot}] no states extracted")
            continue
        p_gt, q_gt = ee[:, :3], ee[:, 3:7]  # xyzw
        print(f"\n==== {robot}  ({len(joints)} samples, urdf={urdf}) ====")
        best = None
        for base in bases:
            for tip in tips:
                try:
                    fk = SerialChainFK(path, base_link=base, tip_link=tip)
                    if fk.dof != 6:
                        print(f"   base={base} tip={tip}: dof={fk.dof} (skip)")
                        continue
                    R, p = fk(joints)
                    dpos = np.linalg.norm(p - p_gt, axis=1)
                    drot = rot_angle_deg(R, q_gt)
                    line = (f"   base={base:10s} tip={tip:22s} "
                            f"pos_err[mm] mean={dpos.mean()*1e3:8.2f} p95={np.percentile(dpos,95)*1e3:8.2f} | "
                            f"rot_err[deg] mean={drot.mean():7.3f} p95={np.percentile(drot,95):7.3f}")
                    print(line)
                    score = dpos.mean() + np.radians(drot.mean())
                    if best is None or score < best[0]:
                        best = (score, base, tip, dpos.mean(), drot.mean())
                except Exception as e:
                    print(f"   base={base} tip={tip}: ERROR {type(e).__name__}: {e}")
        if best:
            print(f"   >>> BEST: base={best[1]} tip={best[2]}  pos={best[3]*1e3:.2f}mm rot={best[4]:.3f}deg")


if __name__ == "__main__":
    main()
