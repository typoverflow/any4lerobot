"""Infer each robot's native gripper -> OpenCV canonical axis relabel from wrist-camera flow.

The canonical gripper frame (design doc) is OpenCV: z = approach (forward, out of the gripper),
x = right, y = down, "aligned with the [wrist] camera". A wrist camera is rigidly mounted to the
tip link, so the constant relabel R_{tip->cam} can be recovered by correlating the tip-frame EEF
translational velocity with the mean optical flow the wrist camera sees (a la vifailback):

    mean image flow (u, v) ~= -K * ( (R_rel @ v_tip)_x , (R_rel @ v_tip)_y )

Least-squares  u ~ a . v_tip ,  v ~ b . v_tip  gives, up to a positive scale K:
    opencv-x axis (in tip coords) = -a/|a| ,  opencv-y axis = -b/|b| ,  opencv-z = x cross y.
Snapping to the nearest signed permutation yields axis_alignment_matrix(x_to, y_to, z_to). The
wrist camera is auto-detected as the stream whose flow best fits (highest R^2).

Run:  python infer_gripper_axes.py --data-dir .../robochallenge/data --robot arx5 --n 4
"""
import argparse
import os
import sys

import cv2
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alignment import transforms_numpy as tn  # noqa: E402
import rc_rrd  # noqa: E402
import convert  # noqa: E402
from rc_fk import SerialChainFK  # noqa: E402

AXES = {"x": (1, 0, 0), "-x": (-1, 0, 0), "y": (0, 1, 0), "-y": (0, -1, 0), "z": (0, 0, 1), "-z": (0, 0, -1)}


def snap_axis(vec):
    """Nearest signed unit axis to ``vec`` -> (name, unit vector)."""
    best = max(AXES.items(), key=lambda kv: np.dot(vec, kv[1]))
    return best[0], np.array(best[1], dtype=float)


def flow_mean(prev_gray, gray):
    # Median over the FULL frame: the dominant motion is the (static) background parallax as the
    # wrist camera translates. The center holds the gripper / held object (moves WITH the camera ->
    # no parallax), so a central crop measures the wrong thing -- the median over everything is
    # robust to that foreground blob and to outliers.
    f = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    return float(np.median(f[..., 0])), float(np.median(f[..., 1]))


def probe_rollout(rrd_path, cfg, fk, probe_fps=10):
    import rerun.dataframe as rd

    rec = rd.load_recording(str(rrd_path))
    prefix = "left_arm" if cfg["dual"] else "arm"
    arm = rc_rrd.read_arm(rec, prefix)
    grid = rc_rrd.grid_times(arm["times"][0], arm["times"][-1], probe_fps)
    idx = rc_rrd.nearest_indices(arm["times"], grid)
    R, p = fk(arm["joints"][idx])                       # tip pose in base frame
    v_world = np.diff(p, axis=0)
    v_tip = np.einsum("tij,tj->ti", np.swapaxes(R[:-1], 1, 2), v_world)  # (T-1, 3)
    # Per-step tip rotation angle (rad): high rotation induces depth-independent flow that
    # corrupts the translation->flow correlation, so we filter those frames out in fit().
    Rrel = np.matmul(np.swapaxes(R[:-1], 1, 2), R[1:])
    tr = np.clip((Rrel[:, 0, 0] + Rrel[:, 1, 1] + Rrel[:, 2, 2] - 1) / 2, -1, 1)
    rot = np.arccos(tr)

    out = {}
    for entity in rc_rrd.camera_entities(rec, cfg["dual"]):
        ns = rc_rrd.NearestFrameStream(rc_rrd.decode_video_frames(rec, entity), (480, 640, 3))
        grays = []
        for t in grid:
            fr = ns.at(t)
            g = cv2.cvtColor(cv2.resize(fr, (160, 120)), cv2.COLOR_RGB2GRAY)
            grays.append(g)
        uv = np.array([flow_mean(grays[i], grays[i + 1]) for i in range(len(grays) - 1)])
        out[entity] = (v_tip, uv, rot)
    return out


def fit(v_tip, uv, rot):
    """Return (a, b, r2) for u ~ a.v_tip, v ~ b.v_tip over translation-dominant frames."""
    speed = np.linalg.norm(v_tip, axis=1)
    # Keep frames with clear translation and little rotation (pure-translation parallax).
    m = (speed > np.percentile(speed, 60)) & (rot < np.percentile(rot, 40))
    if m.sum() < 20:
        m = speed > np.percentile(speed, 60)
    X, U, V = v_tip[m], uv[m, 0], uv[m, 1]
    a, *_ = np.linalg.lstsq(X, U, rcond=None)
    b, *_ = np.linalg.lstsq(X, V, rcond=None)
    r2 = 0.0
    for Y, c in ((U, a), (V, b)):
        ss = np.sum((Y - Y.mean()) ** 2)
        if ss > 0:
            r2 += 1 - np.sum((Y - X @ c) ** 2) / ss
    return a, b, r2 / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--robot", required=True, choices=list(convert.ROBOTS))
    ap.add_argument("--n", type=int, default=4, help="rollouts to probe")
    args = ap.parse_args()
    from pathlib import Path

    cfg = convert.ROBOTS[args.robot]
    urdf = os.path.join(convert._HERE, cfg["urdf"]) if not os.path.isabs(cfg["urdf"]) else cfg["urdf"]
    fk = SerialChainFK(urdf, cfg["base"], cfg["tip"], cfg["tool_offset"])
    jobs = convert.discover_rollouts(Path(args.data_dir), args.robot)
    # spread picks across the list
    picks = jobs[:: max(1, len(jobs) // args.n)][: args.n]

    per_cam = {}
    for j in picks:
        try:
            for entity, triple in probe_rollout(j["rrd"], cfg, fk).items():
                per_cam.setdefault(entity, []).append(triple)
        except Exception as e:
            print(f"  skip {j['rrd'].name}: {type(e).__name__}: {e}")

    print(f"\n==== {args.robot} ====")
    best = None
    for entity, chunks in per_cam.items():
        v = np.concatenate([c[0] for c in chunks])
        uv = np.concatenate([c[1] for c in chunks])
        rot = np.concatenate([c[2] for c in chunks])
        a, b, r2 = fit(v, uv, rot)
        ax, ux = snap_axis(-a / (np.linalg.norm(a) + 1e-9))
        ay, uy = snap_axis(-b / (np.linalg.norm(b) + 1e-9))
        print(f"  {entity}: R2={r2:.3f}  opencv-x~{ax:>2}  opencv-y~{ay:>2}  |a|={np.linalg.norm(a):.1f} |b|={np.linalg.norm(b):.1f}")
        if best is None or r2 > best[0]:
            best = (r2, entity, ux, uy, ax, ay)
    r2, entity, ux, uy, ax, ay = best
    uz = np.cross(ux, uy)
    zname, uz_snap = snap_axis(uz / (np.linalg.norm(uz) + 1e-9))
    # Build R_{native->opencv}: rows = opencv axes in native coords.
    R = np.stack([ux, uy, uz_snap])
    det = np.linalg.det(R)
    # axis_alignment_matrix args = where native x,y,z point in opencv = columns of R.
    col_names = []
    for jcol in range(3):
        nm, _ = snap_axis(R[:, jcol])
        col_names.append(nm)
    print(f"\n  >>> wrist cam = {entity} (R2={r2:.3f})")
    print(f"      opencv axes in native: x={ax}, y={ay}, z={zname}  (det R={det:+.0f})")
    print(f"      gripper_align = {tuple(col_names)}   # axis_alignment_matrix(*this) = R_native->opencv")
    if abs(det - 1) > 1e-6:
        print("      WARNING: not a proper rotation (det != +1); inference ambiguous -> keep native.")


if __name__ == "__main__":
    main()
