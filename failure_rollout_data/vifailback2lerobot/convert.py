#!/usr/bin/env python
"""Convert the ViFailback dataset (dual-arm ALOHA / AgileX Cobot-Magic, HDF5) to LeRobot v3.

Raw layout: ``<raw_dir>/<task_name>/episode_<n>.hdf5`` with, per episode:
    observations/qpos|qvel|effort  (T, 14)  left joints [0:6] + left gripper [6],
                                            right joints [7:13] + right gripper [13]
    action                         (T, 14)  target joint positions (~= qpos at t+1)
    action_eef                     (T, 16)  per arm [x, y, z, qx, qy, qz, qw, gripper]
                                            (quaternion is xyzw), target EEF pose in
                                            each arm's own base frame
    observations/images/*          JPEG bytes, decode to 480x640x3 RGB (stored in RGB
                                            order -- cv2.imdecode output needs NO BGR swap)
    observations/images_depth/*    PNG bytes, decode to 480x640 uint16
    base_action                    (T, 2)   mobile base (linear, angular) velocity command

The arms are AgileX PiPER. ``assets/piper_description.urdf`` (vendored from
agilexrobotics/piper_ros @ humble, src/piper_description/urdf/piper_description.urdf)
drives a minimal numpy forward kinematics over the base_link -> link6 chain, which
reproduces the shipped poses: FK(qpos[t+1]) == action_eef[t] to ~0.1-0.4 mm / <0.1 deg on
moving arms. So the observed EEF *state* is FK(qpos[t]) and the FK-solved EEF *target* is
FK(action[t]) (``action`` = the real target-joint-position command). ``action_eef`` is a
relabeled achieved-next pose used here ONLY to validate/QC the FK solution. Known data quirk:
in some episodes an *idle* arm carries stale qpos/action inconsistent with action_eef
(constant large residual); such episodes are dropped and recorded in meta/qc_warnings.jsonl.

Output follows ../dataset.md: native (untransformed) ``raw_state.*`` / ``raw_target.*`` and
canonically axis-aligned ``state.*`` / ``target.*`` are split; the per-step ACTION is NOT
stored -- it is derived at load time from state.*/target.* (dataset.md sec 3). Dual-arm ->
``left_``/``right_`` prefix; base -> ``base_`` prefix. side in {left, right}:
    observation.images.cam_high|cam_left_wrist|cam_right_wrist        video 480x640x3
    observation.images.*_depth (only with --save-depth)               image 480x640x1 uint16
    raw_state.{side}_joint_pos (6)     qpos joints
    raw_state.{side}_joint_vel (6)     qvel joints (gripper-vel slot dropped)
    raw_state.{side}_eef_xyz (3)       FK(qpos) translation, native base frame
    raw_state.{side}_eef_quat (4)      FK(qpos) link6 orientation, quat xyzw (native frame)
    raw_state.{side}_gripper_state (1) qpos gripper, raw width in meters; larger=more open
    raw_target.{side}_joint_pos (6)    `action` target joints (absolute)
    raw_target.{side}_eef_xyz (3)      FK(action) translation, native base frame
    raw_target.{side}_eef_quat (4)     FK(action) link6 orientation, quat xyzw (native frame)
    raw_target.{side}_gripper_state (1) `action` gripper, raw target width in meters
    raw_target.base_vel (2)            base_action, (linear, angular) velocity command
    state.{side}_joint_pos (6)         qpos joints (frame-independent, copied)
    state.{side}_joint_vel (6)         qvel joints (frame-independent, copied)
    state.{side}_eef_xyz (3)           canonical FK(qpos) translation (world align = identity)
    state.{side}_eef_rot9d (9)         canonical FK(qpos) rotation (gripper -> OpenCV), full row-major matrix
    state.{side}_gripper_state (1)     gripper_width / GRIPPER_MAX, clipped [0, 1], 1=open
    target.{side}_joint_pos (6)        `action` target joints (frame-independent, copied)
    target.{side}_eef_xyz (3)          canonical FK(action) translation (world align = identity)
    target.{side}_eef_rot9d (9)        canonical FK(action) rotation (gripper -> OpenCV), full row-major matrix
    target.{side}_gripper_state (1)    `action` gripper_width / GRIPPER_MAX, clipped [0, 1], 1=open
    target.base_vel (2)                base_action, copied (velocity control -> used directly)
    debug.{side}_gripper_eef_xyz (3)   DEBUG ONLY: state(t)->state(t+1) translation in the
                                       canonical (OpenCV-aligned) gripper frame at t
    debug.{side}_gripper_eef_rot6d (6) DEBUG ONLY: state(t)->state(t+1) rotation, same frame;
                                       last step padded with the identity delta

Load-time action (dataset.md sec 3; documented in README, not stored): joint-position control
-> per-arm dq = target.{side}_joint_pos - state.{side}_joint_pos; gripper ->
target.{side}_gripper_state; base (velocity control) -> target.base_vel directly.

Frames. Each arm's base_link is a ROS FLU frame (x forward, y left, z up), i.e. already the
canonical world frame of ../dataset.md, so the world alignment is the identity and positions
pass through unchanged. The raw_* eef is stored in the NATIVE link6 frame; the canonical
state.*/target.* eef re-bases the gripper onto the canonical OpenCV frame (z=approach, x=right,
y=down) via R_GRIPPER_ALIGN (empirically determined: approach = native +z; wrist-camera
optical-flow gives native x -> camera +y (down), native y -> camera -x):
    R_align = alignment.axis_alignment_matrix("y", "-x", "z")   # R_{e'}^e, det=+1, both arms
    R_canon, p_canon = alignment.align_axis(R_native, p_native, np.eye(3), R_align)
i.e. R_canon = R_native @ R_align.T with positions unchanged (world is already canonical).

Example:
    python convert.py \
        --raw-dir /path/to/ViFailback-Dataset/raw_data \
        --local-dir /path/to/output \
        --repo-id your_id/vifailback \
        --num-proc 8
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Shared backend-agnostic rotation/frame math lives in the repo-root ``alignment`` package. This
# script sits at <repo>/failure_rollout_data/vifailback2lerobot/, so the repo root is three levels up.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from alignment import transforms_numpy as tn  # noqa: E402

URDF_PATH = Path(__file__).parent / "assets" / "piper_description.urdf"

# Canonical-gripper (OpenCV) alignment for the PiPER link6 frame, R_{e'}^e; see "Frames" above.
# Same for both arms. Used only for the debug.* delta features -- stored poses stay native.
R_GRIPPER_ALIGN = tn.axis_alignment_matrix("y", "-x", "z")
ROT6D_IDENTITY = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32)

# Normalizer for the raw gripper channels (qpos[6]/qpos[13]/action[6]/action[13], meters-ish).
# NOT the URDF finger limit (0.035): computed empirically as ~p99 of the recorded values over a
# 300-episode random sample (p95 = 0.076, p99 = 0.095, p99.9 = 0.099, max = 0.099). 0=closed, 1=open.
GRIPPER_MAX = 0.095

def patch_lerobot_for_uint16_depth():
    """Make this lerobot version able to store (H, W, 1) uint16 depth as ``dtype: "image"``.

    Applied only with --save-depth. Two gaps in the stock code path:
    - ``image_writer.image_array_to_pil_image`` only accepts 3-channel images (explicit TODO for
      depth); 1-channel uint16 frames are saved as lossless 16-bit PNGs (PIL mode I;16) instead.
    - ``compute_stats.sample_images`` loads stat samples as uint8 RGB, which would mangle 16-bit
      PNGs; they are loaded as float instead, scaled by 255/65535 so that after the caller's
      fixed /255 normalization the recorded depth stats are fractions of the uint16 full scale.
    """
    import PIL.Image
    import lerobot.datasets.compute_stats as lcs
    import lerobot.datasets.image_writer as liw

    orig_to_pil = liw.image_array_to_pil_image
    orig_sample = lcs.sample_images

    def to_pil(image_array, range_check: bool = True):
        if isinstance(image_array, np.ndarray) and image_array.dtype == np.uint16 and image_array.ndim == 3:
            arr = image_array[..., 0] if image_array.shape[-1] == 1 else image_array[0]
            return PIL.Image.fromarray(arr)  # 2D uint16 -> mode I;16, saved as 16-bit PNG
        return orig_to_pil(image_array, range_check)

    def sample_images(image_paths):
        with PIL.Image.open(image_paths[0]) as im0:
            if im0.mode not in ("I", "I;16"):
                return orig_sample(image_paths)
        sampled = lcs.sample_indices(len(image_paths))
        images = None
        for i, idx in enumerate(sampled):
            arr = np.array(PIL.Image.open(image_paths[idx]), dtype=np.float32)[None]  # (1, H, W)
            arr = lcs.auto_downsample_height_width(arr) * (255.0 / 65535.0)
            if images is None:
                images = np.empty((len(sampled), *arr.shape), dtype=np.float32)
            images[i] = arr
        return images

    liw.image_array_to_pil_image = to_pil
    lcs.sample_images = sample_images


CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
IMG_SHAPE = (480, 640, 3)
DEPTH_SHAPE = (480, 640, 1)
QC_RESIDUAL_WARN_M = 0.005  # warn when mean FK-vs-action_eef residual exceeds 5 mm

np.set_printoptions(precision=3, suppress=True)


# --------------------------------------------------------------------------------------
# Forward kinematics for the PiPER base_link -> link6 chain (6 revolute joints)
# --------------------------------------------------------------------------------------
class PiperFK:
    """Minimal, vectorized URDF FK for a serial revolute chain.

    Validated against the dataset's own ``action_eef``: FK(qpos[t+1]) matches it to
    ~0.1-0.4 mm / <0.1 deg on moving arms (see module docstring).
    """

    def __init__(self, urdf_path: Path, base_link: str = "base_link", tip_link: str = "link6"):
        root = ET.parse(urdf_path).getroot()
        by_child = {}
        for j in root.iter("joint"):
            jtype = j.get("type")
            if jtype not in ("revolute", "continuous", "prismatic", "fixed"):
                continue
            origin = j.find("origin")
            xyz = np.array([float(v) for v in (origin.get("xyz") or "0 0 0").split()]) if origin is not None else np.zeros(3)
            rpy = np.array([float(v) for v in (origin.get("rpy") or "0 0 0").split()]) if origin is not None else np.zeros(3)
            axis_el = j.find("axis")
            axis = np.array([float(v) for v in axis_el.get("xyz").split()]) if axis_el is not None else np.array([1.0, 0.0, 0.0])
            by_child[j.find("child").get("link")] = {
                "type": jtype,
                "parent": j.find("parent").get("link"),
                "origin": self._origin_mat(xyz, rpy),
                "axis": axis / np.linalg.norm(axis),
            }
        chain = []
        link = tip_link
        while link != base_link:
            joint = by_child[link]
            chain.append(joint)
            link = joint["parent"]
        self.chain = chain[::-1]
        n_rev = sum(j["type"] in ("revolute", "continuous") for j in self.chain)
        if n_rev != 6 or any(j["type"] == "prismatic" for j in self.chain):
            raise ValueError(f"unexpected chain {base_link}->{tip_link}: {n_rev} revolute joints")

    @staticmethod
    def _origin_mat(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
        A = np.eye(4)
        A[:3, :3] = tn.rpy_to_matrix(rpy, extrinsic=True)  # URDF origin rpy is fixed-axis XYZ
        A[:3, 3] = xyz
        return A

    def __call__(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """q: (T, 6) joint angles -> (R: (T, 3, 3), p: (T, 3)) pose of link6 in base_link."""
        q = np.asarray(q, dtype=np.float64)
        T_ = np.broadcast_to(np.eye(4), (q.shape[0], 4, 4)).copy()
        qi = 0
        for joint in self.chain:
            T_ = T_ @ joint["origin"]
            if joint["type"] in ("revolute", "continuous"):
                rot = np.broadcast_to(np.eye(4), (q.shape[0], 4, 4)).copy()
                rot[:, :3, :3] = tn.axis_angle_to_matrix(joint["axis"] * q[:, qi : qi + 1])
                T_ = T_ @ rot
                qi += 1
        return T_[:, :3, :3].astype(np.float32), T_[:, :3, 3].astype(np.float32)


# --------------------------------------------------------------------------------------
# Features / episode loading
# --------------------------------------------------------------------------------------
JOINT_NAMES = [f"joint_{i}" for i in range(6)]
ROT6D_NAMES = ["r11", "r21", "r31", "r12", "r22", "r32"]
ROT9D_NAMES = [f"r{row}{col}" for row in range(1, 4) for col in range(1, 4)]
XYZ_NAMES = ["x", "y", "z"]
QUAT_NAMES = ["x", "y", "z", "w"]  # scalar-last (xyzw), the rep ViFailback ships (action_eef)
BASE_VEL_NAMES = ["linear_vel", "angular_vel"]
SIDES = ("left", "right")


def build_features(use_videos: bool = True, save_depth: bool = False) -> dict:
    features = {
        f"observation.images.{cam}": {
            "dtype": "video" if use_videos else "image",
            "shape": IMG_SHAPE,
            "names": ["height", "width", "rgb"],
        }
        for cam in CAMERAS
    }
    if save_depth:
        features.update(
            {
                f"observation.images.{cam}_depth": {
                    "dtype": "image",
                    "shape": DEPTH_SHAPE,
                    "names": ["height", "width", "channel"],
                }
                for cam in CAMERAS
            }
        )
    # raw_state.*: native metrics, no transform / no alignment (dataset.md 2.1). eef is FK(qpos)
    # in the native link6 frame, stored as quat (the rep ViFailback ships via action_eef).
    for side in SIDES:
        features[f"raw_state.{side}_joint_pos"] = {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES}
        features[f"raw_state.{side}_joint_vel"] = {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES}
        features[f"raw_state.{side}_eef_xyz"] = {"dtype": "float32", "shape": (3,), "names": XYZ_NAMES}
        features[f"raw_state.{side}_eef_quat"] = {"dtype": "float32", "shape": (4,), "names": QUAT_NAMES}
        features[f"raw_state.{side}_gripper_state"] = {"dtype": "float32", "shape": (1,), "names": ["gripper"]}
    # raw_target.*: controller command, no transform (dataset.md 2.2). eef is FK(action) native.
    for side in SIDES:
        features[f"raw_target.{side}_joint_pos"] = {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES}
        features[f"raw_target.{side}_eef_xyz"] = {"dtype": "float32", "shape": (3,), "names": XYZ_NAMES}
        features[f"raw_target.{side}_eef_quat"] = {"dtype": "float32", "shape": (4,), "names": QUAT_NAMES}
        features[f"raw_target.{side}_gripper_state"] = {"dtype": "float32", "shape": (1,), "names": ["gripper"]}
    features["raw_target.base_vel"] = {"dtype": "float32", "shape": (2,), "names": BASE_VEL_NAMES}
    # state.*: canonical, axis-aligned (dataset.md 2.3). Joints frame-independent (copied); eef
    # re-based onto canonical world (identity) + OpenCV gripper frame; gripper normalized [0, 1].
    for side in SIDES:
        features[f"state.{side}_joint_pos"] = {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES}
        features[f"state.{side}_joint_vel"] = {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES}
        features[f"state.{side}_eef_xyz"] = {"dtype": "float32", "shape": (3,), "names": XYZ_NAMES}
        features[f"state.{side}_eef_rot9d"] = {"dtype": "float32", "shape": (9,), "names": ROT9D_NAMES}
        features[f"state.{side}_gripper_state"] = {"dtype": "float32", "shape": (1,), "names": ["gripper"]}
    # target.*: canonical target (dataset.md 2.4). Joints frame-independent (copied from raw_target).
    for side in SIDES:
        features[f"target.{side}_joint_pos"] = {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES}
        features[f"target.{side}_eef_xyz"] = {"dtype": "float32", "shape": (3,), "names": XYZ_NAMES}
        features[f"target.{side}_eef_rot9d"] = {"dtype": "float32", "shape": (9,), "names": ROT9D_NAMES}
        features[f"target.{side}_gripper_state"] = {"dtype": "float32", "shape": (1,), "names": ["gripper"]}
    features["target.base_vel"] = {"dtype": "float32", "shape": (2,), "names": BASE_VEL_NAMES}
    for side in SIDES:
        features[f"debug.{side}_gripper_eef_xyz"] = {"dtype": "float32", "shape": (3,), "names": XYZ_NAMES}
        features[f"debug.{side}_gripper_eef_rot6d"] = {"dtype": "float32", "shape": (6,), "names": ROT6D_NAMES}
    return features


def normalize_gripper(g: np.ndarray) -> np.ndarray:
    return np.clip(g / GRIPPER_MAX, 0.0, 1.0).astype(np.float32)


def decode_jpeg_frames(dataset: h5py.Dataset) -> np.ndarray:
    """Decode a (T,) array of JPEG byte strings to (T, H, W, 3) uint8 RGB (stored in RGB order)."""
    frames = []
    for buf in dataset:
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("failed to decode JPEG frame")
        frames.append(img)
    return np.stack(frames)


def decode_depth_frames(dataset: h5py.Dataset) -> np.ndarray:
    """Decode a (T,) array of PNG byte strings to (T, H, W, 1) uint16."""
    frames = []
    for buf in dataset:
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("failed to decode depth frame")
        frames.append(img)
    return np.stack(frames)[..., None]


def fk_action_eef_residual(qpos: np.ndarray, action: np.ndarray, action_eef: np.ndarray, fk: PiperFK) -> dict[str, float]:
    """Per-arm mean FK-vs-action_eef residual in meters (on moving steps when any exist).

    action_eef is a relabeled achieved-next-pose (== FK of the next qpos to ~0.2 mm on healthy
    data). Two consistency checks are returned per arm; both are near zero on healthy data and
    blow up (>0.1 m) on the stale-idle-arm recording artifact, where an idle arm's qpos/action are
    frozen while action_eef stays live:
      "<side>"        state check : mean |FK(qpos[t+1]) - action_eef[t]|  (validated drop detector)
      "<side>_target" target check: mean |FK(action[t]) - action_eef[t]| (solved raw_target vs shipped)
    """
    out = {}
    for side, sl, st in (("left", slice(0, 6), 0), ("right", slice(7, 13), 8)):
        _, p_state = fk(qpos[:, sl])
        _, p_tgt = fk(action[:, sl])
        eef = action_eef[:, st : st + 3]
        moving = np.linalg.norm(np.diff(eef, axis=0), axis=1) > 1e-4     # len T-1 (over transitions)
        res_state = np.linalg.norm(p_state[1:] - eef[:-1], axis=1)      # FK(qpos[t+1]) vs action_eef[t], len T-1
        res_tgt = np.linalg.norm(p_tgt - eef, axis=1)                    # FK(action[t]) vs action_eef[t], len T
        moving_t = np.zeros(len(eef), dtype=bool)                        # len-T mask: steps adjacent to a transition
        moving_t[:-1] |= moving
        moving_t[1:] |= moving
        out[side] = float(res_state[moving].mean()) if moving.any() else float(res_state.mean())
        out[f"{side}_target"] = float(res_tgt[moving_t].mean()) if moving_t.any() else float(res_tgt.mean())
    return out


def episode_qc(path: Path, fk: PiperFK) -> dict[str, float]:
    """QC residuals only (no image decode) -- cheap pre-check for the drop filter."""
    with h5py.File(path, "r") as f:
        return fk_action_eef_residual(f["observations/qpos"][()], f["action"][()], f["action_eef"][()], fk)


def load_episode(path: Path, fk: PiperFK, save_depth: bool = False) -> tuple[dict, dict]:
    """Read one HDF5 episode and return (per-step feature arrays, QC info)."""
    with h5py.File(path, "r") as f:
        qpos = f["observations/qpos"][()].astype(np.float32)
        qvel = f["observations/qvel"][()].astype(np.float32)
        action = f["action"][()].astype(np.float32)
        action_eef = f["action_eef"][()].astype(np.float32)
        base_action = f["base_action"][()].astype(np.float32)
        images = {cam: decode_jpeg_frames(f[f"observations/images/{cam}"]) for cam in CAMERAS}
        depth = (
            {cam: decode_depth_frames(f[f"observations/images_depth/{cam}"]) for cam in CAMERAS}
            if save_depth
            else {}
        )

    data = {f"observation.images.{cam}": images[cam] for cam in CAMERAS}
    data.update({f"observation.images.{cam}_depth": depth[cam] for cam in depth})
    # base_action is a mobile-base velocity command (linear, angular); no base state is shipped, so
    # it lives only in the target groups (velocity control -> used directly as the base action).
    data["raw_target.base_vel"] = base_action
    data["target.base_vel"] = base_action

    eye3 = np.eye(3, dtype=np.float32)
    for side, sl in (("left", slice(0, 7)), ("right", slice(7, 14))):
        joints, grip = qpos[:, sl][:, :6], qpos[:, sl][:, 6]
        joint_vel = qvel[:, sl][:, :6]  # arm joint velocities; gripper-vel slot [.., 6] dropped
        act_joints, act_grip = action[:, sl][:, :6], action[:, sl][:, 6]

        # FK the current joints (state) and the target joints (from `action`); native link6 frame.
        R_state, p_state = fk(joints)
        R_tgt, p_tgt = fk(act_joints)
        # Canonical alignment (dataset.md 2.3/2.4): world already FLU -> identity; gripper -> OpenCV.
        R_state_c, p_state_c = tn.align_axis(R_state, p_state, eye3, R_GRIPPER_ALIGN)
        R_tgt_c, p_tgt_c = tn.align_axis(R_tgt, p_tgt, eye3, R_GRIPPER_ALIGN)

        # raw_state.* (native, no transform)
        data[f"raw_state.{side}_joint_pos"] = joints
        data[f"raw_state.{side}_joint_vel"] = joint_vel
        data[f"raw_state.{side}_eef_xyz"] = p_state
        data[f"raw_state.{side}_eef_quat"] = tn.matrix_to_quaternion(R_state).astype(np.float32)
        data[f"raw_state.{side}_gripper_state"] = grip.astype(np.float32)[:, None]  # raw width (m)
        # raw_target.* (native command / FK-solved target)
        data[f"raw_target.{side}_joint_pos"] = act_joints
        data[f"raw_target.{side}_eef_xyz"] = p_tgt
        data[f"raw_target.{side}_eef_quat"] = tn.matrix_to_quaternion(R_tgt).astype(np.float32)
        data[f"raw_target.{side}_gripper_state"] = act_grip.astype(np.float32)[:, None]  # raw width (m)
        # state.* (canonical)
        data[f"state.{side}_joint_pos"] = joints  # frame-independent, copied
        data[f"state.{side}_joint_vel"] = joint_vel
        data[f"state.{side}_eef_xyz"] = p_state_c.astype(np.float32)
        data[f"state.{side}_eef_rot9d"] = R_state_c.reshape(-1, 9).astype(np.float32)
        data[f"state.{side}_gripper_state"] = normalize_gripper(grip)[:, None]
        # target.* (canonical)
        data[f"target.{side}_joint_pos"] = act_joints  # frame-independent, copied
        data[f"target.{side}_eef_xyz"] = p_tgt_c.astype(np.float32)
        data[f"target.{side}_eef_rot9d"] = R_tgt_c.reshape(-1, 9).astype(np.float32)
        data[f"target.{side}_gripper_state"] = normalize_gripper(act_grip)[:, None]

        # DEBUG ONLY: relative pose from GT state t to GT state t+1, expressed in the canonical
        # (OpenCV-aligned) gripper frame at t. Built from the canonical state pose; the last step
        # (no successor) gets the identity delta.
        dR, dp = tn.gripper_delta_pose(R_state_c[:-1], p_state_c[:-1], R_state_c[1:], p_state_c[1:])
        data[f"debug.{side}_gripper_eef_xyz"] = np.concatenate(
            [dp, np.zeros((1, 3), dtype=np.float32)]
        ).astype(np.float32)
        data[f"debug.{side}_gripper_eef_rot6d"] = np.concatenate(
            [tn.matrix_to_rotation_6d(dR), ROT6D_IDENTITY]
        ).astype(np.float32)

    return data, fk_action_eef_residual(qpos, action, action_eef, fk)


TASK_EP_SUFFIX = re.compile(r"_ep\d+$")


def task_to_instruction(task_dir_name: str) -> str:
    return TASK_EP_SUFFIX.sub("", task_dir_name).replace("_", " ")


def discover_episodes(raw_dir: Path) -> list[tuple[Path, str]]:
    """Sorted (episode_path, instruction) list; deterministic across runs and workers."""
    episodes = []
    for task_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        instruction = task_to_instruction(task_dir.name)
        eps = sorted(task_dir.glob("episode_*.hdf5"), key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
        episodes.extend((ep, instruction) for ep in eps)
    return episodes


# --------------------------------------------------------------------------------------
# Conversion (single worker)
# --------------------------------------------------------------------------------------
def convert_worker(args, episodes: list[tuple[Path, str]], local_dir: Path, repo_id: str, desc: str, position: int = 0):
    sentinel = local_dir / "meta" / ".conversion_complete"
    progress_file = local_dir / "meta" / "_progress.json"
    if args.overwrite and local_dir.exists():
        shutil.rmtree(local_dir)
    if sentinel.exists():
        print(f"[convert] {local_dir.name}: already complete (sentinel present); skipping.")
        return

    # Resume dispatch: a partial dir with loadable metadata is appended to; anything else is rebuilt.
    dataset = None
    start_offset = 0
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
            start_offset = max(dataset.meta.total_episodes, prog_done)
            print(f"[convert] {local_dir.name}: resuming after {start_offset} episodes.")
        except Exception as e:
            print(f"[convert] {local_dir.name}: cannot resume ({type(e).__name__}: {e}); rebuilding.")
            shutil.rmtree(local_dir)
            dataset = None
            start_offset = 0

    if dataset is None:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            robot_type="aloha_agilex_piper",
            root=local_dir,
            fps=args.fps,
            use_videos=args.use_videos,
            features=build_features(args.use_videos, args.save_depth),
            image_writer_processes=args.image_writer_process,
            image_writer_threads=args.image_writer_threads,
        )

    fk = PiperFK(URDF_PATH)
    qc_file = local_dir / "meta" / "qc_warnings.jsonl"
    todo = episodes[start_offset:]
    if args.max_episodes is not None:
        todo = todo[: args.max_episodes]

    def log_qc(record):
        tqdm.write(f"[{desc}] QC (FK vs action_eef): {record}")
        qc_file.parent.mkdir(parents=True, exist_ok=True)
        with open(qc_file, "a") as qf:
            qf.write(json.dumps(record) + "\n")

    consumed = start_offset
    for ep_path, instruction in tqdm(todo, desc=desc, unit="ep", position=position, dynamic_ncols=True):
        try:
            # Cheap pose-only QC pre-check (no image decode): drop episodes where an arm's qpos and
            # action_eef contradict each other (stale-idle-arm recording artifact, ~488/5202
            # episodes concentrated in 6 tasks -- see README).
            qc = episode_qc(ep_path, fk)
            if args.qc_drop_threshold > 0 and max(qc.values()) > args.qc_drop_threshold:
                for side, residual in qc.items():
                    if residual > args.qc_drop_threshold:
                        log_qc({"task": ep_path.parent.name, "episode": ep_path.name, "arm": side,
                                "mean_residual_m": round(residual, 4), "dropped": True})
                consumed += 1
                progress_file.parent.mkdir(parents=True, exist_ok=True)
                progress_file.write_text(json.dumps({"consumed": consumed}))
                continue

            data, qc = load_episode(ep_path, fk, args.save_depth)
            num_frames = len(data["state.left_joint_pos"])
            for i in range(num_frames):
                dataset.add_frame({key: value[i] for key, value in data.items()} | {"task": instruction})
            dataset.save_episode()
            for side, residual in qc.items():
                if residual > QC_RESIDUAL_WARN_M:
                    log_qc({"task": ep_path.parent.name, "episode": ep_path.name, "arm": side,
                            "mean_residual_m": round(residual, 4), "dropped": False})
        except Exception as e:
            if not args.skip_bad_episodes:
                raise
            if dataset.has_pending_frames():
                dataset.clear_episode_buffer()
            tqdm.write(f"[{desc}] skip bad episode {ep_path}: {type(e).__name__}: {e}")
        # Persist progress AFTER the episode is durably saved (or deliberately skipped).
        consumed += 1
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        progress_file.write_text(json.dumps({"consumed": consumed}))

    # Flush parquet footers; without this the last data/meta files are unreadable.
    dataset.finalize()
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(json.dumps({"total_episodes": dataset.meta.total_episodes}))
    return dataset


# --------------------------------------------------------------------------------------
# Orchestrator (parallel shards + merge), mirroring openx2lerobot/openx_rlds.py
# --------------------------------------------------------------------------------------
def run_parallel_conversion(args):
    n = args.num_proc
    base_repo = args.repo_id or "vifailback"
    shards_root = args.local_dir / "_shards"
    tags = [f"shard{i:03d}" for i in range(n)]
    worker_roots = [shards_root / f"vifailback_lerobot_{t}" for t in tags]
    worker_repo_ids = [f"{base_repo}_{t}" for t in tags]
    log_dir = shards_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    sentinel_of = lambda i: worker_roots[i] / "meta" / ".conversion_complete"

    # Cap per-worker allocator arenas and thread pools so n workers don't blow up RSS / cores
    # (same rationale as openx_rlds.py).
    worker_env = {
        **os.environ,
        "HDF5_USE_FILE_LOCKING": "FALSE",
        "MALLOC_ARENA_MAX": "2",
        "MALLOC_TRIM_THRESHOLD_": "131072",
        "OMP_NUM_THREADS": "2",
        "OPENCV_FFMPEG_THREADS": "2",
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
            "--repo-id", worker_repo_ids[i],
            "--shard-tag", tags[i],
            "--num-shards", str(n),
            "--fps", str(args.fps),
            "--image-writer-process", str(args.image_writer_process),
            "--image-writer-threads", str(args.image_writer_threads),
            "--qc-drop-threshold", str(args.qc_drop_threshold),
        ]
        if args.use_videos:
            cmd.append("--use-videos")
        if args.save_depth:
            cmd.append("--save-depth")
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
    incomplete = [i for i in range(n) if not sentinel_of(i).exists()]
    if incomplete:
        raise RuntimeError(
            f"[parallel] shards {incomplete} incomplete; not merging. Inspect {log_dir}/shardNNN.log, "
            f"then re-run the SAME command to resume (finished shards are skipped)."
        )

    aggr_root = args.local_dir / "vifailback_lerobot"
    if aggr_root.exists():
        shutil.rmtree(aggr_root)
    print(f"[parallel] merging {n} shards -> {aggr_root}")
    aggregate_datasets(
        repo_ids=worker_repo_ids,
        aggr_repo_id=base_repo,
        roots=worker_roots,
        aggr_root=aggr_root,
    )
    # Carry the per-shard QC warnings into the merged output before removing the shards.
    with open(aggr_root / "meta" / "qc_warnings.jsonl", "a") as out:
        for root in worker_roots:
            qc = root / "meta" / "qc_warnings.jsonl"
            if qc.exists():
                out.write(qc.read_text())
    shutil.rmtree(shards_root, ignore_errors=True)

    if args.push_to_hub:
        LeRobotDataset(base_repo, root=aggr_root).push_to_hub(
            tags=["LeRobot", "vifailback", "aloha", "piper"], private=False, push_videos=True, license="mit"
        )
    print(f"[parallel] done -> {aggr_root}")
    return aggr_root


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, required=True, help="ViFailback raw_data directory (contains <task>/episode_N.hdf5).")
    parser.add_argument("--local-dir", type=Path, required=True, help="Output directory for the LeRobot dataset.")
    parser.add_argument("--repo-id", type=str, default="vifailback", help="Repository id (required for --push-to-hub).")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--fps", type=int, default=25, help="Control/record frequency (ViFailback tooling default: 25).")
    parser.add_argument("--use-videos", action="store_true", default=True, help="Encode RGB cameras as mp4 (default on).")
    parser.add_argument("--save-depth", action="store_true", help="Also store the uint16 depth streams as image features.")
    parser.add_argument("--image-writer-process", type=int, default=5)
    parser.add_argument("--image-writer-threads", type=int, default=10)
    parser.add_argument("--num-proc", type=int, default=1, help=">1 converts disjoint episode shards in parallel and merges them.")
    parser.add_argument("--shard-tag", type=str, default=None, help="Internal: marks this process as a shard worker.")
    parser.add_argument("--num-shards", type=int, default=None, help="Internal: total shard count for a worker.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Debug: convert at most this many episodes (per worker).")
    parser.add_argument(
        "--qc-drop-threshold",
        type=float,
        default=0.02,
        help="Drop episodes whose per-arm mean FK-vs-action_eef residual exceeds this (meters); "
        "catches the stale-idle-arm artifact (488/5202 episodes, all >0.1 m). 0 disables dropping.",
    )
    parser.add_argument("--skip-bad-episodes", action="store_true", help="Log and skip episodes that raise instead of aborting.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild from scratch even if output exists (disables resume).")
    args = parser.parse_args()

    if args.num_proc > 1 and args.shard_tag is None:
        run_parallel_conversion(args)
        return

    # SIGTERM -> SystemExit so parquet writers close their footers on the way out, keeping the
    # partial dataset resumable. (SIGKILL cannot be handled: it leaves footer-less parquets, the
    # local metadata is then unreadable and the shard is rebuilt from scratch on the next run.)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    if args.save_depth:
        # Must run before LeRobotDataset.create so forked image-writer processes inherit it.
        patch_lerobot_for_uint16_depth()

    episodes = discover_episodes(args.raw_dir)
    if args.shard_tag is not None:
        # Worker mode: take this shard's contiguous slice of the (deterministic) episode list.
        shard_id = int(args.shard_tag[len("shard"):])
        indices = np.array_split(np.arange(len(episodes)), args.num_shards)[shard_id]
        episodes = [episodes[i] for i in indices]
        local_dir = args.local_dir / f"vifailback_lerobot_{args.shard_tag}"
        desc, position = args.shard_tag, shard_id
    else:
        local_dir = args.local_dir / "vifailback_lerobot"
        desc, position = "vifailback", 0

    dataset = convert_worker(args, episodes, local_dir, args.repo_id, desc, position)
    if args.push_to_hub and args.shard_tag is None and dataset is not None:
        dataset.push_to_hub(
            tags=["LeRobot", "vifailback", "aloha", "piper"], private=False, push_videos=True, license="mit"
        )


if __name__ == "__main__":
    main()
