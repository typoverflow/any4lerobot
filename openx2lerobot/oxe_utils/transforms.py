"""
Adapt from https://github.com/openvla/openvla/blob/main/prismatic/vla/datasets/rlds/oxe/transforms.py
transforms.py

Defines a registry of per-dataset standardization transforms for each dataset in Open-X Embodiment.

Transforms adopt the following structure:
    Input: Dictionary of *batched* features (i.e., has leading time dimension)
    Output: Dictionary `step` =>> {
        "observation": {
            <image_keys, depth_image_keys>
            State (in chosen state representation)
        },
        "action": Action (in chosen action representation),
        "language_instruction": str
    }
"""

import os
import sys
from typing import Any, Dict

import tensorflow as tf
from oxe_utils.transform_utils import (
    binarize_gripper_actions,
    invert_gripper_actions,
    rel2abs_gripper_actions,
    relabel_bridge_actions,
)

# The DROID transforms below use the shared, backend-agnostic frame/rotation math from the
# alignment package at the repo root (see design_of_state_and_action_space.md). openx runs with
# CWD=openx2lerobot, so the repo root is not on sys.path by default -- add it before importing.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from alignment import transforms_tf as align_tf  # noqa: E402


# === Canonical-frame helpers (design_of_state_and_action_space.md, "Preprocessing") ===============
# Every dataset standardized below runs on a robot whose BASE frame is already the canonical FLU
# convention (x-forward, y-left, z-up), so the world alignment R_{w'}^w is the identity and positions
# pass through unchanged. Only the GRIPPER frame is re-based onto the canonical OpenCV convention
# (z = approach / out of the gripper, x = finger-open direction "right", y completes "down").
# Per-robot native-EEF -> canonical-gripper rotations R_{e'}^e (verified against URDFs; see commit msg):
_WORLD_ALIGN_IDENTITY = align_tf.axis_alignment_matrix("x", "y", "z")  # FLU base == canonical world
# Franka panda_hand: native +z is the approach axis and the fingers translate along native +y
# (franka_description: panda_finger_joint axis 0 1 0). Mapping finger +y -> canonical +x is a -90 deg
# rotation about the (shared) approach axis z -- i.e. "rotate about z", as expected. NOTE: DROID
# physically mounts a Robotiq 2F-85 whose logged TCP may be yawed, so the in-plane (x/y) part is
# ~medium confidence; z = approach is solid.
_GRIPPER_ALIGN_FRANKA = align_tf.axis_alignment_matrix("-y", "x", "z")
# Bridge/WidowX: the stored rpy is NOT the URDF ee_gripper_link frame (+x approach) -- widowx_envs
# composes it with DEFAULT_ROTATION = [[0,0,1],[0,1,0],[-1,0,0]], so the stored frame reads ~identity
# when the gripper points straight down. Hence stored -z = approach and the finger axis is stored y;
# descent motion maps to canonical +z under this align (verified on bridge_orig). The in-plane SIGN was
# fixed against video on SOAR (same widowx_envs stack, 2026-07-06): at the neutral gripper-down pose
# canonical x must read world-RIGHT (= stored -y) and canonical y world-backward (= stored -x); the
# earlier ("y","x","-z") was 180 degrees off about the approach axis.
_GRIPPER_ALIGN_WIDOWX = align_tf.axis_alignment_matrix("-y", "-x", "-z")
# Google / Everyday-Robots link_gripper_tcp: native +z is the approach axis (TCP is a +z offset), but
# the finger-open axis is NOT documented publicly, so the in-plane orientation is left native
# (identity) -- z = approach is already correct; revisit the x/y roll if the finger axis is recovered.
_GRIPPER_ALIGN_GOOGLE = align_tf.axis_alignment_matrix("x", "y", "z")


def _to_canonical(R_native, p_native, gripper_align):
    """Map a native base-frame EEF pose into the canonical frames.

    World is already canonical FLU for every dataset here, so position is unchanged and the
    orientation is re-based onto the canonical gripper axes: ``R_e^w = R_native @ (R_{e'}^e)^T``.
    Returns ``(R_canonical, p_canonical)``.
    """
    return align_tf.align_axis(R_native, p_native, _WORLD_ALIGN_IDENTITY, gripper_align)


def _eef_delta_fields(R_cur, p_cur, R_tgt, p_tgt, suffix=""):
    """The six frame-aware eef delta fields for current pose e -> target e* (canonical frames).

    ``suffix`` is ``""`` for the default ground-truth-next target, or ``"_command"`` for a commanded
    e* (design doc: the target has two modes; commands are emitted as a parallel ``*_command`` set).
    rpy is extrinsic XYZ throughout (the repo-wide convention).
    """
    world_R, world_p = align_tf.world_delta_pose(R_cur, p_cur, R_tgt, p_tgt)
    grip_R, grip_p = align_tf.gripper_delta_pose(R_cur, p_cur, R_tgt, p_tgt)
    diff_rpy = align_tf.matrix_to_rpy(R_tgt, extrinsic=True) - align_tf.matrix_to_rpy(R_cur, extrinsic=True)
    return {
        f"diff_eef_xyz{suffix}": world_p,
        f"diff_eef_rpy{suffix}": diff_rpy,
        f"world_eef_xyz{suffix}": world_p,
        f"world_eef_rot6d{suffix}": align_tf.matrix_to_rotation_6d(world_R),
        f"gripper_eef_xyz{suffix}": grip_p,
        f"gripper_eef_rot6d{suffix}": align_tf.matrix_to_rotation_6d(grip_R),
    }


def _next_step_delta_fields(R, p):
    """Default action over ALL T steps: the per-step motion to the next pose (obs[t] -> obs[t+1]).

    The last step has no successor, so its target is the current pose repeated -- making the final
    delta a genuine no-op (identity rotation, zero translation) while keeping every field at length T.

    We build the target by repeating the last pose (``R[-1:]``/``p[-1:]``) rather than padding the
    already-shortened delta tensor. The old ``tf.zeros_like(v[-1:])`` pad produced a *zero-length* row
    for T==1 episodes (where the delta tensor ``v`` has length T-1 == 0), leaving xyz fields length 0
    while rot6d fields were length 1 -- which silently dropped every single-timestep episode. The
    repeat-last-target form is byte-for-byte identical for T>=2 and correct for T<=1.
    """
    R_tgt = tf.concat([R[1:], R[-1:]], axis=0)  # next pose; last repeated -> no-op final step
    p_tgt = tf.concat([p[1:], p[-1:]], axis=0)
    return _eef_delta_fields(R, p, R_tgt, p_tgt)


def _diff_with_dummy_last(x):
    """Per-step forward difference ``x[t+1] - x[t]`` over all T steps, with a zero dummy last step."""
    return tf.concat([x[1:] - x[:-1], tf.zeros_like(x[-1:])], axis=0)


def _first_nonempty_instruction(trajectory):
    """DROID ships up to three language instructions and the first is frequently empty; return the
    first non-empty one per step (falling back across ``language_instruction_2`` / ``_3``)."""
    lang = trajectory["language_instruction"]
    for k in ("language_instruction_2", "language_instruction_3"):
        if k in trajectory:
            lang = tf.where(tf.strings.length(lang) > 0, lang, trajectory[k])
    return lang


def droid_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """DROID (Franka). Default e* = ground-truth next pose (last step a dummy no-op); the commanded
    ``action_dict`` pose/joints/gripper are also emitted as ``*_command``. Native euler is extrinsic XYZ.
    Franka hand -> canonical OpenCV gripper via ``_GRIPPER_ALIGN_FRANKA`` (= NVIDIA Cosmos ``_DROID_TO_OPENCV``).
    """
    obs = trajectory["observation"]
    # The wrist Zed Mini is mounted below the arm, i.e. rolled 180 deg about the optical axis, so the
    # raw image x/y point opposite the canonical OpenCV gripper x/y. Rotating the image 180 deg
    # (reverse H and W; a proper rotation, not a mirror) restores image +x ~ gripper +x, +y ~ gripper +y.
    # NOTE: raw DROID wrist intrinsics (not shipped in OXE) would need cx,cy -> W-1-cx, H-1-cy.
    obs["wrist_image_left"] = obs["wrist_image_left"][:, ::-1, ::-1, :]
    eef_xyz, joint = obs["cartesian_position"][:, :3], obs["joint_position"]
    R, p = _to_canonical(align_tf.rpy_to_matrix(obs["cartesian_position"][:, 3:6], extrinsic=True), eef_xyz, _GRIPPER_ALIGN_FRANKA)
    state = {
        "eef_xyz": p,
        "eef_rpy": align_tf.matrix_to_rpy(R, extrinsic=True),
        "eef_rot6d": align_tf.matrix_to_rotation_6d(R),
        "joint_pos": joint,
        "gripper_state": invert_gripper_actions(obs["gripper_position"]),
    }
    action = _next_step_delta_fields(R, p)
    action["diff_joint_pos"] = _diff_with_dummy_last(joint)
    # commanded absolute pose (xyz + extrinsic-XYZ euler) / joints / gripper, aligned the same way
    act = trajectory["action_dict"]
    cmd = act["cartesian_position"]
    R_cmd, p_cmd = _to_canonical(align_tf.rpy_to_matrix(cmd[:, 3:6], extrinsic=True), cmd[:, :3], _GRIPPER_ALIGN_FRANKA)
    action.update(_eef_delta_fields(R, p, R_cmd, p_cmd, suffix="_command"))
    action["diff_joint_pos_command"] = act["joint_position"] - joint
    action["gripper_state"] = invert_gripper_actions(act["gripper_position"])
    trajectory["state"], trajectory["action"] = state, action
    trajectory["language_instruction"] = _first_nonempty_instruction(trajectory)
    return trajectory


def bridge_oxe_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies to version of Bridge V2 in Open X-Embodiment mixture.

    Note =>> In original Bridge V2 dataset, the first timestep has an all-zero action, so we remove it!
    """
    for key in trajectory.keys():
        if key == "traj_metadata":
            continue
        elif key in ["observation", "action"]:
            for key2 in trajectory[key]:
                trajectory[key][key2] = trajectory[key][key2][1:]
        else:
            trajectory[key] = trajectory[key][1:]

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            tf.cast(trajectory["action"]["open_gripper"][:, None], tf.float32),
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    trajectory = relabel_bridge_actions(trajectory)
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def bridge_orig_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """BridgeData V2 (WidowX). state [7] = eef xyz + extrinsic-XYZ rpy + gripper (1=open); no extrinsics.
    Default e* = ground-truth next pose (last step a dummy no-op); NO ``*_command`` (the raw action is the
    unreliable one the repo relabels to the state delta). Stored widowx_envs gripper frame ->
    canonical OpenCV gripper via ``_GRIPPER_ALIGN_WIDOWX``.
    """
    obs = trajectory["observation"]
    eef_xyz = obs["state"][:, :3]
    R, p = _to_canonical(align_tf.rpy_to_matrix(obs["state"][:, 3:6], extrinsic=True), eef_xyz, _GRIPPER_ALIGN_WIDOWX)
    state = {
        "eef_xyz": p,
        "eef_rpy": align_tf.matrix_to_rpy(R, extrinsic=True),
        "eef_rot6d": align_tf.matrix_to_rotation_6d(R),
        "gripper_state": obs["state"][:, 6:7],
    }
    action = _next_step_delta_fields(R, p)
    # commanded gripper, binarized; already 1=open / 0=closed (no inversion)
    action["gripper_state"] = binarize_gripper_actions(trajectory["action"][:, 6])[:, None]
    trajectory["state"], trajectory["action"] = state, action
    # language passes through unchanged from BridgeData V2's step-level top-level instruction (length T)
    trajectory["language_instruction"] = trajectory["language_instruction"]
    return trajectory


def ppgm_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["cartesian_position"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["gripper_position"][:, -1:]
    return trajectory


def rt1_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """RT-1 / fractal (Google Robot, no joints, no extrinsics). ``base_pose_tool_reached`` = xyz + quat
    (xyzw). Default e* = ground-truth next pose (last step a dummy no-op); the base-frame command
    ``world_vector`` + ``rotation_delta`` is emitted as ``*_command``. Tool approach axis is already
    canonical +z and the finger axis is undocumented, so ``_GRIPPER_ALIGN_GOOGLE`` is identity.
    """
    obs = trajectory["observation"]
    eef_xyz = obs["base_pose_tool_reached"][:, :3]
    R_native = align_tf.quaternion_to_matrix(obs["base_pose_tool_reached"][:, 3:7])  # (x, y, z, w)
    R, p = _to_canonical(R_native, eef_xyz, _GRIPPER_ALIGN_GOOGLE)
    state = {
        "eef_xyz": p,
        "eef_rpy": align_tf.matrix_to_rpy(R, extrinsic=True),
        "eef_rot6d": align_tf.matrix_to_rotation_6d(R),
        "gripper_state": invert_gripper_actions(obs["gripper_closed"]),
    }
    action = _next_step_delta_fields(R, p)
    # command e*: p + world_vector ; R = dR_world @ R (world-frame delta, left-multiply)
    act = trajectory["action"]
    dR_world = align_tf.rpy_to_matrix(act["rotation_delta"], extrinsic=True)
    R_cmd, p_cmd = _to_canonical(tf.matmul(dR_world, R_native), eef_xyz + act["world_vector"], _GRIPPER_ALIGN_GOOGLE)
    action.update(_eef_delta_fields(R, p, R_cmd, p_cmd, suffix="_command"))
    # gripper: relative closedness command -> absolute (0=closed, 1=open)
    action["gripper_state"] = rel2abs_gripper_actions(act["gripper_closedness_action"][:, 0])[:, None]
    trajectory["state"], trajectory["action"] = state, action
    trajectory["language_instruction"] = obs["natural_language_instruction"]
    return trajectory


def kuka_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def taco_play_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """CALVIN / taco_play (Franka). ``robot_obs`` [15] = eef xyz (0:3) + extrinsic-XYZ rpy (3:6) +
    gripper width (6) + joints (7:14) + gripper action (14); pybullet poses, no extrinsics. Default e*
    = ground-truth next pose (last step a dummy no-op); the absolute-pose command ``action.actions`` [0:6]
    is emitted as ``*_command`` (CALVIN's scaled ``rel_actions_*`` are not used). Franka hand -> canonical
    OpenCV gripper via ``_GRIPPER_ALIGN_FRANKA``.
    """
    obs = trajectory["observation"]
    robot_obs = obs["robot_obs"]
    eef_xyz, joint = robot_obs[:, 0:3], robot_obs[:, 7:14]
    R, p = _to_canonical(align_tf.rpy_to_matrix(robot_obs[:, 3:6], extrinsic=True), eef_xyz, _GRIPPER_ALIGN_FRANKA)
    state = {
        "eef_xyz": p,
        "eef_rpy": align_tf.matrix_to_rpy(R, extrinsic=True),
        "eef_rot6d": align_tf.matrix_to_rotation_6d(R),
        "joint_pos": joint,
        "gripper_state": robot_obs[:, 6:7],  # sensed gripper width; larger = more open
    }
    action = _next_step_delta_fields(R, p)
    action["diff_joint_pos"] = _diff_with_dummy_last(joint)
    # commanded absolute target pose: actions[:, 0:3] xyz, actions[:, 3:6] extrinsic-XYZ rpy
    cmd = trajectory["action"]["actions"]
    R_cmd, p_cmd = _to_canonical(align_tf.rpy_to_matrix(cmd[:, 3:6], extrinsic=True), cmd[:, 0:3], _GRIPPER_ALIGN_FRANKA)
    action.update(_eef_delta_fields(R, p, R_cmd, p_cmd, suffix="_command"))
    # commanded gripper (absolute), 1=open / -1=closed -> clip to 1=open / 0=closed
    action["gripper_state"] = tf.clip_by_value(cmd[:, 6:7], 0.0, 1.0)
    trajectory["state"], trajectory["action"] = state, action
    trajectory["language_instruction"] = obs["natural_language_instruction"]
    return trajectory


def jaco_play_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state_eef"] = trajectory["observation"]["end_effector_cartesian_pos"][:, :6]
    trajectory["observation"]["state_gripper"] = trajectory["observation"]["end_effector_cartesian_pos"][:, -1:]

    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            tf.zeros_like(trajectory["action"]["world_vector"]),
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def berkeley_cable_routing_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            tf.zeros_like(trajectory["action"]["world_vector"][:, :1]),
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def roboturk_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert absolute gripper action, +1 = open, 0 = close
    gripper_action = invert_gripper_actions(tf.clip_by_value(trajectory["action"]["gripper_closedness_action"], 0, 1))

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action,
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def nyu_door_opening_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def viola_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, None]
    gripper_action = tf.clip_by_value(gripper_action, 0, 1)
    gripper_action = invert_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action,
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def berkeley_autolab_ur5_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # flip wrist_image from bgr to rgb
    trajectory["observation"]["hand_image"] = trajectory["observation"]["hand_image"][..., ::-1]

    trajectory["observation"]["state"] = trajectory["observation"]["robot_state"][:, 6:14]
    trajectory["observation"]["depth"] = trajectory["observation"].pop("image_with_depth")

    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def toto_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            tf.cast(trajectory["action"]["open_gripper"][:, None], tf.float32),
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def language_table_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # default to "open" gripper
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"]),
            tf.ones_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )

    # decode language instruction
    instruction_bytes = trajectory["observation"]["instruction"]
    instruction_encoded = tf.strings.unicode_encode(instruction_bytes, output_encoding="UTF-8")
    # Remove trailing padding --> convert RaggedTensor to regular Tensor.
    trajectory["language_instruction"] = tf.strings.split(instruction_encoded, "\x00")[:, :1].to_tensor()[:, 0]
    return trajectory


def pusht_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            trajectory["action"]["gripper_closedness_action"][:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def stanford_kuka_multimodal_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["depth_image"] = trajectory["observation"]["depth_image"][..., 0]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tf.zeros_like(trajectory["action"][:, :3]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def nyu_rot_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][..., :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][..., -1:]
    trajectory["action"] = trajectory["action"][..., :7]
    return trajectory


def stanford_hydra_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # flip image & wrist_image from bgr to rgb
    trajectory["observation"]["image"] = trajectory["observation"]["image"][..., ::-1]
    trajectory["observation"]["wrist_image"] = trajectory["observation"]["wrist_image"][..., ::-1]

    # invert gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ),
        axis=-1,
    )

    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            trajectory["observation"]["state"][:, 7:10],
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -3:-2]
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def austin_buds_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)),
        ),
        axis=-1,
    )

    trajectory["observation"]["state"] = trajectory["observation"]["state"][:, :8]
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def nyu_franka_play_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["depth"] = tf.cast(trajectory["observation"]["depth"][..., 0], tf.float32)
    trajectory["observation"]["depth_additional_view"] = tf.cast(
        trajectory["observation"]["depth_additional_view"][..., 0], tf.float32
    )
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, -6:]

    # clip gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, -8:-2],
            tf.clip_by_value(trajectory["action"][:, -2:-1], 0, 1),
        ),
        axis=-1,
    )

    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def maniskill_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][..., 7:8]
    return trajectory


def furniture_bench_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :7],
            trajectory["observation"]["state"][:, -1:],
        ),
        axis=-1,
    )

    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)),
        ),
        axis=-1,
    )
    return trajectory


def cmu_franka_exploration_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def ucsd_kitchen_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["joint_state"] = trajectory["observation"]["state"][:, :7]
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def ucsd_pick_place_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tf.zeros_like(trajectory["action"][:, :3]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def austin_sailor_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)),
        ),
        axis=-1,
    )

    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def austin_sirius_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)),
        ),
        axis=-1,
    )

    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def bc_z_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """BC-Z (Google Robot, no joints, no extrinsics). ``present/xyz`` + ``present/axis_angle`` (rotation
    vector) in the base frame. Default e* = ground-truth next pose (last step a dummy no-op). The
    ``future/{xyz,axis_angle}_residual`` command is FAR-HORIZON (~5-7 steps, ~9x per-step), emitted as
    ``*_command`` (a multi-step target, not a 1-step command). Tool approach is already canonical +z and
    the finger axis is undocumented, so ``_GRIPPER_ALIGN_GOOGLE`` is identity.
    """
    obs = trajectory["observation"]
    eef_xyz = obs["present/xyz"]
    R_native = align_tf.axis_angle_to_matrix(obs["present/axis_angle"])
    R, p = _to_canonical(R_native, eef_xyz, _GRIPPER_ALIGN_GOOGLE)
    state = {
        "eef_xyz": p,
        "eef_rpy": align_tf.matrix_to_rpy(R, extrinsic=True),
        "eef_rot6d": align_tf.matrix_to_rotation_6d(R),
        # present/sensed_close is continuous in [~0.2, 1] with 1 = fully closed -> invert to 1=open
        "gripper_state": invert_gripper_actions(obs["present/sensed_close"]),
    }
    action = _next_step_delta_fields(R, p)
    # far-horizon command: present + first future residual (xyz = base-frame translation; axis_angle =
    # base-frame delta rotation, left-multiplied onto R)
    fut = trajectory["action"]
    R_cmd_native = tf.matmul(align_tf.axis_angle_to_matrix(fut["future/axis_angle_residual"][:, :3]), R_native)
    R_cmd, p_cmd = _to_canonical(R_cmd_native, eef_xyz + fut["future/xyz_residual"][:, :3], _GRIPPER_ALIGN_GOOGLE)
    action.update(_eef_delta_fields(R, p, R_cmd, p_cmd, suffix="_command"))
    # commanded immediate gripper, binary {0,1} with 1=closed -> invert to 1=open
    action["gripper_state"] = invert_gripper_actions(tf.cast(fut["future/target_close"][:, :1], tf.float32))
    trajectory["state"], trajectory["action"] = state, action
    trajectory["language_instruction"] = obs["natural_language_instruction"]
    return trajectory


def tokyo_pr2_opening_fridge_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def tokyo_pr2_tabletop_manipulation_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def utokyo_xarm_pick_place_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return trajectory


def utokyo_xarm_bimanual_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., -7:]
    return trajectory


def robo_net_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :4],
            tf.zeros_like(trajectory["observation"]["state"][:, :2]),
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :4],
            tf.zeros_like(trajectory["action"][:, :2]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def berkeley_mvp_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["gripper"] = trajectory["observation"]["gripper"][:, None]
    return trajectory


def berkeley_rpt_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["gripper"] = trajectory["observation"]["gripper"][:, None]
    return trajectory


def kaist_nonprehensible_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state"] = trajectory["observation"]["state"][:, -7:]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            tf.zeros_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    return trajectory


def stanford_mask_vit_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["end_effector_pose"][:, :4],
            tf.zeros_like(trajectory["observation"]["end_effector_pose"][:, :2]),
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["end_effector_pose"][:, -1:]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :4],
            tf.zeros_like(trajectory["action"][:, :2]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def tokyo_lsmo_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def dlr_sara_pour_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return trajectory


def dlr_sara_grid_clamp_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state"] = trajectory["observation"]["state"][:, :6]
    return trajectory


def dlr_edan_shared_control_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ),
        axis=-1,
    )
    return trajectory


def asu_table_top_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["ground_truth_states"]["EE"]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def robocook_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def imperial_wristcam_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def iamlab_pick_insert_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["observation"]["joint_state"] = trajectory["observation"]["state"][:, :7]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, 7:8]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            trajectory["action"][:, 7:8],
        ),
        axis=-1,
    )
    return trajectory


def uiuc_d3field_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    return trajectory


def utaustin_mutex_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # flip image & wrist_image from bgr to rgb
    trajectory["observation"]["image"] = trajectory["observation"]["image"][..., ::-1]
    trajectory["observation"]["wrist_image"] = trajectory["observation"]["wrist_image"][..., ::-1]

    trajectory["observation"]["state"] = trajectory["observation"]["state"][:, :8]

    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)),
        ),
        axis=-1,
    )

    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def berkeley_fanuc_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # flip image & wrist_image from bgr to rgb
    trajectory["observation"]["image"] = trajectory["observation"]["image"][..., ::-1]
    trajectory["observation"]["wrist_image"] = trajectory["observation"]["wrist_image"][..., ::-1]

    trajectory["observation"]["joint_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, 6:7]

    # dataset does not store gripper actions, so use gripper state info, invert so +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            invert_gripper_actions(trajectory["observation"]["gripper_state"]),
        ),
        axis=-1,
    )
    return trajectory


def cmu_playing_with_food_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def playfusion_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            trajectory["action"][:, -4:],
        ),
        axis=-1,
    )
    return trajectory


def cmu_stretch_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            tf.zeros_like(trajectory["observation"]["state"][:, :3]),
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def gnm_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["position"],
            tf.zeros_like(trajectory["observation"]["state"][:, :3]),
            trajectory["observation"]["yaw"],
        ),
        axis=-1,
    )
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    return trajectory


def fmb_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # flip image from bgr to rgb
    trajectory["observation"]["image_wrist_1"] = trajectory["observation"]["image_wrist_1"][..., ::-1]
    trajectory["observation"]["image_wrist_2"] = trajectory["observation"]["image_wrist_2"][..., ::-1]
    trajectory["observation"]["image_side_1"] = trajectory["observation"]["image_side_1"][..., ::-1]
    trajectory["observation"]["image_side_2"] = trajectory["observation"]["image_side_2"][..., ::-1]
    
    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["eef_pose"],
            trajectory["observation"]["state_gripper_pose"][..., None],
        ),
        axis=-1,
    )
    return trajectory


def dobbe_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def roboset_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]

    # gripper action is in -1...1 --> clip to 0...1, flip
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))

    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :7],
            gripper_action,
        ),
        axis=-1,
    )
    return trajectory


def rh20t_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["tcp_base"],
            tf.cast(trajectory["action"]["gripper"][:, None], tf.float32),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["tcp_base"],
            trajectory["observation"]["gripper_width"][..., None],
        ),
        axis=-1,
    )
    return trajectory


def tdroid_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["cartesian_position"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["gripper_position"][:, -1:]
    return trajectory


def libero_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # gripper action is in -1 (open)...1 (close) --> clip to 0...1, flip --> +1 = open, 0 = close
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            gripper_action,
        ],
        axis=1,
    )
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -2:]  # 2D gripper state
    return trajectory


# === Registry ===
OXE_STANDARDIZATION_TRANSFORMS = {
    "bridge_oxe": bridge_oxe_dataset_transform,
    "bridge_orig": bridge_orig_dataset_transform,
    "bridge_dataset": bridge_orig_dataset_transform,
    "ppgm": ppgm_dataset_transform,
    "ppgm_static": ppgm_dataset_transform,
    "ppgm_wrist": ppgm_dataset_transform,
    "fractal20220817_data": rt1_dataset_transform,
    "kuka": kuka_dataset_transform,
    "taco_play": taco_play_dataset_transform,
    "jaco_play": jaco_play_dataset_transform,
    "berkeley_cable_routing": berkeley_cable_routing_dataset_transform,
    "roboturk": roboturk_dataset_transform,
    "nyu_door_opening_surprising_effectiveness": nyu_door_opening_dataset_transform,
    "viola": viola_dataset_transform,
    "berkeley_autolab_ur5": berkeley_autolab_ur5_dataset_transform,
    "toto": toto_dataset_transform,
    "language_table": language_table_dataset_transform,
    "columbia_cairlab_pusht_real": pusht_dataset_transform,
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": stanford_kuka_multimodal_dataset_transform,
    "nyu_rot_dataset_converted_externally_to_rlds": nyu_rot_dataset_transform,
    "stanford_hydra_dataset_converted_externally_to_rlds": stanford_hydra_dataset_transform,
    "austin_buds_dataset_converted_externally_to_rlds": austin_buds_dataset_transform,
    "nyu_franka_play_dataset_converted_externally_to_rlds": nyu_franka_play_dataset_transform,
    "maniskill_dataset_converted_externally_to_rlds": maniskill_dataset_transform,
    "furniture_bench_dataset_converted_externally_to_rlds": furniture_bench_dataset_transform,
    "cmu_franka_exploration_dataset_converted_externally_to_rlds": cmu_franka_exploration_dataset_transform,
    "ucsd_kitchen_dataset_converted_externally_to_rlds": ucsd_kitchen_dataset_transform,
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": ucsd_pick_place_dataset_transform,
    "austin_sailor_dataset_converted_externally_to_rlds": austin_sailor_dataset_transform,
    "austin_sirius_dataset_converted_externally_to_rlds": austin_sirius_dataset_transform,
    "bc_z": bc_z_dataset_transform,
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds": tokyo_pr2_opening_fridge_dataset_transform,
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": tokyo_pr2_tabletop_manipulation_dataset_transform,
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": utokyo_xarm_pick_place_dataset_transform,
    "utokyo_xarm_bimanual_converted_externally_to_rlds": utokyo_xarm_bimanual_dataset_transform,
    "robo_net": robo_net_dataset_transform,
    "berkeley_mvp_converted_externally_to_rlds": berkeley_mvp_dataset_transform,
    "berkeley_rpt_converted_externally_to_rlds": berkeley_rpt_dataset_transform,
    "kaist_nonprehensile_converted_externally_to_rlds": kaist_nonprehensible_dataset_transform,
    "stanford_mask_vit_converted_externally_to_rlds": stanford_mask_vit_dataset_transform,
    "tokyo_u_lsmo_converted_externally_to_rlds": tokyo_lsmo_dataset_transform,
    "dlr_sara_pour_converted_externally_to_rlds": dlr_sara_pour_dataset_transform,
    "dlr_sara_grid_clamp_converted_externally_to_rlds": dlr_sara_grid_clamp_dataset_transform,
    "dlr_edan_shared_control_converted_externally_to_rlds": dlr_edan_shared_control_dataset_transform,
    "asu_table_top_converted_externally_to_rlds": asu_table_top_dataset_transform,
    "stanford_robocook_converted_externally_to_rlds": robocook_dataset_transform,
    "imperialcollege_sawyer_wrist_cam": imperial_wristcam_dataset_transform,
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": iamlab_pick_insert_dataset_transform,
    "uiuc_d3field": uiuc_d3field_dataset_transform,
    "utaustin_mutex": utaustin_mutex_dataset_transform,
    "berkeley_fanuc_manipulation": berkeley_fanuc_dataset_transform,
    "cmu_playing_with_food": cmu_playing_with_food_dataset_transform,
    "cmu_play_fusion": playfusion_dataset_transform,
    "cmu_stretch": cmu_stretch_dataset_transform,
    "berkeley_gnm_recon": gnm_dataset_transform,
    "berkeley_gnm_cory_hall": gnm_dataset_transform,
    "berkeley_gnm_sac_son": gnm_dataset_transform,
    "droid": droid_transform,
    "fmb_dataset": fmb_dataset_transform,
    "dobbe": dobbe_dataset_transform,
    "roboset": roboset_dataset_transform,
    "rh20t_rlds": rh20t_dataset_transform,
    ### T-DROID datasets
    "tdroid_carrot_in_bowl": tdroid_dataset_transform,
    "tdroid_pour_corn_in_pot": tdroid_dataset_transform,
    "tdroid_flip_pot_upright": tdroid_dataset_transform,
    "tdroid_move_object_onto_plate": tdroid_dataset_transform,
    "tdroid_knock_object_over": tdroid_dataset_transform,
    "tdroid_cover_object_with_towel": tdroid_dataset_transform,
    ### DROID Finetuning datasets
    "droid_wipe": droid_transform,
    ### LIBERO datasets (modified versions)
    "libero_spatial_no_noops": libero_dataset_transform,
    "libero_object_no_noops": libero_dataset_transform,
    "libero_goal_no_noops": libero_dataset_transform,
    "libero_10_no_noops": libero_dataset_transform,
}
