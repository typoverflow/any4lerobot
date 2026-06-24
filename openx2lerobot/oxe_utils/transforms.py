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


def _droid_state_and_action(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Shared DROID state/action builder, implementing design_of_state_and_action_space.md.

    Target e* selection (see the doc's two modes): DROID's raw schema does NOT expose a commanded
    absolute eef/joint pose -- ``action_dict`` only carries ``gripper_position`` (a commanded
    gripper target). We therefore fall back to mode 2 and use the *ground-truth next* pose as e*.
    Per that mode we do NOT pad: the last step has no successor, so it is DISCARDED, leaving every
    field at length T-1 (action[t] is the motion from obs[t] to obs[t+1]). The gripper action keeps
    DROID's genuine *commanded* target since it is the one command the dataset does provide.

    Field names follow design_of_state_and_action_space.md. The world/body deltas are the unfolded
    ``world_body_eef_motion`` written directly with the backend-agnostic ``alignment`` utilities
    (no padding branch).
    """
    obs = trajectory["observation"]
    eef_xyz = obs["cartesian_position"][:, :3]
    eef_rpy = obs["cartesian_position"][:, 3:6]
    joint = obs["joint_position"]

    # DROID's native euler is extrinsic (fixed-axis) XYZ -- its own code decodes poses with
    # scipy from_euler("xyz") -- so build rotations with extrinsic=True for a faithful rotation.
    R = align_tf.rpy_to_matrix(eef_rpy, extrinsic=True)  # [T, 3, 3] world-frame orientations

    # --- State (world frame): eef_xyz, eef_rpy, eef_rot6d, joint_pos, gripper_state ---
    state = {
        "eef_xyz": eef_xyz,
        "eef_rpy": eef_rpy,
        "joint_pos": joint,
        "gripper_state": invert_gripper_actions(obs["gripper_position"]),
        "eef_rot6d": align_tf.matrix_to_rotation_6d(R),
    }

    # --- Action: per-step delta from obs[t] (current pose e) to obs[t+1] (target pose e*) ---
    # world-frame delta: R_{e->e*}^w = R_{t+1} R_t^T, p = p_{t+1} - p_t
    world_R, world_p = align_tf.world_delta(R[:-1], eef_xyz[:-1], R[1:], eef_xyz[1:])
    # body-frame delta: R_{e->e*}^e = R_t^T R_{t+1}, p = R_t^T (p_{t+1} - p_t)
    body_R, body_p = align_tf.relative_pose(R[:-1], eef_xyz[:-1], R[1:], eef_xyz[1:])
    action = {
        # literal finite differences (translation only), for completeness
        "diff_eef_xyz": world_p,
        "diff_joint_pos": joint[1:] - joint[:-1],
        # frame-aware eef deltas
        "world_eef_xyz": world_p,
        "world_eef_rot6d": align_tf.matrix_to_rotation_6d(world_R),
        "body_eef_xyz": body_p,
        "body_eef_rot6d": align_tf.matrix_to_rotation_6d(body_R),
        # commanded gripper target (the one command DROID provides), inverted to 1=open / 0=closed
        "gripper_state": invert_gripper_actions(trajectory["action_dict"]["gripper_position"])[:-1],
    }

    # Discard the last step (no padding): drop it from state and observation too so every field is
    # length T-1 and action[t] stays paired with obs[t].
    trajectory["state"] = {k: v[:-1] for k, v in state.items()}
    trajectory["observation"] = {k: v[:-1] for k, v in trajectory["observation"].items()}
    trajectory["action"] = action
    return trajectory


def droid_baseact_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    DROID dataset transformation for actions expressed in *base* frame of the robot.
    """

    def rand_swap_exterior_images(img1, img2):
        """
        Randomly swaps the two exterior images (for training with single exterior input).
        """
        return tf.cond(tf.random.uniform(shape=[]) > 0.5, lambda: (img1, img2), lambda: (img2, img1))

    trajectory["observation"]["exterior_image_1_left"], trajectory["observation"]["exterior_image_2_left"] = (
        rand_swap_exterior_images(
            trajectory["observation"]["exterior_image_1_left"],
            trajectory["observation"]["exterior_image_2_left"],
        )
    )
    return _droid_state_and_action(trajectory)


def droid_finetuning_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    DROID dataset transformation for actions expressed in *base* frame of the robot.
    """
    return _droid_state_and_action(trajectory)


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
    """Original BridgeData V2 (project website), standardized per design_of_state_and_action_space.md.

    Raw schema: observation.state [7] = eef xyz + extrinsic-XYZ rpy + gripper (1=open); action [7] =
    commanded delta + gripper. There are NO camera extrinsics in this dataset.

    Target e* selection (the doc's two modes): BridgeData V2's stored ``action`` is a *command* that
    does not match the achieved state difference, and the repo's canonical ``relabel_bridge_actions``
    already replaces it with the achieved-state finite difference. We follow that and use mode 2 --
    the ground-truth *next* pose as e*. Per mode 2 we do NOT pad: the last step has no successor, so it
    is DISCARDED (every field length T-1, action[t] = motion from obs[t] to obs[t+1]). The gripper
    action keeps the dataset's commanded gripper (binarized); state/action gripper are already
    1=open / 0=closed, so no inversion is applied.

    Output schema matches the fractal transform (state {eef_xyz, eef_rpy, eef_rot6d, gripper_state};
    world/body/diff eef action), differing only in mode (ground-truth next vs command).
    """
    obs = trajectory["observation"]
    eef_xyz = obs["state"][:, :3]
    eef_rpy = obs["state"][:, 3:6]

    # Bridge's native euler is extrinsic (fixed-axis) XYZ -- verified to match scipy 'xyz' (~1e-7)
    # and clearly reject intrinsic 'XYZ'. Build rotations with extrinsic=True for a faithful R.
    R = align_tf.rpy_to_matrix(eef_rpy, extrinsic=True)  # [T, 3, 3] world-frame orientations

    # --- State (world frame): eef_xyz, eef_rpy, eef_rot6d, gripper_state (already 1=open) ---
    state = {
        "eef_xyz": eef_xyz,
        "eef_rpy": eef_rpy,
        "gripper_state": obs["state"][:, 6:7],
        "eef_rot6d": align_tf.matrix_to_rotation_6d(R),
    }

    # --- Action: per-step delta from obs[t] (current pose e) to obs[t+1] (target pose e*) ---
    # world-frame delta: R_{e->e*}^w = R_{t+1} R_t^T, p = p_{t+1} - p_t
    world_R, world_p = align_tf.world_delta(R[:-1], eef_xyz[:-1], R[1:], eef_xyz[1:])
    # body-frame delta: R_{e->e*}^e = R_t^T R_{t+1}, p = R_t^T (p_{t+1} - p_t)
    body_R, body_p = align_tf.relative_pose(R[:-1], eef_xyz[:-1], R[1:], eef_xyz[1:])
    action = {
        # literal finite differences (translation / euler), for completeness
        "diff_eef_xyz": world_p,
        "diff_eef_rpy": eef_rpy[1:] - eef_rpy[:-1],
        # frame-aware eef deltas
        "world_eef_xyz": world_p,
        "world_eef_rot6d": align_tf.matrix_to_rotation_6d(world_R),
        "body_eef_xyz": body_p,
        "body_eef_rot6d": align_tf.matrix_to_rotation_6d(body_R),
        # commanded gripper target, binarized; already 1=open / 0=closed (no inversion)
        "gripper_state": binarize_gripper_actions(trajectory["action"][:, 6])[:-1, None],
    }

    # Discard the last step (no padding): drop it from state and observation too so every field is
    # length T-1 and action[t] stays paired with obs[t].
    trajectory["state"] = {k: v[:-1] for k, v in state.items()}
    trajectory["observation"] = {k: v[:-1] for k, v in trajectory["observation"].items()}
    trajectory["action"] = action
    # BridgeData V2 carries the instruction at step level (not under observation); truncate to match.
    trajectory["language_instruction"] = trajectory["language_instruction"][:-1]
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
    """RT-1 / fractal20220817_data, standardized per design_of_state_and_action_space.md.

    Google Robot, base(world)-frame EEF control, no joint sensing. The raw schema has NO camera
    extrinsics. ``base_pose_tool_reached`` is the achieved EEF pose as xyz + quaternion in
    (x, y, z, w) order (verified against the data). Unlike DROID, fractal ships genuine commands, so
    we use the doc's mode 1: the target e* is the commanded next pose, recovered from the raw action
    ``world_vector`` (commanded translation) + ``rotation_delta`` (commanded rpy rotation) -- both in
    the base/world frame (verified: world-frame R_{t+1} R_t^T matches rotation_delta). Every step
    carries a command, so nothing is discarded.
    """
    obs = trajectory["observation"]
    eef_xyz = obs["base_pose_tool_reached"][:, :3]
    eef_quat = obs["base_pose_tool_reached"][:, 3:7]  # (x, y, z, w)
    R = align_tf.quaternion_to_matrix(eef_quat)  # [T, 3, 3] world-frame orientations
    # store orientation as extrinsic-XYZ rpy (consistent with eef_rot6d and the rest of the repo)
    eef_rpy = align_tf.matrix_to_rpy(R, extrinsic=True)

    # --- State (world frame): eef_xyz, eef_rpy, eef_rot6d, gripper_state (no joints on this robot) ---
    state = {
        "eef_xyz": eef_xyz,
        "eef_rpy": eef_rpy,
        "gripper_state": invert_gripper_actions(obs["gripper_closed"]),
        "eef_rot6d": align_tf.matrix_to_rotation_6d(R),
    }

    # --- Action (mode 1, commanded next pose e*): world delta is the raw command ---
    #   p_{e*} = p_e + world_vector ; R_{e*} = dR_world @ R_e  (world-frame delta, left-multiply)
    act = trajectory["action"]
    dR_world = align_tf.rpy_to_matrix(act["rotation_delta"], extrinsic=True)
    p_estar = eef_xyz + act["world_vector"]
    R_estar = tf.matmul(dR_world, R)
    world_R, world_p = align_tf.world_delta(R, eef_xyz, R_estar, p_estar)  # -> (dR_world, world_vector)
    body_R, body_p = align_tf.relative_pose(R, eef_xyz, R_estar, p_estar)
    # literal componentwise rpy difference between the target and current orientation
    diff_eef_rpy = align_tf.matrix_to_rpy(R_estar, extrinsic=True) - eef_rpy
    # gripper: relative closedness command -> absolute (0=closed, 1=open), repo convention
    gripper_cmd = rel2abs_gripper_actions(act["gripper_closedness_action"][:, 0])[:, None]
    trajectory["action"] = {
        "diff_eef_xyz": world_p,  # literal world-frame translation (= world_eef_xyz under mode 1)
        "diff_eef_rpy": diff_eef_rpy,
        "world_eef_xyz": world_p,
        "world_eef_rot6d": align_tf.matrix_to_rotation_6d(world_R),
        "body_eef_xyz": body_p,
        "body_eef_rot6d": align_tf.matrix_to_rotation_6d(body_R),
        "gripper_state": gripper_cmd,
    }
    trajectory["state"] = state
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
    """CALVIN / taco_play (Franka), standardized per design_of_state_and_action_space.md.

    Raw schema: ``observation.robot_obs`` [15] = eef xyz (0:3) + eef rpy (3:6) + gripper width (6) +
    joint positions (7:14) + gripper action (14). Poses come from pybullet, whose euler is extrinsic
    (fixed-axis) XYZ. There are NO camera extrinsics.

    Target e* selection (the doc's two modes): taco_play ships an absolute pose command
    (``action.actions`` [0:6], in meters/radians) that differs materially from the achieved motion
    (~0.03 m commanded gap vs ~0.0035 m achieved per step), so a real command exists. But its joints
    have NO joint command, and the relative actions (``rel_actions_world`` / ``rel_actions_gripper``)
    are CALVIN's *scaled + clipped* normalized actions (~[-1, 1], ~40x the physical displacement),
    not usable as geometry. For a uniform, well-defined eef *and* joint delta we follow DROID and use
    mode 2 -- the ground-truth *next* pose as e*. Per mode 2 we do NOT pad: the last step has no
    successor, so it is DISCARDED (every field length T-1, action[t] = motion obs[t]->obs[t+1]). The
    gripper action keeps the dataset's commanded gripper (``action.actions[:, 6]``, 1=open / -1=closed
    -> clipped to 1=open / 0=closed).
    """
    obs = trajectory["observation"]
    robot_obs = obs["robot_obs"]
    eef_xyz = robot_obs[:, 0:3]
    eef_rpy = robot_obs[:, 3:6]
    joint = robot_obs[:, 7:14]
    lang = obs["natural_language_instruction"]

    # CALVIN/taco_play poses come from pybullet -> extrinsic (fixed-axis) XYZ euler.
    R = align_tf.rpy_to_matrix(eef_rpy, extrinsic=True)  # [T, 3, 3] world-frame orientations

    # --- State (world frame): eef_xyz, eef_rpy, eef_rot6d, joint_pos, gripper_state (gripper width) ---
    state = {
        "eef_xyz": eef_xyz,
        "eef_rpy": eef_rpy,
        "joint_pos": joint,
        "gripper_state": robot_obs[:, 6:7],  # sensed gripper width; larger = more open
        "eef_rot6d": align_tf.matrix_to_rotation_6d(R),
    }

    # --- Action: per-step delta from obs[t] (current pose e) to obs[t+1] (target pose e*) ---
    world_R, world_p = align_tf.world_delta(R[:-1], eef_xyz[:-1], R[1:], eef_xyz[1:])
    body_R, body_p = align_tf.relative_pose(R[:-1], eef_xyz[:-1], R[1:], eef_xyz[1:])
    action = {
        # literal finite differences (translation / euler / joints), for completeness
        "diff_eef_xyz": world_p,
        "diff_eef_rpy": eef_rpy[1:] - eef_rpy[:-1],
        "diff_joint_pos": joint[1:] - joint[:-1],
        # frame-aware eef deltas
        "world_eef_xyz": world_p,
        "world_eef_rot6d": align_tf.matrix_to_rotation_6d(world_R),
        "body_eef_xyz": body_p,
        "body_eef_rot6d": align_tf.matrix_to_rotation_6d(body_R),
        # commanded gripper target (absolute action), 1=open / -1=closed -> clip to 1=open / 0=closed
        "gripper_state": tf.clip_by_value(trajectory["action"]["actions"][:-1, 6:7], 0.0, 1.0),
    }

    # Discard the last step (no padding): drop it from state and observation too so every field is
    # length T-1 and action[t] stays paired with obs[t].
    trajectory["state"] = {k: v[:-1] for k, v in state.items()}
    trajectory["observation"] = {k: v[:-1] for k, v in trajectory["observation"].items()}
    trajectory["action"] = action
    trajectory["language_instruction"] = lang[:-1]
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
    """BC-Z, standardized per design_of_state_and_action_space.md.

    Google Robot, base(world)-frame EEF control, no joint sensing. The raw schema has NO camera
    extrinsics. The current EEF orientation ``present/axis_angle`` is an axis-angle (rotation vector)
    in the robot/base frame; ``present/xyz`` is the base-frame position.

    Target e* selection (the doc's two modes): although BC-Z exposes ``future/{xyz,axis_angle}_residual``
    (10 "future actions", each an additive delta to the current pose), the first block is NOT a
    single-step command -- empirically ``future/xyz_residual[:3]`` best matches the achieved
    displacement ~5-7 steps ahead (~0.033 m vs ~0.0035 m achieved per step, ~9x), i.e. it is a
    far-horizon target. Using it as the per-step e* would inflate every action ~9x and break
    state-action consistency. So we use mode 2 -- the ground-truth *next* pose as e* -- like DROID and
    bridge. Per mode 2 we do NOT pad: the last step has no successor, so it is DISCARDED (every field
    length T-1, action[t] = motion obs[t]->obs[t+1]). The gripper action keeps BC-Z's commanded
    immediate gripper target (``future/target_close[:, 0]``), which is a genuine 1-step binary command.
    """
    obs = trajectory["observation"]
    eef_xyz = obs["present/xyz"]  # [T, 3] base-frame position
    eef_aa = obs["present/axis_angle"]  # [T, 3] axis-angle (rotation vector)
    R = align_tf.axis_angle_to_matrix(eef_aa)  # [T, 3, 3] world-frame orientations
    # store orientation as extrinsic-XYZ rpy (consistent with eef_rot6d and the rest of the repo)
    eef_rpy = align_tf.matrix_to_rpy(R, extrinsic=True)
    lang = obs["natural_language_instruction"]

    # --- State (world frame): eef_xyz, eef_rpy, eef_rot6d, gripper_state (no joints on this robot) ---
    # present/sensed_close is continuous in [~0.2, 1] with 1 = fully closed -> invert to 1=open.
    state = {
        "eef_xyz": eef_xyz,
        "eef_rpy": eef_rpy,
        "gripper_state": invert_gripper_actions(obs["present/sensed_close"]),
        "eef_rot6d": align_tf.matrix_to_rotation_6d(R),
    }

    # --- Action: per-step delta from obs[t] (current pose e) to obs[t+1] (target pose e*) ---
    # world-frame delta: R_{e->e*}^w = R_{t+1} R_t^T, p = p_{t+1} - p_t
    world_R, world_p = align_tf.world_delta(R[:-1], eef_xyz[:-1], R[1:], eef_xyz[1:])
    # body-frame delta: R_{e->e*}^e = R_t^T R_{t+1}, p = R_t^T (p_{t+1} - p_t)
    body_R, body_p = align_tf.relative_pose(R[:-1], eef_xyz[:-1], R[1:], eef_xyz[1:])
    action = {
        # literal finite differences (translation / euler), for completeness
        "diff_eef_xyz": world_p,
        "diff_eef_rpy": eef_rpy[1:] - eef_rpy[:-1],
        # frame-aware eef deltas
        "world_eef_xyz": world_p,
        "world_eef_rot6d": align_tf.matrix_to_rotation_6d(world_R),
        "body_eef_xyz": body_p,
        "body_eef_rot6d": align_tf.matrix_to_rotation_6d(body_R),
        # commanded immediate gripper target, binary {0,1} with 1=closed -> invert to 1=open; discard last
        "gripper_state": invert_gripper_actions(tf.cast(trajectory["action"]["future/target_close"][:-1, :1], tf.float32)),
    }

    # Discard the last step (no padding): drop it from state and observation too so every field is
    # length T-1 and action[t] stays paired with obs[t].
    trajectory["state"] = {k: v[:-1] for k, v in state.items()}
    trajectory["observation"] = {k: v[:-1] for k, v in trajectory["observation"].items()}
    trajectory["action"] = action
    trajectory["language_instruction"] = lang[:-1]
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
    "droid": droid_baseact_transform,
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
    "droid_wipe": droid_finetuning_transform,
    ### LIBERO datasets (modified versions)
    "libero_spatial_no_noops": libero_dataset_transform,
    "libero_object_no_noops": libero_dataset_transform,
    "libero_goal_no_noops": libero_dataset_transform,
    "libero_10_no_noops": libero_dataset_transform,
}
