

STATE_NAMES = {
    "eef_xyz": ["x", "y", "z"],
    "eef_rpy": ["roll", "pitch", "yaw"],
    "eef_quat": ["x", "y", "z", "w"],
    "eef_rot6d": ["rot1", "rot2", "rot3", "rot4", "rot5", "rot6"],
    "joint_pos": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    "gripper_state": ["gripper"],
}


ACTION_NAMES = {
    "eef_xyz": ["x", "y", "z"],
    "eef_rpy": ["roll", "pitch", "yaw"],
    "eef_quat": ["x", "y", "z", "w"],
    "eef_rot6d": ["rot1", "rot2", "rot3", "rot4", "rot5", "rot6"],
    "diff_eef_xyz": ["x", "y", "z"],
    "diff_eef_rpy": ["roll", "pitch", "yaw"],
    "diff_joint_pos": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    "world_eef_xyz": ["x", "y", "z"],
    "world_eef_rpy": ["roll", "pitch", "yaw"],
    "world_eef_rot6d": ["rot1", "rot2", "rot3", "rot4", "rot5", "rot6"],
    "gripper_eef_xyz": ["x", "y", "z"],
    "gripper_eef_rot6d": ["rot1", "rot2", "rot3", "rot4", "rot5", "rot6"],
    "gripper_state": ["gripper"],
}

# Per design_of_state_and_action_space.md, the default action target e* is the ground-truth next pose.
# When a dataset also ships a meaningful pose *command*, the same delta fields are emitted a second time
# with a ``_command`` suffix (computed against the commanded e*). Register those variants automatically.
_COMMAND_ACTION_KEYS = (
    "diff_eef_xyz", "diff_eef_rpy", "diff_joint_pos",
    "world_eef_xyz", "world_eef_rpy", "world_eef_rot6d",
    "gripper_eef_xyz", "gripper_eef_rot6d",
)
ACTION_NAMES.update({f"{k}_command": ACTION_NAMES[k] for k in _COMMAND_ACTION_KEYS})