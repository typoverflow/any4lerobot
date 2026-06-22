

STATE_NAMES = {
    "eef_xyz": ["x", "y", "z"],
    "eef_rpy": ["roll", "pitch", "yaw"],
    "eef_quat": ["x", "y", "z", "w"],
    "eef_rot6d": ["rot1", "rot2", "rot3", "rot4", "rot5", "rot6"],
    "joint_position": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    "gripper_state": ["gripper"],
}


ACTION_NAMES = {
    "eef_xyz": ["x", "y", "z"],
    "eef_rpy": ["roll", "pitch", "yaw"],
    "eef_quat": ["x", "y", "z", "w"],
    "eef_rot6d": ["rot1", "rot2", "rot3", "rot4", "rot5", "rot6"],
    "joint_position": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    "world_eef_xyz": ["x", "y", "z"],
    "world_eef_rpy": ["roll", "pitch", "yaw"],
    "world_eef_rot6d": ["rot1", "rot2", "rot3", "rot4", "rot5", "rot6"],
    "body_eef_xyz": ["x", "y", "z"],
    "body_eef_rot6d": ["rot1", "rot2", "rot3", "rot4", "rot5", "rot6"],
    "gripper_state": ["gripper"],
    "command_eef_xyz": ["x", "y", "z"],
    "command_eef_rpy": ["roll", "pitch", "yaw"],
    "command_eef_quat": ["x", "y", "z", "w"],
    "command_eef_rot6d": ["rot1", "rot2", "rot3", "rot4", "rot5", "rot6"],
    "command_joint_position": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    "command_gripper_state": ["gripper"],
}