"""
Copied from https://github.com/openvla/openvla/blob/main/prismatic/vla/datasets/rlds/utils/data_utils.py
"""

from typing import Any, Dict

import tensorflow as tf


def binarize_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    """
    Converts gripper actions from continuous to binary values (0 and 1).

    We exploit that fact that most of the time, the gripper is fully open (near 1.0) or fully closed (near 0.0). As it
    transitions between the two, it sometimes passes through a few intermediate values. We relabel those intermediate
    values based on the state that is reached _after_ those intermediate values.

    In the edge case that the trajectory ends with an intermediate value, we give up on binarizing and relabel that
    chunk of intermediate values as the last action in the trajectory.

    The `scan_fn` implements the following logic:
        new_actions = np.empty_like(actions)
        carry = actions[-1]
        for i in reversed(range(actions.shape[0])):
            if in_between_mask[i]:
                carry = carry
            else:
                carry = float(open_mask[i])
            new_actions[i] = carry
    """
    open_mask, closed_mask = actions > 0.95, actions < 0.05
    in_between_mask = tf.logical_not(tf.logical_or(open_mask, closed_mask))
    is_open_float = tf.cast(open_mask, tf.float32)

    def scan_fn(carry, i):
        return tf.cond(in_between_mask[i], lambda: tf.cast(carry, tf.float32), lambda: is_open_float[i])

    return tf.scan(scan_fn, tf.range(tf.shape(actions)[0]), actions[-1], reverse=True)


def invert_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    return 1 - actions


def rel2abs_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    """
    Converts relative gripper actions (+1 for closing, -1 for opening) to absolute actions (0 = closed; 1 = open).

    Assumes that the first relative gripper is not redundant (i.e. close when already closed)!
    """
    # Note =>> -1 for closing, 1 for opening, 0 for no change
    opening_mask, closing_mask = actions < -0.1, actions > 0.1
    thresholded_actions = tf.where(opening_mask, 1, tf.where(closing_mask, -1, 0))

    def scan_fn(carry, i):
        return tf.cond(thresholded_actions[i] == 0, lambda: carry, lambda: thresholded_actions[i])

    # If no relative grasp, assumes open for whole trajectory
    start = -1 * thresholded_actions[tf.argmax(thresholded_actions != 0, axis=0)]
    start = tf.cond(start == 0, lambda: 1, lambda: start)

    # Note =>> -1 for closed, 1 for open
    new_actions = tf.scan(scan_fn, tf.range(tf.shape(actions)[0]), start)
    new_actions = tf.cast(new_actions, tf.float32) / 2 + 0.5

    return new_actions


# =================================================================================================
# Rotation representation conversions: {euler, quaternion, rotation_6d}
#
# All functions operate on *batched* tensors, converting along the last axis (arbitrary leading
# dims, e.g. a leading time dimension, are supported).
#
# WHY THESE CONVENTIONS:
# euler + rotation_6d follow pytorch3d, because the DROID dataset's own training pipeline
# (droid-dataset/droid_policy_learning, robomimic/scripts/conversion/convert_droid.py) defines its
# `*_rot_6d` action/state features via pytorch3d's `euler_angles_to_rot_6d(..., convention="XYZ")`.
# To make the 6D features we emit here numerically match DROID's published representation, we must
# use the *same* math:
#   - euler convention "XYZ" is *intrinsic* and composes R = Rx(rx) @ Ry(ry) @ Rz(rz).
#     (tensorflow_graphics instead composes R = Rz @ Ry @ Rx, i.e. the reversed order, which would
#      silently produce different 6D numbers for the same input angles.)
#   - rotation_6d is the first two *rows* of the matrix, flattened; the matrix is recovered by
#     Gram-Schmidt (Zhou et al. 2019), exactly as pytorch3d does.
# The QUATERNION is the one place we depart from pytorch3d: we store it scalar-LAST (x, y, z, w),
# NOT pytorch3d's scalar-first (w, x, y, z) -- see the note above quaternion_to_matrix for why
# (repo-wide / ROS / scipy / tensorflow_graphics convention). It is the same Hamilton quaternion.
# Everything below is pure TF so the module carries no extra dependency, and the
# euler<->matrix<->{quat,6d} round-trips have been numerically verified to machine precision.
#
# Conventions
# -----------
#   euler        : [..., 3] angles ``(rx, ry, rz)`` in radians; default convention "XYZ" (intrinsic),
#                  R = Rx(rx) @ Ry(ry) @ Rz(rz). Matches pytorch3d / DROID.
#   quaternion   : [..., 4] in ``(x, y, z, w)`` order (scalar-last), matching ROS / scipy / tfg.
#   rotation_6d  : [..., 6] the first two rows of the rotation matrix, flattened (Zhou et al. 2019).
# =================================================================================================
ROTATION_REPRESENTATIONS = ("euler", "quaternion", "rotation_6d")
_EULER_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def _stack_matrix(flat) -> tf.Tensor:
    """Stack a length-9 list of [...] tensors (row-major) into a [..., 3, 3] matrix."""
    rows = [tf.stack(flat[i : i + 3], axis=-1) for i in (0, 3, 6)]
    return tf.stack(rows, axis=-2)


# --- rotation matrix <-> 6D (Zhou et al. 2019, pytorch3d "rows" convention) ----------------------
def rotation_6d_to_matrix(d6: tf.Tensor) -> tf.Tensor:
    """Convert a 6D rotation representation [..., 6] to a rotation matrix [..., 3, 3] via Gram-Schmidt."""
    d6 = tf.cast(d6, tf.float32)
    a1, a2 = d6[..., 0:3], d6[..., 3:6]
    b1 = tf.math.l2_normalize(a1, axis=-1)
    # Make b2 orthogonal to b1 (subtract the projection), then normalize; b3 completes the frame.
    a2_proj = a2 - tf.reduce_sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = tf.math.l2_normalize(a2_proj, axis=-1)
    b3 = tf.linalg.cross(b1, b2)
    # The 6D vector holds the first two *rows*, so stack b1, b2, b3 as rows -> [..., 3, 3].
    return tf.stack((b1, b2, b3), axis=-2)


def matrix_to_rotation_6d(matrix: tf.Tensor) -> tf.Tensor:
    """Convert a rotation matrix [..., 3, 3] to the 6D representation [..., 6] (first two rows)."""
    matrix = tf.cast(matrix, tf.float32)
    return tf.concat((matrix[..., 0, :], matrix[..., 1, :]), axis=-1)


# --- rotation matrix <-> euler (pure-TF port of pytorch3d, default convention "XYZ") -------------
def _axis_angle_rotation(axis: str, angle: tf.Tensor) -> tf.Tensor:
    """Elementary right-handed rotation matrix [..., 3, 3] about a single named axis."""
    cos, sin = tf.cos(angle), tf.sin(angle)
    one, zero = tf.ones_like(angle), tf.zeros_like(angle)
    if axis == "X":
        flat = [one, zero, zero, zero, cos, -sin, zero, sin, cos]
    elif axis == "Y":
        flat = [cos, zero, sin, zero, one, zero, -sin, zero, cos]
    elif axis == "Z":
        flat = [cos, -sin, zero, sin, cos, zero, zero, zero, one]
    else:
        raise ValueError(f"Invalid rotation axis: {axis!r}")
    return _stack_matrix(flat)


def euler_to_matrix(euler: tf.Tensor, convention: str = "XYZ") -> tf.Tensor:
    """Convert euler angles [..., 3] to a rotation matrix [..., 3, 3] (intrinsic, e.g. R = Rx @ Ry @ Rz)."""
    euler = tf.cast(euler, tf.float32)
    angles = tf.unstack(euler, axis=-1)
    mats = [_axis_angle_rotation(axis, ang) for axis, ang in zip(convention, angles)]
    return tf.matmul(tf.matmul(mats[0], mats[1]), mats[2])


def _angle_from_tan(axis: str, other_axis: str, data: tf.Tensor, horizontal: bool, tait_bryan: bool) -> tf.Tensor:
    """pytorch3d helper: recover one euler angle from a row/column of the rotation matrix via atan2."""
    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ("XY", "YZ", "ZX")
    if horizontal == even:
        return tf.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return tf.atan2(-data[..., i2], data[..., i1])
    return tf.atan2(data[..., i2], -data[..., i1])


def matrix_to_euler(matrix: tf.Tensor, convention: str = "XYZ") -> tf.Tensor:
    """Convert a rotation matrix [..., 3, 3] to euler angles [..., 3] for the given convention."""
    matrix = tf.cast(matrix, tf.float32)
    i0 = _EULER_AXIS_INDEX[convention[0]]
    i2 = _EULER_AXIS_INDEX[convention[2]]
    tait_bryan = i0 != i2
    if tait_bryan:
        central = tf.asin(matrix[..., i0, i2] * (-1.0 if (i0 - i2) in (-1, 2) else 1.0))
    else:
        central = tf.acos(tf.clip_by_value(matrix[..., i0, i0], -1.0, 1.0))
    # NOTE: pytorch3d passes a *column* (matrix[..., i2]) for the first angle but a *row*
    # (matrix[..., i0, :]) for the third -- the asymmetry is load-bearing, do not "simplify" it.
    o0 = _angle_from_tan(convention[0], convention[1], matrix[..., :, i2], horizontal=False, tait_bryan=tait_bryan)
    o2 = _angle_from_tan(convention[2], convention[1], matrix[..., i0, :], horizontal=True, tait_bryan=tait_bryan)
    return tf.stack((o0, central, o2), axis=-1)


# --- rotation matrix <-> quaternion (Hamilton, scalar-LAST (x, y, z, w)) -------------------------
# NOTE: We use scalar-last (x, y, z, w), NOT pytorch3d's scalar-first (w, x, y, z). The math is the
# same Hamilton quaternion; only the storage order differs. We store xyzw because that is what the
# rest of this codebase and the surrounding ecosystem use: every other quaternion-handling transform
# here feeds tensorflow_graphics' ``tft.euler.from_quaternion`` (xyzw), and ROS / scipy default to
# xyzw as well. Caveat: the upstream OXE POS_QUAT datasets are not internally consistent -- e.g.
# maniskill stores tcp_pose as [x, y, z, qw, qx, qy, qz] (wxyz), while furniture_bench (Isaac Gym
# Preview) is xyzw -- so "consistent with all source datasets" is impossible; xyzw is the repo-wide
# convention we standardize on.
def quaternion_to_matrix(quaternion: tf.Tensor) -> tf.Tensor:
    """Convert a quaternion [..., 4] (x, y, z, w) to a rotation matrix [..., 3, 3]."""
    quaternion = tf.cast(quaternion, tf.float32)
    i, j, k, r = tf.unstack(quaternion, axis=-1)
    two_s = 2.0 / tf.reduce_sum(quaternion * quaternion, axis=-1)  # 2/|q|^2, so unnormalized q is fine
    flat = [
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ]
    return _stack_matrix(flat)


def matrix_to_quaternion(matrix: tf.Tensor) -> tf.Tensor:
    """Convert a rotation matrix [..., 3, 3] to a quaternion [..., 4] (x, y, z, w), standardized to w >= 0."""
    matrix = tf.cast(matrix, tf.float32)
    m00, m11, m22 = matrix[..., 0, 0], matrix[..., 1, 1], matrix[..., 2, 2]

    def _sqrt_pos(x):
        return tf.sqrt(tf.maximum(x, 0.0))

    w = _sqrt_pos(1.0 + m00 + m11 + m22) / 2.0
    x = _sqrt_pos(1.0 + m00 - m11 - m22) / 2.0
    y = _sqrt_pos(1.0 - m00 + m11 - m22) / 2.0
    z = _sqrt_pos(1.0 - m00 - m11 + m22) / 2.0
    # Recover signs from the off-diagonal terms (copysign), then standardize the global sign to w >= 0.
    x = tf.abs(x) * tf.sign(matrix[..., 2, 1] - matrix[..., 1, 2])
    y = tf.abs(y) * tf.sign(matrix[..., 0, 2] - matrix[..., 2, 0])
    z = tf.abs(z) * tf.sign(matrix[..., 1, 0] - matrix[..., 0, 1])
    quat = tf.stack((x, y, z, w), axis=-1)  # scalar-last
    return tf.where(quat[..., 3:4] < 0, -quat, quat)


# --- direct cross conversions --------------------------------------------------------------------
def euler_to_quaternion(euler: tf.Tensor) -> tf.Tensor:
    """Convert euler angles [..., 3] to a quaternion [..., 4] (x, y, z, w)."""
    return matrix_to_quaternion(euler_to_matrix(euler))


def quaternion_to_euler(quaternion: tf.Tensor) -> tf.Tensor:
    """Convert a quaternion [..., 4] (x, y, z, w) to euler angles [..., 3]."""
    return matrix_to_euler(quaternion_to_matrix(quaternion))


def euler_to_rotation_6d(euler: tf.Tensor) -> tf.Tensor:
    """Convert euler angles [..., 3] to the 6D rotation representation [..., 6]."""
    return matrix_to_rotation_6d(euler_to_matrix(euler))


def rotation_6d_to_euler(d6: tf.Tensor) -> tf.Tensor:
    """Convert the 6D rotation representation [..., 6] to euler angles [..., 3]."""
    return matrix_to_euler(rotation_6d_to_matrix(d6))


def quaternion_to_rotation_6d(quaternion: tf.Tensor) -> tf.Tensor:
    """Convert a quaternion [..., 4] (x, y, z, w) to the 6D rotation representation [..., 6]."""
    return matrix_to_rotation_6d(quaternion_to_matrix(quaternion))


def rotation_6d_to_quaternion(d6: tf.Tensor) -> tf.Tensor:
    """Convert the 6D rotation representation [..., 6] to a quaternion [..., 4] (x, y, z, w)."""
    return matrix_to_quaternion(rotation_6d_to_matrix(d6))


# --- generic dispatcher --------------------------------------------------------------------------
# (from_rep, to_rep) -> conversion fn. Identity conversions return the input unchanged.
_ROTATION_CONVERTERS = {
    ("euler", "quaternion"): euler_to_quaternion,
    ("euler", "rotation_6d"): euler_to_rotation_6d,
    ("quaternion", "euler"): quaternion_to_euler,
    ("quaternion", "rotation_6d"): quaternion_to_rotation_6d,
    ("rotation_6d", "euler"): rotation_6d_to_euler,
    ("rotation_6d", "quaternion"): rotation_6d_to_quaternion,
}


def convert_rotation(rotation: tf.Tensor, from_rep: str, to_rep: str) -> tf.Tensor:
    """Convert ``rotation`` from ``from_rep`` to ``to_rep``.

    Args:
        rotation: batched rotation tensor whose last axis matches ``from_rep``
            (euler: 3, quaternion: 4, rotation_6d: 6).
        from_rep: one of ``{"euler", "quaternion", "rotation_6d"}``.
        to_rep: one of ``{"euler", "quaternion", "rotation_6d"}``.

    Returns:
        The rotation expressed in ``to_rep``.
    """
    for name, rep in (("from_rep", from_rep), ("to_rep", to_rep)):
        if rep not in ROTATION_REPRESENTATIONS:
            raise ValueError(f"{name}={rep!r} must be one of {ROTATION_REPRESENTATIONS}")
    if from_rep == to_rep:
        return tf.cast(rotation, tf.float32)
    return _ROTATION_CONVERTERS[(from_rep, to_rep)](rotation)


# === Bridge-V2 =>> Dataset-Specific Transform ===
def relabel_bridge_actions(traj: Dict[str, Any]) -> Dict[str, Any]:
    """Relabels actions to use reached proprioceptive state; discards last timestep (no-action)."""
    movement_actions = traj["observation"]["state"][1:, :6] - traj["observation"]["state"][:-1, :6]
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], traj)
    traj_truncated["action"] = tf.concat([movement_actions, traj["action"][:-1, -1:]], axis=1)

    return traj_truncated
