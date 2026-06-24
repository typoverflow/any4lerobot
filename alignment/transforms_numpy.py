"""State / action space conversions -- NumPy backend.

Self-contained NumPy implementation of the rotation-representation conversions,
SE(3) frame math, and model-output -> controller-input deployment helpers
described in ``design_of_state_and_action_space.md``. The Torch and TensorFlow
siblings (``transforms_torch.py`` / ``transforms_tf.py``) expose the *same*
function names, signatures, and conventions -- pick the file for your framework.

All functions are batched on the *last* axis (rotations) or last two axes
(matrices); arbitrary leading dims (e.g. a time axis [T, ...]) pass through.

Conventions
-----------
  euler       : [..., 3] (rx, ry, rz) radians, convention "XYZ" = *intrinsic*,
                R = Rx(rx) @ Ry(ry) @ Rz(rz). Matches pytorch3d / DROID's
                published ``*_rot_6d`` features.
  rpy         : [..., 3] (roll, pitch, yaw) about (X, Y, Z). ``extrinsic=False``
                composes intrinsic XYZ (= euler above); ``extrinsic=True`` composes
                fixed-axis XYZ = Rz(yaw) @ Ry(pitch) @ Rx(roll) (ROS tf / scipy
                'xyz' / pybullet / xArm). MUST match how the source made its rpy.
  quaternion  : [..., 4] scalar-LAST (x, y, z, w) -- ROS / scipy / tfg order.
  rotation_6d : [..., 6] first two *rows* of R, flattened (Zhou et al. 2019),
                recovered by Gram-Schmidt.
  frame names : R_a^b ("R_a_to_b") is the orientation of frame a expressed in b;
                a vector converts as v^b = R_a^b v^a. Subscript e = end-effector
                (body) frame, w = world, c = the frame the model was trained in.
"""

from __future__ import annotations

import numpy as np

ROTATION_REPRESENTATIONS = ("euler", "quaternion", "rotation_6d", "matrix", "axis_angle")
_EULER_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def _f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _matvec(R: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply rotation matrix [..., 3, 3] to vector [..., 3] -> [..., 3]."""
    return np.matmul(R, v[..., None])[..., 0]


# --- rotation matrix <-> 6D (Zhou et al. 2019, "rows" convention) --------------------------------
def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """6D rotation [..., 6] -> rotation matrix [..., 3, 3] via Gram-Schmidt."""
    d6 = _f32(d6)
    a1, a2 = d6[..., 0:3], d6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2_proj = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2_proj / np.linalg.norm(a2_proj, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """Rotation matrix [..., 3, 3] -> 6D [..., 6] (first two rows)."""
    matrix = _f32(matrix)
    return np.concatenate((matrix[..., 0, :], matrix[..., 1, :]), axis=-1)


# --- rotation matrix <-> euler (port of pytorch3d, default convention "XYZ") ----------------------
def _axis_angle_rotation(axis: str, angle: np.ndarray) -> np.ndarray:
    cos, sin = np.cos(angle), np.sin(angle)
    one, zero = np.ones_like(angle), np.zeros_like(angle)
    if axis == "X":
        flat = [one, zero, zero, zero, cos, -sin, zero, sin, cos]
    elif axis == "Y":
        flat = [cos, zero, sin, zero, one, zero, -sin, zero, cos]
    elif axis == "Z":
        flat = [cos, -sin, zero, sin, cos, zero, zero, zero, one]
    else:
        raise ValueError(f"Invalid rotation axis: {axis!r}")
    rows = [np.stack(flat[i : i + 3], axis=-1) for i in (0, 3, 6)]
    return np.stack(rows, axis=-2)


def euler_to_matrix(euler: np.ndarray, convention: str = "XYZ") -> np.ndarray:
    """Euler angles [..., 3] -> rotation matrix [..., 3, 3] (intrinsic, R = Ra @ Rb @ Rc)."""
    euler = _f32(euler)
    mats = [_axis_angle_rotation(axis, euler[..., i]) for i, axis in enumerate(convention)]
    return np.matmul(np.matmul(mats[0], mats[1]), mats[2])


def _angle_from_tan(axis: str, other_axis: str, data: np.ndarray, horizontal: bool, tait_bryan: bool):
    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ("XY", "YZ", "ZX")
    if horizontal == even:
        return np.arctan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return np.arctan2(-data[..., i2], data[..., i1])
    return np.arctan2(data[..., i2], -data[..., i1])


def matrix_to_euler(matrix: np.ndarray, convention: str = "XYZ") -> np.ndarray:
    """Rotation matrix [..., 3, 3] -> euler angles [..., 3] for ``convention``."""
    matrix = _f32(matrix)
    i0, i2 = _EULER_AXIS_INDEX[convention[0]], _EULER_AXIS_INDEX[convention[2]]
    tait_bryan = i0 != i2
    if tait_bryan:
        central = np.arcsin(matrix[..., i0, i2] * (-1.0 if (i0 - i2) in (-1, 2) else 1.0))
    else:
        central = np.arccos(np.clip(matrix[..., i0, i0], -1.0, 1.0))
    # pytorch3d asymmetry (column for first angle, row for third) is load-bearing.
    o0 = _angle_from_tan(convention[0], convention[1], matrix[..., :, i2], horizontal=False, tait_bryan=tait_bryan)
    o2 = _angle_from_tan(convention[2], convention[1], matrix[..., i0, :], horizontal=True, tait_bryan=tait_bryan)
    return np.stack((o0, central, o2), axis=-1)


# --- rotation matrix <-> quaternion (Hamilton, scalar-LAST (x, y, z, w)) --------------------------
def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Quaternion [..., 4] (x, y, z, w) -> rotation matrix [..., 3, 3]."""
    quaternion = _f32(quaternion)
    i, j, k, r = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    two_s = 2.0 / np.sum(quaternion * quaternion, axis=-1)  # 2/|q|^2; unnormalized q is fine
    flat = [
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ]
    rows = [np.stack(flat[m : m + 3], axis=-1) for m in (0, 3, 6)]
    return np.stack(rows, axis=-2)


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Rotation matrix [..., 3, 3] -> quaternion [..., 4] (x, y, z, w), standardized to w >= 0."""
    matrix = _f32(matrix)
    m00, m11, m22 = matrix[..., 0, 0], matrix[..., 1, 1], matrix[..., 2, 2]

    def _sqrt_pos(x):
        return np.sqrt(np.maximum(x, 0.0))

    w = _sqrt_pos(1.0 + m00 + m11 + m22) / 2.0
    x = _sqrt_pos(1.0 + m00 - m11 - m22) / 2.0
    y = _sqrt_pos(1.0 - m00 + m11 - m22) / 2.0
    z = _sqrt_pos(1.0 - m00 - m11 + m22) / 2.0
    x = np.abs(x) * np.sign(matrix[..., 2, 1] - matrix[..., 1, 2])
    y = np.abs(y) * np.sign(matrix[..., 0, 2] - matrix[..., 2, 0])
    z = np.abs(z) * np.sign(matrix[..., 1, 0] - matrix[..., 0, 1])
    quat = np.stack((x, y, z, w), axis=-1)  # scalar-last
    return np.where(quat[..., 3:4] < 0, -quat, quat)


# --- rotation matrix <-> axis-angle (rotation vector v = axis * angle, ||v|| = angle in rad) -----
def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Axis-angle / rotation vector [..., 3] -> rotation matrix [..., 3, 3] (Rodrigues' formula)."""
    axis_angle = _f32(axis_angle)
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)  # [..., 1]; ||v|| = rotation angle
    axis = axis_angle / np.maximum(angle, 1e-8)  # safe at angle=0: axis->0, R->I below
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    a = angle[..., 0]
    s, c = np.sin(a), np.cos(a)
    C = 1.0 - c
    flat = [
        c + x * x * C, x * y * C - z * s, x * z * C + y * s,
        y * x * C + z * s, c + y * y * C, y * z * C - x * s,
        z * x * C - y * s, z * y * C + x * s, c + z * z * C,
    ]
    rows = [np.stack(flat[m : m + 3], axis=-1) for m in (0, 3, 6)]
    return np.stack(rows, axis=-2)


def matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
    """Rotation matrix [..., 3, 3] -> axis-angle / rotation vector [..., 3] (via quaternion)."""
    quat = matrix_to_quaternion(matrix)  # (x, y, z, w), w >= 0
    xyz = quat[..., :3]
    w = quat[..., 3:4]
    sin_half = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sin_half, w)
    # rotvec = (xyz / sin_half) * angle; at angle->0, angle/sin_half -> 2 (Taylor), so scale -> 2
    scale = np.where(sin_half < 1e-8, 2.0, angle / np.maximum(sin_half, 1e-8))
    return xyz * scale


def rpy_to_matrix(rpy: np.ndarray, extrinsic: bool = False) -> np.ndarray:
    """RPY angles [..., 3] = (roll, pitch, yaw) -> rotation matrix [..., 3, 3].

    ``extrinsic=False`` (default): intrinsic XYZ, R = Rx(roll) @ Ry(pitch) @ Rz(yaw)
    (pytorch3d / DROID ``*_rot_6d``). ``extrinsic=True``: fixed-axis XYZ,
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (ROS tf / scipy 'xyz' / pybullet / xArm),
    which equals intrinsic ZYX on the reversed angle order.
    """
    rpy = _f32(rpy)
    if extrinsic:
        return euler_to_matrix(rpy[..., ::-1], convention="ZYX")
    return euler_to_matrix(rpy, convention="XYZ")


def matrix_to_rpy(matrix: np.ndarray, extrinsic: bool = False) -> np.ndarray:
    """Inverse of ``rpy_to_matrix``: rotation matrix [..., 3, 3] -> RPY [..., 3] = (roll, pitch, yaw).

    ``extrinsic`` must match the convention the angles are meant to compose under (see
    ``rpy_to_matrix``): ``False`` = intrinsic XYZ, ``True`` = extrinsic (fixed-axis) XYZ.
    """
    if extrinsic:
        return matrix_to_euler(matrix, convention="ZYX")[..., ::-1]
    return matrix_to_euler(matrix, convention="XYZ")


# --- generic rep <-> matrix + dispatcher ---------------------------------------------------------
def to_matrix(rotation: np.ndarray, rep: str, *, convention: str = "XYZ", extrinsic: bool = False) -> np.ndarray:
    """Convert any supported rotation representation to a rotation matrix [..., 3, 3]."""
    if rep == "matrix":
        return _f32(rotation)
    if rep == "euler":
        return euler_to_matrix(rotation, convention=convention)
    if rep == "rpy":
        return rpy_to_matrix(rotation, extrinsic=extrinsic)
    if rep == "quaternion":
        return quaternion_to_matrix(rotation)
    if rep == "rotation_6d":
        return rotation_6d_to_matrix(rotation)
    if rep == "axis_angle":
        return axis_angle_to_matrix(rotation)
    raise ValueError(f"rep={rep!r} must be one of {ROTATION_REPRESENTATIONS} (or 'rpy')")


def from_matrix(matrix: np.ndarray, rep: str, *, convention: str = "XYZ") -> np.ndarray:
    """Convert a rotation matrix [..., 3, 3] to the requested representation."""
    if rep == "matrix":
        return _f32(matrix)
    if rep == "euler":
        return matrix_to_euler(matrix, convention=convention)
    if rep == "quaternion":
        return matrix_to_quaternion(matrix)
    if rep == "rotation_6d":
        return matrix_to_rotation_6d(matrix)
    if rep == "axis_angle":
        return matrix_to_axis_angle(matrix)
    raise ValueError(f"rep={rep!r} must be one of {ROTATION_REPRESENTATIONS}")


def convert_rotation(rotation: np.ndarray, from_rep: str, to_rep: str) -> np.ndarray:
    """Convert ``rotation`` between any two representations (via rotation matrix)."""
    return from_matrix(to_matrix(rotation, from_rep), to_rep)


# === SE(3) frame math (see design_of_state_and_action_space.md) ==================================
def relative_pose(R_cur: np.ndarray, p_cur: np.ndarray, R_tgt: np.ndarray, p_tgt: np.ndarray):
    """Body-frame delta from current pose e to target pose e* (the stored representation).

    Returns (R_delta_body, p_delta_body), i.e. the rotation/translation of the relative
    transform T_{e*}^e = (T_e^w)^{-1} T_{e*}^w:
        R_{e->e*}^e = (R_e^w)^T R_{e*}^w
        p_{e->e*}^e = (R_e^w)^T (p_{e*}^w - p_e^w)
    All inputs are world-frame (R [..., 3, 3], p [..., 3]).
    """
    R_cur = _f32(R_cur)
    Rc_T = np.swapaxes(R_cur, -1, -2)
    R_delta = np.matmul(Rc_T, _f32(R_tgt))
    p_delta = _matvec(Rc_T, _f32(p_tgt) - _f32(p_cur))
    return R_delta, p_delta


def world_delta(R_cur: np.ndarray, p_cur: np.ndarray, R_tgt: np.ndarray, p_tgt: np.ndarray):
    """World-frame delta from current pose e to target pose e*.

    Returns (R_delta_world, p_delta_world):
        R_{e->e*}^w = R_{e*}^w (R_e^w)^T
        p_{e->e*}^w = p_{e*}^w - p_e^w
    """
    R_cur, R_tgt = _f32(R_cur), _f32(R_tgt)
    R_delta = np.matmul(R_tgt, np.swapaxes(R_cur, -1, -2))
    p_delta = _f32(p_tgt) - _f32(p_cur)
    return R_delta, p_delta


def change_delta_frame(R_delta: np.ndarray, p_delta: np.ndarray, R_src_to_dst: np.ndarray):
    """Re-express a delta (rotation + translation) from frame ``src`` into frame ``dst``.

    ``R_src_to_dst`` = R_src^dst. Applies the conjugation / rotation rules from the doc:
        R_delta^dst = R_src^dst R_delta^src (R_src^dst)^T
        p_delta^dst = R_src^dst p_delta^src
    With src=e, dst=c this is exactly the load-time body->frame-c conversion.
    """
    Rsd = _f32(R_src_to_dst)
    R_out = np.matmul(np.matmul(Rsd, _f32(R_delta)), np.swapaxes(Rsd, -1, -2))
    p_out = _matvec(Rsd, _f32(p_delta))
    return R_out, p_out


# === Deployment: model output (frame c) -> controller input =====================================
def world_pose_from_model_delta(R_delta_c, p_delta_c, R_e_w, p_e_w, R_c_w):
    """Case 1: controller wants the absolute target pose in the WORLD frame.

    The model emits a delta in frame c: (R_{e->e*}^c, p_{e->e*}^c). Given the current
    EEF pose (R_e^w, p_e^w) and the orientation of frame c in world R_c^w, returns the
    absolute target (R_{e*}^w, p_{e*}^w). For a model trained in the world frame pass
    R_c_w = I; for one trained in the body frame pass R_c_w = R_e^w.
    """
    R_delta_w, p_delta_w = change_delta_frame(R_delta_c, p_delta_c, R_c_w)  # src=c -> dst=w
    R_tgt = np.matmul(R_delta_w, _f32(R_e_w))
    p_tgt = _f32(p_e_w) + p_delta_w
    return R_tgt, p_tgt


def body_delta_from_model_delta(R_delta_c, p_delta_c, R_e_w, R_c_w):
    """Case 2: controller wants the relative motion in the BODY (gripper) frame e.

    Converts the model's frame-c delta into the body frame, returning
    (R_{e->e*}^e, p_{e->e*}^e). Needs only the current EEF orientation R_e^w and the
    frame-c orientation R_c^w, via R_c^e = (R_e^w)^T R_c^w.
    """
    R_c_e = np.matmul(np.swapaxes(_f32(R_e_w), -1, -2), _f32(R_c_w))  # R_c^e = R_w^e R_c^w
    return change_delta_frame(R_delta_c, p_delta_c, R_c_e)
