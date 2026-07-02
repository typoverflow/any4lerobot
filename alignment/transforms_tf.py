"""State / action space conversions -- TensorFlow backend.

Self-contained pure-TF implementation of the rotation-representation conversions,
SE(3) frame math, and model-output -> controller-input deployment helpers
described in ``design_of_state_and_action_space.md``. Mirrors the NumPy and Torch
siblings (``transforms_numpy.py`` / ``transforms_torch.py``) exactly -- same
function names, signatures, and conventions. Pure TF, so it runs inside
``tf.data`` graphs with no extra dependency (this is what ``openx2lerobot`` uses).

All functions are batched on the *last* axis (rotations) or last two axes
(matrices); arbitrary leading dims (e.g. a time axis [T, ...]) pass through.

Conventions
-----------
  euler       : [..., 3] (rx, ry, rz) radians, convention "XYZ" = *intrinsic*,
                R = Rx(rx) @ Ry(ry) @ Rz(rz). Matches pytorch3d / DROID.
  rpy         : [..., 3] (roll, pitch, yaw) about (X, Y, Z). ``extrinsic=False``
                composes intrinsic XYZ; ``extrinsic=True`` composes fixed-axis XYZ
                = Rz(yaw) @ Ry(pitch) @ Rx(roll) (ROS tf / scipy 'xyz' / pybullet).
  quaternion  : [..., 4] scalar-LAST (x, y, z, w).
  rotation_6d : [..., 6] first two *rows* of R, flattened (Zhou et al. 2019).
  frame names : R_a^b ("R_a_to_b") is the orientation of frame a expressed in b;
                v^b = R_a^b v^a. e = end-effector (body), w = world, c = the frame
                the model was trained in.
"""

from __future__ import annotations

import tensorflow as tf

ROTATION_REPRESENTATIONS = ("euler", "quaternion", "rotation_6d", "matrix", "axis_angle")
_EULER_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def _f32(x) -> tf.Tensor:
    return tf.cast(x, tf.float32)


def _stack_matrix(flat) -> tf.Tensor:
    """Stack a length-9 list of [...] tensors (row-major) into a [..., 3, 3] matrix."""
    rows = [tf.stack(flat[i : i + 3], axis=-1) for i in (0, 3, 6)]
    return tf.stack(rows, axis=-2)


def _matvec(R: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
    """Apply rotation matrix [..., 3, 3] to vector [..., 3] -> [..., 3]."""
    return tf.squeeze(tf.matmul(R, v[..., None]), axis=-1)


# --- rotation matrix <-> 6D (Zhou et al. 2019, "rows" convention) --------------------------------
def rotation_6d_to_matrix(d6: tf.Tensor) -> tf.Tensor:
    """6D rotation [..., 6] -> rotation matrix [..., 3, 3] via Gram-Schmidt."""
    d6 = _f32(d6)
    a1, a2 = d6[..., 0:3], d6[..., 3:6]
    b1 = tf.math.l2_normalize(a1, axis=-1)
    a2_proj = a2 - tf.reduce_sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = tf.math.l2_normalize(a2_proj, axis=-1)
    b3 = tf.linalg.cross(b1, b2)
    return tf.stack((b1, b2, b3), axis=-2)


def matrix_to_rotation_6d(matrix: tf.Tensor) -> tf.Tensor:
    """Rotation matrix [..., 3, 3] -> 6D [..., 6] (first two rows)."""
    matrix = _f32(matrix)
    return tf.concat((matrix[..., 0, :], matrix[..., 1, :]), axis=-1)


# --- rotation matrix <-> euler (port of pytorch3d, default convention "XYZ") ----------------------
def _axis_angle_rotation(axis: str, angle: tf.Tensor) -> tf.Tensor:
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
    """Euler angles [..., 3] -> rotation matrix [..., 3, 3] (intrinsic, R = Ra @ Rb @ Rc)."""
    euler = _f32(euler)
    mats = [_axis_angle_rotation(axis, euler[..., i]) for i, axis in enumerate(convention)]
    return tf.matmul(tf.matmul(mats[0], mats[1]), mats[2])


def _angle_from_tan(axis: str, other_axis: str, data: tf.Tensor, horizontal: bool, tait_bryan: bool):
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
    """Rotation matrix [..., 3, 3] -> euler angles [..., 3] for ``convention``."""
    matrix = _f32(matrix)
    i0, i2 = _EULER_AXIS_INDEX[convention[0]], _EULER_AXIS_INDEX[convention[2]]
    tait_bryan = i0 != i2
    if tait_bryan:
        central = tf.asin(matrix[..., i0, i2] * (-1.0 if (i0 - i2) in (-1, 2) else 1.0))
    else:
        central = tf.acos(tf.clip_by_value(matrix[..., i0, i0], -1.0, 1.0))
    # pytorch3d asymmetry (column for first angle, row for third) is load-bearing.
    o0 = _angle_from_tan(convention[0], convention[1], matrix[..., :, i2], horizontal=False, tait_bryan=tait_bryan)
    o2 = _angle_from_tan(convention[2], convention[1], matrix[..., i0, :], horizontal=True, tait_bryan=tait_bryan)
    return tf.stack((o0, central, o2), axis=-1)


# --- rotation matrix <-> quaternion (Hamilton, scalar-LAST (x, y, z, w)) --------------------------
def quaternion_to_matrix(quaternion: tf.Tensor) -> tf.Tensor:
    """Quaternion [..., 4] (x, y, z, w) -> rotation matrix [..., 3, 3]."""
    quaternion = _f32(quaternion)
    i, j, k, r = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    two_s = 2.0 / tf.reduce_sum(quaternion * quaternion, axis=-1)  # 2/|q|^2; unnormalized q is fine
    flat = [
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ]
    return _stack_matrix(flat)


def matrix_to_quaternion(matrix: tf.Tensor) -> tf.Tensor:
    """Rotation matrix [..., 3, 3] -> quaternion [..., 4] (x, y, z, w), standardized to w >= 0."""
    matrix = _f32(matrix)
    m00, m11, m22 = matrix[..., 0, 0], matrix[..., 1, 1], matrix[..., 2, 2]

    def _sqrt_pos(x):
        return tf.sqrt(tf.maximum(x, 0.0))

    w = _sqrt_pos(1.0 + m00 + m11 + m22) / 2.0
    x = _sqrt_pos(1.0 + m00 - m11 - m22) / 2.0
    y = _sqrt_pos(1.0 - m00 + m11 - m22) / 2.0
    z = _sqrt_pos(1.0 - m00 - m11 + m22) / 2.0
    x = tf.abs(x) * tf.sign(matrix[..., 2, 1] - matrix[..., 1, 2])
    y = tf.abs(y) * tf.sign(matrix[..., 0, 2] - matrix[..., 2, 0])
    z = tf.abs(z) * tf.sign(matrix[..., 1, 0] - matrix[..., 0, 1])
    quat = tf.stack((x, y, z, w), axis=-1)  # scalar-last
    return tf.where(quat[..., 3:4] < 0, -quat, quat)


# --- rotation matrix <-> axis-angle (rotation vector v = axis * angle, ||v|| = angle in rad) -----
def axis_angle_to_matrix(axis_angle: tf.Tensor) -> tf.Tensor:
    """Axis-angle / rotation vector [..., 3] -> rotation matrix [..., 3, 3] (Rodrigues' formula)."""
    axis_angle = _f32(axis_angle)
    angle = tf.norm(axis_angle, axis=-1, keepdims=True)  # [..., 1]; ||v|| = rotation angle
    axis = axis_angle / tf.maximum(angle, 1e-8)  # safe at angle=0: axis->0, R->I below
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    a = angle[..., 0]
    s, c = tf.sin(a), tf.cos(a)
    C = 1.0 - c
    flat = [
        c + x * x * C, x * y * C - z * s, x * z * C + y * s,
        y * x * C + z * s, c + y * y * C, y * z * C - x * s,
        z * x * C - y * s, z * y * C + x * s, c + z * z * C,
    ]
    return _stack_matrix(flat)


def matrix_to_axis_angle(matrix: tf.Tensor) -> tf.Tensor:
    """Rotation matrix [..., 3, 3] -> axis-angle / rotation vector [..., 3] (via quaternion)."""
    quat = matrix_to_quaternion(matrix)  # (x, y, z, w), w >= 0
    xyz = quat[..., :3]
    w = quat[..., 3:4]
    sin_half = tf.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * tf.atan2(sin_half, w)
    # rotvec = (xyz / sin_half) * angle; at angle->0, angle/sin_half -> 2 (Taylor), so scale -> 2
    scale = tf.where(sin_half < 1e-8, tf.fill(tf.shape(sin_half), 2.0), angle / tf.maximum(sin_half, 1e-8))
    return xyz * scale


def rpy_to_matrix(rpy: tf.Tensor, extrinsic: bool = False) -> tf.Tensor:
    """RPY angles [..., 3] = (roll, pitch, yaw) -> rotation matrix [..., 3, 3].

    ``extrinsic=False`` (default): intrinsic XYZ, R = Rx(roll) @ Ry(pitch) @ Rz(yaw)
    (pytorch3d / DROID). ``extrinsic=True``: fixed-axis XYZ,
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (ROS tf / scipy 'xyz' / pybullet / xArm),
    which equals intrinsic ZYX on the reversed angle order.
    """
    rpy = _f32(rpy)
    if extrinsic:
        return euler_to_matrix(rpy[..., ::-1], convention="ZYX")
    return euler_to_matrix(rpy, convention="XYZ")


def matrix_to_rpy(matrix: tf.Tensor, extrinsic: bool = False) -> tf.Tensor:
    """Inverse of ``rpy_to_matrix``: rotation matrix [..., 3, 3] -> RPY [..., 3] = (roll, pitch, yaw).

    ``extrinsic`` must match the convention the angles are meant to compose under (see
    ``rpy_to_matrix``): ``False`` = intrinsic XYZ, ``True`` = extrinsic (fixed-axis) XYZ.
    """
    if extrinsic:
        return matrix_to_euler(matrix, convention="ZYX")[..., ::-1]
    return matrix_to_euler(matrix, convention="XYZ")


# --- generic rep <-> matrix + dispatcher ---------------------------------------------------------
def to_matrix(rotation: tf.Tensor, rep: str, *, convention: str = "XYZ", extrinsic: bool = False) -> tf.Tensor:
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


def from_matrix(matrix: tf.Tensor, rep: str, *, convention: str = "XYZ") -> tf.Tensor:
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


def convert_rotation(rotation: tf.Tensor, from_rep: str, to_rep: str) -> tf.Tensor:
    """Convert ``rotation`` between any two representations (via rotation matrix)."""
    return from_matrix(to_matrix(rotation, from_rep), to_rep)


# === Axis-orientation alignment (native frames -> canonical; see design doc: Preprocessing) =======
_AXIS_VECTORS = {
    "x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0),
    "z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0),
}


def _axis_alignment_rows(x_to: str, y_to: str, z_to: str):
    """Validate a signed-axis spec and return the alignment matrix as row-major python lists.

    Each of ``x_to``/``y_to``/``z_to`` names where the SOURCE (native) x/y/z axis points in the
    DESTINATION (canonical) frame -- one of 'x','-x','y','-y','z','-z'. Those three unit vectors are
    the COLUMNS of the returned matrix, so ``v^dst = R v^src``. Raises unless the spec is a proper
    rotation (a signed permutation of x,y,z with det = +1; det = -1 is a reflection / mixed handedness).
    """
    spec = (x_to, y_to, z_to)
    for a in spec:
        if a not in _AXIS_VECTORS:
            raise ValueError(f"axis {a!r} must be one of {sorted(_AXIS_VECTORS)}")
    if sorted(a.lstrip("-") for a in spec) != ["x", "y", "z"]:
        raise ValueError(f"axes {spec} are not orthonormal: each of x, y, z must appear exactly once")
    cols = [_AXIS_VECTORS[a] for a in spec]
    rows = [[cols[k][r] for k in range(3)] for r in range(3)]  # row-major; columns = source axes
    (m00, m01, m02), (m10, m11, m12), (m20, m21, m22) = rows
    det = (m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20)
           + m02 * (m10 * m21 - m11 * m20))
    if abs(det - 1.0) > 1e-6:
        raise ValueError(f"axes {spec} give det={det:+.0f}; need a proper rotation (+1). "
                         "det = -1 is a reflection (mixed handedness) -- flip one axis.")
    return rows


def axis_alignment_matrix(x_to: str, y_to: str, z_to: str) -> tf.Tensor:
    """Constant rotation ``R_{src->dst}`` mapping a frame's axes onto a target convention.

    Build once per dataset to bring the native world frame to canonical FLU (x-fwd, y-left, z-up) and,
    if the gripper axes differ, the native gripper frame to canonical OpenCV (z-fwd, x-right, y-down);
    feed the result(s) to ``align_axis``. Args: see ``_axis_alignment_rows``.

    Example -- native Forward-Right-Down world -> canonical Forward-Left-Up:
        axis_alignment_matrix("x", "-y", "-z")  # -> diag(1, -1, -1)
    """
    return tf.constant(_axis_alignment_rows(x_to, y_to, z_to), dtype=tf.float32)


def align_axis(R, p, R_world_align, R_gripper_align=None):
    """Map native-frame poses into the canonical frames (design doc: Preprocessing -> Axis alignment).

    Applies the constant alignment rotations R_{w'}^w (world) and, optionally, R_{e'}^e (gripper):
        p^w   = R_{w'}^w p^{w'}
        R_e^w = R_{w'}^w R_{e'}^{w'} (R_{e'}^e)^T
    ``R_world_align`` = R_{w'}^w and ``R_gripper_align`` = R_{e'}^e come from ``axis_alignment_matrix``
    (pass ``R_gripper_align=None`` when the gripper axes are already canonical, i.e. only re-base the
    world). Inputs are native world-frame poses R [..., 3, 3], p [..., 3]; returns the aligned (R, p).
    Joint angles are frame-independent and need no alignment.
    """
    Rw = _f32(R_world_align)
    R_aligned = tf.matmul(Rw, _f32(R))
    if R_gripper_align is not None:
        R_aligned = tf.matmul(R_aligned, tf.linalg.matrix_transpose(_f32(R_gripper_align)))
    p_aligned = _matvec(Rw, _f32(p))
    return R_aligned, p_aligned


# === SE(3) frame math (see design_of_state_and_action_space.md) ==================================
def gripper_delta_pose(R_cur: tf.Tensor, p_cur: tf.Tensor, R_tgt: tf.Tensor, p_tgt: tf.Tensor):
    """Gripper-frame delta from current pose e to target pose e* (the stored representation).

    Returns (R_delta_gripper, p_delta_gripper), i.e. the rotation/translation of the relative
    transform T_{e*}^e = (T_e^w)^{-1} T_{e*}^w:
        R_{e->e*}^e = (R_e^w)^T R_{e*}^w
        p_{e->e*}^e = (R_e^w)^T (p_{e*}^w - p_e^w)
    All inputs are world-frame (R [..., 3, 3], p [..., 3]).
    """
    R_cur = _f32(R_cur)
    Rc_T = tf.linalg.matrix_transpose(R_cur)
    R_delta = tf.matmul(Rc_T, _f32(R_tgt))
    p_delta = _matvec(Rc_T, _f32(p_tgt) - _f32(p_cur))
    return R_delta, p_delta


def world_delta_pose(R_cur: tf.Tensor, p_cur: tf.Tensor, R_tgt: tf.Tensor, p_tgt: tf.Tensor):
    """World-frame delta from current pose e to target pose e*.

    Returns (R_delta_world, p_delta_world):
        R_{e->e*}^w = R_{e*}^w (R_e^w)^T
        p_{e->e*}^w = p_{e*}^w - p_e^w
    """
    R_cur, R_tgt = _f32(R_cur), _f32(R_tgt)
    R_delta = tf.matmul(R_tgt, tf.linalg.matrix_transpose(R_cur))
    p_delta = _f32(p_tgt) - _f32(p_cur)
    return R_delta, p_delta


def change_delta_pose_frame(R_delta: tf.Tensor, p_delta: tf.Tensor, R_src_to_dst: tf.Tensor):
    """Re-express a delta (rotation + translation) from frame ``src`` into frame ``dst``.

    ``R_src_to_dst`` = R_src^dst. Applies the conjugation / rotation rules from the doc:
        R_delta^dst = R_src^dst R_delta^src (R_src^dst)^T
        p_delta^dst = R_src^dst p_delta^src
    With src=e, dst=c this is exactly the load-time gripper->frame-c conversion.
    """
    Rsd = _f32(R_src_to_dst)
    R_out = tf.matmul(tf.matmul(Rsd, _f32(R_delta)), tf.linalg.matrix_transpose(Rsd))
    p_out = _matvec(Rsd, _f32(p_delta))
    return R_out, p_out


# === Deployment: model output (frame c) -> controller input =====================================
def world_pose_from_model_delta(R_delta_c, p_delta_c, R_e_w, p_e_w, R_c_w):
    """Case 1: controller wants the absolute target pose in the WORLD frame.

    The model emits a delta in frame c: (R_{e->e*}^c, p_{e->e*}^c). Given the current
    EEF pose (R_e^w, p_e^w) and the orientation of frame c in world R_c^w, returns the
    absolute target (R_{e*}^w, p_{e*}^w). For a model trained in the world frame pass
    R_c_w = I; for one trained in the gripper frame pass R_c_w = R_e^w.
    """
    R_delta_w, p_delta_w = change_delta_pose_frame(R_delta_c, p_delta_c, R_c_w)  # src=c -> dst=w
    R_tgt = tf.matmul(R_delta_w, _f32(R_e_w))
    p_tgt = _f32(p_e_w) + p_delta_w
    return R_tgt, p_tgt


def gripper_delta_from_model_delta(R_delta_c, p_delta_c, R_e_w, R_c_w):
    """Case 2: controller wants the relative motion in the GRIPPER frame e.

    Converts the model's frame-c delta into the gripper frame, returning
    (R_{e->e*}^e, p_{e->e*}^e). Needs only the current EEF orientation R_e^w and the
    frame-c orientation R_c^w, via R_c^e = (R_e^w)^T R_c^w.
    """
    R_c_e = tf.matmul(tf.linalg.matrix_transpose(_f32(R_e_w)), _f32(R_c_w))  # R_c^e = R_w^e R_c^w
    return change_delta_pose_frame(R_delta_c, p_delta_c, R_c_e)
