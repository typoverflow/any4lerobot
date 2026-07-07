"""Minimal, vectorized URDF forward kinematics for RoboChallenge arms.

Generalized from ``vifailback2lerobot/convert.py::PiperFK``: parse a URDF, walk the
serial chain ``base_link -> tip_link``, and evaluate the pose of ``tip_link`` in
``base_link`` for a batch of joint configurations. Pure NumPy, no external FK library
(none is installed in the target env). Shared rotation math comes from the repo-root
``alignment`` package.

The RoboChallenge crawled ``.rrd`` files only store joint angles (no EEF pose), so the
end-effector pose is recovered here by FK. Each robot's chain (base link, tip link,
revolute joint order) is validated against the released HF ``ee_positions`` ground truth
before bulk conversion -- see ``validate_fk.py``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# Repo-root ``alignment`` package (this file is imported with the repo root on sys.path).
from alignment import transforms_numpy as tn


class SerialChainFK:
    """Vectorized URDF FK over the serial chain ``base_link -> tip_link``.

    Auto-discovers the chain by following ``child -> parent`` joint links from the tip
    back to the base. ``revolute``/``continuous`` joints consume one input angle each (in
    chain order, base -> tip); ``fixed`` joints contribute only their static origin
    transform. Prismatic joints are rejected (none of these arms use them).
    """

    def __init__(
        self,
        urdf_path: Path | str,
        base_link: str,
        tip_link: str,
        tool_offset=(0.0, 0.0, 0.0),
    ):
        # ``tool_offset``: fixed translation (m) from ``tip_link`` to the real end-effector
        # (TCP), expressed in the tip frame. Used when the URDF tip is the wrist but the
        # dataset's ee_positions is measured at a point further along the tool (DOS-W1:
        # ~99 mm along tip x). Calibrated against HF ee_positions in validate_fk.py.
        self.tool_offset = np.asarray(tool_offset, dtype=np.float64)
        root = ET.parse(str(urdf_path)).getroot()
        by_child: dict[str, dict] = {}
        for j in root.iter("joint"):
            jtype = j.get("type")
            if jtype not in ("revolute", "continuous", "prismatic", "fixed"):
                continue
            origin = j.find("origin")
            xyz = (
                np.array([float(v) for v in (origin.get("xyz") or "0 0 0").split()])
                if origin is not None
                else np.zeros(3)
            )
            rpy = (
                np.array([float(v) for v in (origin.get("rpy") or "0 0 0").split()])
                if origin is not None
                else np.zeros(3)
            )
            axis_el = j.find("axis")
            axis = (
                np.array([float(v) for v in axis_el.get("xyz").split()])
                if axis_el is not None
                else np.array([1.0, 0.0, 0.0])
            )
            norm = np.linalg.norm(axis)
            child = j.find("child").get("link")
            by_child[child] = {
                "name": j.get("name"),
                "type": jtype,
                "parent": j.find("parent").get("link"),
                "origin": self._origin_mat(xyz, rpy),
                "axis": axis / norm if norm > 0 else axis,  # fixed joints may declare a zero axis
            }

        chain = []
        link = tip_link
        while link != base_link:
            if link not in by_child:
                raise ValueError(
                    f"cannot reach base_link={base_link!r} from tip_link={tip_link!r}: "
                    f"link {link!r} has no parent joint in the URDF"
                )
            joint = by_child[link]
            chain.append(joint)
            link = joint["parent"]
        self.chain = chain[::-1]
        if any(j["type"] == "prismatic" for j in self.chain):
            raise ValueError(f"chain {base_link}->{tip_link} contains a prismatic joint")
        self.base_link = base_link
        self.tip_link = tip_link
        self.dof = sum(j["type"] in ("revolute", "continuous") for j in self.chain)
        self.joint_names = [j["name"] for j in self.chain if j["type"] in ("revolute", "continuous")]

    @staticmethod
    def _origin_mat(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
        A = np.eye(4)
        A[:3, :3] = tn.rpy_to_matrix(rpy, extrinsic=True)  # URDF origin rpy is fixed-axis XYZ
        A[:3, 3] = xyz
        return A

    def __call__(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """q: (T, dof) joint angles -> (R: (T,3,3), p: (T,3)) pose of tip in base frame."""
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        if q.shape[-1] != self.dof:
            raise ValueError(f"expected q[..., {self.dof}] for {self.tip_link}, got {q.shape}")
        T_ = np.broadcast_to(np.eye(4), (q.shape[0], 4, 4)).copy()
        qi = 0
        for joint in self.chain:
            T_ = T_ @ joint["origin"]
            if joint["type"] in ("revolute", "continuous"):
                rot = np.broadcast_to(np.eye(4), (q.shape[0], 4, 4)).copy()
                rot[:, :3, :3] = tn.axis_angle_to_matrix(joint["axis"] * q[:, qi : qi + 1])
                T_ = T_ @ rot
                qi += 1
        R = T_[:, :3, :3]
        p = T_[:, :3, 3]
        if np.any(self.tool_offset):
            p = p + np.einsum("tij,j->ti", R, self.tool_offset)
        return R.astype(np.float32), p.astype(np.float32)
