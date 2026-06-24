"""Probe + validate template for standardizing an OXE/RLDS dataset.

Two phases:
  PHASE A (probe)    -- inspect raw conventions BEFORE writing the transform
                        (rotation rep, euler extrinsic/intrinsic, action frame,
                         command-vs-ground-truth, gripper polarity, ranges).
  PHASE B (validate) -- after writing <name>_dataset_transform, assert it
                        reconstructs e* from real episodes (~1e-7).

Run with the Python env that has TF + tfds installed (this project's `lerobot` conda env):
  <path-to-lerobot-python> probe_template.py

Edit the CONFIG block, then the slicing in each phase to match the dataset's schema.
See SKILL.md for the checklist this script answers.
"""
import sys
import numpy as np

sys.path.insert(0, "/localscratch/cgao304/dev/any4lerobot/openx2lerobot")
sys.path.insert(0, "/localscratch/cgao304/dev/any4lerobot")

import tensorflow as tf
import tensorflow_datasets as tfds
from scipy.spatial.transform import Rotation as Rsp
import alignment.transforms_numpy as A

# ----------------------------------------------------------------------------- CONFIG
DATA_DIR = "/localscratch/cgao304/dev/datasets/rlds/<dataset>/<version>"
N_EPISODES = 4
RUN_VALIDATE = False  # flip True once <name>_dataset_transform exists
# from oxe_utils.transforms import name_dataset_transform as TRANSFORM
# from oxe_utils.configs import OXE_DATASET_CONFIGS; CFG = OXE_DATASET_CONFIGS["<dataset>"]
# from oxe_utils.constants import STATE_NAMES, ACTION_NAMES
# -----------------------------------------------------------------------------

builder = tfds.builder_from_directory(DATA_DIR)
ds = builder.as_dataset(split="train").take(N_EPISODES)


def episode_arrays(ep):
    steps = list(ep["steps"])
    T = len(steps)
    obs = {k: np.stack([np.asarray(s["observation"][k]) for s in steps]) for k in steps[0]["observation"].keys()
           if np.asarray(steps[0]["observation"][k]).ndim <= 1}  # skip images
    act = np.stack([np.asarray(s["action"]) for s in steps])
    return T, obs, act


# ============================================================ PHASE A: PROBE
print("=" * 70, "\nPHASE A — raw convention probe\n", "=" * 70)
for ei, ep in enumerate(ds):
    T, obs, act = episode_arrays(ep)
    # EDIT these slices to the dataset's schema:
    state = obs["state"]                 # e.g. [T, 7]
    xyz = state[:, :3]
    rpy = state[:, 3:6]                  # or a quaternion -> see below
    grip = state[:, 6]

    print(f"\n--- ep{ei}  T={T} ---")
    print("obs keys:", list(obs.keys()))
    print("xyz range :", xyz.min(0).round(3), xyz.max(0).round(3))
    print("rpy range :", rpy.min(0).round(3), rpy.max(0).round(3), "(radians? ~[-pi,pi])")
    print("grip range:", round(float(grip.min()), 3), round(float(grip.max()), 3),
          "(1=open or 1=closed?)")
    print("action range:", act.min(0).round(3), act.max(0).round(3))
    print("action[0] (first-step sentinel?):", act[0].round(4))

    # --- 1.3 euler: extrinsic vs intrinsic ---
    R_ext = A.rpy_to_matrix(rpy.astype(np.float32), extrinsic=True)
    print("euler vs scipy 'xyz' (extrinsic):", np.abs(R_ext - Rsp.from_euler("xyz", rpy).as_matrix()).max(),
          " vs 'XYZ' (intrinsic):", np.abs(R_ext - Rsp.from_euler("XYZ", rpy).as_matrix()).max(),
          "-> smaller wins")

    # --- 1.2 quaternion order (uncomment if orientation is a quat at state[:,3:7]) ---
    # q = state[:, 3:7]
    # R_xyzw = A.quaternion_to_matrix(q)            # alignment expects xyzw
    # R_wxyz = A.quaternion_to_matrix(q[:, [1, 2, 3, 0]])
    # dR = lambda Rm: np.matmul(Rm[1:], np.transpose(Rm[:-1], (0, 2, 1)))  # world R_{t+1}R_t^T
    # cmd = A.rpy_to_matrix(act[:-1, 3:6], extrinsic=True)  # if action carries a rotation command
    # print("xyzw vs cmd:", np.abs(A.matrix_to_rotation_6d(dR(R_xyzw)) - A.matrix_to_rotation_6d(cmd)).max(),
    #       " wxyz vs cmd:", np.abs(A.matrix_to_rotation_6d(dR(R_wxyz)) - A.matrix_to_rotation_6d(cmd)).max())

    # --- 1.6 command (mode 1) vs ground-truth finite difference (mode 2) ---
    dxyz = xyz[1:] - xyz[:-1]
    print("action[:-1,:3] vs state-diff xyz  (small => mode-2 finite diff):",
          np.abs(act[:-1, :3] - dxyz).max())
    print("  (materially != 0 => raw action is a real command => mode 1 available)")

# ============================================================ PHASE B: VALIDATE
if RUN_VALIDATE:
    print("\n" + "=" * 70, "\nPHASE B — validate transform reconstructs e*\n", "=" * 70)
    ds2 = builder.as_dataset(split="train").take(N_EPISODES)
    for ei, ep in enumerate(ds2):
        steps = list(ep["steps"])
        T = len(steps)
        traj = {
            "observation": {k: tf.stack([s["observation"][k] for s in steps])
                            for k in steps[0]["observation"].keys()},
            "action": tf.stack([s["action"] for s in steps]),
            "language_instruction": tf.stack([s["language_instruction"] for s in steps]),
        }
        raw_state = np.asarray(tf.stack([s["observation"]["state"] for s in steps]))
        out = TRANSFORM(traj)  # noqa: F821
        st = {k: np.asarray(v) for k, v in out["state"].items()}
        ac = {k: np.asarray(v) for k, v in out["action"].items()}
        xyz, rpy = raw_state[:, :3], raw_state[:, 3:6]
        R_all = A.rpy_to_matrix(rpy, extrinsic=True)

        # MODE 2 lengths (T-1). For mode 1 change to == T.
        assert all(v.shape[0] == T - 1 for v in st.values()), "state len"
        assert all(v.shape[0] == T - 1 for v in ac.values()), "action len"

        # body delta reconstructs e*
        bodyR = A.rotation_6d_to_matrix(ac["body_eef_rot6d"])
        R_rec = np.matmul(R_all[:-1], bodyR)
        p_rec = xyz[:-1] + np.matmul(R_all[:-1], ac["body_eef_xyz"][..., None])[..., 0]
        print(f"ep{ei}: body->R_estar {np.abs(R_rec - R_all[1:]).max():.2e}  "
              f"body->p_estar {np.abs(p_rec - xyz[1:]).max():.2e}  "
              f"world==diff {np.abs(ac['world_eef_xyz'] - ac['diff_eef_xyz']).max():.2e}")

    for k in out["state"]:
        assert k in STATE_NAMES, f"STATE_NAMES missing {k}"  # noqa: F821
    for k in out["action"]:
        assert k in ACTION_NAMES, f"ACTION_NAMES missing {k}"  # noqa: F821
    # state_encoding/action_encoding define the out-of-box observation.state / action vectors
    sdim = sum(CFG["state_encoding"].values()); adim = sum(CFG["action_encoding"].values())  # noqa: F821
    print(f"observation.state dim={sdim}  action dim={adim}\nVALIDATE OK")
