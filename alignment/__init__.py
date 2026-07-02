"""State / action space conversion utilities.

One self-contained module per array backend, each exposing the *same* function
names, signatures, and conventions (see ``design_of_state_and_action_space.md``):

    transforms_numpy.py   NumPy        -- deployment / scripts / tests
    transforms_torch.py   PyTorch      -- training / online inference
    transforms_tf.py      TensorFlow   -- offline tf.data pipelines (openx2lerobot)

Import the one matching your framework, e.g.::

    from alignment import transforms_numpy as T   # or transforms_torch / transforms_tf
    R = T.rpy_to_matrix(rpy, extrinsic=True)
    gripper_R, gripper_p = T.gripper_delta_pose(R_e, p_e, R_estar, p_estar)
    R_estar_w, p_estar_w = T.world_pose_from_model_delta(R_dc, p_dc, R_e, p_e, R_c_w)

The submodules are intentionally NOT imported here so that importing ``alignment``
does not require all three frameworks to be installed.
"""

__all__ = ["transforms_numpy", "transforms_torch", "transforms_tf"]
