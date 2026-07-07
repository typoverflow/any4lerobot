# RoboArena → LeRobot v3

Converts the [RoboArena](https://robo-arena.github.io/) dataset — autonomous generalist-policy
rollouts on the DROID **Franka Panda** setup, each scored for success and annotated with a pairwise
policy preference — into LeRobot v3. The output follows the unified convention in
[`../dataset.md`](../dataset.md): each pose is stored twice, once as the native `raw_state.*` /
`raw_target.*` (no transform) and once as the canonically axis-aligned `state.*` / `target.*`. The
per-step **action is not stored** — it is computed at load time from `state.*` / `target.*`
(`dataset.md` §3). A small `debug.*` set holds a precomputed ground-truth-next reference in the
canonical gripper frame.

Paper: *RoboArena: Distributed Real-World Evaluation of Generalist Robot Policies*
([arXiv:2506.18123](https://arxiv.org/pdf/2506.18123)).

## Layout of the raw data

Two dated dumps, each converted into its **own** LeRobot dataset:

```text
<raw_dir>/                                     # e.g. .../roboarena/DataDump_08-05-2025
├── global_metadata.yaml                       # policy_index -> {open_source, action_space}
└── evaluation_sessions/
    └── <session_uuid>/
        ├── metadata.yaml                       # instruction, preference, per-policy scores
        └── <LETTER>_<policy_name>/             # ONE episode = one policy's rollout
            ├── *_npz_file.npz                  # proprio + actions (object array `data`, len T)
            └── *_video_{left,right,wrist}.mp4  # some sessions omit `left`
```

Each **policy rollout is one LeRobot episode**. Every step in the npz `data` array carries
`cartesian_position` (6: xyz + extrinsic-XYZ rpy, base frame), `joint_position` (7),
`gripper_position` (1, continuous), and `action` (8). The action is `action[:7]` = 7 arm joint-space
command values + `action[7]` = a binary gripper command. There is **no** commanded cartesian target;
the joint command's unit depends on the policy (see the mixed-mode note below).

## Output features

Fields are grouped per `dataset.md`. `raw_*` are native (no transform); the unprefixed `state.*` /
`target.*` are canonically axis-aligned. Joints are frame-independent, so their canonical copies equal
the raw ones.

| feature | shape | notes |
| --- | --- | --- |
| `observation.images.left` / `right` / `wrist` | (288, 512, 3) video | stored **as shipped** (un-flipped); a missing camera is padded with **black frames** (see `cameras_present`) |
| `raw_state.joint_pos` | (7) | raw `joint_position` |
| `raw_state.eef_xyz` | (3) | raw `cartesian_position[:3]`, base frame |
| `raw_state.eef_rpy` | (3) | raw extrinsic-XYZ rpy passthrough (the shipped rotation representation) |
| `raw_state.gripper_state` | (1) | **raw continuous** `gripper_position` (see gripper note) |
| `raw_target.joint_pos` | (7) | `action[:7]` for a `joint_position` policy, else `0.0` (absolute rad target) |
| `raw_target.joint_vel` | (7) | `action[:7]` for a `joint_velocity` policy, else `0.0` (absolute rad/s) |
| `raw_target.gripper_state` | (1) | **raw commanded** gripper, binary `{0, 1}` (see gripper note) |
| `state.joint_pos` | (7) | canonical joint position (= `raw_state.joint_pos`) |
| `state.eef_xyz` | (3) | canonical eef translation (base already FLU → identity) |
| `state.eef_rot6d` | (6) | canonical eef rotation (Franka hand → OpenCV gripper), as rot6d |
| `state.gripper_state` | (1) | `1 − gripper_position`, canonical **0 = closed, 1 = open** |
| `target.joint_pos` | (7) | canonical joint-position target (= `raw_target.joint_pos`) |
| `target.joint_vel` | (7) | canonical joint-velocity target (= `raw_target.joint_vel`) |
| `target.gripper_state` | (1) | `1 − raw` binary gripper command, canonical **0 = closed, 1 = open** |
| `control_is_position` | (1) | `1.0` if the policy is `joint_position`, `0.0` if `joint_velocity`; constant per episode |
| `debug.gripper_eef_xyz` | (3) | GT-next Δtranslation in the canonical gripper frame (debug only) |
| `debug.gripper_eef_rot6d` | (6) | GT-next Δrotation, canonical gripper frame; last step is a no-op |
| `binary_success` | (1) | episode binary success label, constant per episode |
| `partial_success` | (1) | episode partial-success score ∈ [0, 1], constant per episode |

There is **no top-level `observation.state` / `action` vector**: fields are kept separate so you can
assemble the exact state/action encoding you want at load time.

### Computing the action at load time (`dataset.md` §3)

The action space follows the policy's control mode, selected per episode by `control_is_position`:

- **`joint_velocity`** (`control_is_position == 0`): the action is `target.joint_vel` directly
  (already an absolute velocity; no differencing).
- **`joint_position`** (`control_is_position == 1`): the action is the per-step joint delta
  `Δq = target.joint_pos − state.joint_pos`.

In both modes the gripper action is `target.gripper_state`.

### ⚠️ Action space is heterogeneous (`joint_velocity` vs `joint_position`)

The joint command's **unit depends on the policy**. The 02-03-2026 dump mixes two control modes (8
`joint_velocity` policies, 7 `joint_position` policies; the 08-05-2025 dump is uniformly
`joint_velocity`):

- `joint_velocity` → the command is a joint **velocity** (rad/s).
- `joint_position` → the command is an **absolute next-step joint target** (rad); empirically
  `action[:7] ≈ joint_position[t+1]` to ~0.02 rad.

To keep a fixed schema, the command is routed to the matching field — `raw_target.joint_vel` /
`target.joint_vel` for velocity policies, `raw_target.joint_pos` / `target.joint_pos` for position
policies — and the inactive field is **0-filled**. Because `0.0` is itself a valid value, always
split/select by `control_is_position` (mirrored per-episode by `action_space` in the sidecar). **Do
not mix the two modes in one normalized action vector.**

### ⚠️ Gripper polarity

DROID `gripper_position` is stored with **larger ≈ more closed** (0 = open, 1 = closed). The two
namespaces differ:

- `raw_state.gripper_state` / `raw_target.gripper_state` — the **raw DROID polarity, un-inverted**
  (continuous state ≈ `[0, 1]`; commanded gripper binary `{0, 1}`).
- `state.gripper_state` / `target.gripper_state` — inverted to the **canonical** convention
  `1 − raw`, so **0 = closed, 1 = open** (`dataset.md` §2.3; matches openx2lerobot's DROID
  `invert_gripper_actions`).

## Frames & the `debug.*` fields

The DROID base frame is already the canonical world frame (Forward-Left-Up), so the world alignment
`R_{w'}^w` is the identity and positions pass through unchanged. Only the **gripper** frame is
re-based: the Franka `panda_hand` EEF frame is mapped onto the canonical OpenCV gripper frame
(z = approach, x = finger-open "right", y = down) via the constant rotation
`R_{e'}^e = axis_alignment_matrix("-y", "x", "z")` (identical to openx2lerobot's `_GRIPPER_ALIGN_FRANKA`).
`state.eef_{xyz,rot6d}` are the canonical pose; to reproduce the alignment from `raw_state`:

```python
from alignment import transforms_numpy as tn
R = tn.rpy_to_matrix(raw_state_eef_rpy, extrinsic=True)      # DROID rpy is extrinsic XYZ
R_align = tn.axis_alignment_matrix("-y", "x", "z")           # Franka EEF -> canonical OpenCV gripper
R_canon, p_canon = tn.align_axis(R, raw_state_eef_xyz, np.eye(3), R_align)
# R_canon == rotation_6d_to_matrix(state_eef_rot6d), p_canon == state_eef_xyz
```

The `debug.gripper_eef_{xyz,rot6d}` fields hold the ground-truth next-step relative motion
(`state[t] → state[t+1]`) expressed in the canonical gripper frame at `t`, with an identity/no-op
final step. They are redundant with the load-time action derived from `state.*` (a `joint_position`
policy's eef motion is not the commanded action), but are kept as a precomputed frame sanity-check.

### Wrist camera & co-training

All camera views are stored **exactly as shipped** — the wrist view is **not** rotated during
conversion. The **wrist** camera (DROID Zed-Mini) is physically mounted rolled 180°, so a consumer
**co-training RoboArena with other datasets** should apply a 180° image rotation to the wrist
(`frame[::-1, ::-1, :]` — reverse H and W, a proper rotation, not a mirror) to align its image axes
with the canonical gripper frame. `decode_video(..., flip180=True)` performs exactly this.

## Per-episode metadata

`meta/roboarena_metadata.jsonl` has one JSON line per episode (`episode_index` aligned with the
dataset) recording provenance and labels: `session_id`, `policy_letter`, `policy_name`,
`language_instruction`, `binary_success`, `partial_success`, `duration`, `preference` (session-level
A/B/TIE), `evaluator_name`, `evaluation_location`, session timestamps, `longform_feedback`,
`open_source`, `action_space`, `cameras_present` (which cameras were real vs black-padded), and
`num_frames`. `binary_success` / `partial_success` are also broadcast as per-frame features.

## Usage

```bash
# Convert both dumps (each -> its own <name>_lerobot dir). Resumable: re-run to continue.
bash convert.sh

# Or one dump directly:
python convert.py \
    --raw-dir /path/to/roboarena/DataDump_08-05-2025 \
    --local-dir /path/to/output \
    --num-proc 8 --skip-bad-episodes
```

Key flags: `--num-proc N` (parallel shards + merge), `--max-episodes K` (debug subset),
`--overwrite` (rebuild, disables resume), `--skip-bad-episodes` (log & continue past unreadable
episodes/videos), `--push-to-hub`. Video decoding uses **pyav** (torchcodec is unreliable in this
env). Default `fps=15` (DROID control rate).
