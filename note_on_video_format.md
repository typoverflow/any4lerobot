# Note on Video Formats: ABC vs. LeRobot

Focus: **seek to the middle of a video, read a short consecutive chunk, fast.**

## ABC format (`/localscratch/cgao304/dev/abc`)
Encoded for a *deterministic, synthesizable* frame index:
- **libx264**, **GOP=30 fixed** (`keyint=min-keyint=30, scenecut=0`), keyframe at every `i%30==0`.
- **No B-frames** (`-bf 0`) → decode order = display order, monotonic PTS.
- **CFR**, fixed timebase `1/15360`, **512 ticks/frame** → `pts(i) = 512*i` exactly.
- **+faststart** (moov atom at front, for remote reads).
- **Cameras vstacked** into one MP4; split on decode by row-slicing.

Decode trick (`train_loop.py`): build the index in pure Python and pass it to torchcodec,
so it seeks exactly **without scanning the file**:
```python
frames = [{"pts": 512*i, "duration": 512, "key_frame": 1 if i%30==0 else 0}
          for i in range(episode_length)]
decoder = VideoDecoder(path, custom_frame_mappings=json.dumps({"frames": frames}))
```
Remote: just pre-downloads via `aws s3 cp` (the paper's file-like object isn't implemented).

## LeRobot v0.5.1 (`lerobot/datasets/video_utils.py`)
- **libsvtav1 (AV1)** default, **GOP=2** (hardcoded), CRF=30, yuv420p.
- **B-frames not controlled** (codec default, usually on).
- **faststart** applied only at the concatenation step (final files usually have it).
- **One MP4 per camera**; **many episodes concatenated per file** (episode span stored as
  `from/to_timestamp`).
- Decode: **torchcodec default**, `seek_mode="approximate"` (skips scan, but estimates
  frame position from `average_fps` → can mis-land). Frame-index random access via
  `get_frames_at(indices=...)`; decoders cached; consecutive frames cheap.

## Comparison
| | ABC | LeRobot |
|---|---|---|
| codec | libx264 | libsvtav1 |
| GOP | 30 fixed | 2 |
| B-frames | off | on (default) |
| faststart | initial encode | concat step |
| cameras | vstacked, 1 file | 1 file/camera |
| file↔episode | 1:1 | many:1 |
| seek | exact, synthesized `custom_frame_mappings` | `approximate` (fps estimate) |

## Takeaways for your use case
- **GOP drives mid-video seek cost** (decode from nearest keyframe up to target). LeRobot's
  **GOP=2 is already near-optimal**; ABC's GOP=30 is *worse* (chosen for file size). Don't copy 30.
- LeRobot **already** does torchcodec random access + cheap consecutive frames — the pattern is supported.
- Genuinely additive from ABC: **exact `custom_frame_mappings`** (exact start frame, no scan;
  supported by installed torchcodec) and **disabling B-frames**. For LeRobot's many-episodes-per-file
  layout, generate the mapping once with `ffprobe`, not ABC-style synthesis.
- Out of scope: vstack cameras, remote file-like reader (high blast radius).
- Env caveat: torchcodec native lib fails to load in the `lerobot` env (`libavutil.so.60` missing) —
  fix before decode-side tests.
