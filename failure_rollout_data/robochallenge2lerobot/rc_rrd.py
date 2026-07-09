"""Read RoboChallenge Rerun (``.rrd``) rollouts: joint/gripper scalars + H.264 video.

The crawled preview ``.rrd`` files carry, per arm, ``cur_joint/joint_1..6`` and
``cur_gripper`` (scalars logged ~138 Hz on the ``log_time`` timeline), plus three H.264
(``avc1``) video streams at ~28 fps (per-camera frame counts differ slightly). This module
reads those signals and resamples everything onto a uniform target-fps grid by
nearest-neighbour, decoding video one frame at a time so memory stays O(1) in the frame
count (a full 2-min episode would otherwise be ~10 GB of decoded RGB).

Rerun 0.26 dataframe API; PyAV for H.264 decode.
"""

from __future__ import annotations

import numpy as np
import rerun.dataframe as rd

# Camera entity paths, in a stable order, by arm layout.
SINGLE_ARM_CAMERAS = ["/videos_1", "/videos_2", "/videos_3"]
DUAL_ARM_CAMERAS = ["/videos_front", "/videos_left", "/videos_right"]


def _log_time_seconds(col) -> np.ndarray:
    """Rerun ``log_time`` column (arrow timestamps) -> float seconds (vectorized)."""
    arr = col.combine_chunks() if hasattr(col, "combine_chunks") else col
    return arr.to_numpy(zero_copy_only=False).astype("datetime64[ns]").astype("int64") / 1e9


def read_arm(rec: rd.Recording, prefix: str) -> dict:
    """Read one arm's joints+gripper. ``prefix`` is 'arm', 'left_arm', or 'right_arm'.

    The seven scalars (``joint_1..6``, ``cur_gripper``) are each logged on their OWN
    ``log_time`` rows -- staggered by sub-millisecond, never on a shared row -- at ~138 Hz.
    We read each channel as an independent (time, value) series and resample the others onto
    ``joint_1``'s timeline by nearest-neighbour (< ~4 ms error). Returns dict with ``times``
    (T,), ``joints`` (T, 6), ``gripper`` (T,).
    """
    jcols = [f"/{prefix}/cur_joint/joint_{i}" for i in range(1, 7)]
    gcol = f"/{prefix}/cur_gripper"

    def series(entity):
        # One view per entity: the log_time index then holds only that channel's ~13k dense
        # rows (no nulls), far cheaper than filtering the 7x-larger sparse union.
        tbl = rec.view(index="log_time", contents=entity).select().read_all()
        vcol = [n for n in tbl.column_names if n.startswith(entity + ":")]
        if not vcol:
            raise KeyError(f"{entity} not found")
        t = _log_time_seconds(tbl.column("log_time"))
        vals = tbl.column(vcol[0]).combine_chunks().flatten().to_numpy(zero_copy_only=False)
        order = np.argsort(t)
        return t[order], vals.astype(np.float64)[order]

    ref_t, j1 = series(jcols[0])
    joints = np.empty((len(ref_t), 6), dtype=np.float64)
    joints[:, 0] = j1
    for k, jc in enumerate(jcols[1:], start=1):
        t, v = series(jc)
        joints[:, k] = v[nearest_indices(t, ref_t)]
    gt, gv = series(gcol)
    grip = gv[nearest_indices(gt, ref_t)]
    return {"times": ref_t, "joints": joints, "gripper": grip}


def camera_entities(rec: rd.Recording, dual: bool) -> list[str]:
    """Camera entity paths actually present in this recording, in canonical order."""
    present = {c.entity_path for c in rec.schema().component_columns()}
    wanted = DUAL_ARM_CAMERAS if dual else SINGLE_ARM_CAMERAS
    return [e for e in wanted if e in present]


def read_video_times(rec: rd.Recording, entity: str) -> np.ndarray:
    """Sorted float-second log times of the frames of one video stream (empty if absent)."""
    import pyarrow.compute as pc

    tbl = rec.view(index="log_time", contents=entity + "/**").select().read_all()
    scol = [n for n in tbl.column_names if n.endswith(":VideoStream:sample")]
    if not scol:
        return np.empty(0, dtype=np.float64)
    mask = pc.is_valid(tbl.column(scol[0]))
    return np.sort(_log_time_seconds(tbl.column("log_time").filter(mask)))


def decode_video_frames(rec: rd.Recording, entity: str):
    """Yield ``(time_seconds, frame_rgb_uint8[H,W,3])`` for one stream, in time order.

    Decodes the concatenated H.264 (Annex-B) samples one frame at a time. Frames are paired
    with the sorted sample log times by decode order (these streams are decode==display order).
    """
    import av
    import pyarrow.compute as pc

    tbl = rec.view(index="log_time", contents=entity + "/**").select().read_all()
    scol = [n for n in tbl.column_names if n.endswith(":VideoStream:sample")]
    if not scol:
        return
    mask = pc.is_valid(tbl.column(scol[0]))
    times = _log_time_seconds(tbl.column("log_time").filter(mask))
    order = np.argsort(times)
    times = times[order]
    samples = tbl.column(scol[0]).filter(mask).to_pylist()
    samples = [samples[i] for i in order]
    buf = b"".join(bytes(s[0] if isinstance(s, list) else s) for s in samples)

    codec = av.CodecContext.create("h264", "r")
    idx = 0
    for packet in codec.parse(buf):
        for frame in codec.decode(packet):
            img = frame.to_ndarray(format="rgb24")
            yield (float(times[idx]) if idx < len(times) else float(times[-1]), img)
            idx += 1
    for frame in codec.decode(None):  # flush
        img = frame.to_ndarray(format="rgb24")
        yield (float(times[idx]) if idx < len(times) else float(times[-1]), img)
        idx += 1


def grid_times(t_start: float, t_end: float, fps: int) -> np.ndarray:
    """Uniform time grid over [t_start, t_end] at ``fps`` (>= 1 sample)."""
    n = max(1, int(round((t_end - t_start) * fps)) + 1)
    return t_start + np.arange(n, dtype=np.float64) / fps


def nearest_indices(src_times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    """For each query time, index of the nearest source time (both need not be sorted-src)."""
    src = np.asarray(src_times, dtype=np.float64)
    q = np.asarray(query_times, dtype=np.float64)
    pos = np.searchsorted(src, q)
    pos = np.clip(pos, 1, len(src) - 1)
    left, right = src[pos - 1], src[pos]
    choose_left = (q - left) <= (right - q)
    return np.where(choose_left, pos - 1, pos)


def resample_linear(src_times: np.ndarray, values: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    """Linearly interpolate ``values`` (sampled at ``src_times``) onto ``query_times``.

    ``values`` may be (T,) or (T, d); interpolation is per-column. ``src_times`` must be sorted
    ascending (``read_arm`` returns it so). Unlike ``nearest_indices``, this removes the staircase
    quantization of nearest-neighbour resampling -- important when the signal is later differenced
    (the FK-EEF delta action), where NN steps show up as high-frequency jitter.
    """
    src = np.asarray(src_times, dtype=np.float64)
    q = np.asarray(query_times, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    if vals.ndim == 1:
        return np.interp(q, src, vals)
    return np.stack([np.interp(q, src, vals[:, k]) for k in range(vals.shape[1])], axis=1)


def smooth_butter(x: np.ndarray, fps: float, cutoff_hz: float, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth low-pass along axis 0 (returns float64, same shape as ``x``).

    ``filtfilt`` is zero-phase, so the smoothed signal stays time-aligned with the original -- the
    per-step delta computed from it (the action) is not lagged. Short signals (whole rollout shorter
    than ``filtfilt``'s pad length) are returned unfiltered so the rollout still converts.
    """
    from scipy.signal import butter, filtfilt  # lazy: scipy import is ~0.3 s

    x = np.asarray(x, dtype=np.float64)
    b, a = butter(order, cutoff_hz / (fps / 2.0))
    padlen = 3 * max(len(a), len(b))  # filtfilt's default; it errors when len(x) <= padlen
    if x.shape[0] <= padlen:
        return x
    return filtfilt(b, a, x, axis=0)


class NearestFrameStream:
    """Pull the frame nearest a monotonically-increasing query time from a frame generator.

    Holds at most two decoded frames at a time (current + lookahead), so a whole episode
    streams through in O(1) frame memory. Query times must be non-decreasing.
    """

    def __init__(self, gen, fallback_shape):
        self._gen = gen
        self._fallback = np.zeros(fallback_shape, dtype=np.uint8)
        self._cur = next(gen, None)  # (t, frame) or None
        self._nxt = next(gen, None)

    def at(self, t: float) -> np.ndarray:
        if self._cur is None:
            return self._fallback
        # Advance while the lookahead frame is at least as close to t as the current one.
        while self._nxt is not None and abs(self._nxt[0] - t) <= abs(self._cur[0] - t):
            self._cur = self._nxt
            self._nxt = next(self._gen, None)
        return self._cur[1]
