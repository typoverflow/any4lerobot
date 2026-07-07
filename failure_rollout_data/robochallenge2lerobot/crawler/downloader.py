"""Incremental, resumable downloader for RoboChallenge rrd files."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

import config
from metadata import make_session, run_dir, write_json


class RateLimiter:
    """Token-bucket bandwidth limiter shared across download threads."""

    def __init__(self, bytes_per_sec: float):
        self.rate = bytes_per_sec
        self.capacity = bytes_per_sec  # allow up to ~1s of burst
        self.tokens = bytes_per_sec
        self.last = time.monotonic()
        self.lock = threading.Lock()

    def throttle(self, nbytes: int) -> None:
        """Block until nbytes may pass under the configured rate."""
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(
                    self.capacity, self.tokens + (now - self.last) * self.rate
                )
                self.last = now
                if self.tokens >= nbytes:
                    self.tokens -= nbytes
                    return
                wait = (nbytes - self.tokens) / self.rate
            time.sleep(min(wait, 1.0))


class StateStore:
    """Ledger of downloaded rollouts, persisted to state.json."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        if path.exists():
            with open(path) as f:
                self.state = json.load(f)
        else:
            self.state = {}

    def is_done(self, rollout_id: str, dest: Path) -> bool:
        entry = self.state.get(rollout_id)
        if not entry or entry.get("status") != "done":
            return False
        expected = entry.get("size")
        return dest.exists() and (expected is None or dest.stat().st_size == expected)

    def mark_done(self, rollout_id: str, url: str, dest: Path, size: int) -> None:
        with self.lock:
            self.state[rollout_id] = {
                "status": "done",
                "url": url,
                "path": str(dest),
                "size": size,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            self._save()

    def _save(self) -> None:
        write_json(self.path, self.state)


def download_file(
    session: requests.Session, url: str, dest: Path, limiter: RateLimiter = None
) -> int:
    """Stream url to dest with resume support. Returns final size in bytes."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    last_err = None
    for attempt in range(config.MAX_RETRIES):
        try:
            offset = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with session.get(
                url, headers=headers, stream=True, timeout=config.DOWNLOAD_TIMEOUT
            ) as resp:
                if offset and resp.status_code == 200:
                    # Server ignored the Range header; restart from scratch.
                    offset = 0
                elif offset and resp.status_code == 416:
                    # Range not satisfiable: partial file is corrupt or complete.
                    total = _total_size(session, url)
                    if total is not None and part.stat().st_size == total:
                        part.replace(dest)
                        return total
                    part.unlink()
                    continue
                resp.raise_for_status()

                if resp.status_code == 206:
                    total = int(
                        resp.headers["Content-Range"].rsplit("/", 1)[-1]
                    )
                else:
                    total = int(resp.headers.get("Content-Length", 0)) or None

                mode = "ab" if offset else "wb"
                with open(part, mode) as f:
                    for chunk in resp.iter_content(chunk_size=config.CHUNK_SIZE):
                        if limiter is not None:
                            limiter.throttle(len(chunk))
                        f.write(chunk)

            size = part.stat().st_size
            if total is not None and size != total:
                raise IOError(f"size mismatch: got {size}, expected {total}")
            part.replace(dest)
            return size
        except (requests.RequestException, IOError) as err:
            last_err = err
            time.sleep(config.RETRY_BACKOFF ** attempt)
    raise RuntimeError(f"Failed to download {url}: {last_err}")


def _total_size(session: requests.Session, url: str):
    resp = session.head(url, timeout=config.HTTP_TIMEOUT)
    length = resp.headers.get("Content-Length")
    return int(length) if length else None


def collect_jobs(records: list, data_dir: Path, state: StateStore) -> list:
    """Return a job dict for every rollout not yet downloaded."""
    jobs = []
    for record in records:
        base = run_dir(data_dir, record["challenge"], record) / "rollouts"
        for rollout in record["rollouts"]:
            url = rollout["data_url"]
            rollout_id = rollout["rollout_id"]
            if not url or not rollout_id:
                continue
            dest = base / f"{rollout_id}.rrd"
            if not state.is_done(rollout_id, dest):
                jobs.append(
                    {
                        "rollout_id": rollout_id,
                        "url": url,
                        "dest": dest,
                        "label": (
                            f"{record['user_name']} / {record['task_name']} "
                            f"/ run {record['run_id'][:8]} / rollout {rollout_id[:8]}"
                        ),
                    }
                )
    return jobs


def download_all(
    records: list,
    data_dir: Path,
    workers: int = config.DEFAULT_WORKERS,
    rate_limit_mbps: float = config.DEFAULT_RATE_LIMIT_MBPS,
):
    """Download every missing rollout in records. Returns (n_ok, n_failed)."""
    state = StateStore(data_dir / "state.json")
    jobs = collect_jobs(records, data_dir, state)
    total = len(jobs)
    if not total:
        print("Nothing to download: everything is up to date.")
        return 0, 0

    limiter = RateLimiter(rate_limit_mbps * 1e6) if rate_limit_mbps > 0 else None
    rate_note = f", capped at {rate_limit_mbps:g} MB/s total" if limiter else ""
    print(f"Downloading {total} rollout file(s) with {workers} worker(s){rate_note}...")
    n_ok = n_failed = 0
    counter_lock = threading.Lock()

    def worker(job):
        part = job["dest"].with_suffix(job["dest"].suffix + ".part")
        resume_note = (
            f" (resuming from {part.stat().st_size / 1e6:.1f} MB)"
            if part.exists()
            else ""
        )
        print(f"start  {job['label']}{resume_note}", flush=True)
        session = make_session()
        size = download_file(session, job["url"], job["dest"], limiter=limiter)
        state.mark_done(job["rollout_id"], job["url"], job["dest"], size)
        return size

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                size = future.result()
                with counter_lock:
                    n_ok += 1
                    done = n_ok + n_failed
                print(
                    f"[{done}/{total}] done   {job['label']} ({size / 1e6:.1f} MB)",
                    flush=True,
                )
            except Exception as err:
                with counter_lock:
                    n_failed += 1
                    done = n_ok + n_failed
                print(f"[{done}/{total}] FAILED {job['label']}: {err}", flush=True)

    print(f"Done: {n_ok} downloaded, {n_failed} failed.")
    return n_ok, n_failed
