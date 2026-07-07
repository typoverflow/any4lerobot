"""Fetch and normalize RoboChallenge run metadata."""

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

import config


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = config.USER_AGENT
    return session


def fetch_json(session: requests.Session, url: str):
    last_err = None
    for attempt in range(config.MAX_RETRIES):
        try:
            resp = session.get(url, timeout=config.HTTP_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as err:
            last_err = err
            time.sleep(config.RETRY_BACKOFF ** attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def parse_arena(rrd_url: str) -> str:
    """Extract the arena machine name from an rrd URL.

    e.g. https://video-preview.robochallenge.ai/arena_data_rc_arx5_3/2026-06-29/....rrd
    -> arena_data_rc_arx5_3
    """
    match = re.search(r"video-preview\.robochallenge\.ai/([^/]+)/", rrd_url or "")
    return match.group(1) if match else ""


def build_task_catalog(benchmark_list: list) -> dict:
    """Map task_name -> task info (tags, robot, description, ...)."""
    catalog = {}
    for benchmark in benchmark_list:
        for task in benchmark.get("tasks", []):
            tags = task.get("task_tag", [])
            robots = [t for t in tags if t in config.KNOWN_ROBOTS]
            catalog[task["task_name"]] = {
                "benchmark_name": benchmark.get("benchmark_name", ""),
                "robot": robots[0] if robots else "",
                "task_tags": tags,
                "task_description": task.get("task_description", ""),
                "prompt": task.get("prompt", ""),
                "task_training_data": task.get("task_training_data", ""),
                "scoring": task.get("scoring", ""),
            }
    return catalog


def normalize_run(benchmark: str, run: dict, task_catalog: dict) -> dict:
    """Produce one normalized metadata record for a run."""
    task_info = task_catalog.get(run.get("task_name", ""), {})
    rollouts = []
    arenas = set()
    for rollout in run.get("rollouts", []):
        url = rollout.get("rollout_details", "")
        arena = parse_arena(url)
        if arena:
            arenas.add(arena)
        rollouts.append(
            {
                "rollout_id": rollout.get("rollout_id", ""),
                "status": rollout.get("status", ""),
                "score": rollout.get("score"),
                "completion": rollout.get("completion"),
                "comments": rollout.get("comments", ""),
                "data_url": url,
            }
        )

    exec_time = run.get("time")
    exec_time_iso = (
        datetime.fromtimestamp(exec_time, tz=timezone.utc).isoformat()
        if exec_time
        else ""
    )

    return {
        "challenge": benchmark,
        "benchmark_name": task_info.get("benchmark_name", ""),
        "task_id": run.get("task_id", ""),
        "task_name": run.get("task_name", ""),
        "user_name": run.get("user_name", ""),
        "run_id": run.get("run_id", ""),
        "model_name": run.get("model_name", ""),
        "display_name": run.get("display_name", ""),
        "is_multi_task_model": run.get("is_multi_task_model"),
        "is_ranked": run.get("is_ranked"),
        "status": run.get("status", ""),
        "score": run.get("score"),
        "success_rate": run.get("success_rate"),
        "execution_time_unix": exec_time,
        "execution_time_utc": exec_time_iso,
        "hardware": {
            "robot": task_info.get("robot", ""),
            "arenas": sorted(arenas),
            "task_tags": task_info.get("task_tags", []),
        },
        "task_model": run.get("task_model", {}),
        "task_description": task_info.get("task_description", ""),
        "prompt": task_info.get("prompt", ""),
        "rollouts": rollouts,
    }


def run_dir(data_dir: Path, benchmark: str, record: dict) -> Path:
    task_folder = f"{record['task_name']}_{record['task_id']}"
    return data_dir / benchmark / task_folder / record["run_id"]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def diff_runs(old_runs, new_runs) -> dict:
    """Compare two runs snapshots; describe what is new in new_runs."""
    if old_runs is None:
        return {"baseline": True, "new_runs": [], "new_rollouts": 0, "new_users": []}

    old_run_ids = {r.get("run_id") for r in old_runs}
    old_rollout_ids = {
        ro.get("rollout_id") for r in old_runs for ro in r.get("rollouts", [])
    }
    old_users = {r.get("user_name") for r in old_runs}

    new_run_list = [r for r in new_runs if r.get("run_id") not in old_run_ids]
    new_rollouts = sum(
        1
        for r in new_runs
        for ro in r.get("rollouts", [])
        if ro.get("rollout_id") not in old_rollout_ids
    )
    new_users = sorted(
        {r.get("user_name") for r in new_run_list} - old_users
    )
    return {
        "baseline": False,
        "new_runs": [
            {
                "run_id": r.get("run_id", ""),
                "user_name": r.get("user_name", ""),
                "task_name": r.get("task_name", ""),
                "n_rollouts": len(r.get("rollouts", [])),
            }
            for r in new_run_list
        ],
        "new_rollouts": new_rollouts,
        "new_users": new_users,
    }


def refresh_metadata(benchmark: str, data_dir: Path, session=None):
    """Fetch fresh metadata, write raw snapshots + per-run metadata + index.

    Returns (records, diff) where records is the list of normalized run
    records and diff describes what changed since the previous fetch.
    """
    session = session or make_session()

    benchmark_list = fetch_json(session, config.benchmark_list_url(benchmark))
    runs = fetch_json(session, config.runs_url(benchmark))

    raw_dir = data_dir / "raw"
    runs_path = raw_dir / config.RUNS_FILES[benchmark]
    old_runs = None
    if runs_path.exists():
        with open(runs_path) as f:
            old_runs = json.load(f)
    diff = diff_runs(old_runs, runs)

    write_json(raw_dir / "benchmark_list.json", benchmark_list)
    write_json(runs_path, runs)

    task_catalog = build_task_catalog(benchmark_list)
    records = [normalize_run(benchmark, run, task_catalog) for run in runs]

    for record in records:
        write_json(run_dir(data_dir, benchmark, record) / "metadata.json", record)

    write_json(data_dir / benchmark / "index.json", records)
    return records, diff
