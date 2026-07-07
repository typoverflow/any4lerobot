#!/usr/bin/env python3
"""RoboChallenge crawler CLI.

Usage:
  python crawl.py                 # refresh metadata, then download all missing rrd files
  python crawl.py metadata        # refresh metadata only
  python crawl.py download [...]  # refresh metadata + incremental download

Download options:
  --task NAME_OR_ID   only runs for this task (task_name or task_id)
  --user NAME         only runs whose user_name contains NAME (case-insensitive)
  --workers N         parallel downloads (default 4)
  --limit N           cap the number of rollout files downloaded this invocation
  --dry-run           show what would be downloaded, don't download
"""

import argparse
import sys
from pathlib import Path

import config
from downloader import StateStore, collect_jobs, download_all
from metadata import refresh_metadata


def filter_records(records, task=None, user=None):
    if task:
        task_lower = task.lower()
        records = [
            r
            for r in records
            if r["task_name"].lower() == task_lower or r["task_id"].lower() == task_lower
        ]
    if user:
        user_lower = user.lower()
        records = [r for r in records if user_lower in r["user_name"].lower()]
    return records


AVG_RRD_MB = 56  # rough average rrd size, for volume estimates


def print_summary(records, diff, data_dir):
    n_rollouts = sum(len(r["rollouts"]) for r in records)
    users = {r["user_name"] for r in records}

    print("=== Update summary ===")
    print(f"Total on server: {len(records)} runs, {len(users)} users, {n_rollouts} rollouts")

    if diff["baseline"]:
        print("First fetch: no previous snapshot, all runs recorded as baseline.")
    elif not diff["new_runs"] and not diff["new_rollouts"]:
        print("New since last fetch: nothing — no new runs or rollouts.")
    else:
        new_gb = diff["new_rollouts"] * AVG_RRD_MB / 1024
        print(
            f"New since last fetch: {len(diff['new_runs'])} run(s), "
            f"{diff['new_rollouts']} rollout(s) (~{new_gb:.1f} GB)"
        )
        if diff["new_users"]:
            print(f"New users: {', '.join(diff['new_users'])}")
        for r in diff["new_runs"][:20]:
            print(
                f"  + {r['user_name']} / {r['task_name']} "
                f"({r['n_rollouts']} rollouts, run {r['run_id']})"
            )
        if len(diff["new_runs"]) > 20:
            print(f"  ... and {len(diff['new_runs']) - 20} more run(s)")

    state = StateStore(data_dir / "state.json")
    pending = collect_jobs(records, data_dir, state)
    pending_gb = len(pending) * AVG_RRD_MB / 1024
    print(
        f"Not yet downloaded (all filters aside): {len(pending)} rollout(s) "
        f"(~{pending_gb:.1f} GB)"
    )
    print("======================")


def cmd_metadata(args):
    data_dir = Path(args.data_dir)
    records, diff = refresh_metadata(args.benchmark, data_dir)
    print(f"Metadata refreshed -> {data_dir / args.benchmark / 'index.json'}")
    print_summary(records, diff, data_dir)
    return records


def cmd_download(args):
    data_dir = Path(args.data_dir)
    records = cmd_metadata(args)
    records = filter_records(records, task=args.task, user=args.user)
    if not records:
        print("No runs match the given filters.")
        return

    state = StateStore(data_dir / "state.json")
    jobs = collect_jobs(records, data_dir, state)
    if args.limit:
        jobs = jobs[: args.limit]

    if args.dry_run:
        est_gb = len(jobs) * AVG_RRD_MB / 1024
        print(f"Dry run: {len(jobs)} rollout file(s) to download (~{est_gb:.1f} GB).")
        for job in jobs[:20]:
            print(f"  {job['label']}  {job['url']}")
        if len(jobs) > 20:
            print(f"  ... and {len(jobs) - 20} more")
        return

    # Restrict records to the selected jobs by rebuilding a rollout-id filter.
    if args.limit:
        keep = {job["rollout_id"] for job in jobs}
        records = [
            {**r, "rollouts": [ro for ro in r["rollouts"] if ro["rollout_id"] in keep]}
            for r in records
        ]

    n_ok, n_failed = download_all(
        records, data_dir, workers=args.workers, rate_limit_mbps=args.rate_limit
    )
    if n_failed:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="RoboChallenge run crawler")
    parser.add_argument(
        "--benchmark",
        default=config.DEFAULT_BENCHMARK,
        choices=sorted(config.API_BASES),
        help="benchmark key (default: %(default)s)",
    )
    parser.add_argument(
        "--data-dir",
        default=str(config.DATA_DIR),
        help="output directory (default: %(default)s)",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("metadata", help="refresh metadata only")

    dl = sub.add_parser("download", help="refresh metadata, then download rrd files")
    dl.add_argument("--task", help="filter by task_name or task_id")
    dl.add_argument("--user", help="filter by user_name substring")
    dl.add_argument("--workers", type=int, default=config.DEFAULT_WORKERS)
    dl.add_argument("--limit", type=int, help="max rollout files this invocation")
    dl.add_argument("--dry-run", action="store_true")
    dl.add_argument(
        "--rate-limit",
        type=float,
        default=config.DEFAULT_RATE_LIMIT_MBPS,
        help="total download bandwidth cap in MB/s, 0 = unlimited "
        "(default: %(default)s)",
    )

    args = parser.parse_args()

    if args.command == "metadata":
        cmd_metadata(args)
    else:
        if args.command is None:
            # bare invocation = metadata + download everything
            args.task = args.user = args.limit = None
            args.workers = config.DEFAULT_WORKERS
            args.rate_limit = config.DEFAULT_RATE_LIMIT_MBPS
            args.dry_run = False
        cmd_download(args)


if __name__ == "__main__":
    main()
