#!/usr/bin/env python3
"""Upload completed LIBERO Plus LeRobot v3 datasets to Hugging Face."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

PARTITIONS = (
    "libero_plus_10",
    "libero_plus_object",
    "libero_plus_goal",
    "libero_plus_spatial",
)
DEFAULT_ROOT = Path("/scratch/cgao304/dev/FastWAM/data/datasets/lerobot_v3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("partitions", nargs="*", metavar="PARTITION")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--namespace", default="typoverflow")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    invalid = sorted(set(args.partitions) - set(PARTITIONS))
    if invalid:
        parser.error(
            f"unknown partition(s): {', '.join(invalid)}; "
            f"choose from {', '.join(PARTITIONS)}"
        )
    return args


def main() -> None:
    args = parse_args()
    partitions = args.partitions or list(PARTITIONS)
    api = HfApi()
    for partition in partitions:
        folder = (args.root / partition).resolve()
        required = [
            folder / "meta" / "info.json",
            folder / "meta" / "stats.json",
            folder / "meta" / "conversion_config.json",
            folder / "meta" / "conversion_validation.json",
            folder / "meta" / "noop_audit.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete conversion; missing: {missing}")
        completed = json.loads((folder / "meta" / "info.json").read_text())
        repo_id = f"{args.namespace}/{partition}"
        print(
            f"uploading {folder} -> {repo_id} "
            f"({completed['total_episodes']} episodes, {completed['total_frames']} frames)"
        )
        api.create_repo(repo_id, repo_type="dataset", private=args.private, exist_ok=True)
        api.upload_large_folder(
            repo_id=repo_id,
            folder_path=folder,
            repo_type="dataset",
            num_workers=args.workers,
            print_report=True,
            print_report_every=60,
        )
        remote_files = api.list_repo_files(repo_id, repo_type="dataset")
        if "meta/.conversion_complete" in remote_files:
            api.delete_file(
                path_in_repo="meta/.conversion_complete",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Remove private conversion sentinel",
            )
        refs = api.list_repo_refs(repo_id, repo_type="dataset")
        if any(tag.name == "v3.0" for tag in refs.tags):
            api.delete_tag(repo_id, tag="v3.0", repo_type="dataset")
        api.create_tag(repo_id, tag="v3.0", revision="main", repo_type="dataset")
        print(f"uploaded https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
