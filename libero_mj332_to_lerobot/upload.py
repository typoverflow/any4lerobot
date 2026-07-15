#!/usr/bin/env python3
"""Upload LIBERO MuJoCo 3.3.2 v3 partitions to separate HF dataset repos."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_ROOT = Path("/scratch/cgao304/dev/FastWAM/data/datasets/lerobot_v3")
PARTITIONS = (
    "libero_10_no_noops_lerobot",
    "libero_goal_no_noops_lerobot",
    "libero_object_no_noops_lerobot",
    "libero_spatial_no_noops_lerobot",
)
REPOSITORIES = {
    "libero_10_no_noops_lerobot": "libero_10_mj332",
    "libero_goal_no_noops_lerobot": "libero_goal_mj332",
    "libero_object_no_noops_lerobot": "libero_object_mj332",
    "libero_spatial_no_noops_lerobot": "libero_spatial_mj332",
}


def prepare_upload_cache(folder: Path, repo_id: str) -> None:
    """Keep resumable upload metadata only when it belongs to this destination."""
    cache_root = folder / ".cache" / "huggingface"
    upload_cache = cache_root / "upload"
    repo_marker = cache_root / "upload_repo_id"
    cached_repo = repo_marker.read_text().strip() if repo_marker.is_file() else None
    if cached_repo != repo_id:
        shutil.rmtree(upload_cache, ignore_errors=True)
        cache_root.mkdir(parents=True, exist_ok=True)
        repo_marker.write_text(repo_id + "\n")


def wait_for_remote_files(
    api: HfApi, repo_id: str, required: list[str], attempts: int = 10
) -> list[str]:
    """Allow a short window for Hub file-listing eventual consistency."""
    remote_files: list[str] = []
    for attempt in range(attempts):
        remote_files = api.list_repo_files(repo_id, repo_type="dataset")
        if all(path in remote_files for path in required):
            return remote_files
        if attempt + 1 < attempts:
            time.sleep(2)
    return remote_files


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
    root = args.root.resolve()
    partitions = args.partitions or list(PARTITIONS)
    api = HfApi()
    for partition in partitions:
        folder = root / REPOSITORIES[partition]
        required = [
            folder / "README.md",
            folder / "meta" / "info.json",
            folder / "meta" / "stats.json",
            folder / "meta" / "conversion_config.json",
            folder / "meta" / "conversion_validation.json",
            folder / "meta" / "noop_audit.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete conversion; missing: {missing}")
        info = json.loads((folder / "meta" / "info.json").read_text())
        repo_id = f"{args.namespace}/{REPOSITORIES[partition]}"
        print(
            f"uploading {folder} -> {repo_id} "
            f"({info['total_episodes']} episodes, {info['total_frames']} frames)"
        )
        prepare_upload_cache(folder, repo_id)
        api.create_repo(
            repo_id, repo_type="dataset", private=args.private, exist_ok=True
        )
        api.upload_large_folder(
            repo_id=repo_id,
            folder_path=folder,
            repo_type="dataset",
            num_workers=args.workers,
            print_report=True,
            print_report_every=60,
        )
        required_remote = [
            "README.md",
            "meta/info.json",
            "meta/stats.json",
            "meta/conversion_config.json",
            "meta/conversion_validation.json",
            "meta/noop_audit.json",
        ]
        remote_files = wait_for_remote_files(api, repo_id, required_remote)
        missing_remote = [path for path in required_remote if path not in remote_files]
        if missing_remote:
            raise RuntimeError(
                f"upload verification failed for {repo_id}; missing: {missing_remote}"
            )
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
