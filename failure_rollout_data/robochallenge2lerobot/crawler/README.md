# RoboChallenge Crawler

Downloads all submitted runs (metadata + rerun `.rrd` data) for RoboChallenge
benchmarks from https://robochallenge.ai. Defaults to the **Table 30 V2**
challenge. Only dependency: `requests`.

## How it works

The site serves everything as static JSON, no auth needed:

- `https://robochallenge.ai/api/v2/benchmark/benchmark_list.json` — task catalog
  (descriptions, robot tags, prompts)
- `https://robochallenge.ai/api/v2/runs/runs_table30_v2.json` — every run with
  its rollouts; each rollout's `rollout_details` field is the `.rrd` file URL
  on `video-preview.robochallenge.ai`

(`--benchmark table_30` switches to the older challenge on `/api/v1`.)

## Usage

```bash
# 1. Refresh metadata only (fast, ~2 MB)
python crawl.py metadata

# 2. See what a download would fetch
python crawl.py download --dry-run

# 3. Download with filters
python crawl.py download --task arrange_flowers            # by task_name or task_id
python crawl.py download --user "Yuze Xuan"                # by user substring
python crawl.py download --task 9d9d --limit 20 --workers 8

# 4. Download EVERYTHING (~5700 rollouts, several hundred GB — run detached)
nohup python crawl.py download > crawl.log 2>&1 &
```

Every metadata fetch (including dry runs) prints an update summary: totals on
the server, what is new since the previous fetch (runs / rollouts / users,
with an estimated volume), and how many rollouts are still not downloaded.

Downloads are bandwidth-capped to be polite to the server: all workers share
a global limit of 16 MB/s by default. Tune with `--rate-limit MB_PER_S`
(`--rate-limit 0` disables the cap — not recommended for full crawls).

## Incremental updates

Just re-run `python crawl.py download`. Every invocation re-fetches the runs
JSON, diffs it against `data/state.json` (rollout_id -> downloaded file +
size), and only downloads rollouts that are new or incomplete. New submissions
are picked up automatically; existing files are never re-downloaded.
Interrupted downloads leave a `.part` file that resumes via HTTP Range.
Metadata files are always rewritten so scores stay current.

## Output layout

```
data/
  raw/                            # raw API snapshots
    benchmark_list.json
    runs_table30_v2.json
  table30_v2/
    index.json                    # all runs, one normalized record each
    <task_name>_<task_id>/
      <run_id>/
        metadata.json             # challenge, task, user, run_id, hardware
                                  # (robot + arena), model, score, exec time,
                                  # rollout list with data URLs
        rollouts/
          <rollout_id>.rrd        # rerun data (open with app.rerun.io, v0.24.1)
  state.json                      # incremental download ledger
```

Hardware info per run: `robot` (ARX5 / UR5 / W1 / ALOHA) comes from the task's
tags in the benchmark catalog; `arenas` is the physical arena machine parsed
from the rrd URLs (e.g. `arena_data_rc_arx5_3`).

## Files

- `crawl.py` — CLI entry point
- `config.py` — API endpoints, paths, retry/concurrency settings
- `metadata.py` — fetch + normalize run metadata
- `downloader.py` — resumable, parallel rrd downloader + state ledger
