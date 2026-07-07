"""Configuration for the RoboChallenge crawler."""

from pathlib import Path

SITE_BASE = "https://robochallenge.ai"

# Each benchmark is served from its own static API prefix.
API_BASES = {
    "table_30": f"{SITE_BASE}/api/v1",
    "table30_v2": f"{SITE_BASE}/api/v2",
}

# Filename of the runs listing per benchmark (as used by the site frontend).
RUNS_FILES = {
    "table_30": "runs_list.json",
    "table30_v2": "runs_table30_v2.json",
}

DEFAULT_BENCHMARK = "table30_v2"

# Robot tags that appear in benchmark task_tag lists.
KNOWN_ROBOTS = {"ARX5", "UR5", "W1", "ALOHA"}

DATA_DIR = Path(__file__).resolve().parent / "data"

HTTP_TIMEOUT = 60  # seconds, per request
DOWNLOAD_TIMEOUT = 300  # seconds, per rrd download request
MAX_RETRIES = 5
RETRY_BACKOFF = 2.0  # exponential backoff base, seconds
CHUNK_SIZE = 1024 * 1024  # 1 MiB streaming chunks
DEFAULT_WORKERS = 4
# Global download bandwidth cap shared by all workers (MB/s); 0 = unlimited.
DEFAULT_RATE_LIMIT_MBPS = 8.0

USER_AGENT = "robochallenge-crawler/1.0"


def api_base(benchmark: str) -> str:
    return API_BASES[benchmark]


def benchmark_list_url(benchmark: str) -> str:
    return f"{api_base(benchmark)}/benchmark/benchmark_list.json"


def runs_url(benchmark: str) -> str:
    return f"{api_base(benchmark)}/runs/{RUNS_FILES[benchmark]}"
