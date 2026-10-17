"""Rate-limited, disk-cached HTTP access to SEC endpoints.

Every SEC response the pipeline uses (filing documents, XBRL company facts, submissions histories) is
fetched once and stored gzip-compressed under data/raw/, so re-running parsing or experiments never
re-hits SEC and the exact bytes the paper's numbers come from can be frozen and hashed.

SEC fair-access policy allows at most 10 requests/second; we stay at 8 with a process-wide token bucket
so the downloader can use a thread pool safely.
"""

import gzip
import os
import threading
import time

import requests

_RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "raw")


class _RateLimiter:
    def __init__(self, per_second: float):
        self._interval = 1.0 / per_second
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                time.sleep(self._next - now)
                now = self._next
            self._next = now + self._interval


SEC_LIMITER = _RateLimiter(per_second=8.0)
_session = requests.Session()


def raw_path(kind: str, key: str, ext: str) -> str:
    return os.path.join(_RAW_DIR, kind, f"{key}.{ext}.gz")


def sec_get(url: str, user_agent: str, timeout: int = 60, retries: int = 5) -> requests.Response | None:
    """GET a SEC URL under the shared rate limit. Returns None on 404, raises after repeated failures."""
    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
    for attempt in range(retries):
        SEC_LIMITER.wait()
        try:
            response = _session.get(url, headers=headers, timeout=timeout)
        except requests.exceptions.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
            continue
        if response.status_code == 404:
            return None
        if response.status_code in (429, 503) or response.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        response.raise_for_status()
        return response
    raise RuntimeError(f"SEC request failed after {retries} attempts: {url}")


def cached_fetch(kind: str, key: str, ext: str, url: str, user_agent: str) -> str | None:
    """Return the cached body for (kind, key), downloading it once if absent. A 404 is cached as an
    empty marker so it is not retried on every run."""
    path = raw_path(kind, key, ext)
    if os.path.exists(path):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            body = f.read()
        return body or None
    response = sec_get(url, user_agent)
    body = response.text if response is not None else ""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        f.write(body)
    os.replace(tmp, path)
    return body or None
