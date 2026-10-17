"""Stage 2 -- download. For every discovered candidate, cache (once) under data/raw/:
  filings/   the merger proxy / SC 14D9 primary document
  xbrl/      the target's XBRL Company Facts JSON
  subs/      the target's EDGAR submissions history (used for completion / termination status and to
             locate the target's latest 10-K)
  tenk/      the primary document of the target's most recent 10-K filed BEFORE the deal filing date
             (business description for the text-based peer baseline)
Runs with a thread pool under the shared 8 req/s SEC rate limit; safe to interrupt and resume.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion.http_cache import cached_fetch
from src.utils.config import load_config
from src.utils.logging_utils import get_logger

logger = get_logger("download_filings")
USER_AGENT = load_config()["data_sources"]["sec_edgar"]["user_agent"]
CANDIDATES_PATH = os.path.join("data", "processed", "candidates.json")
ARCHIVES = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{fn}"


def filing_key(c: dict) -> str:
    return f"{c['cik']}_{c['accession_number']}"


def submissions(cik: str) -> dict | None:
    body = cached_fetch("subs", cik, "json", f"https://data.sec.gov/submissions/CIK{cik}.json", USER_AGENT)
    return json.loads(body) if body else None


def submission_rows(cik: str, subs: dict) -> list[dict]:
    """Flatten 'recent' plus any paged older submission files into one list of filing rows."""
    tables = [subs.get("filings", {}).get("recent", {})]
    for extra in subs.get("filings", {}).get("files", []):
        name = extra["name"]
        body = cached_fetch("subs", name.replace(".json", ""), "json", f"https://data.sec.gov/submissions/{name}", USER_AGENT)
        if body:
            tables.append(json.loads(body))
    rows = []
    for t in tables:
        for i in range(len(t.get("accessionNumber", []))):
            rows.append({k: t[k][i] for k in ("accessionNumber", "filingDate", "form", "primaryDocument")})
    return rows


def latest_10k_before(rows: list[dict], filing_date: str) -> dict | None:
    tenks = [r for r in rows if r["form"] in ("10-K", "10-K405", "10-KSB", "20-F", "40-F") and r["filingDate"] < filing_date]
    return max(tenks, key=lambda r: r["filingDate"]) if tenks else None


def process(c: dict) -> str:
    cik_int = str(int(c["cik"]))
    cached_fetch("filings", filing_key(c), "html",
                 ARCHIVES.format(cik=cik_int, acc=c["accession_number"].replace("-", ""), fn=c["primary_filename"]), USER_AGENT)
    cached_fetch("xbrl", c["cik"], "json", f"https://data.sec.gov/api/xbrl/companyfacts/CIK{c['cik']}.json", USER_AGENT)
    subs = submissions(c["cik"])
    if subs:
        tenk = latest_10k_before(submission_rows(c["cik"], subs), c["filing_date"])
        if tenk and tenk["primaryDocument"]:
            cached_fetch("tenk", f"{c['cik']}_{tenk['accessionNumber']}", "html",
                         ARCHIVES.format(cik=cik_int, acc=tenk["accessionNumber"].replace("-", ""), fn=tenk["primaryDocument"]),
                         USER_AGENT)
    return c["cik"]


def main():
    with open(CANDIDATES_PATH) as f:
        candidates = json.load(f)
    logger.info(f"downloading raw data for {len(candidates)} candidates")
    failures = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(process, c): c for c in candidates}
        for n, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as exc:
                failures.append({"cik": futures[future]["cik"], "error": str(exc)[:200]})
            if n % 100 == 0:
                logger.info(f"{n}/{len(candidates)} done, {len(failures)} failures")
    with open(os.path.join("data", "processed", "download_failures.json"), "w") as f:
        json.dump(failures, f, indent=1)
    logger.info(f"finished: {len(candidates) - len(failures)} ok, {len(failures)} failed")


if __name__ == "__main__":
    main()
