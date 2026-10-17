"""Stage 1 -- discovery. Query SEC EDGAR full-text search for definitive merger proxies (DEFM14A/DEFM14C)
and cash-tender-offer recommendation statements (SC 14D9), 2013-2025, and save one candidate per target CIK.

Queries are split by calendar year so no single query approaches EDGAR's result cap, and only the
primary document of an original (non-amended) filing is kept: hits whose `file_type` is an exhibit or
whose form ends in "/A" are dropped. Output: data/processed/candidates.json plus discovery stats.
"""

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion.http_cache import sec_get
from src.utils.config import load_config
from src.utils.logging_utils import get_logger

logger = get_logger("discover_filings")
_cfg = load_config()["data_sources"]["sec_edgar"]
USER_AGENT = _cfg["user_agent"]
EFTS_URL = "https://efts.sec.gov/LATEST/search-index"

PROXY_FORMS = ["DEFM14A", "DEFM14C"]
TENDER_FORMS = ["SC 14D9"]
PROXY_QUERIES = _cfg["proxy_queries"]
TENDER_QUERIES = _cfg["tender_queries"]
START_YEAR, END_YEAR = _cfg["start_year"], _cfg["end_year"]
END_DATE_LAST_YEAR = _cfg["end_date"]

OUT_PATH = os.path.join("data", "processed", "candidates.json")
STATS_PATH = os.path.join("data", "processed", "discovery_stats.json")


def _search(query: str, forms: list[str], start: str, end: str) -> list[dict]:
    from urllib.parse import urlencode

    hits, offset = [], 0
    while True:
        params = {"q": query, "forms": ",".join(forms), "dateRange": "custom", "startdt": start, "enddt": end, "from": offset}
        response = sec_get(f"{EFTS_URL}?{urlencode(params)}", USER_AGENT, timeout=30)
        payload = response.json() if response is not None else {}
        page = payload.get("hits", {}).get("hits", [])
        hits.extend(page)
        total = payload.get("hits", {}).get("total", {}).get("value", 0)
        offset += len(page)
        if not page or offset >= total or offset >= 9900:
            return hits


def _normalize(hit: dict) -> dict | None:
    source = hit["_source"]
    form = source.get("form") or ""
    if form.endswith("/A") or source.get("file_type") != form:
        return None
    if not source.get("ciks") or not (source.get("sics") or [None])[0]:
        return None
    accession, filename = hit["_id"].split(":", 1)
    return {
        "cik": source["ciks"][0],
        "company_name": source["display_names"][0],
        "form": form,
        "filing_date": source["file_date"],
        "sic": source["sics"][0],
        "biz_state": (source.get("biz_states") or [None])[0],
        "inc_state": (source.get("inc_states") or [None])[0],
        "accession_number": accession,
        "primary_filename": filename,
    }


def main():
    raw_hits = Counter()
    by_accession = {}
    for forms, queries in ((PROXY_FORMS, PROXY_QUERIES), (TENDER_FORMS, TENDER_QUERIES)):
        for year in range(START_YEAR, END_YEAR + 1):
            start = f"{year}-01-01"
            end = END_DATE_LAST_YEAR if year == END_YEAR else f"{year}-12-31"
            for query in queries:
                hits = _search(query, forms, start, end)
                raw_hits[forms[0]] += len(hits)
                for hit in hits:
                    record = _normalize(hit)
                    if record:
                        by_accession[record["accession_number"]] = record
            logger.info(f"{forms[0]} {year}: {len(by_accession)} unique primary filings so far")

    # Keep EVERY original primary filing. The one-deal-per-target choice is made at parse time (earliest
    # filing per CIK that passes the M&A gate), because the earliest SC 14D9 is often only a short
    # pre-commencement communication, not the recommendation statement itself.
    candidates = sorted(by_accession.values(), key=lambda r: (r["filing_date"], r["accession_number"]))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(candidates, f, indent=1)
    stats = {
        "raw_hits": dict(raw_hits),
        "unique_primary_filings": len(by_accession),
        "unique_target_ciks": len({c["cik"] for c in candidates}),
        "by_form": dict(Counter(c["form"] for c in candidates)),
    }
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(json.dumps(stats))


if __name__ == "__main__":
    main()
