"""Discover and download real M&A merger filings (DEFM14A/DEFM14C/S-4) from SEC EDGAR full-text search."""

import re
import time

import requests

EDGAR_FULL_TEXT_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_no_leading_zeros}/{accession_nodash}/{filename}"

REQUEST_DELAY_SECONDS = 0.4  # stay well under SEC's fair-access rate limits


def _headers(user_agent: str) -> dict:
    return {"User-Agent": user_agent, "Accept": "application/json"}


def search_merger_filings(
    query: str,
    forms: list[str],
    start_date: str,
    end_date: str,
    user_agent: str,
    max_results: int = 100,
) -> list[dict]:
    """Query SEC EDGAR full-text search for merger-related filings and return normalized hit records."""
    results: list[dict] = []
    page_size = 10
    offset = 0
    while len(results) < max_results:
        params = {
            "q": query,
            "forms": ",".join(forms),
            "dateRange": "custom",
            "startdt": start_date,
            "enddt": end_date,
            "from": offset,
        }
        payload = None
        for attempt in range(5):
            try:
                response = requests.get(
                    EDGAR_FULL_TEXT_SEARCH_URL, params=params, headers=_headers(user_agent), timeout=20
                )
                response.raise_for_status()
                payload = response.json()
                break
            except (requests.exceptions.HTTPError, requests.exceptions.RequestException):
                if attempt == 4:
                    raise
                time.sleep(0.5 * (attempt + 1))
        if payload is None:
            break
        hits = payload.get("hits", {}).get("hits", [])
        if not hits:
            break
        for hit in hits:
            source = hit["_source"]
            doc_id = hit["_id"]  # format "<accession-with-dashes>:<filename>"
            accession, filename = doc_id.split(":")
            results.append(
                {
                    "cik": source["ciks"][0].lstrip("0"),
                    "cik_padded": source["ciks"][0],
                    "company_name": source["display_names"][0],
                    "form": source.get("form"),
                    "filing_date": source.get("file_date"),
                    "sic": (source.get("sics") or [None])[0],
                    "accession_number": accession,
                    "primary_filename": filename,
                }
            )
            if len(results) >= max_results:
                break
        offset += page_size
        time.sleep(REQUEST_DELAY_SECONDS)
        if offset > payload.get("hits", {}).get("total", {}).get("value", 0):
            break
    return results


def build_filing_url(cik: str, accession_number: str, filename: str) -> str:
    return EDGAR_ARCHIVES_URL.format(
        cik_no_leading_zeros=cik.lstrip("0"),
        accession_nodash=accession_number.replace("-", ""),
        filename=filename,
    )


def get_company_submissions(cik: str, user_agent: str) -> dict:
    response = requests.get(
        EDGAR_SUBMISSIONS_URL.format(cik=cik.zfill(10)), headers=_headers(user_agent), timeout=20
    )
    response.raise_for_status()
    return response.json()


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def download_filing_text(url: str, user_agent: str) -> str:
    """Download a filing document and strip HTML markup down to plain text."""
    for attempt in range(4):
        try:
            response = requests.get(url, headers=_headers(user_agent), timeout=30)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    time.sleep(REQUEST_DELAY_SECONDS)
    html = response.text
    html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    text = _TAG_RE.sub(" ", html)
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&#x[0-9a-fA-F]+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    text = _WS_RE.sub(" ", text)
    text = re.sub(r"\n ", "\n", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()
