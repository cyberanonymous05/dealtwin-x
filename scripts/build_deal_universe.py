"""Stage 3 -- parse. Build the deal universe from the raw cache only (no network).

For every candidate: clean the filing text, apply the M&A gate, extract deal terms / stated premium /
acquirer attributes / legal-term flags, compute point-in-time XBRL fundamentals, look up FRED context,
determine completion status from the target's later EDGAR filings, extract the 10-K business description,
and select legal-clause passages (sentence embeddings on GPU).

Every excluded candidate gets a reason, so the parsing yield reported in the paper comes straight from
data/processed/parse_stats.json.
"""

import gzip
import json
import os
import re
import sys
from collections import Counter
from datetime import date, timedelta
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.data_ingestion.fred_data import get_macro_snapshot
from src.data_ingestion.http_cache import raw_path
from src.data_ingestion.xbrl_facts import extract_financial_fundamentals
from src.dealdna.clause_extraction import (
    extract_acquirer_attributes, extract_deal_terms, extract_legal_flags, extract_stated_premium_pct,
    html_to_text, ma_gate,
)
from src.utils.logging_utils import get_logger

logger = get_logger("build_deal_universe")
CANDIDATES_PATH = os.path.join("data", "processed", "candidates.json")
OUT_PATH = os.path.join("data", "processed", "deal_universe.json")
STATS_PATH = os.path.join("data", "processed", "parse_stats.json")
EXCLUDED_SIC = {"6770"}  # blank-check companies (SPAC de-SPAC proxies: the economic target is private)
DEREGISTRATION_FORMS = {"25-NSE", "25", "15-12B", "15-12G", "15-15D", "15F-12B", "15F-12G", "15F-15D"}


def _read(kind: str, key: str, ext: str) -> str | None:
    path = raw_path(kind, key, ext)
    if not os.path.exists(path):
        return None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        body = f.read()
    return body or None


def _submission_rows(cik: str) -> list[dict]:
    body = _read("subs", cik, "json")
    if not body:
        return []
    subs = json.loads(body)
    tables = [subs.get("filings", {}).get("recent", {})]
    for extra in subs.get("filings", {}).get("files", []):
        extra_body = _read("subs", extra["name"].replace(".json", ""), "json")
        if extra_body:
            tables.append(json.loads(extra_body))
    rows = []
    for t in tables:
        for i in range(len(t.get("accessionNumber", []))):
            rows.append({k: t[k][i] for k in ("accessionNumber", "filingDate", "form", "primaryDocument")})
    return rows


def completion_status(rows: list[dict], filing_date: str) -> str:
    """'completed' if the target deregistered within 18 months of the deal filing; 'terminated' if it
    instead kept filing periodic reports more than 12 months later; otherwise 'unknown' (e.g. pending)."""
    start = date.fromisoformat(filing_date)
    later = [r for r in rows if r["filingDate"] > filing_date]
    if any(r["form"] in DEREGISTRATION_FORMS and date.fromisoformat(r["filingDate"]) <= start + timedelta(days=548) for r in later):
        return "completed"
    if any(r["form"] in ("10-K", "10-Q") and date.fromisoformat(r["filingDate"]) > start + timedelta(days=365) for r in later):
        return "terminated"
    return "unknown"


_ITEM1_RE = re.compile(r"item\s*1\.?\s*[-:.]?\s*business\b", re.I)
_ITEM1_END_RE = re.compile(r"item\s*1a\.?\s*[-:.]?\s*risk\s+factors|item\s*2\.?\s*[-:.]?\s*properties", re.I)


def business_description(rows: list[dict], cik: str, filing_date: str) -> str:
    tenks = [r for r in rows if r["form"] in ("10-K", "10-K405", "10-KSB", "20-F", "40-F") and r["filingDate"] < filing_date]
    if not tenks:
        return ""
    tenk = max(tenks, key=lambda r: r["filingDate"])
    html = _read("tenk", f"{cik}_{tenk['accessionNumber']}", "html")
    if not html:
        return ""
    text = html_to_text(html)
    # The first "Item 1. Business" match is usually the table of contents; take the occurrence followed by
    # the longest stretch of text before Item 1A / Item 2.
    best = ""
    for m in _ITEM1_RE.finditer(text):
        end = _ITEM1_END_RE.search(text, m.end())
        section = text[m.end(): end.start() if end else m.end() + 40_000]
        if len(section) > len(best):
            best = section
    return re.sub(r"\s+", " ", best)[:20_000].strip()


def enterprise_value(fundamentals: dict, terms: dict) -> float | None:
    """Cash price x shares + long-term debt - cash, for ALL-CASH deals only. For stock and cash-and-stock
    deals the per-share cash figure is not the value of the consideration (audit: most price errors were
    in such deals), and free data does not give the acquirer's historical share price."""
    shares = fundamentals.get("shares_outstanding")
    if terms.get("consideration_mix") != "cash" or not terms.get("cash_per_share") or not shares:
        return None
    if not fundamentals.get("reports_in_usd", True):
        return None
    ev = terms["cash_per_share"] * shares + (fundamentals.get("long_term_debt") or 0) - (fundamentals.get("cash_and_equivalents") or 0)
    rev = fundamentals.get("revenues")
    if ev <= 0 or (rev and not 0.05 <= ev / rev <= 100):
        return None  # implausible multiple: almost always a mis-parsed price or share count
    return ev


def parse_candidate(c: dict) -> dict:
    try:
        return _parse_candidate(c)
    except Exception as exc:  # one malformed filing must not abort the corpus build
        return {"excluded": "parse_error", "cik": c["cik"], "form": c["form"], "error": str(exc)[:200]}


def _parse_candidate(c: dict) -> dict:
    """CPU part of parsing. Returns {'excluded': reason} or a record (passages added later on GPU)."""
    if c["sic"] in EXCLUDED_SIC:
        return {"excluded": "spac_blank_check", "cik": c["cik"], "form": c["form"]}
    html = _read("filings", f"{c['cik']}_{c['accession_number']}", "html")
    if not html:
        return {"excluded": "download_failed", "cik": c["cik"], "form": c["form"]}
    text = html_to_text(html)
    reason = ma_gate(text, c['form'])
    if reason:
        return {"excluded": reason, "cik": c["cik"], "form": c["form"]}

    facts_body = _read("xbrl", c["cik"], "json")
    fundamentals = extract_financial_fundamentals(json.loads(facts_body) if facts_body else None, c["filing_date"])
    if fundamentals["revenues"] is None and fundamentals["total_assets"] is None:
        return {"excluded": "no_point_in_time_xbrl", "cik": c["cik"], "form": c["form"]}

    terms = extract_deal_terms(text)
    rows = _submission_rows(c["cik"])
    ev = enterprise_value(fundamentals, terms)

    from src.dealdna.semantic_clause_classifier import segment_candidate_sentences

    return {
        "deal_id": f"{c['cik']}_{c['accession_number']}",
        "company_name": c["company_name"],
        "cik": c["cik"],
        "sic": c["sic"],
        "form": c["form"],
        "deal_type": "tender_offer" if c["form"] == "SC 14D9" else "merger_proxy",
        "filing_date": c["filing_date"],
        "source_url": f"https://www.sec.gov/Archives/edgar/data/{int(c['cik'])}/{c['accession_number'].replace('-', '')}/{c['primary_filename']}",
        "biz_state": c.get("biz_state"),
        **fundamentals,
        "ev_estimate": ev,
        "deal_premium": extract_stated_premium_pct(text),
        **terms,
        **extract_acquirer_attributes(text, terms["consideration_mix"]),
        "legal_flags": extract_legal_flags(text),
        "completion_status": completion_status(rows, c["filing_date"]),
        "business_description": business_description(rows, c["cik"], c["filing_date"]),
        "macro_snapshot": get_macro_snapshot(c["filing_date"]),
        "_candidate_sentences": segment_candidate_sentences(text),
    }


def main():
    with open(CANDIDATES_PATH) as f:
        candidates = json.load(f)
    logger.info(f"parsing {len(candidates)} candidates from the raw cache")
    with Pool(processes=min(24, os.cpu_count())) as pool:
        results = pool.map(parse_candidate, candidates, chunksize=8)

    # One deal per target: the earliest filing per CIK that passes every gate. Filings of a target that
    # already has a deal are not counted as exclusions.
    deals, seen = [], set()
    for r in sorted((r for r in results if "excluded" not in r), key=lambda r: (r["filing_date"], r["deal_id"])):
        if r["cik"] not in seen:
            seen.add(r["cik"])
            deals.append(r)
    excluded = Counter((r["form"], r["excluded"]) for r in results if "excluded" in r and r["cik"] not in seen)
    excluded_ciks = {r["cik"] for r in results if "excluded" in r and r["cik"] not in seen}
    logger.info(f"{len(deals)} deals passed parsing; selecting clause passages on GPU")

    from src.dealdna.semantic_clause_classifier import KEYWORD_ANCHORS, encode_sentences, select_from_embeddings
    import numpy as np

    # Pass 1: purely semantic selection (no lexical gate).
    sentence_cache = []
    for n, d in enumerate(deals, 1):
        sentences = d.pop("_candidate_sentences")
        emb = encode_sentences(sentences) if sentences else np.zeros((0, 384), dtype=np.float32)
        sentence_cache.append((sentences, emb))
        picked = select_from_embeddings(sentences, emb) if sentences else []
        d["clause_passages_ungated"] = [p["text"] for p in picked]
        d["clause_passage_types_ungated"] = [p["type"] for p in picked]
        if n % 200 == 0:
            logger.info(f"clause selection {n}/{len(deals)}")

    # Selector precision per clause type against the keyword-anchored silver standard, on 120 random
    # filings: a selected passage of type t counts as correct if it contains t's keyword anchor.
    rng = np.random.default_rng(0)
    sample = rng.choice(len(deals), size=min(120, len(deals)), replace=False)
    hits, totals = Counter(), Counter()
    for i in sample:
        for text, ctype in zip(deals[i]["clause_passages_ungated"], deals[i]["clause_passage_types_ungated"]):
            totals[ctype] += 1
            hits[ctype] += bool(KEYWORD_ANCHORS[ctype].search(text))
    precision = {t: {"precision": round(hits[t] / totals[t], 3), "n_passages": totals[t]} for t in sorted(totals)}
    overall = round(sum(hits.values()) / max(1, sum(totals.values())), 3)
    gate = tuple(sorted(precision, key=lambda t: precision[t]["precision"])[:3])
    logger.info(f"ungated selector precision {overall}; lexical gate on {gate}")

    # Pass 2: final passages with the lexical gate on the three lowest-precision clause types.
    for d, (sentences, emb) in zip(deals, sentence_cache):
        picked = select_from_embeddings(sentences, emb, lexical_gate=gate) if sentences else []
        d["clause_passages"] = [p["text"] for p in picked]
        d["clause_passage_types"] = [p["type"] for p in picked]
    del sentence_cache
    gated_hits, gated_totals = Counter(), Counter()
    for i in sample:
        for text, ctype in zip(deals[i]["clause_passages"], deals[i]["clause_passage_types"]):
            gated_totals[ctype] += 1
            gated_hits[ctype] += bool(KEYWORD_ANCHORS[ctype].search(text))
    clause_stats = {
        "silver_standard_filings": int(len(sample)),
        "ungated": {"per_type": precision, "overall": overall},
        "lexical_gate_types": list(gate),
        "gated": {"per_type": {t: {"precision": round(gated_hits[t] / gated_totals[t], 3), "n_passages": gated_totals[t]}
                               for t in sorted(gated_totals)},
                  "overall": round(sum(gated_hits.values()) / max(1, sum(gated_totals.values())), 3)},
        "deals_with_passage": int(sum(1 for d in deals if d["clause_passages"])),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(deals, f, default=str)
    stats = {
        "n_candidate_filings": len(candidates),
        "n_candidate_targets": len({c["cik"] for c in candidates}),
        "n_targets_excluded": len(excluded_ciks),
        "n_candidates_by_form": dict(Counter(c["form"] for c in candidates)),
        "n_deals": len(deals),
        "n_deals_by_type": dict(Counter(d["deal_type"] for d in deals)),
        "excluded_filings_of_unmatched_targets": {f"{form}|{reason}": n for (form, reason), n in sorted(excluded.items())},
        "completion_status": dict(Counter(d["completion_status"] for d in deals)),
        "clause_selector": clause_stats,
    }
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
