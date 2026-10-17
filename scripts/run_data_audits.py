"""Data-quality audits reported in the paper.

1. Point-in-time XBRL: for each deal, compare the revenue chosen under our rule (fact FILED on or before
   the deal date) with the value an end-date-only rule would choose (period ENDED before the deal, any
   filing date, latest filing wins). A difference means the end-date-only rule would have used a value
   disclosed or restated after announcement -- look-ahead leakage.
2. Deal-price audit sample: 60 random deals with an extracted cash price, with the sentence the price was
   taken from, written to results/audit_deal_price_sample.json for manual verification.
Output: results/data_audits.json
"""

import gzip
import json
import os
import re
import sys
from datetime import date

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion.http_cache import raw_path
from src.data_ingestion.xbrl_facts import _TAGS, _days
from src.dealdna.clause_extraction import html_to_text

RESULTS = os.environ.get("DTX_RESULTS", "results")


def end_date_only_revenue(facts, as_of):
    for taxonomy, tags in _TAGS["revenues"].items():
        for tag in tags:
            data = facts.get(taxonomy, {}).get(tag)
            if not data:
                continue
            entries = data.get("units", {}).get("USD") or next(iter(data.get("units", {}).values()), [])
            annual = [e for e in entries if e.get("val") and e["val"] > 0 and e.get("end", "9999") <= as_of
                      and (d := _days(e)) and 350 <= d <= 380]
            if annual:
                latest_end = max(e["end"] for e in annual)
                return max((e for e in annual if e["end"] == latest_end), key=lambda e: e.get("filed", ""))
    return None


def main():
    deals = json.load(open(os.path.join("data", "processed", "deal_universe.json")))
    n_cmp = n_diff = n_late = 0
    rel_gaps = []
    for d in deals:
        prov = (d.get("xbrl_provenance") or {}).get("revenues", "")
        if not prov.endswith(":annual") or d.get("revenues") is None:
            continue
        with gzip.open(raw_path("xbrl", d["cik"], "json"), "rt") as f:
            facts = json.load(f).get("facts", {})
        alt = end_date_only_revenue(facts, d["filing_date"])
        if alt is None:
            continue
        n_cmp += 1
        if alt["filed"] > d["filing_date"]:
            n_late += 1
        if abs(alt["val"] - d["revenues"]) > 1e-6 * max(1.0, abs(d["revenues"])):
            n_diff += 1
            rel_gaps.append(abs(alt["val"] - d["revenues"]) / abs(d["revenues"]))
    audits = {"point_in_time": {
        "deals_compared": n_cmp,
        "end_date_rule_uses_fact_filed_after_deal": n_late,
        "share_filed_after_deal": round(n_late / max(1, n_cmp), 4),
        "revenue_value_differs": n_diff,
        "share_value_differs": round(n_diff / max(1, n_cmp), 4),
        "median_relative_gap_when_different": round(float(np.median(rel_gaps)), 4) if rel_gaps else None,
    }}

    rng = np.random.default_rng(2026)  # fresh sample: seed 7 was the development sample used to fix the extractor
    priced = [d for d in deals if d.get("cash_per_share")]
    sample = []
    for i in rng.choice(len(priced), size=min(60, len(priced)), replace=False):
        d = priced[i]
        with gzip.open(raw_path("filings", d["deal_id"], "html"), "rt", errors="replace") as f:
            text = html_to_text(f.read())
        price = f"{d['cash_per_share']:,.2f}"
        m = re.search(r"[^.]{0,220}\$\s?" + re.escape(price) + r"[^.]{0,160}", text)
        sample.append({"company": d["company_name"], "form": d["form"], "filing_date": d["filing_date"],
                       "url": d["source_url"], "extracted_cash_per_share": d["cash_per_share"],
                       "context": re.sub(r"\s+", " ", m.group(0)).strip() if m else None})
    json.dump(sample, open(os.path.join(RESULTS, "audit_deal_price_sample.json"), "w"), indent=1)
    audits["deal_price_audit_sample_size"] = len(sample)
    json.dump(audits, open(os.path.join(RESULTS, "data_audits.json"), "w"), indent=1)
    print(json.dumps(audits, indent=1))


if __name__ == "__main__":
    main()
