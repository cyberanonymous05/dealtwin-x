"""Write a blind human-review sheet for the 60-filing deal-price audit.

Fill `human_correct` with Y/N (and `true_price` if N) after opening `url` and
checking the per-share cash consideration. Then run score_human_audit.py.
"""
import csv, json, pathlib, random

root = pathlib.Path(__file__).resolve().parents[1]
rows = json.load(open(root / "results/audit_deal_price_sample.json"))
random.Random(0).shuffle(rows)  # order differs from the file the original grader saw
out = root / "human_audit/price_audit_sheet.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "company", "form", "filing_date", "url", "extracted_cash_per_share",
                "context", "all_cash_deal(Y/N)", "human_correct(Y/N)", "true_price", "notes"])
    for i, r in enumerate(rows, 1):
        w.writerow([i, r["company"], r["form"], r["filing_date"], r["url"],
                    r["extracted_cash_per_share"], r["context"], "", "", "", ""])
print("wrote", out, len(rows), "rows")
