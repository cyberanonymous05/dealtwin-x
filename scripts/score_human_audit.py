"""Score the filled human audit sheet; prints overall and all-cash accuracy with Wilson 95% CIs."""
import csv, math, pathlib, sys

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                    pathlib.Path(__file__).resolve().parents[1] / "human_audit/price_audit_sheet.csv")
rows = [r for r in csv.DictReader(open(path)) if r["human_correct(Y/N)"].strip()]
def report(name, rs):
    k = sum(r["human_correct(Y/N)"].strip().upper() == "Y" for r in rs)
    lo, hi = wilson(k, len(rs))
    print(f"{name}: {k}/{len(rs)} = {k/max(len(rs),1):.1%}  (95% CI {lo:.1%}-{hi:.1%})")
report("overall", rows)
report("all-cash", [r for r in rows if r["all_cash_deal(Y/N)"].strip().upper() == "Y"])
