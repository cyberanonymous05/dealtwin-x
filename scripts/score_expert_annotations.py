"""Score a filled expert sheet: mean rating of cascade top-5 vs SIC back-off top-5, paired over queries,
with a bootstrap 95% CI and a sign test. Also Spearman correlation between rating and realized-multiple proximity is
left to the caller."""
import csv, json, pathlib, random, sys
from math import comb

root = pathlib.Path(__file__).resolve().parents[1] / "human_audit"
rows = {r["item"]: r for r in csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1 else root / "expert_annotation_sheet.csv"))}
key = json.load(open(root / "expert_annotation_KEY_do_not_share.json"))
col = "comparability_1to5 (fill for CANDIDATE rows)"
by_q = {}
for k in key:
    v = rows[k["item"]][col].strip()
    if v:
        by_q.setdefault(k["query"], []).append((int(v), k))
diffs = []
for q, items in by_q.items():
    c = [v for v, k in items if k["in_cascade"]]; s = [v for v, k in items if k["in_sic_backoff"]]
    if c and s:
        diffs.append(sum(c) / len(c) - sum(s) / len(s))
n = len(diffs); mean = sum(diffs) / n
rng = random.Random(0)
boot = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(10000))
wins, losses = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs); m = wins + losses
p = min(1.0, 2 * sum(comb(m, i) for i in range(min(wins, losses) + 1)) / 2 ** m) if m else 1.0
print(f"{n} queries; mean rating diff (cascade - SIC) = {mean:+.3f}  95% CI [{boot[250]:+.3f}, {boot[9750]:+.3f}]")
print(f"cascade better in {wins}, worse in {losses}, tied {n-m}; sign-test p = {p:.3f}")
