"""Blind expert-annotation set: do the cascade's top-5 precedents differ from the SIC back-off screen's top-5
in a way an M&A practitioner can see?

Reuses the outcome-construct, industry-fold protocol of run_retrieval_study.py. For N sampled queries whose two
top-5 lists differ, the union of the candidates is shuffled and shown WITHOUT system labels. The expert rates
each candidate's comparability to the query (1 = not comparable ... 5 = highly comparable).
The system key is written to a separate file. Score with score_expert_annotations.py.

Usage: python scripts/make_expert_annotation_set.py [n_queries=30]
"""
import csv, json, os, random, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_retrieval_study import BLOCKS, context, load  # noqa: E402
from src.evaluation.engine import causal_mask, make_folds  # noqa: E402
from src.retrieval import retrievers as R  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
K = 5
deals, attrs, emb = load()
ctx, rel = context(deals, attrs, emb, "outcome")
pool = ~np.isnan(attrs["log_multiple"])
mask = causal_mask(attrs["date_ord"], True, pool)
tiebreak = np.random.default_rng(0).random(len(deals))
systems = {"sic_backoff": R.sic_backoff(ctx), "cascade": R.lambdamart_cascade(ctx, BLOCKS)}

tops = {s: {} for s in systems}
for train_idx, test_idx in make_folds(attrs, "sic"):
    for s, retr in systems.items():
        scores, _ = retr(np.asarray(train_idx), np.asarray(test_idx))
        for row, q in enumerate(test_idx):
            if not mask[q].any():
                continue
            sc = np.where(mask[q], scores[row], -np.inf)
            order = np.lexsort((tiebreak, -sc))[:K]
            tops[s][int(q)] = [int(i) for i in order if mask[q][i]]

rng = random.Random(0)
both = [q for q in tops["cascade"] if q in tops["sic_backoff"]
        and len(tops["cascade"][q]) == K and set(tops["cascade"][q]) != set(tops["sic_backoff"][q])]
queries = rng.sample(sorted(both), min(N, len(both)))


def desc(i):
    d = deals[i]
    rev = d.get("revenues")
    return [d["company_name"], d["sic"], d["filing_date"], d["deal_type"],
            f"{rev/1e6:,.0f}" if rev else "n/a", d["source_url"]]


os.makedirs("human_audit", exist_ok=True)
key = []
with open("human_audit/expert_annotation_sheet.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["item", "role", "company", "SIC", "filing_date", "deal_type", "revenue_USDm", "url",
                "comparability_1to5 (fill for CANDIDATE rows)", "notes"])
    for n, q in enumerate(queries, 1):
        w.writerow([f"Q{n}", "QUERY"] + desc(q) + ["", ""])
        cands = sorted(set(tops["cascade"][q]) | set(tops["sic_backoff"][q]))
        rng.shuffle(cands)
        for j, c in enumerate(cands, 1):
            w.writerow([f"Q{n}.{j}", "CANDIDATE"] + desc(c) + ["", ""])
            key.append({"item": f"Q{n}.{j}", "query": q, "candidate": c,
                        "in_cascade": c in tops["cascade"][q], "in_sic_backoff": c in tops["sic_backoff"][q],
                        "cascade_rank": tops["cascade"][q].index(c) + 1 if c in tops["cascade"][q] else None,
                        "sic_rank": tops["sic_backoff"][q].index(c) + 1 if c in tops["sic_backoff"][q] else None})
json.dump(key, open("human_audit/expert_annotation_KEY_do_not_share.json", "w"), indent=1)
print(len(queries), "queries,", len(key), "candidate rows")
