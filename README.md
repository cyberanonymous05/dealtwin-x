# DealTwin-X

Multi-modal transaction embeddings ("DealDNA") for M&A precedent-transaction retrieval, evaluated under a
causal-temporal protocol against naive and practitioner-style industry/size baselines. This repository
reproduces every number, table and figure of the revised CICT-2026 paper (Paper ID 899).

All data comes from free public sources: SEC EDGAR full-text search, SEC XBRL Company Facts, EDGAR
submissions histories and FRED. Nothing is hand-curated.

## Reproduce

```bash
pip install -r requirements.txt
echo "FRED_API_KEY=<your key>" > .env          # free key: https://fred.stlouisfed.org/docs/api/api_key.html
# set data_sources.sec_edgar.user_agent in config/config.yaml to "<name> <email>" (SEC requirement)
bash run_all.sh
```

`run_all.sh` runs nine stages. Only stages 1–2 touch the network. Everything later reads the
gzip-compressed raw cache in `data/raw/`. The raw cache is not in this repository (too large); stages 1–2
rebuild it from SEC EDGAR. The processed snapshot (`data/processed/`, `data/embeddings/`) and all
results are included; check them with `sha256sum -c MANIFEST.sha256` (raw entries need stage 2 first).

| Stage | Script | Output |
|---|---|---|
| 1 discovery | `scripts/discover_filings.py` | `data/processed/candidates.json`, `discovery_stats.json` |
| 2 download | `scripts/download_filings.py` | `data/raw/{filings,xbrl,subs,tenk}/*.gz` (8 req/s limit) |
| 3 parse | `scripts/build_deal_universe.py` | `data/processed/deal_universe.json`, `parse_stats.json` |
| 4 embeddings | `scripts/build_embeddings.py` | `data/embeddings/*.npy` (GPU recommended) |
| 5 retrieval | `scripts/run_retrieval_study.py` | `results/retrieval/<run>.json` (per-query metrics) |
| 6 valuation | `scripts/run_valuation_study.py` | `results/valuation.json` |
| 7 statistics | `scripts/run_significance_study.py` | `results/significance.json` |
| 8 paper numbers | `scripts/make_paper_tables.py` | `results/paper_numbers.json`, `paper/figures/` |
| 9 manifest | `scripts/write_manifest.py` | `MANIFEST.sha256` |

Approximate runtime on one RTX 3090 / 28 cores: about 35 min of download, then 30 min to parse and embed,
and about 1–2 h for the experiments.

## What the pipeline does

* **Corpus.** DEFM14A/DEFM14C merger proxies and SC 14D9 cash-tender-offer statements, 2013-01-01 to
  2025-06-30. One deal is kept per target: the earliest filing that passes the M&A gate. The gate
  excludes SPAC business combinations, acquirer-side proxies (no Item 402(t) golden-parachute vote but a
  share-issuance proposal), filings without a merger agreement (mini-tenders, partial or foreign offers)
  and asset sales. Every exclusion reason is counted in `parse_stats.json`.
* **Point-in-time fundamentals.** An XBRL fact is used only if it was *filed* on or before the deal
  filing date. When a period was reported more than once, the earliest-filed value wins, so
  restatements are ignored. Fallbacks: IFRS, trailing twelve months, and bank revenue (interest +
  noninterest income).
* **Completion status.** A deal counts as completed if the target deregistered (Form 25/15) within 18
  months. It counts as terminated if the target kept filing 10-K/10-Q reports for more than 12 months
  afterwards.
* **DealDNA blocks.**
  * industry (SIC)
  * financial
  * structure
  * acquirer (sponsor/strategic, listed, cross-border)
  * macro (FRED, plus recency)
  * legal: clause passages embedded with FinBERT-tone, LEGAL-BERT or MPNet
  * business: the target's 10-K Item 1, embedded with MPNet
* **Relevance constructs.** These are proxies, not ground truth:
  * heuristic
  * practitioner rubric
  * concordance
  * outcome-grounded (realized EV/Revenue similarity)
  * legal-aware: agreement on ten regex-extracted legal terms, independent of the clause selector
* **Leakage control.** Under the multiple-based constructs, every transaction-value-derived feature is
  removed: EV/Revenue, EV/EBITDA, premium and termination fee.
* **Protocol.**
  * A query retrieves only deals filed on or before it.
  * Folds are 5-fold GroupKFold by SIC2, or an expanding time window.
  * Imputation, scaling and every learned component are fitted on the training fold, and learned
    retrievers train only on pairs whose query and candidate are both in the training fold.
* **Retrievers.**
  * naive (SIC2 one-hot + size)
  * SIC 2/3/4-digit back-off
  * within-industry size bins
  * 10-K text peers
  * SIC2 + text
  * linear block weights
  * LambdaMART (LightGBM)
  * LambdaMART cascade: SIC back-off proposes the top 50, LambdaMART re-ranks them using the realized
    multiples of those earlier precedents and an in-fold predicted multiple for the target (never the
    target's own multiple; `tests/test_cascade.py`)
* **Statistics.**
  * cluster bootstrap over (SIC2 × year) blocks
  * Holm correction over the entire family of reported comparisons
  * paired Cliff's δ
  * Wilcoxon as a secondary check

## Tests

`python -m pytest -q tests` covers the point-in-time XBRL rule, the causal mask, the time folds, leakage
control, fold-local preprocessing, the M&A gate, the metrics, the Holm correction and the cluster bootstrap.

## Notes

* FRED's ICE BofA high-yield series is now limited to the last three years, so the credit-spread
  feature uses Moody's Baa − 10-year Treasury (`BAA10Y`).
* `scripts/legacy/` holds the scripts of the earlier 200-deal prototype. They are not used for the paper.
