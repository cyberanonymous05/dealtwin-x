#!/usr/bin/env bash
# Reproduce every number in the DealTwin-X paper from scratch.
#   Stages 1-2 hit SEC EDGAR (rate-limited to 8 req/s; ~35 min). Skip them by restoring the frozen
#   data/raw snapshot from the archive (see README) -- everything after stage 2 reads only the cache.
set -euo pipefail
cd "$(dirname "$0")"
set -a; [ -f .env ] && . ./.env; set +a          # FRED_API_KEY

python -I scripts/discover_filings.py           # 1. candidate filings  -> data/processed/candidates.json
python -I scripts/download_filings.py           # 2. raw cache          -> data/raw/{filings,xbrl,subs,tenk}
python -I scripts/build_deal_universe.py        # 3. parse              -> data/processed/deal_universe.json, parse_stats.json
python -I scripts/build_embeddings.py           # 4. text embeddings    -> data/embeddings/*.npy (GPU recommended)
python -I scripts/run_retrieval_study.py        # 5. retrieval runs     -> results/retrieval/*.json
python -I scripts/run_valuation_study.py        # 6. valuation          -> results/valuation.json
python -I scripts/run_valuation_tests.py        #    k-NN vs industry median significance -> results/valuation_tests.json
python -I scripts/run_significance_study.py     # 7. statistics         -> results/significance.json
python -I scripts/make_paper_tables.py          # 8. paper numbers      -> results/paper_numbers.json, paper/figures
python -I scripts/make_pipeline_figure.py       #    pipeline figure    -> paper/figures/fig0_pipeline.png
python -I scripts/run_data_audits.py            #    audits             -> results/data_audits.json, audit_deal_price_sample.json
python -I scripts/write_manifest.py             # 9. SHA-256 manifest   -> MANIFEST.sha256
python -m pytest -q tests
(cd paper && tectonic main.tex)                 # 10. compile the paper (or: pdflatex main.tex)
