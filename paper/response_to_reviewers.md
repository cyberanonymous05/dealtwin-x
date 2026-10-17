# Response to Reviewers: CICT-2026, Paper ID 899
**DealTwin-X: Multi-Modal Embeddings for M&A Precedent Retrieval** (Track 1: AI, ML & Data Analytics)

We thank both reviewers for unusually careful and constructive reviews. Reviewer 1's comments are numbered R1.1–R1.10 as in the review; Reviewer 2's single paragraph is split into R2.1–R2.9 below. **`main_annotated.pdf` marks every place a comment is addressed with a blue tag such as [R1.5]**; `main.pdf` is the clean version. Addressing them required more than editing. We rebuilt the data pipeline and re-ran every experiment under a stricter protocol: point-in-time data, leakage control, fold-local preprocessing, a labelled candidate pool, stronger baselines and clustered inference with Holm correction over every comparison. As a result, every number in the paper has changed. Some conclusions became stronger and one became more modest, and we report both. The principal changes are:

* **Corpus.** The corpus was rebuilt from the same sources, forms (DEFM14A/14C, SC 14D9) and period (2013–2025). It now contains **2,416 deals** (1,947 merger proxies and 469 tender offers), and the parsing yield and every exclusion reason are reported. XBRL fundamentals are strictly point-in-time. The deal-price extractor was audited and improved, and enterprise value is computed only for all-cash deals.
* **Stronger baselines.** We added four-digit SIC back-off, within-industry size bins, 10-K text peers and SIC2 + text.
* **Redesigned ranker.** The LambdaMART ranker now uses fine SIC match indicators, feature-difference features and a business-description block. The *submitted* ranker design is re-run on the new corpus for a like-for-like comparison.
* **Main result.** The improved ranker beats the naive baseline with a large effect, whereas the submitted design barely does. However, on its own it only *ties* a hierarchical SIC + size screen, and the paper states this plainly. A two-stage **cascade**, fixed in advance, re-ranks that screen's top 50 candidates using the realized multiples of the earlier precedents and an in-fold predicted multiple for the target. It **beats the screen** by a small but consistent margin (NDCG@5 +0.012 under SIC folds, Holm p = 0.086; +0.015 under temporal folds, Holm p = 0.003). It also gives the best k-NN valuations (R² 0.40 vs. 0.22 for the industry median). An ablation shows that the whole gain comes from the precedent multiples.

---

## Reviewer #1

**R1.1: Validate the relevance constructs or acknowledge that they are proxies.**
No practitioner judgements were available. The constructs are now presented throughout as *proxies, not ground truth* (Sec. V, Limitations). We explain which constructs reuse retriever attributes, and we treat the outcome-grounded construct as primary. We added a fifth, **legal-aware** construct that is nearly independent of the others (|τ| ≤ 0.15). We also release a labelling template for practitioner judgements.
*Where:* Sec. I, V (constructs), VII (limitations).

**R1.2: Practical significance of the gains; when does the ranker help?**
A new subsection, VI-C, separates statistical from practical significance. The binary metrics barely move (8.29 vs. 8.22 relevant precedents per top-10 list), because industry-first retrievers saturate them. The gain is in ordering *within* the industry: WI-NDCG@5 improves from 0.649 to 0.747 (+15%). The gain holds both in industries with dispersed multiples (δ = 0.55) and in clustered ones (δ = 0.44). Practically, most of the gain is obtained by a four-digit-SIC back-off screen with a size tie-break, and the paper says so. The ranker adds value over that screen when it also consults the realized multiples of the screened precedents (the cascade, +0.012 to +0.015 NDCG@5).
*Where:* Sec. VI-C, Tables IV–V.

**R1.3: Why does retrieval not transfer to valuation? Try alternatives.**
With industry-aware retrievers, k-NN valuation now **beats** the causal industry median. The best retriever is the new cascade: R² 0.402 vs. 0.219, absolute log error −0.084, Holm p < 10⁻¹¹, δ = 0.21, small (Table VIII). Linear DealDNA follows at 0.357. The top-8 rank-weighted setting was fixed in advance. We report the full 15-configuration grid (k ∈ {3, 5, 8, 10, 20} × mean/median/rank-weighted); the cascade is best in all 15 (maximum R² 0.411). We tried a valuation-objective retriever trained to minimize neighbour–query multiple distance, and it did **not** help (R² 0.257). Plain LambdaMART is weaker (R² 0.329) because NDCG without precedent multiples rewards industry ordering rather than homogeneity of the neighbours' multiples. This is what the cascade addresses.
*Where:* Sec. VI-F, Table VIII.

**R1.4: Limited contribution of the legal block; improve selection or add text.**
We improved selection with a lexical gate on the three lowest-precision clause types. We compared three encoders (FinBERT, LEGAL-BERT, MPNet), added a 10-K business-description block, and added the legal-aware construct. Under the outcome construct the legal block still carries no signal: weight ≈ 0.03 of 6, ablation −0.002, not significant. Under the legal-aware construct it carries a **small but significant** signal (ablation −0.008 NDCG@5, Holm p < 10⁻⁵, negligible effect).
*Where:* Sec. IV (blocks, clause selection), VI-E, Table VII.

**R1.5: Per-clause-type selector performance; impact on the negative finding.**
New Table II reports precision for all eight types before and after gating. Before gating, earn-out/CVR scores 0.04 and regulatory approval 0.24; after gating, overall precision rises from 0.59 to 0.84. We state explicitly that gated precision is 1.0 by construction and that the silver standard is keyword-based. The Limitations section bounds the legal conclusion accordingly.
*Where:* Sec. IV, Table II; VI-E; VII.

**R1.6: Missing data and imputation; sensitivity analyses.**
Three treatments were compared: median imputation, median plus missingness indicators, and dropping the premium and EV-derived features. They change WI-NDCG@5 by at most 0.012. For the surrogate, R² is 0.375, 0.377 and 0.335 (native missing-value handling). Deal premium (64.9% missing) is used only where the target does not depend on the multiple, and EV only for all-cash deals.
*Where:* Sec. III, VI-G (missing data).

**R1.7: Surrogate under-prediction; error by range; intended use.**
New Table III gives the error by realized-multiple quintile: the top quintile is under-predicted by 12.9× and its ±1σ coverage is 25%. We state the intended use (an explanation aid, not a valuation tool) and the range where it is unreliable (above about 6× revenue). The surrogate is also evaluated under temporal cross-validation (R² 0.48).
*Where:* Sec. VI-A, Table III.

**R1.8: External validity.**
A new limitation lists the excluded deal types: S-4 stock mergers, private targets, non-US deals, asset sales and carve-outs. It also notes that multiples cover all-cash deals only. Subgroup results (proxies vs. tender offers, strategic vs. sponsor, completed vs. terminated) are reported in Sec. VI-G and are consistent.
*Where:* Sec. VI-G (subgroups), VII (external validity).

**R1.9: Repository and reproduction instructions.**
The complete repository includes a README and a single `run_all.sh` that reproduces every number from a frozen, SHA-256-hashed raw cache. It also contains per-query results for every table, the audit samples and 42 unit tests.
> **Authors: insert the GitHub URL and Zenodo DOI** (two red TODOs in the Data Availability section).
*Where:* Data and Code Availability; Sec. VI-G (reproducibility).

**R1.10: Statistical vs. practical significance; language and consistency.**
Statistical and practical significance are addressed in Sec. VI-C. The manuscript was copy-edited throughout. We also fixed an effect-size labelling error: the submitted paper called δ = 0.27 "medium", but under the stated thresholds it is small.
*Where:* Abstract; Sec. V (metrics); VI-C.

---

## Reviewer #2

**R2.1: Better baselines (finer SIC/NAICS, size bins, text-based peers); robustness of the gain.**
We added SIC 2/3/4-digit back-off, within-industry revenue quintiles, 10-K business-description text peers (Hoberg & Phillips style) and SIC2 + text. SEC data provide SIC but not NAICS codes. The reviewer's concern is confirmed: **SIC back-off is far stronger than the naive baseline** (NDCG@5 0.777 vs. 0.683). The improved LambdaMART ties it (+0.0005, CI [−0.006, 0.007], Holm p = 1). The abstract, contributions and conclusion now say this explicitly. To test whether learning can add anything beyond such a screen, we added a pre-specified two-stage cascade: the screen proposes 50 candidates and LambdaMART re-ranks them using the precedents' realized multiples. The cascade beats the screen under every construct. Under the outcome construct the margin is small (+0.012 NDCG@5, CI [0.006, 0.018], Holm p = 0.086; +0.015 under temporal folds, Holm p = 0.003; WI-NDCG@10 Holm p < 10⁻³). Under concordance it beats the size-bin screen by +0.060 (Holm p < 10⁻³), where plain LambdaMART tied. Only the *earlier* deals' multiples are used, never the query's own; a unit test checks this.
*Where:* Sec. IV (baselines), V (cascade), Tables IV–VI, VIII.

**R2.2: Within-industry measures with significance; moderate the abstract and conclusion.**
WI-NDCG@5/@10 are now primary metrics and are tested in the same Holm family. Abstract and conclusion are moderated. We also explain why MAP, MRR, P@5 and R@10 are saturated for industry-first retrievers (Sec. V).
*Where:* Abstract, Sec. I, VI-B–VI-D, VIII; Tables IV–VI.

**R2.3: Remove multiple-derived features under the outcome construct.**
Under the multiple-based constructs, EV/Revenue, EV/EBITDA, deal premium **and the termination fee** (negotiated as a share of equity value) are removed. We also found and fixed a second, subtler leak: retrievers could score well by predicting which candidates *have* a label (e.g. "cash deals have multiples"). Under these constructs the candidate pool is therefore restricted to deals with a multiple (Sec. IV), and the paper reports this.
*Where:* Sec. IV (leakage control).

**R2.4: Clustered/block bootstrap; correction over all statistics; scaling, imputation and ranking inside folds.**
We use a cluster bootstrap over (SIC2 × filing year) blocks with bootstrap-SE z-tests, and Holm correction over **all 880 comparisons** (every retriever pair × metric × construct, plus ablations and the dispersion split). Wilcoxon is kept as a secondary check. Imputation, scaling, size bins, block weights and the ranker are fitted on the training fold, and learned retrievers train only on pairs whose query *and* candidate are in the training fold.
*Where:* Sec. IV, V (protocol, statistical inference).

**R2.5: Neural retriever: describe or remove.**
Removed from all claims; a single sentence notes that it was explored during development.
*Where:* Sec. V (learned retrievers).

**R2.6: Reformulate the legal claim; add a legal-aware relevance measure.**
Reformulated as "no signal found *with these constructs and this clause selector*". A legal-aware construct was added, built from ten regex legal-term flags that are independent of the clause selector. Under that construct a small legal-text signal is detected (see R1.4).
*Where:* Sec. V (construct 5), VI-D, VI-E, Table VII.

**R2.7: Industry-median R² puzzle (−0.06 vs. 0.21); temporal CV for the surrogate.**
The difference comes from the split. Under GroupKFold by SIC the test industry never appears in training, so the "industry median" falls back to the global median for 100% of test deals (R² −0.05). Under temporal folds, same-industry history exists for 97.4% of test deals (R² 0.23). The surrogate is now evaluated under both schemes (R² 0.38 / 0.48).
*Where:* Sec. VI-A.

**R2.8: Parsing rate, terminated deals, point-in-time XBRL, public data/DOI.**
* *Parsing yield:* 3,841 filings for 3,336 targets yield 2,416 deals (72.4%). Every exclusion reason is counted: no merger agreement 652, SPAC 320, acquirer-side proxy 104, too short 11, no point-in-time XBRL 5, asset sale 3.
* *Deal-price accuracy:* an independent audit of 60 random filings found 83% correct overall and 92% correct on the all-cash deals used for EV.
* *Terminated deals:* detected from deregistration filings; 165 (6.8%) were terminated. They are retained and analysed separately, with consistent results.
* *Point-in-time XBRL:* a fact is used only if filed before the deal. An end-date-only rule would have used a post-deal filing for 25.6% of deals and a different revenue value for 14.0%.
* *Public data:* the snapshot is released with its hash manifest (see R1.9).
*Where:* Sec. III, VII; Data and Code Availability.

**R2.9: Naming consistency (DealTwin-X vs. DealTwinX).**
"DealTwin-X" is used throughout.
*Where:* throughout.

---

### Other corrections
* "Five-block" corrected (the submitted 855 dimensions were six blocks); the revised embedding has seven blocks.
* The legal-weight inconsistency in the submission (0.11 in the text vs. 0.025–0.045 in the threshold sweep) is resolved: it is now 0.030–0.032 throughout.
* FRED's ICE BofA high-yield series is now restricted to three years, so the credit-spread feature uses Moody's Baa − 10-year Treasury.
* "Dr." removed from the author block, per IEEE style.
