"""Tests for the leakage, point-in-time and causal guarantees the revised evaluation relies on."""

import numpy as np

from src.data_ingestion.xbrl_facts import extract_financial_fundamentals
from src.dealdna.clause_extraction import extract_legal_flags, ma_gate
from src.evaluation.engine import causal_mask, make_folds, query_metrics
from src.evaluation.features import VALUE_DERIVED, NumericBlocks
from src.evaluation.relevance import build_relevance
from src.evaluation.stats import holm, paired


def _facts(entries):
    return {"facts": {"us-gaap": {"Revenues": {"units": {"USD": entries}}}}}


def test_xbrl_ignores_facts_filed_after_the_deal():
    entries = [
        {"start": "2019-01-01", "end": "2019-12-31", "val": 100.0, "filed": "2020-02-15", "form": "10-K"},
        # restatement of the same period, filed after the deal date: must be ignored
        {"start": "2019-01-01", "end": "2019-12-31", "val": 999.0, "filed": "2020-09-01", "form": "10-K"},
        # next fiscal year, period ended before the deal but filed after it: must be ignored
        {"start": "2020-01-01", "end": "2020-03-31", "val": 50.0, "filed": "2020-07-01", "form": "10-Q"},
    ]
    assert extract_financial_fundamentals(_facts(entries), "2020-06-01")["revenues"] == 100.0


def test_causal_mask_admits_only_earlier_or_same_day_deals():
    m = causal_mask(np.array([3.0, 1.0, 2.0, 2.0]))
    assert not m[1].any()                     # earliest deal has no admissible precedent
    assert m[0, 1] and m[0, 2] and m[0, 3]    # latest deal sees everything else
    assert m[2, 3] and m[3, 2] and not m[2, 2]


def test_time_folds_never_train_on_the_future():
    attrs = {"date_ord": np.arange(60.0), "sic2": np.array(["10"] * 60)}
    for train, test in make_folds(attrs, "time"):
        assert attrs["date_ord"][train].max() < attrs["date_ord"][test].min()


def test_value_derived_features_are_dropped():
    deal = {"revenues": 10.0, "ev_estimate": 50.0, "ebitda_estimate": 5.0, "deal_premium": 0.3,
            "termination_fee_usd": 1.0, "filing_date": "2020-01-01", "consideration_mix": "cash"}
    blocks = NumericBlocks([deal, deal], drop=VALUE_DERIVED).blocks
    assert not VALUE_DERIVED & {f for names in blocks.values() for f in names}


def test_preprocessing_statistics_come_from_training_fold_only():
    deals = [{"revenues": r, "filing_date": "2020-01-01"} for r in (10.0, 20.0, 1e9)]
    scaled = NumericBlocks(deals).fit_transform(np.array([0, 1]))["financial"][:, 0]
    assert abs(scaled[0] + scaled[1]) < 1e-9 and scaled[2] > 10   # test deal is an outlier, not part of the fit


def test_outcome_relevance_is_industry_gated_and_graded():
    attrs = {"sic": np.array(["2834", "2834", "3571"]), "log_multiple": np.array([1.0, 1.1, 1.0])}
    rel = build_relevance("outcome", attrs)
    assert rel[0, 1] == 3.0 and rel[0, 2] == 0.0 and rel[0, 0] == 0.0


def test_query_metrics_perfect_ranking():
    rel = np.array([0.0, 3.0, 1.0, 0.0])
    m = query_metrics(np.array([0.0, 9.0, 5.0, 1.0]), rel, np.array([False, True, True, True]),
                      np.array([True, True, True, True]), np.zeros(4))
    assert m["ndcg@5"] == 1.0 and m["map"] == 1.0 and m["mrr"] == 1.0


def test_holm_is_monotone_and_bounded():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == 0.03 and adj["c"] == 0.06 and adj["b"] == 0.06


def test_cluster_bootstrap_detects_consistent_improvement():
    a = {i: 0.6 for i in range(200)}
    b = {i: 0.5 for i in range(200)}
    res = paired(a, b, np.array([str(i % 20) for i in range(200)]), n_boot=2000)
    assert res["diff"] > 0.09 and res["p_cluster_bootstrap"] <= 0.001 and res["cliffs_delta"] == 1.0


def test_ma_gate_rejects_acquirer_side_and_spac_proxies():
    target = "Agreement and Plan of Merger. " * 5 + "advisory vote on merger-related compensation. " + "x" * 6000
    acquirer = "Agreement and Plan of Merger. " * 5 + "the share issuance proposal. " + "x" * 6000
    spac = "merger agreement " * 5 + "trust account " * 4 + "business combination " * 6 + "x" * 6000
    assert ma_gate(target, "DEFM14A") is None
    assert ma_gate(acquirer, "DEFM14A") == "acquirer_side_proxy"
    assert ma_gate(spac, "DEFM14A") == "spac_business_combination"


def test_legal_flags():
    flags = extract_legal_flags("The agreement includes a go-shop period and a reverse termination fee; HSR Act clearance.")
    assert flags["go_shop"] and flags["reverse_termination_fee"] and flags["hsr_required"] and not flags["cvr"]


def test_enterprise_value_only_for_all_cash_deals():
    from scripts.build_deal_universe import enterprise_value

    f = {"shares_outstanding": 10.0, "long_term_debt": 5.0, "cash_and_equivalents": 2.0, "revenues": 50.0, "reports_in_usd": True}
    assert enterprise_value(f, {"consideration_mix": "cash", "cash_per_share": 10.0}) == 103.0
    assert enterprise_value(f, {"consideration_mix": "cash_and_stock", "cash_per_share": 10.0}) is None
    assert enterprise_value(f, {"consideration_mix": "stock", "cash_per_share": 10.0}) is None
    assert enterprise_value(f, {"consideration_mix": "cash", "cash_per_share": 1e6}) is None  # implausible multiple


def test_candidate_pool_excludes_unlabelled_deals():
    m = causal_mask(np.array([1.0, 2.0, 3.0]), candidate_pool=np.array([True, False, True]))
    assert m[2, 0] and not m[2, 1]          # a deal without a multiple can never be retrieved
