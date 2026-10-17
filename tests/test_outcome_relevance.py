from src.evaluation.cross_validation import build_outcome_based_relevance


def _deal(sic, ev, rev, filing_date="2022-01-01"):
    return {"sic": sic, "ev_estimate": ev, "revenues": rev, "filing_date": filing_date}


def test_same_industry_similar_multiple_scores_high():
    deals = [_deal("7372", 300, 100), _deal("7372", 320, 105)]  # ~3.0x and ~3.05x
    relevance = build_outcome_based_relevance(deals)
    assert relevance[0, 1] >= 2.5


def test_same_industry_very_different_multiple_scores_lower_than_similar():
    same_group_similar = [_deal("7372", 300, 100), _deal("7379", 320, 105)]  # ~3.0x and ~3.05x, same group
    same_group_different = [_deal("7372", 300, 100), _deal("7379", 3000, 100)]  # 3x vs 30x, same group
    similar_score = build_outcome_based_relevance(same_group_similar)[0, 1]
    different_score = build_outcome_based_relevance(same_group_different)[0, 1]
    assert different_score < similar_score


def test_different_industry_scores_zero_regardless_of_multiple():
    deals = [_deal("7372", 300, 100), _deal("2834", 300, 100)]  # identical multiple, different industry
    relevance = build_outcome_based_relevance(deals)
    assert relevance[0, 1] == 0.0


def test_missing_multiple_scores_zero():
    deals = [_deal("7372", None, None), _deal("7372", 300, 100)]
    relevance = build_outcome_based_relevance(deals)
    assert relevance[0, 1] == 0.0
    assert relevance[1, 0] == 0.0
