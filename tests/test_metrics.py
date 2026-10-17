import math

from src.evaluation.metrics import (
    average_precision,
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([2, 1, 1, 0, 0], k=5) == 1.0


def test_ndcg_worst_ranking_is_less_than_best():
    best = ndcg_at_k([2, 1, 0, 0, 0], k=5)
    worst = ndcg_at_k([0, 0, 0, 1, 2], k=5)
    assert worst < best


def test_ndcg_no_relevant_items_is_zero():
    assert ndcg_at_k([0, 0, 0], k=5) == 0.0


def test_dcg_matches_hand_computed_value():
    # relevances [3, 2] at ranks 1, 2 -> 3/log2(2) + 2/log2(3)
    expected = 3 / math.log2(2) + 2 / math.log2(3)
    assert math.isclose(dcg_at_k([3, 2], k=5), expected)


def test_precision_at_k_counts_hits_in_window():
    assert precision_at_k([1, 0, 1, 0, 0], k=5) == 0.4
    assert precision_at_k([1, 1, 0, 0, 0], k=2) == 1.0


def test_recall_at_k_divides_by_total_relevant():
    assert recall_at_k([1, 0, 1, 0], total_relevant=4, k=4) == 0.5
    assert recall_at_k([0, 0], total_relevant=0, k=2) == 0.0


def test_average_precision_single_hit_at_rank_one():
    assert average_precision([1, 0, 0]) == 1.0


def test_average_precision_no_hits_is_zero():
    assert average_precision([0, 0, 0]) == 0.0


def test_average_precision_matches_hand_computed_value():
    # hits at ranks 1 and 3 -> precision@1=1.0, precision@3=2/3 -> mean
    expected = (1.0 + 2 / 3) / 2
    assert math.isclose(average_precision([1, 0, 1]), expected)
