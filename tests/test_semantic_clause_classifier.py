from src.dealdna.semantic_clause_classifier import classify_clause_passages

SAMPLE_TEXT = """
The Company Board considered possible alternatives to the acquisition by Parent, including the
possibility of continuing to operate the Company as an independent enterprise. The Merger Agreement
provides that the Company shall not encourage or facilitate a competing acquisition proposal from any
third party, subject to the Board's fiduciary duties. If the Merger Agreement is terminated under
certain circumstances, the Company will be required to pay Parent a fee of $50,000,000. No event has
occurred that constitutes a Material Adverse Effect on the Company's business or financial condition.
Consummation of the Merger is subject to the receipt of required antitrust approvals under the HSR Act.
Stockholders who do not vote in favor of the Merger may be entitled to exercise appraisal rights under
Delaware law. Each share of Company common stock will be converted into the right to receive a fixed
exchange ratio of shares of Parent common stock.
"""


def test_classify_clause_passages_returns_nonempty_list():
    passages = classify_clause_passages(SAMPLE_TEXT)
    assert len(passages) > 0


def test_classify_clause_passages_finds_no_solicitation_without_literal_keyword():
    # this sentence never says "solicitation" or "no-shop" -- only a semantic classifier should catch it
    passages = classify_clause_passages(SAMPLE_TEXT)
    assert any("encourage or facilitate a competing acquisition proposal" in p for p in passages)


def test_classify_clause_passages_finds_appraisal_rights():
    passages = classify_clause_passages(SAMPLE_TEXT)
    assert any("appraisal rights" in p for p in passages)


def test_classify_clause_passages_handles_empty_text():
    assert classify_clause_passages("") == []
