from src.dealdna.clause_extraction import extract_deal_terms, extract_stated_premium_pct

SAMPLE_TEXT = """
Pursuant to the Agreement and Plan of Merger, each share of Company common stock will be converted
into the right to receive $42.50 per share, without interest, per share in cash. This is an
all-cash transaction. The Company may be required to pay Parent a termination fee of approximately
$18.5 million if the board changes its recommendation. No Material Adverse Effect has occurred.
In the Offer, holders receive $42.50 per share in cash. Our board determined that $42.50 per share is fair.
The parties agree there is no earn-out or contingent value right associated with this transaction.
The Merger Consideration represents a 28% premium to the closing price of Company Common Stock on
March 3, 2024, the last full trading day prior to the announcement of the transaction. It also
represents a 45% premium to the 52-week high closing price of Company Common Stock.
"""


def test_extract_stated_premium_pct_finds_unaffected_price_premium():
    assert extract_stated_premium_pct(SAMPLE_TEXT) == 0.28


def test_extract_stated_premium_pct_returns_none_without_unaffected_context():
    assert extract_stated_premium_pct("a 45% premium to the 52-week high closing price") is None


def test_extract_deal_terms_finds_cash_per_share():
    terms = extract_deal_terms(SAMPLE_TEXT)
    assert terms["cash_per_share"] == 42.50


def test_extract_deal_terms_finds_termination_fee():
    terms = extract_deal_terms(SAMPLE_TEXT)
    assert terms["termination_fee_usd"] == 18_500_000.0


def test_extract_deal_terms_classifies_all_cash():
    terms = extract_deal_terms(SAMPLE_TEXT)
    assert terms["consideration_mix"] == "cash"


def test_extract_deal_terms_detects_earnout_keyword():
    terms = extract_deal_terms(SAMPLE_TEXT)
    assert terms["has_earnout"] is True


def test_extract_deal_terms_handles_empty_text():
    terms = extract_deal_terms("")
    assert terms["cash_per_share"] is None
    assert terms["consideration_mix"] == "unknown"


def test_one_off_price_is_not_taken_as_deal_price():
    text = "The stock traded at $17.25 per share last year. " + "The merger consideration is $30.00 per share in cash. " * 3
    assert extract_deal_terms(text)["cash_per_share"] == 30.00


def test_par_values_dividends_and_preferred_prices_are_not_deal_prices():
    noise = ("common stock, par value $0.20 per share. " * 4 + "a special dividend of $0.40 per share. " * 4
             + "a redemption price of $25.00 per share of Series B Preferred. " * 4)
    deal = "you will be entitled to receive $41.00 in cash, without interest, for each share. " * 3
    assert extract_deal_terms(noise)["cash_per_share"] is None
    assert extract_deal_terms(noise + deal)["cash_per_share"] == 41.00
