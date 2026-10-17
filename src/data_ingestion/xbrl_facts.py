"""Point-in-time financial fundamentals from the SEC XBRL Company Facts API.

Point-in-time rule: a fact is usable for a deal only if it was FILED on or before the deal's filing date
(the `filed` field of each XBRL entry), not merely if its period ENDED before it. Filtering on period end
alone silently admits values first disclosed -- or restated -- after the deal was announced, which is
look-ahead leakage. When several filings report the same period, the earliest-filed value (the one an
analyst could actually see at the time) wins; later restatements are ignored.

Taxonomies: us-gaap first, then ifrs-full (foreign private issuers). Flow items prefer the latest annual
(10-K / 20-F / 40-F) value; if that annual period ended more than 15 months before the deal, a trailing-
twelve-month value is built as annual + current YTD - prior-year YTD from 10-Q facts.
"""

from datetime import date, timedelta

_TAGS = {
    "revenues": {
        "us-gaap": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                    "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"],
        "ifrs-full": ["Revenue", "RevenueFromContractsWithCustomers"],
    },
    "net_income": {"us-gaap": ["NetIncomeLoss", "ProfitLoss"], "ifrs-full": ["ProfitLoss"]},
    "operating_income": {"us-gaap": ["OperatingIncomeLoss"], "ifrs-full": ["ProfitLossFromOperatingActivities"]},
    "depreciation_amortization": {
        "us-gaap": ["DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet",
                    "DepreciationAndAmortization", "Depreciation"],
        "ifrs-full": ["DepreciationAndAmortisationExpense", "DepreciationExpense"],
    },
    "total_assets": {"us-gaap": ["Assets"], "ifrs-full": ["Assets"]},
    "total_liabilities": {"us-gaap": ["Liabilities"], "ifrs-full": ["Liabilities"]},
    "cash_and_equivalents": {
        "us-gaap": ["CashAndCashEquivalentsAtCarryingValue", "Cash", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
        "ifrs-full": ["CashAndCashEquivalents"],
    },
    "long_term_debt": {"us-gaap": ["LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"],
                       "ifrs-full": ["NoncurrentPortionOfNoncurrentBorrowings", "Borrowings"]},
    "stockholders_equity": {"us-gaap": ["StockholdersEquity"], "ifrs-full": ["Equity"]},
}
_FLOW = {"revenues", "net_income", "operating_income", "depreciation_amortization"}
_ANNUAL_FORMS = ("10-K", "20-F", "40-F", "10-KT")


def _days(entry: dict) -> int | None:
    try:
        return (date.fromisoformat(entry["end"]) - date.fromisoformat(entry["start"])).days
    except (KeyError, ValueError, TypeError):
        return None


def _visible(entries: list[dict], as_of: str) -> list[dict]:
    """Entries filed on or before as_of; for each (start, end) period keep the EARLIEST-filed value."""
    first = {}
    for e in entries:
        if e.get("val") is None or not e.get("filed") or e["filed"] > as_of or e.get("end", "9999") > as_of:
            continue
        key = (e.get("start"), e["end"])
        if key not in first or e["filed"] < first[key]["filed"]:
            first[key] = e
    return list(first.values())


def _units(tag_data: dict) -> list[dict]:
    units = tag_data.get("units", {})
    if "USD" in units:
        return units["USD"]
    # IFRS filers report in their own currency; values are kept only for ratio features
    # (margins, leverage) where currency cancels, and flagged so size features can skip them.
    return next(iter(units.values()), [])


def _flow_value(entries: list[dict], as_of: str, positive: bool = False) -> tuple[float | None, str | None]:
    visible = _visible(entries, as_of)
    annual = [e for e in visible if (d := _days(e)) and 350 <= d <= 380]
    if positive:
        annual = [e for e in annual if e["val"] > 0]
    if not annual:
        return None, None
    latest = max(annual, key=lambda e: e["end"])
    stale = (date.fromisoformat(as_of) - date.fromisoformat(latest["end"])).days > 456
    if stale:
        ytd = [e for e in visible if (d := _days(e)) and 80 <= d <= 280 and e["end"] > latest["end"]]
        if ytd:
            cur = max(ytd, key=lambda e: e["end"])
            prior_end = (date.fromisoformat(cur["end"]) - timedelta(days=365)).isoformat()
            prior = [e for e in visible if _days(e) and abs(_days(e) - _days(cur)) <= 10
                     and abs((date.fromisoformat(e["end"]) - date.fromisoformat(prior_end)).days) <= 10]
            if prior:
                return latest["val"] + cur["val"] - prior[0]["val"], "ttm"
    return latest["val"], "annual"


def _instant_value(entries: list[dict], as_of: str) -> float | None:
    visible = [e for e in _visible(entries, as_of) if not e.get("start")]
    return max(visible, key=lambda e: e["end"])["val"] if visible else None


def _revenue_growth(facts: dict, as_of: str) -> float | None:
    for taxonomy, tags in _TAGS["revenues"].items():
        for tag in tags:
            data = facts.get(taxonomy, {}).get(tag)
            if not data:
                continue
            annual = sorted((e for e in _visible(_units(data), as_of) if (d := _days(e)) and 350 <= d <= 380),
                            key=lambda e: e["end"], reverse=True)
            if len(annual) >= 2 and annual[1]["val"]:
                gap = (date.fromisoformat(annual[0]["end"]) - date.fromisoformat(annual[1]["end"])).days
                if 330 <= gap <= 400:
                    return (annual[0]["val"] - annual[1]["val"]) / abs(annual[1]["val"])
    return None


def extract_financial_fundamentals(company_facts: dict | None, as_of_date: str) -> dict:
    facts = (company_facts or {}).get("facts", {})
    result = {k: None for k in _TAGS}
    provenance = {}
    for name, by_taxonomy in _TAGS.items():
        for taxonomy, tags in by_taxonomy.items():
            for tag in tags:
                data = facts.get(taxonomy, {}).get(tag)
                if not data:
                    continue
                entries = _units(data)
                if name in _FLOW:
                    value, kind = _flow_value(entries, as_of_date, positive=(name == "revenues"))
                else:
                    value, kind = _instant_value(entries, as_of_date), "instant"
                if value is not None:
                    result[name] = float(value)
                    provenance[name] = f"{taxonomy}:{tag}:{kind}"
                    break
            if result[name] is not None:
                break

    if result["revenues"] is None:
        # Banks and thrifts report no "Revenues" tag; their revenue is interest income + noninterest income.
        parts = []
        for tag in ("InterestAndDividendIncomeOperating", "NoninterestIncome"):
            data = facts.get("us-gaap", {}).get(tag)
            parts.append(_flow_value(_units(data), as_of_date)[0] if data else None)
        if parts[0] is not None:
            result["revenues"] = float(parts[0] + (parts[1] or 0.0))
            provenance["revenues"] = "us-gaap:bank_interest_plus_noninterest_income"

    ebitda = None
    if result["operating_income"] is not None and result["depreciation_amortization"] is not None:
        ebitda = result["operating_income"] + result["depreciation_amortization"]
    result["ebitda_estimate"] = ebitda
    result["ebitda_margin"] = ebitda / result["revenues"] if ebitda is not None and result["revenues"] else None
    result["revenue_growth"] = _revenue_growth(facts, as_of_date)

    shares = facts.get("dei", {}).get("EntityCommonStockSharesOutstanding")
    result["shares_outstanding"] = _instant_value(shares["units"].get("shares", []), as_of_date) if shares else None
    if result["shares_outstanding"] is None:
        alt = facts.get("us-gaap", {}).get("CommonStockSharesOutstanding")
        result["shares_outstanding"] = _instant_value(alt["units"].get("shares", []), as_of_date) if alt else None

    result["reports_in_usd"] = not any(p.startswith("ifrs-full") for p in provenance.values()) or bool(
        facts.get("ifrs-full", {}).get("Revenue", {}).get("units", {}).get("USD"))
    result["xbrl_provenance"] = provenance
    return result
