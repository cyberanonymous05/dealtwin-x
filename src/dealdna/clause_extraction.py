"""Pattern-based extraction from raw merger-proxy / SC 14D9 text: numeric deal terms, stated premium,
acquirer attributes, and binary legal-term flags.

Numeric terms (cash per share, exchange ratio, termination fee) are inherently pattern-based. Which
passages represent which *type* of legal clause for the text encoders is decided by sentence-embedding
similarity in semantic_clause_classifier.py, not here. The legal-term flags below are deliberately
independent of that selector: they feed the legal-aware relevance construct, so a legal signal can be
evaluated without relying on the passages the encoders see.
"""

import re
from collections import Counter

# ---------------------------------------------------------------------------- text cleaning
_SCRIPT_RE = re.compile(r"(?is)<(script|style)[^>]*>.*?</\1>")
_BLOCK_RE = re.compile(r"(?i)</?(p|div|br|tr|li|h\d|table)[^>]*>")
_TAG_RE = re.compile(r"<[^>]+>")


def html_to_text(html: str) -> str:
    html = _SCRIPT_RE.sub(" ", html)
    html = _BLOCK_RE.sub("\n", html)
    text = _TAG_RE.sub(" ", html)
    for entity, repl in (("&nbsp;", " "), ("&#160;", " "), ("&amp;", "&"), ("&#8217;", "'"), ("&#8220;", '"'),
                         ("&#8221;", '"'), ("&#8212;", "-"), ("&#8211;", "-"), ("&rsquo;", "'"), ("&ldquo;", '"'),
                         ("&rdquo;", '"'), ("&mdash;", "-"), ("&ndash;", "-"), ("&#36;", "$"), ("&#37;", "%")):
        text = text.replace(entity, repl)
    text = re.sub(r"&#x?[0-9a-fA-F]+;", " ", text)
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n\s*\n\s*", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------- M&A gate
_MERGER_AGREEMENT_RE = re.compile(r"agreement\s+and\s+plan\s+of\s+merger|merger\s+agreement|plan\s+of\s+merger", re.I)
_ASSET_SALE_RE = re.compile(r"sale\s+of\s+(?:all\s+or\s+)?substantially\s+all\s+(?:of\s+)?(?:the\s+)?(?:company'?s\s+)?assets", re.I)


_SPAC_RE = re.compile(r"trust\s+account", re.I)
_BUSINESS_COMBINATION_RE = re.compile(r"business\s+combination", re.I)
# Target-company proxies must carry the Item 402(t) advisory vote on merger-related ("golden parachute")
# compensation; acquirer-side proxies instead ask holders to approve the issuance of the buyer's shares.
_GOLDEN_PARACHUTE_RE = re.compile(
    r"(?:advisory|non-binding)[^.]{0,120}?(?:compensation|golden\s+parachute)[^.]{0,200}?(?:in\s+connection\s+with|based\s+on\s+or\s+otherwise\s+relat|relat\w+\s+to)\s+(?:the\s+)?(?:merger|transaction)"
    r"|golden\s+parachute\s+compensation|merger[\s-]related\s+(?:named\s+executive\s+officer\s+)?compensation",
    re.I,
)
_SHARE_ISSUANCE_RE = re.compile(r"(?:share|stock)\s+issuance\s+proposal|approve\s+the\s+issuance\s+of\s+(?:shares|common\s+stock)", re.I)


def ma_gate(text: str, form: str = "") -> str | None:
    """None if the filing is a target-side document for a negotiated whole-company acquisition,
    else an exclusion reason."""
    head = text[:300_000]
    if len(text) < 5000:
        return "text_too_short"
    if len(_MERGER_AGREEMENT_RE.findall(head)) < 3:
        return "no_merger_agreement"
    if len(_SPAC_RE.findall(head)) >= 3 and len(_BUSINESS_COMBINATION_RE.findall(head)) >= 5:
        return "spac_business_combination"
    if len(_ASSET_SALE_RE.findall(head)) > len(_MERGER_AGREEMENT_RE.findall(head)):
        return "asset_sale"
    if form == "DEFM14A" and not _GOLDEN_PARACHUTE_RE.search(head) and _SHARE_ISSUANCE_RE.search(head):
        return "acquirer_side_proxy"
    return None


# ---------------------------------------------------------------------------- numeric deal terms
_CASH_PER_SHARE_RE = re.compile(
    r"\$\s?([\d,]{1,7}\.\d{2,4})\s*(?:in\s+cash[,\s]*)?(?:,?\s*without\s+interest[,\s]*)?(?:net\s+to\s+the\s+seller[^.]{0,40}?)?per\s+(?:share|Share)",
    re.I,
)
_CASH_PER_SHARE_RE2 = re.compile(r"per\s+share\s+(?:merger\s+)?(?:consideration|price)\s+of\s+\$\s?([\d,]{1,7}\.\d{2,4})", re.I)
_CASH_PER_SHARE_RE3 = re.compile(
    r"\$\s?([\d,]{1,7}\.\d{2,4})\s+in\s+cash[^.$]{0,160}?for\s+each\s+(?:outstanding\s+)?(?:share|common\s+share)", re.I
)
_EXCHANGE_RATIO_RE = re.compile(r"(\d+\.\d{2,6})\s+(?:of\s+a\s+)?shares?\s+of[^.]{0,60}?common\s+stock", re.I)
_FEE_PATTERNS = [
    re.compile(r"termination\s+fee\s+(?:of|equal\s+to|in\s+the\s+amount\s+of)\s+(?:approximately\s+)?\$\s?([\d,]+(?:\.\d+)?)\s*(million|billion)?", re.I),
    re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*(million|billion)?\s+(?:company\s+)?termination\s+fee", re.I),
    re.compile(r"(?:pay|paying)\s+(?:to\s+\w+\s+)?a\s+(?:termination\s+)?fee\s+of\s+\$\s?([\d,]+(?:\.\d+)?)\s*(million|billion)?", re.I),
]
_REVERSE_FEE_RE = re.compile(r"reverse\s+termination\s+fee|parent\s+termination\s+fee", re.I)
_ALL_CASH_RE = re.compile(r"all[\s-]cash\s+(?:transaction|merger|deal)|paid\s+entirely\s+in\s+cash|\bin\s+cash,?\s+without\s+interest", re.I)
_ALL_STOCK_RE = re.compile(r"all[\s-]stock\s+(?:transaction|merger)|stock[\s-]for[\s-]stock", re.I)
_EARNOUT_RE = re.compile(r"earn[\s-]?out|contingent\s+value\s+rights?|\bCVRs?\b", re.I)
_MAC_RE = re.compile(r"Material\s+Adverse\s+(?:Effect|Change)", re.I)


def _money(amount: str, unit: str | None) -> float:
    value = float(amount.replace(",", ""))
    if unit:
        value *= 1e9 if unit.lower() == "billion" else 1e6
    return value


# A per-share dollar figure counts as a deal-price candidate only in a consideration context, and never
# when it is a par value, dividend, redemption / liquidation preference, option exercise price, or a quoted
# trading price (audit of 50 filings: these were the failure modes of an unconditioned regex).
_PRICE_CONTEXT_RE = re.compile(
    r"in\s+cash|net\s+to\s+the\s+(?:seller|holder|stockholder)|merger\s+consideration|offer\s+price|purchase\s+price|"
    r"per\s+share\s+(?:merger\s+|cash\s+)?(?:consideration|price)|right\s+to\s+receive|entitled\s+to\s+receive|"
    r"price\s+per\s+share\s+equal\s+to|will\s+receive|to\s+acquire\s+all",
    re.I,
)
# Words directly attached to the dollar figure (within ~40 characters) that make it something other than
# the deal price ...
_PRICE_ATTACHED_EXCLUDE_RE = re.compile(
    r"par\s+value|dividend|distribution|redemption|redeem|liquidation\s+preference|exercise\s+price|strike|"
    r"closing\s+(?:sale\s+)?price|trading\s+price|traded|52-week|high\s+of|low\s+of|at\s+a\s+cost\s+of|"
    r"preferred|warrant|conversion\s+price",
    re.I,
)
# ... and sentence-level markers of negotiation history or valuation analysis (earlier bids in the
# "Background of the Merger", fairness-opinion ranges), which repeat superseded or hypothetical prices.
_PRICE_SENTENCE_EXCLUDE_RE = re.compile(
    r"propos|non-binding|indicat\w+\s+of\s+interest|range|implied|valued\s+the\s+company|ex-dividend|reference\s+range",
    re.I,
)


def _price_candidates(text: str) -> list[tuple[float, int]]:
    """(price, position) for every per-share dollar figure in a deal-consideration context."""
    out, seen = [], set()
    for pattern in (_CASH_PER_SHARE_RE, _CASH_PER_SHARE_RE2, _CASH_PER_SHARE_RE3):
        for m in pattern.finditer(text):
            if m.start(1) in seen:  # the same dollar figure matched by two patterns counts once
                continue
            seen.add(m.start(1))
            before = text[max(0, m.start() - 200): m.start()]
            if not _PRICE_CONTEXT_RE.search(before[-160:] + text[m.start(): m.end() + 80]):
                continue
            sentence = before
            cut = max(sentence.rfind(". "), sentence.rfind("; "), sentence.rfind("\n"))
            if cut >= 0:
                sentence = sentence[cut + 2:]
            if _PRICE_ATTACHED_EXCLUDE_RE.search(sentence[-40:] + text[m.start(): m.end()]):
                continue
            if _PRICE_SENTENCE_EXCLUDE_RE.search(sentence + text[m.start(): m.end() + 60]):
                continue
            value = float(m.group(1).replace(",", ""))
            if 0.05 <= value <= 5000:
                out.append((value, m.start()))
    return out


def extract_deal_terms(text: str) -> dict:
    candidates = _price_candidates(text)
    # The deal price is the per-share figure the document repeats most often (it is quoted in the summary,
    # the background, every fairness-opinion section and the merger-agreement summary); one-off figures
    # are historical trading prices, analyst ranges or option strike prices.
    cash_per_share = None
    # The deal price is stated at the front of the document (cover letter, summary term sheet, Q&A: the
    # first quarter, at most 80k characters); later sections narrate competing bids and superseded offers.
    # Decide from that head region whenever a figure repeats there, otherwise from the whole document.
    head = [(p, pos) for p, pos in candidates if pos < min(0.25 * len(text), 80_000)]
    if head and max(Counter(p for p, _ in head).values()) >= 2:
        candidates = head
    if candidates:
        counts = Counter(p for p, _ in candidates)
        first_seen = {}
        for p, pos in candidates:
            first_seen.setdefault(p, pos)
        top = max(counts.values())
        if top >= 2:
            # most repeated figure; ties go to the figure stated first (cover letter / summary term sheet)
            cash_per_share = min((p for p, c in counts.items() if c == top), key=lambda p: first_seen[p])

    ratios = [float(r) for r in _EXCHANGE_RATIO_RE.findall(text) if 0.01 <= float(r) <= 50]
    exchange_ratio = Counter(ratios).most_common(1)[0][0] if ratios else None

    fees = []
    for pattern in _FEE_PATTERNS:
        for amount, unit in pattern.findall(text[:400_000]):
            value = _money(amount, unit)
            if 1e5 <= value <= 5e10:
                fees.append(value)
    termination_fee_usd = Counter(fees).most_common(1)[0][0] if fees else None

    is_all_cash, is_all_stock = bool(_ALL_CASH_RE.search(text)), bool(_ALL_STOCK_RE.search(text))
    if is_all_cash and not is_all_stock and exchange_ratio is None:
        mix = "cash"
    elif is_all_stock and not is_all_cash:
        mix = "stock"
    elif cash_per_share is not None and exchange_ratio is not None:
        mix = "cash_and_stock"
    elif cash_per_share is not None:
        mix = "cash"
    elif exchange_ratio is not None:
        mix = "stock"
    else:
        mix = "unknown"

    return {
        "cash_per_share": cash_per_share,
        "exchange_ratio": exchange_ratio,
        "termination_fee_usd": termination_fee_usd,
        "consideration_mix": mix,
        "has_earnout": bool(_EARNOUT_RE.search(text)),
        "mac_clause_mentions": len(_MAC_RE.findall(text)),
    }


# ---------------------------------------------------------------------------- stated premium
_STATED_PREMIUM_RES = [
    re.compile(r"(\d+(?:\.\d+)?)\s?%\s+premium\s+(?:to|over)\s+the(?:\s+[\w-]+){0,6}?\s+(?:closing|stock|share)\s+(?:stock\s+|share\s+)?price", re.I),
    re.compile(r"premium\s+of\s+(?:approximately\s+)?(\d+(?:\.\d+)?)\s?%\s+(?:to|over)\s+the(?:\s+[\w-]+){0,8}?\s+(?:closing|stock|share)\s+(?:stock\s+|share\s+)?price", re.I),
    re.compile(r"premium\s+of\s+(?:approximately\s+)?(\d+(?:\.\d+)?)\s?%\s+(?:to|over)\s+[A-Z][\w&.,'\- ]{0,40}?\s+(?:stock|share)\s+price", re.I),
]
_UNAFFECTED_RE = re.compile(
    r"last\s+(?:full\s+)?trading\s+day\s+(?:prior|before)|trading\s+day\s+immediately\s+pr(?:ior|eceding)|"
    r"day\s+immediately\s+pr(?:ior|eceding)|(?:prior|before)\s+to\s+the\s+(?:public\s+)?announcement|unaffected",
    re.I,
)


def extract_stated_premium_pct(text: str, context_window: int = 200) -> float | None:
    candidates = []
    for pattern in _STATED_PREMIUM_RES:
        for m in pattern.finditer(text):
            if _UNAFFECTED_RE.search(text[m.end(): m.end() + context_window]):
                value = float(m.group(1)) / 100.0
                if 0.0 <= value <= 5.0:
                    candidates.append((m.start(), value))
    return min(candidates)[1] if candidates else None


# ---------------------------------------------------------------------------- acquirer attributes
_SPONSOR_RE = re.compile(
    r"equity\s+commitment\s+letter|private\s+equity|funds?\s+(?:managed|advised|sponsored)\s+by|"
    r"affiliates?\s+of\s+[A-Z][\w&.\- ]{1,40}(?:Capital|Partners|Equity|Investors|Holdings)\b|\bsponsors?\b(?=[^.]{0,80}(?:equity|financing))",
    re.I,
)
_LISTED_PARENT_RE = re.compile(
    r"(?:Parent|acquir\w+|Buyer)(?:'s)?\s+common\s+stock\s+(?:is|are)\s+(?:listed|traded)\s+on\s+(?:the\s+)?(?:NYSE|New\s+York\s+Stock|Nasdaq|NASDAQ)",
    re.I,
)
_FOREIGN_LAW_RE = re.compile(
    r"(?:organized|incorporated|existing)\s+under\s+the\s+laws\s+of\s+(?!the\s+State\s+of|Delaware|Nevada|Maryland|New\s+York|"
    r"Pennsylvania|Massachusetts|California|Texas|Ohio|Virginia|Florida|Washington|Minnesota|New\s+Jersey|Georgia|Indiana|"
    r"Michigan|Colorado|North\s+Carolina|Tennessee|Wisconsin|Missouri|Oregon|Utah|Arizona|Iowa|Kansas|Kentucky|Louisiana|"
    r"Oklahoma|Connecticut|Illinois|the\s+United\s+States)(?:the\s+)?(England|Wales|Japan|Germany|France|Canada|Ontario|"
    r"British\s+Columbia|Quebec|Alberta|Ireland|the\s+Netherlands|Switzerland|Israel|China|the\s+People's\s+Republic|Hong\s+Kong|"
    r"Singapore|India|Korea|the\s+Republic\s+of\s+Korea|Australia|Sweden|Denmark|Norway|Finland|Belgium|Luxembourg|Spain|Italy|"
    r"Bermuda|the\s+Cayman\s+Islands|Jersey|Guernsey|Taiwan|Mexico|Brazil|Austria)",
    re.I,
)
_CFIUS_RE = re.compile(r"\bCFIUS\b|Committee\s+on\s+Foreign\s+Investment", re.I)


def extract_acquirer_attributes(text: str, consideration_mix: str) -> dict:
    head = text[:400_000]
    sponsor_hits = len(_SPONSOR_RE.findall(head))
    is_sponsor = sponsor_hits >= 3 and consideration_mix not in ("stock", "cash_and_stock")
    listed = bool(_LISTED_PARENT_RE.search(head)) or consideration_mix in ("stock", "cash_and_stock")
    cross_border = bool(_FOREIGN_LAW_RE.search(head)) or len(_CFIUS_RE.findall(head)) >= 2
    return {
        "acquirer_type": "sponsor" if is_sponsor else "strategic",
        "acquirer_listed": listed,
        "cross_border": cross_border,
    }


# ---------------------------------------------------------------------------- legal-term flags
_LEGAL_FLAGS = {
    "go_shop": re.compile(r"go[\s-]shop", re.I),
    "reverse_termination_fee": _REVERSE_FEE_RE,
    "hsr_required": re.compile(r"Hart[\s-]Scott[\s-]Rodino|\bHSR\s+Act\b", re.I),
    "cfius_review": _CFIUS_RE,
    "majority_of_minority": re.compile(r"majority\s+of\s+the\s+(?:outstanding\s+)?(?:shares\s+)?(?:held\s+by\s+)?(?:unaffiliated|disinterested|minority)", re.I),
    "voting_or_support_agreement": re.compile(r"(?:voting|support|tender)\s+(?:and\s+support\s+)?agreements?", re.I),
    "financing_condition": re.compile(r"financing\s+condition|condition(?:ed)?\s+(?:up)?on\s+(?:the\s+)?(?:receipt\s+of\s+)?financing", re.I),
    "top_up_option": re.compile(r"top[\s-]up\s+option", re.I),
    "section_251h": re.compile(r"Section\s+251\s*\(h\)", re.I),
    "cvr": re.compile(r"contingent\s+value\s+rights?|\bCVRs?\b", re.I),
}


def extract_legal_flags(text: str) -> dict:
    head = text[:500_000]
    return {name: bool(pattern.search(head)) for name, pattern in _LEGAL_FLAGS.items()}
