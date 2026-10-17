"""Pull market data (pre-announcement volatility, run-up) around a deal announcement date via yfinance. Best-effort: many acquisition targets delist and lose Yahoo Finance history availability, so callers must tolerate None results."""

import numpy as np
import yfinance as yf


def get_price_history(ticker: str, start: str, end: str):
    try:
        data = yf.Ticker(ticker).history(start=start, end=end)
    except Exception:
        return None
    return data if not data.empty else None


def get_pre_announcement_volatility(ticker: str, announcement_date: str, lookback_days: int = 60) -> float | None:
    """Annualized realized volatility of daily returns in the lookback_days window before announcement."""
    import pandas as pd

    end = pd.Timestamp(announcement_date)
    start = end - pd.Timedelta(days=int(lookback_days * 1.6))  # pad for weekends/holidays
    history = get_price_history(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if history is None or len(history) < 5:
        return None
    returns = history["Close"].pct_change().dropna()
    if returns.empty:
        return None
    return float(returns.std() * np.sqrt(252))


def get_deal_premium(ticker: str, announcement_date: str, offer_price_per_share: float, lookback_days: int = 1):
    """Premium of the offer price over the unaffected price shortly before announcement."""
    import pandas as pd

    end = pd.Timestamp(announcement_date)
    start = end - pd.Timedelta(days=int(lookback_days * 3 + 5))
    history = get_price_history(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if history is None or history.empty or offer_price_per_share is None:
        return None
    unaffected_price = float(history["Close"].iloc[-1])
    if unaffected_price <= 0:
        return None
    return (offer_price_per_share - unaffected_price) / unaffected_price
