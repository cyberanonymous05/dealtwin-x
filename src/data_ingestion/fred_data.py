"""Macroeconomic context from FRED, as of each deal's filing date.

Each series is downloaded in full once and cached as CSV under data/raw/fred/, then looked up locally
(last observation on or before the date), instead of one API call per deal per series.
"""

import os

import pandas as pd

from src.utils.config import load_config

_RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "raw", "fred")
DEFAULT_SERIES = {s["id"]: s["name"] for s in load_config()["data_sources"]["fred"]["series"]}
_series_cache: dict[str, pd.Series] = {}


def _load(series_id: str) -> pd.Series:
    if series_id in _series_cache:
        return _series_cache[series_id]
    path = os.path.join(_RAW, f"{series_id}.csv")
    if not os.path.exists(path):
        from fredapi import Fred

        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            raise RuntimeError("FRED_API_KEY not set. Add it to your .env file.")
        os.makedirs(_RAW, exist_ok=True)
        Fred(api_key=api_key).get_series(series_id).dropna().rename(series_id).to_csv(path)
    series = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].dropna().sort_index()
    _series_cache[series_id] = series
    return series


def get_macro_snapshot(as_of_date: str, series_map: dict[str, str] | None = None) -> dict:
    snapshot = {}
    for series_id, name in (series_map or DEFAULT_SERIES).items():
        series = _load(series_id).loc[:as_of_date]
        snapshot[name] = float(series.iloc[-1]) if not series.empty else None
    return snapshot
