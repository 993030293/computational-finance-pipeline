from __future__ import annotations

import numpy as np
import pandas as pd

from cfpipeline.cleaning import clean_daily_prices, engineer_features


def test_clean_daily_prices_fixes_high_low_and_pct_chg() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["000001", "000001", "000001"],
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "open": [10, 11, 12],
            "high": [10, 10, 13],
            "low": [9, 12, 11],
            "close": [10, 11, 12.1],
            "volume": [100, 110, 120],
            "amount": [1000, 1210, 1452],
            "pct_chg": [np.nan, np.nan, np.nan],
        }
    )
    cleaned = clean_daily_prices(df)
    assert cleaned.loc[1, "high"] == 12
    assert cleaned.loc[1, "low"] == 10
    assert round(float(cleaned.loc[1, "pct_chg"]), 6) == 10.0


def test_engineer_features_adds_technical_columns(make_daily_sample) -> None:
    df = make_daily_sample(symbols=2)
    features, stats = engineer_features(df)
    for col in ["ret", "log_ret", "ret_w", "ma_5", "ma_20", "ema12", "ema26", "macd", "vol_20", "rsi_14"]:
        assert col in features.columns
    assert stats["after"]["symbols"] == 2
    assert features["ma_20"].notna().any()
