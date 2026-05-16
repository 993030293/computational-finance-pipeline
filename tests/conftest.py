from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def make_daily_sample():
    def factory(symbols: int = 6, start: str = "2020-01-01", end: str = "2022-12-31") -> pd.DataFrame:
        dates = pd.date_range(start, end, freq="B")
        frames = []
        for idx in range(symbols):
            symbol = f"{idx + 1:06d}"
            trend = 1 + 0.00015 * idx
            base = 10 + idx
            steps = np.arange(len(dates))
            close = base * (1 + trend * 0.0005) ** steps * (1 + 0.02 * np.sin(steps / 19 + idx))
            open_ = close * (1 - 0.001)
            high = close * 1.01
            low = close * 0.99
            volume = 100000 + idx * 1000 + steps * 10
            amount = close * volume
            frames.append(
                pd.DataFrame(
                    {
                        "symbol": symbol,
                        "date": dates,
                        "open": open_,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume,
                        "amount": amount,
                        "pct_chg": pd.Series(close).pct_change().to_numpy() * 100,
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    return factory
