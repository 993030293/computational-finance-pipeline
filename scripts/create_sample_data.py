from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_sample(symbols: int = 8, start: str = "2019-01-01", end: str = "2023-12-31") -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="B")
    frames = []
    for idx in range(symbols):
        symbol = f"{idx + 1:06d}"
        steps = np.arange(len(dates))
        drift = 0.00018 + idx * 0.00003
        cycle = 0.018 * np.sin(steps / (18 + idx) + idx)
        close = (10 + idx) * np.exp(drift * steps) * (1 + cycle)
        open_ = close * (1 - 0.001)
        high = close * 1.012
        low = close * 0.988
        volume = 100000 + idx * 7500 + steps * (12 + idx)
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


def main() -> int:
    out = Path("examples/sample_data/processed")
    out.mkdir(parents=True, exist_ok=True)
    make_sample().to_csv(out / "daily_price_50.csv", index=False, encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
