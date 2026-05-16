# Baseline Summary

The migrated baseline was produced by the original scripts under `legacy/`.

## Data

- Stock universe: 5,458 A-share symbols.
- Daily panel: 431,246 rows, 300 symbols.
- Daily date range: 2020-01-02 to 2025-12-16.
- Factor panel: 11,937 rows, 300 symbols.
- Factor date range: 2021-03-31 to 2025-10-31.

## Backtest Baseline

The existing `outputs/proj5_output` baseline has 40 monthly periods.

| Strategy | Final NAV | Annual Return | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|
| long_only | 2.141338 | 25.66% | 1.222180 | -15.53% |
| short_only | 0.593218 | -14.50% | -0.682753 | -55.97% |
| long_short | 3.343105 | 43.63% | 2.288222 | -10.96% |
| benchmark_ew | 1.105654 | 3.06% | 0.169608 | -22.49% |

The new pipeline writes fresh runs to `outputs/latest` by default so this baseline stays intact.
