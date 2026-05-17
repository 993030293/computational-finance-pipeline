# Data Card

## Dataset

The project uses A-share stock universe and daily OHLCV data acquired through AkShare. The migrated full local dataset contains 300 symbols and 431,246 daily rows from 2020-01-02 to 2025-12-16.

## Intended Use

This dataset is used to demonstrate data cleaning, factor construction, statistical validation, supervised learning, and backtesting methodology in a reproducible research pipeline.

## Not Intended For

- Live trading decisions.
- Claims of deployable alpha.
- Regulatory or investment advice.

## Processing

The pipeline standardizes column names, coerces numeric/date types, removes duplicate symbol-date rows, fixes `high < low` inconsistencies, computes missing percentage changes from close prices, and creates technical indicators.

## Biases and Limitations

- Universe construction uses a current/snapshot list unless historical constituents are supplied.
- Delisting, suspension, liquidity, and corporate-action details are simplified.
- AkShare endpoint availability may change over time.
- Data quality depends on third-party source coverage.

## Access and Reproducibility

Large migrated data is kept locally in `data/` and excluded from git. The repository includes `examples/sample_data/` so CI and reviewers can run the full pipeline without private or heavy artifacts.

## License and Source Terms

The repository source code is MIT licensed. Market data acquired through AkShare or its upstream providers may be subject to separate terms, redistribution limits, and exchange/vendor restrictions. This repository does not redistribute the full migrated market dataset. Users are responsible for checking the licensing and usage terms of any data source they fetch or provide.
