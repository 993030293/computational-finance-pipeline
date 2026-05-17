# Resume and Interview Notes

This document converts the project into concise descriptions for different technical audiences. It should not be treated as a claim that the strategy is production-ready or suitable for investment decisions.

## Software Engineering Version

Built a modular Python research pipeline with a typed configuration layer, CLI commands, CI quality gates, pre-commit hooks, versioned run outputs, atomic artifact writes, and manifest-based audit trails. The project supports reproducible sample runs without private data and uses tests to verify config validation, artifact generation, leakage controls, and core pipeline behavior.

Possible resume bullet:

- Engineered a production-style Python CLI pipeline with typed config validation, ruff/mypy/pytest-cov CI gates, atomic writes, run manifests, and versioned outputs for reproducible computational finance experiments.

## Data Engineering Version

Implemented a reliable market-data ingestion and transformation workflow with symbol-level caching, checkpoint/resume, bounded concurrency, rate limiting, raw and processed data layers, quality reports, and stage-level I/O contracts. Large generated artifacts are excluded from git while sample data keeps CI and external review reproducible.

Possible resume bullet:

- Built a resumable data pipeline for A-share OHLCV data with symbol-level cache, checkpoint recovery, rate-limited concurrent fetches, data quality reports, and auditable stage outputs.

## Quantitative Research Version

Developed a factor research and backtesting pipeline covering factor construction, Rank IC, ICIR, bootstrap confidence intervals, Fama-MacBeth summaries, purged walk-forward ML validation, transaction-cost-aware backtests, decision-focused portfolio optimization, validation-only tuning, market stress tests, and benchmark registry reporting.

Possible resume bullet:

- Designed a leakage-aware quantitative research pipeline with statistical factor validation, purged walk-forward ML experiments, validation-only hyperparameter tuning, benchmark registry, and gross/net performance diagnostics.

## Interview Walkthrough

1. Problem framing: the goal is not to market a trading system, but to make quant research reproducible, testable, and auditable.
2. Data layer: explain AkShare ingestion, sample data, cache/resume, checkpoint files, and data quality reports.
3. Config and reproducibility: show typed config validation, fail-fast config loading, `resolved_config.yaml`, and `run_manifest.json`.
4. Research workflow: describe factors, IC analysis, backtest, ML, optimizer, tuning, and stress tests.
5. Leakage controls: explain chronological splits, purged walk-forward embargo, and validation-only tuning.
6. Engineering quality: show CI matrix, ruff, mypy, pytest-cov, pre-commit, and atomic writes.
7. Limitations: discuss survivorship bias, simplified costs, no order book simulation, non-stationarity, and why metrics are research diagnostics.

## Risk and Limitation Talking Points

- The public sample data is for reproducibility testing, not statistical conclusions.
- The full local dataset may be survivorship-biased unless historical constituents are supplied.
- Transaction costs and slippage are simplified basis-point assumptions.
- Stress tests approximate market mechanisms; they are not a full exchange or limit-order-book simulator.
- ML metrics can be unstable under regime changes; purged validation may reduce headline metrics but improves credibility.
- Test metrics are reporting-only and must not be used for hyperparameter selection.

## Short Project Description

Computational Finance Pipeline is a reproducible Python CLI project for factor research, supervised learning experiments, decision-aware portfolio optimization, and market stress testing. It emphasizes typed configuration, CI quality gates, sample-data reproducibility, leakage-aware validation, versioned outputs, and audit manifests rather than live trading claims.
