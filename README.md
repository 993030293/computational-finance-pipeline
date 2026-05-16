# Computational Finance Pipeline

[![CI](https://github.com/993030293/computational-finance-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/993030293/computational-finance-pipeline/actions/workflows/ci.yml)

An admissions-oriented data science project that turns raw A-share price data into a reproducible factor research, statistical testing, machine learning, and portfolio backtesting workflow.

**Research report:** [`reports/research_report.md`](reports/research_report.md)

**Admissions packet:** [`reports/admissions_packet.md`](reports/admissions_packet.md)

## Quickstart

Run the complete pipeline on the public sample dataset:

```powershell
git clone https://github.com/993030293/computational-finance-pipeline.git
cd computational-finance-pipeline
python -m pip install -e ".[dev]"
python scripts/create_sample_data.py
cfp run-all --skip-fetch --config configs/sample.yaml
```

## Why This Project Matters

This project demonstrates the core skills expected in data science and statistics graduate study:

- reproducible Python engineering with a CLI, configuration files, tests, and CI;
- time-series experimental design without random data leakage;
- statistical factor validation with IC, bootstrap confidence intervals, Fama-MacBeth regressions, and robustness checks;
- supervised learning experiments with walk-forward validation;
- honest reporting of transaction costs, turnover, limitations, and non-production assumptions.

## Dataset

- Market: A-share daily OHLCV data collected with AkShare.
- Migrated baseline: 431,246 daily rows, 300 symbols, 2020-01-02 to 2025-12-16.
- Factor panel: monthly cross-section, 11,937 rows, 2021-03-31 to 2025-10-31.
- Large local data lives in `data/` and is ignored by git.
- A small public sample lives in `examples/sample_data/` for CI and clone-and-run demos.

See `DATA_CARD.md` for source, quality, and bias notes.

## Methods

The pipeline has eight stages:

```text
fetch -> clean -> factors -> backtest -> ml -> decision -> tune -> stress
```

## Method Map

```text
Raw A-share OHLCV
  -> cleaning and technical indicators
  -> factor panel and ML features
  -> statistical validation
  -> factor/ML prediction scores
  -> LANCER-inspired decision optimizer
  -> MEHA-inspired chronological hyperparameter tuning
  -> EvoMarket-inspired market mechanism stress tests
  -> application-ready reports
```

- `clean`: type normalization, duplicate checks, price sanity checks, winsorized returns, MA/EMA/MACD/RSI/volatility features.
- `factors`: VALUE, MOM_12_1, QUALITY, SIZE plus REVERSAL_1M, VOL_1M, ILLIQUIDITY.
- `factor validation`: Rank IC, ICIR, bootstrap CI, IC decay, quantile group returns, Fama-MacBeth summaries.
- `backtest`: long-only, short-basket, long-short, equal-weight benchmark, turnover, transaction costs, net performance, split-period results.
- `ml`: linear/ridge/lasso/logistic/random-forest/gradient-boosting models with expanding-window walk-forward validation.
- `decision`: independent mean-variance portfolio optimizer that maps factor/ML scores into long-only portfolio weights.
- `tune`: chronological grid search that selects hyperparameters on validation performance only.
- `stress`: market-mechanism sensitivity tests for cost, liquidity, price-limit, and T+1 approximations.

## Baseline Result

The default factor strategy keeps the original baseline behavior while adding net-cost reporting.

| Strategy | Final NAV | Annual Return | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|
| long_only | 2.141338 | 25.66% | 1.222180 | -15.53% |
| short_only | 0.593218 | -14.50% | -0.682753 | -55.97% |
| long_short | 3.343105 | 43.63% | 2.288222 | -10.96% |
| benchmark_ew | 1.105654 | 3.06% | 0.169608 | -22.49% |

These results are not presented as a live trading claim. They are a research baseline for demonstrating statistical modeling and reproducible analysis.

## Research Report

The main admissions-facing research writeup is `reports/research_report.md`. It presents the project as an 8-12 page English research report covering data quality, factor validation, supervised learning, decision-focused portfolio optimization, validation-only tuning, market stress tests, limitations, and reproducibility.

If a local PDF toolchain is available, the same report can also be exported to `reports/research_report.pdf`.

## Reproduce

Install:

```powershell
git clone https://github.com/993030293/computational-finance-pipeline.git
cd computational-finance-pipeline
python -m pip install -e ".[dev]"
```

Run with the small sample dataset:

```powershell
python scripts/create_sample_data.py
cfp run-all --skip-fetch --config configs/sample.yaml
```

Run with the migrated full local dataset:

```powershell
cfp run-all --skip-fetch --config configs/default.yaml
```

Run individual advanced stages:

```powershell
cfp decision --config configs/default.yaml
cfp tune --config configs/default.yaml
cfp stress --config configs/default.yaml
```

Without installing the package:

```powershell
$env:PYTHONPATH = "src"
python -m cfpipeline run-all --skip-fetch --config configs/sample.yaml
```

## Outputs

- Factor research: `outputs/latest/project4/`
- Gross and net backtest: `outputs/latest/proj5_output/`
- ML experiment: `outputs/latest/ml/`
- Decision optimizer: `outputs/latest/decision/`
- Bilevel-style tuning: `outputs/latest/tuning/`
- Market stress tests: `outputs/latest/stress/`
- Research report: `reports/research_report.md`
- Admissions packet: `reports/admissions_packet.md`
- Optional report PDF: `reports/research_report.pdf`
- Related work adaptation: `reports/related_work.md`
- Application summaries: `reports/application_materials.md`

## Tests

```powershell
pytest
```

GitHub Actions runs tests and a CLI smoke test on sample data.

## Limitations

- The dataset is survivorship-biased if the AkShare universe snapshot is used without historical constituents.
- Transaction costs are simplified bps assumptions, not exchange-level execution simulation.
- The monthly factor backtest is educational research, not a production trading system.
- ML results should be interpreted as a statistical learning experiment under non-stationarity.
