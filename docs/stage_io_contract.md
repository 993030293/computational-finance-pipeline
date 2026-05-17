# Stage I/O Contract

Each stage accepts the merged runtime config and returns a dictionary of named output paths. Full `cfp run-all` executions record stage status, inputs, outputs, and metrics summaries in `run_manifest.json`.

| Stage | Input Files | Output Files | Config Section | Failure Behavior | Tests |
|---|---|---|---|---|---|
| `fetch` | AkShare stock universe and daily quote endpoints | `raw/stock_universe.csv`, `raw/daily_price.csv`, `processed/daily_price_panel.csv`, `processed/daily_price_50.csv`, `database/financial_data.db`, `FETCH_REPORT.md`, `fetch_checkpoint.csv` | `fetch`, `data` | Symbol-level failures are recorded in checkpoint and report; successful symbols are preserved; run fails only if no daily data is available | `tests/test_acquire.py` |
| `clean` | `processed/daily_price_panel.csv`, `processed/daily_price_50.csv`, `processed/daily_price.csv`, or `raw/daily_price.csv` from input or output dirs | `processed/daily_price_panel.csv`, `processed/daily_price_50.csv`, `processed/tech_indicators.csv`, `QUALITY_REPORT.md`, `CLEANING_STEPS.json`, EDA figures | `cleaning`, `data` | Missing input raises `FileNotFoundError`; malformed columns are normalized where possible; required missing fields are filled before cleaning | `tests/test_cleaning.py`, `tests/test_integration.py` |
| `factors` | `processed/tech_indicators.csv` or cleaned daily price panel | `project4/factors.csv`, IC series and summaries, correlation matrix, IC decay, group returns, Fama-MacBeth summary, direction report, Project 4 PDF/ZIP | `factors`, `research`, `data` | Missing price panel raises `FileNotFoundError`; insufficient valid rows produce empty diagnostic tables rather than silent success where supported | `tests/test_factors.py`, `tests/test_research.py` |
| `backtest` | `project4/factors.csv`, cleaned daily price panel | `proj5_output/portfolio_returns.csv`, net returns, NAV, performance metrics, turnover, drawdown, rolling metrics, summary report | `backtest`, `research`, `data` | Missing factors/prices raise `FileNotFoundError`; invalid weights raise `ValueError`; outputs distinguish gross and net metrics | `tests/test_backtest.py`, `tests/test_integration.py` |
| `ml` | `project4/factors.csv` | `ml/ml_dataset.csv`, predictions, model metrics, validation comparison, feature importance, ML portfolio returns, ML report | `ml`, `validation`, `data` | Missing feature/target columns raise `KeyError`; chronological split with optional purge prevents train/test overlap | `tests/test_ml.py`, `tests/test_validation.py` |
| `decision` | `project4/factors.csv`, optional `ml/ml_predictions.csv` | `decision/decision_returns.csv`, `decision_weights.csv`, `decision_metrics.csv`, metadata, report | `decision`, `backtest`, `data` | Falls back to factor score if ML predictions are unavailable; optimizer errors are handled by conservative equal weights where applicable | `tests/test_decision.py` |
| `tune` | `project4/factors.csv` | `tuning/tuning_results.csv`, `selected_params.csv`, `test_performance.csv`, `selected_returns.csv`, metadata, report | `tuning`, `research`, `decision`, `data` | Selection uses validation split only; test split is reporting-only; empty validation falls back to first candidate | `tests/test_tuning_stress.py`, `tests/test_validation.py` |
| `stress` | `project4/factors.csv`, cleaned daily price panel | `stress/market_stress_returns.csv`, `market_stress_metrics.csv`, `market_stress_report.md` | `stress`, `backtest`, `data` | Missing inputs raise `FileNotFoundError`; stress regimes produce diagnostic return series and metrics | `tests/test_tuning_stress.py` |
| `benchmarks` | Backtest, ML, decision, tuning, and stress outputs | `benchmarks/benchmark_registry.csv`, `gross_net_comparison.csv`, `validation_method_comparison.csv`, `STABILITY_REPORT.md` | `validation`, `backtest`, `decision`, `data` | Missing optional outputs produce partial registry tables; test metrics are marked reporting-only | `tests/test_validation.py` |

## Artifact Rules

- CSV, JSON, and Markdown stage outputs should use atomic write helpers from `src/cfpipeline/artifacts.py`.
- Full runs write to `outputs/runs/<run_id>/`.
- `outputs/latest/` is a Windows-compatible copy of the latest successful full run.
- Large data and generated outputs are ignored by git.
