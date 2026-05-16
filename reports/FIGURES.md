# Figure Index

Generated figures are written under `outputs/<run>/`.

## Factor Research

- `project4/ic_ts_*.png`: cumulative IC by factor.
- `project4/ic_hist_*.png`: IC distribution by factor.
- `project4/factor_corr_heatmap.png`: standardized factor correlation.

## Backtest

- `proj5_output/cumret_curves__*.png`: cumulative strategy NAV.
- `proj5_output/drawdown_long_short__*.png`: long-short drawdown.
- `proj5_output/rolling_long_short__*.png`: rolling volatility and Sharpe.

## Tables

- `project4/ic_significance.csv`: bootstrap CI and t-statistics.
- `project4/fama_macbeth_summary.csv`: cross-sectional regression coefficient summaries.
- `project4/factor_group_return_summary.csv`: factor quantile return summaries.
- `proj5_output/robustness_quantiles.csv`: quantile robustness table.
- `ml/ml_model_metrics.csv`: supervised learning comparison.
- `ml/ml_feature_importance.csv`: model feature importance.
- `decision/decision_metrics.csv`: factor-score and ML-score decision optimizer comparison.
- `tuning/tuning_results.csv`: validation-only hyperparameter search results.
- `stress/market_stress_metrics.csv`: market mechanism stress-test results.

## Research Report Tables

`reports/research_report.md` embeds the key admissions-facing tables directly so a reviewer does not need local access to ignored large outputs:

- Rank IC summary and bootstrap significance table.
- Gross and net baseline backtest table.
- Quantile robustness table.
- Supervised learning model comparison table.
- Decision optimizer metrics table.
- Chronological tuning train/validation/test table.
- Market mechanism stress-test table.

If Pandoc and a PDF engine are available, the Markdown report can be exported to `reports/research_report.pdf`.
