# Admissions Packet

This packet turns the project into reusable application material for data science, statistics, computational finance, and applied machine learning programs.

## One-Page Project Summary

**Title:** Reproducible Statistical Learning Pipeline for A-Share Factor Research

**Problem.** Financial factor research is a noisy empirical setting: signals are weak, returns are non-stationary, and naive backtests can overstate evidence. This project asks whether cross-sectional A-share factors can rank next-month returns, and whether those predictions remain useful after chronological validation, portfolio decision constraints, and market-friction stress tests.

**Dataset.** The migrated full dataset contains 431,246 daily OHLCV observations for 300 A-share symbols from 2020-01-02 to 2025-12-16. The monthly factor panel contains 11,937 observations from 2021-03-31 to 2025-10-31. Large data is excluded from git; a small sample dataset supports public CI and clone-and-run reproduction.

**Methods.** The pipeline implements data cleaning, technical indicators, factor construction, Rank IC, bootstrap confidence intervals, Fama-MacBeth summaries, transaction-cost-aware backtesting, supervised learning with expanding-window validation, long-only decision optimization, validation-only hyperparameter tuning, and market-mechanism stress tests.

**Key Results.** The reproduced factor baseline has a gross long-short Sharpe of 2.288 and max drawdown of -10.96% over 40 monthly periods. The ML layer finds modest but usable rank signal, with logistic classification reaching Rank IC 0.122 and AUC 0.559. The decision optimizer converts ML scores into long-only weights with net Sharpe 2.838 in the sample. High-cost and liquidity-stress regimes reduce apparent performance, which supports cautious interpretation.

**Contribution.** The value of the project is not a trading claim. It demonstrates reproducible empirical design, statistical validation, model comparison, optimization-aware evaluation, and honest limitations in a single engineered Python package.

**Reproduce.**

```powershell
git clone https://github.com/993030293/computational-finance-pipeline.git
cd computational-finance-pipeline
python -m pip install -e ".[dev]"
python scripts/create_sample_data.py
cfp run-all --skip-fetch --config configs/sample.yaml
```

## Resume Bullets

- Built a reproducible Python pipeline for A-share equity research, covering data cleaning, factor engineering, statistical validation, walk-forward machine learning, portfolio optimization, stress testing, pytest coverage, CLI automation, and GitHub Actions CI.
- Evaluated cross-sectional return signals using Rank IC, bootstrap confidence intervals, Fama-MacBeth regressions, quantile portfolios, transaction-cost-aware backtests, and validation-only hyperparameter tuning.
- Integrated factor scores and ML predictions into a decision-aware long-only optimizer with explicit risk, turnover, and concentration penalties, then tested sensitivity to cost, liquidity, price-limit, and T+1 assumptions.

## SOP Paragraph

One project that shaped my interest in data science was a computational finance pipeline I rebuilt from course scripts into a reproducible research system. I used Python, pandas, scikit-learn, and statistical testing to study whether cross-sectional equity factors could rank next-month returns in A-share data. Beyond implementing a backtest, I focused on research design: chronological train/validation/test splits, walk-forward evaluation, bootstrap confidence intervals, Fama-MacBeth summaries, transaction-cost reporting, robustness checks, and clear documentation of limitations. I also added a decision-aware optimization layer and market-friction stress tests to connect statistical prediction with downstream decisions. This project helped me connect statistical modeling, software engineering, and honest empirical analysis, which is the type of work I hope to deepen in graduate study.

## Professor Email Summary

Dear Professor [Name],

I am applying to [Program] and wanted to share a project that reflects my interests in statistical learning, reproducible research, and financial data analysis. I rebuilt a computational finance coursework project into a Python package and CLI that turns daily A-share OHLCV data into a monthly factor panel, validates signals using Rank IC, bootstrap intervals, IC decay, Fama-MacBeth summaries, and quantile portfolios, and compares factor scores with supervised learning models under expanding-window validation.

The project also includes a decision-aware portfolio optimizer, validation-only hyperparameter tuning, and market-friction stress tests for cost, liquidity, price-limit, and T+1 assumptions. I present the results as a methodological study rather than a trading claim, with explicit limitations around survivorship bias, simplified execution, and non-stationarity.

Repository: https://github.com/993030293/computational-finance-pipeline

Best regards,
[Your Name]

## Interview Talking Points

- Why time-series data cannot be randomly shuffled for model validation.
- Why Rank IC and AUC can be modest but still useful in cross-sectional selection.
- Why a high Sharpe backtest is not enough without cost, turnover, liquidity, and validation checks.
- Why decision-focused evaluation is different from pure prediction accuracy.
- What would be needed to move from this research pipeline to a production-grade trading system.

## Limitations to State Clearly

- The 300-stock universe may contain survivorship bias.
- Transaction costs and slippage are simplified bps assumptions.
- Monthly rebalancing does not model intraday execution or queue priority.
- The market stress module is not a full limit-order-book simulator.
- The sample has only 40 monthly factor observations, so inference is limited.
