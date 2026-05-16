# Executive Summary

I built a reproducible Python pipeline for A-share factor research, statistical testing, supervised learning, and portfolio simulation. The project starts with raw OHLCV data, cleans and validates it, constructs monthly cross-sectional factors, evaluates factor significance with Rank IC, bootstrap confidence intervals, Fama-MacBeth summaries, and robustness checks, then compares factor scores against machine-learning predictions under walk-forward validation.

The project is designed as a data science and statistics admissions portfolio piece. It emphasizes experimental discipline over trading claims: no random time-series shuffling, explicit transaction costs, turnover reporting, train/validation/test splits, model comparison tables, feature importance, tests, CI, and clone-and-run sample data.
