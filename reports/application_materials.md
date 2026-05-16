# Application Materials

## Resume Bullet

Built a reproducible Python data science pipeline for A-share equity research, including data cleaning, factor construction, bootstrap and Fama-MacBeth statistical validation, transaction-cost-aware backtesting, walk-forward ML, decision-aware portfolio optimization, validation-only hyperparameter tuning, market mechanism stress tests, pytest coverage, CLI automation, and GitHub Actions CI.

## SOP Paragraph

One project that shaped my interest in data science was a computational finance pipeline I rebuilt from course scripts into a reproducible research system. I used Python, pandas, scikit-learn, and statistical testing to study whether cross-sectional equity factors could rank next-month returns in A-share data. Beyond implementing a backtest, I focused on research design: chronological train/validation/test splits, walk-forward evaluation, bootstrap confidence intervals, Fama-MacBeth summaries, transaction-cost reporting, robustness checks, and clear documentation of limitations. This project helped me connect statistical modeling, software engineering, and honest empirical analysis, which is the type of work I hope to deepen in graduate study.

## Professor-Facing Summary

This project is a reproducible empirical finance and data science workflow. It transforms raw daily market data into a monthly factor panel, evaluates factor validity through Rank IC, IC decay, quantile portfolios, bootstrap confidence intervals, and Fama-MacBeth regressions, then compares transparent factor scores with supervised learning models under expanding-window validation. The repository includes a CLI, tests, CI, sample data, data/model cards, and a research report. I present the project as evidence of statistical reasoning, reproducible computation, and careful treatment of empirical limitations rather than as a trading product.

## Final Review Checklist

- README explains problem, data, method, results, limitations, and reproduction.
- Sample data runs without private files.
- Tests and GitHub Actions pass.
- Research report is readable without opening code.
- Claims avoid live-trading overstatement.
- Large local data remains excluded from git.
