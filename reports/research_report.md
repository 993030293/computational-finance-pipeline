# A Reproducible Statistical Learning Pipeline for A-Share Factor Research

## Abstract

This project studies whether transparent cross-sectional equity signals can rank next-month returns in a 300-stock A-share universe, and whether those predictions can be connected to portfolio decisions under realistic validation discipline and market frictions. The work is framed as a data science and statistics project, not as a live trading claim. It contributes a reproducible Python pipeline that combines data cleaning, factor construction, statistical validation, supervised learning, decision-aware portfolio optimization, chronological hyperparameter tuning, and market-mechanism stress testing.

The empirical baseline reproduces the original factor backtest with a gross long-short Sharpe ratio of 2.288 and maximum drawdown of -10.96% over 40 monthly periods. The enhanced research layer adds transaction costs, walk-forward machine learning, LANCER-inspired decision optimization, MEHA-inspired validation-only tuning, and EvoMarket-inspired stress tests for cost, liquidity, price-limit, and T+1 approximations. The results show that simple factor and ML signals can produce meaningful rank information in this sample, but performance is sensitive to universe construction, friction assumptions, and validation design. The project is therefore best interpreted as evidence of statistical modeling, experimental design, and reproducible engineering ability.

**Keywords:** factor research, statistical learning, walk-forward validation, portfolio optimization, market stress testing, reproducible data science.

## 1. Introduction and Research Question

Equity factor research is a useful setting for a data science portfolio project because it combines noisy observational data, temporal dependence, cross-sectional ranking, model validation, and downstream decisions. A naive project might stop after reporting a high backtest Sharpe ratio. This project instead asks a more careful question:

> Can a reproducible pipeline connect factor signals and machine learning predictions to portfolio decisions while preserving chronological validation and explicitly testing market-friction sensitivity?

The research design is intentionally broader than a single trading strategy. The goal is to demonstrate the full empirical workflow expected in graduate-level data science and statistics work:

- build a clean and documented dataset from messy market files;
- define factors and prediction targets without future leakage;
- evaluate signals using rank correlation, confidence intervals, cross-sectional regressions, and group returns;
- compare supervised learning models under time-series validation;
- translate predictions into portfolio weights with explicit risk and turnover penalties;
- tune hyperparameters using validation performance only;
- test whether conclusions survive plausible frictions and institutional constraints.

The main contribution is not an original alpha claim. It is an integrated empirical system that makes the link from data to statistical evidence, from evidence to decisions, and from decisions to limitations visible and reproducible.

## 2. Data and Reproducibility

The migrated full dataset contains 431,246 daily OHLCV observations for 300 A-share symbols from 2020-01-02 to 2025-12-16. The factor panel is built at monthly frequency and contains 11,937 cross-sectional observations from 2021-03-31 to 2025-10-31. Large local data and generated outputs are intentionally ignored by git, while a small sample dataset is included for clone-and-run tests.

The pipeline uses a configuration-driven structure:

```text
fetch -> clean -> factors -> backtest -> ml -> decision -> tune -> stress
```

The cleaning stage standardizes column names, parses dates, coerces numeric fields, de-duplicates symbol-date rows, fixes `high < low` observations by swapping inconsistent high/low values, computes missing percentage changes from close prices when needed, and generates technical indicators including moving averages, EMA, MACD, RSI, and rolling volatility. Daily returns are winsorized for robustness; log returns are computed only when current and prior prices are positive.

Reproducibility is built into the project rather than documented as an afterthought. The repository provides a Python package, CLI commands, YAML configs, unit tests, GitHub Actions, a dependency lock, sample data, and generated reports. A reviewer without the private full dataset can run:

```powershell
python scripts/create_sample_data.py
cfp run-all --skip-fetch --config configs/sample.yaml
```

The full local experiment can be reproduced with:

```powershell
cfp run-all --skip-fetch --config configs/default.yaml
```

## 3. Factor Construction and Statistical Validation

The factor panel contains seven standardized monthly signals:

- `VALUE`: valuation-style signal derived from price-level information;
- `MOM_12_1`: intermediate-term momentum excluding the most recent month;
- `QUALITY`: stability and return-quality proxy;
- `SIZE`: liquidity/size proxy;
- `REVERSAL_1M`: short-term reversal signal;
- `VOL_1M`: realized volatility signal;
- `ILLIQUIDITY`: illiquidity proxy.

Each factor is winsorized and cross-sectionally standardized by month. The prediction target is next-month return, so factor values at month `t` are evaluated against forward returns at `t+1`. Predictive validity is assessed with monthly Spearman rank information coefficient (Rank IC), ICIR, bootstrap confidence intervals, IC decay, factor correlations, quantile group returns, and Fama-MacBeth-style cross-sectional regressions.

### 3.1 Rank IC Summary

| Factor | Mean IC | Std. IC | Months | ICIR |
|---|---:|---:|---:|---:|
| z_ILLIQUIDITY | -0.1242 | 0.2070 | 40 | -0.6003 |
| z_MOM_12_1 | -0.0606 | 0.1343 | 40 | -0.4515 |
| z_QUALITY | -0.0816 | 0.1432 | 40 | -0.5701 |
| z_REVERSAL_1M | 0.0686 | 0.1458 | 40 | 0.4709 |
| z_SIZE | -0.1654 | 0.1811 | 40 | -0.9131 |
| z_VALUE | 0.1145 | 0.1544 | 40 | 0.7420 |
| z_VOL_1M | 0.1232 | 0.1427 | 40 | 0.8635 |

The strongest average rank relationships in this sample are `z_SIZE`, `z_VOL_1M`, `z_VALUE`, and `z_ILLIQUIDITY`, with negative average ICs for several factors. The project records factor-direction handling explicitly instead of silently assuming that every raw factor should be long-positive.

### 3.2 Statistical Significance

| Factor | Mean IC | t-stat | Bootstrap CI Low | Bootstrap CI High |
|---|---:|---:|---:|---:|
| z_ILLIQUIDITY | -0.1242 | -3.7964 | -0.1835 | -0.0617 |
| z_MOM_12_1 | -0.0606 | -2.8554 | -0.1027 | -0.0180 |
| z_QUALITY | -0.0816 | -3.6054 | -0.1244 | -0.0352 |
| z_REVERSAL_1M | 0.0686 | 2.9780 | 0.0250 | 0.1110 |
| z_SIZE | -0.1654 | -5.7747 | -0.2176 | -0.1092 |
| z_VALUE | 0.1145 | 4.6930 | 0.0660 | 0.1598 |
| z_VOL_1M | 0.1232 | 5.4612 | 0.0796 | 0.1678 |

The bootstrap intervals do not cross zero for these individual IC summaries. This does not imply that the strategy is production-ready. It means that, under this dataset and monthly sampling design, the cross-sectional rank signal is statistically visible enough to motivate model comparison and portfolio simulation.

### 3.3 Fama-MacBeth-Style Cross-Sectional Checks

| Term | Mean Coef. | t-stat | Months |
|---|---:|---:|---:|
| z_VALUE | 0.0064 | 1.1471 | 40 |
| z_MOM_12_1 | 0.0056 | 1.7292 | 40 |
| z_QUALITY | 0.0025 | 0.8573 | 40 |
| z_SIZE | 0.0025 | 0.2531 | 40 |
| z_REVERSAL_1M | 0.0029 | 1.1002 | 40 |
| z_VOL_1M | 0.0013 | 0.3149 | 40 |
| z_ILLIQUIDITY | -0.0145 | -1.6115 | 40 |

The regression-style checks are weaker than the Rank IC results, which is informative. Rank-based signals may be useful for ordering securities, while linear cross-sectional coefficient estimates remain noisy over only 40 monthly observations. This distinction is important because it keeps the interpretation statistical rather than promotional.

## 4. Baseline Portfolio Backtest

The baseline strategy sorts stocks by the composite factor score, forms long and short baskets, compares the result with an equal-weight benchmark, and reports both gross and net-cost performance. The table below is the full local-data baseline from `configs/default.yaml`.

| Strategy | Gross Ann. Return | Gross Sharpe | Gross Max DD | Net Ann. Return | Net Sharpe | Net Max DD |
|---|---:|---:|---:|---:|---:|---:|
| long_only | 25.66% | 1.2222 | -15.53% | 24.83% | 1.1826 | -15.67% |
| short_only | -14.50% | -0.6828 | -55.97% | -15.02% | -0.7075 | -56.67% |
| long_short | 43.63% | 2.2882 | -10.96% | 41.85% | 2.1939 | -11.35% |
| benchmark_ew | 3.06% | 0.1696 | -22.49% | 3.03% | 0.1679 | -22.49% |

The long-short baseline has strong historical performance in this sample, but the result should be treated as a research benchmark. The universe is a migrated 300-stock panel rather than a complete historical constituent database, and execution is simplified to monthly rebalancing with bps-level transaction cost assumptions.

Robustness across quantile choices shows that concentration changes both return and drawdown:

| Top Quantile | Periods | Final NAV | Mean Monthly Return | Monthly Volatility | Max Drawdown |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 40 | 4.2901 | 4.02% | 8.00% | -20.14% |
| 0.20 | 40 | 3.3431 | 3.21% | 5.50% | -10.96% |
| 0.30 | 40 | 2.7170 | 2.62% | 4.36% | -7.40% |

The 10% portfolio has the highest final NAV but also the largest drawdown. The default 20% portfolio is a more balanced research baseline, not an optimized live allocation.

## 5. Supervised Learning Experiments

The machine learning layer reframes the same factor panel as a supervised prediction problem. The target is future one-month return or top-bucket classification, and models are evaluated with expanding-window walk-forward validation. Random time-series shuffling is not allowed.

The implemented model set includes linear regression, ridge, lasso, logistic classification, random forest, and gradient boosting. Metrics include RMSE, MAE, Rank IC, AUC, precision at top quantile, feature importance, and portfolio performance from ML scores.

| Model | Observations | RMSE | MAE | Rank IC | AUC | Precision at Top |
|---|---:|---:|---:|---:|---:|---:|
| gradient_boosting | 6,587 | 0.1223 | 0.0760 | 0.0871 | 0.5408 | 0.5027 |
| lasso | 6,587 | 0.1193 | 0.0738 | 0.1123 | 0.5550 | 0.5163 |
| linear | 6,587 | 0.1195 | 0.0740 | 0.1128 | 0.5546 | 0.5133 |
| logistic | 6,587 | n/a | n/a | 0.1216 | 0.5588 | 0.5194 |
| random_forest | 6,587 | 0.1226 | 0.0780 | 0.0439 | 0.5170 | 0.4814 |
| ridge | 6,587 | 0.1195 | 0.0740 | 0.1128 | 0.5546 | 0.5125 |

The main result is not that a complex model dominates. Linear, ridge, lasso, and logistic models are competitive with tree ensembles, and random forest underperforms on rank metrics in this experiment. That outcome is plausible in a small, noisy, non-stationary financial panel: regularized linear models can be more stable than flexible learners when the validation horizon is short.

Portfolio evaluation from ML scores reinforces the same point. Logistic and lasso scores produce stronger long-short results than the tree models in this run:

| Model | Long-Short Ann. Return | Long-Short Sharpe | Long-Short Max DD |
|---|---:|---:|---:|
| gradient_boosting | 19.64% | 1.2666 | -10.32% |
| lasso | 35.42% | 1.8460 | -15.77% |
| linear | 32.48% | 1.7216 | -16.15% |
| logistic | 35.43% | 1.9704 | -14.97% |
| random_forest | 17.21% | 1.8729 | -6.11% |

These ML results are best interpreted as statistical learning diagnostics. The AUC values are only modestly above 0.5, but small rank advantages can still matter in cross-sectional portfolio construction.

## 6. Decision-Focused Portfolio Optimization

The decision layer is inspired by LANCER's decision-focused learning perspective: prediction models should be evaluated by the quality of downstream decisions, not only by statistical prediction loss. This project does not copy or depend on LANCER source code. It implements an independent mean-variance optimizer that maps factor or ML scores into long-only portfolio weights.

The optimizer maximizes a score-based objective with penalties for variance, turnover, and concentration:

```text
predicted_return - risk_aversion * variance - turnover_penalty * turnover - concentration_penalty * concentration
```

Weights are non-negative and sum to one. The optimizer uses `scipy.optimize.minimize`, avoiding heavier dependencies such as deep learning frameworks or convex optimization packages. The decision layer compares `factor_score` and `ml_score` under the same portfolio construction interface.

| Source | Return Type | Ann. Return | Ann. Vol | Sharpe | Max Drawdown |
|---|---|---:|---:|---:|---:|
| factor_score | gross | 32.18% | 31.32% | 1.0276 | -19.94% |
| factor_score | net | 31.10% | 31.35% | 0.9919 | -20.13% |
| ml_score | gross | 66.24% | 23.17% | 2.8584 | -8.28% |
| ml_score | net | 65.79% | 23.18% | 2.8378 | -8.28% |

The ML-score optimizer performs strongly in this sample. The correct interpretation is cautious: the decision optimizer shows that the ML ranking can be useful after accounting for risk and turnover penalties, but the result remains dependent on the migrated universe, monthly horizon, and simplified execution assumptions.

## 7. Chronological Hyperparameter Tuning

The tuning module is inspired by MEHA's bilevel optimization framing. The project does not implement Moreau-envelope hypergradients. Instead, it uses a lightweight chronological grid search as a practical bilevel proxy:

- the inner level builds factor scores and portfolios under candidate parameters;
- the outer level selects parameters using validation performance only;
- the held-out test period is used only after selection.

The search space covers factor weight sets, risk aversion, turnover penalty, and top quantile. This is designed to demonstrate validation discipline rather than to maximize the headline result.

| Split | Periods | Ann. Return | Ann. Vol | Sharpe | Max Drawdown | Mean Turnover |
|---|---:|---:|---:|---:|---:|---:|
| train | 16 | -5.57% | 29.54% | -0.1886 | -29.04% | 0.5030 |
| validation | 9 | 81.12% | 30.64% | 2.6472 | -11.25% | 0.5191 |
| test | 15 | 77.63% | 33.90% | 2.2900 | -17.50% | 0.4841 |

Selected parameters:

| Candidate | Risk Aversion | Turnover Penalty | Top Quantile | Factor Weights |
|---:|---:|---:|---:|---|
| 42 | 5.0 | 0.15 | 0.10 | VALUE=1, MOM_12_1=1, QUALITY=1, SIZE=1 |

The train split is weak, validation is strong, and test remains strong in this run. That pattern is encouraging but not conclusive. The sample has only 40 monthly periods, so a few market regimes can dominate results. The important methodological point is that the test result was not used to choose candidate 42.

## 8. Market Mechanism Stress Tests

Traditional daily or monthly backtests cannot generate true counterfactual limit-order-book trajectories. EvoMarket motivates a higher standard: market evaluation should consider mechanisms, frictions, institutional rules, and intervention-style comparisons. This project does not implement a full market simulator. It adds market-mechanism sensitivity tests that are appropriate for a one-month data science project:

- higher transaction cost regime;
- liquidity-stress filtering;
- price-limit-aware tradability filter;
- T+1-style delayed rebalance approximation.

| Regime | Ann. Return | Ann. Vol | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|
| gross_return | 43.63% | 19.07% | 2.2882 | -10.96% |
| net_return | 41.85% | 19.08% | 2.1939 | -11.35% |
| high_cost_return | 34.94% | 19.12% | 1.8277 | -12.90% |
| liquidity_stress_return | 32.51% | 17.37% | 1.8714 | -12.91% |
| price_limit_filtered_return | 44.51% | 19.04% | 2.3380 | -12.08% |
| t_plus_one_delay_return | 38.13% | 20.12% | 1.8951 | -15.24% |

The stress tests lower the strategy's apparent attractiveness under higher costs, liquidity stress, and delayed execution. The price-limit filter slightly improves return in this sample, which should not be overread; filtering can remove both bad and good trades depending on regime. The key value is diagnostic: results are no longer reported as a single frictionless curve.

## 9. Results and Discussion

The project produces three broad findings.

First, the factor layer contains statistically visible rank information. Multiple IC confidence intervals are away from zero, and the baseline long-short strategy outperforms the equal-weight benchmark in the migrated sample. At the same time, Fama-MacBeth summaries are weaker, reminding us that rank ordering and linear return explanation are different claims.

Second, supervised learning adds value mainly through stable ranking rather than high raw prediction accuracy. AUC values around 0.55 would look modest in many classification settings, but cross-sectional portfolio construction can benefit from small but persistent rank advantages. Regularized linear models and logistic classification are competitive with more flexible models.

Third, decision-aware evaluation changes the project from "which model predicts better" to "which signal leads to better constrained decisions." The decision optimizer and tuning layer make risk, turnover, concentration, and validation discipline explicit. The market-stress layer then tests whether the result is robust to more conservative assumptions.

For a graduate application, the most important result is the integrated workflow. The repository demonstrates the ability to build a complete empirical system, make statistical claims carefully, and explain limitations honestly.

## 10. Limitations

Several limitations prevent this from being interpreted as live trading evidence:

- The universe is a migrated 300-stock panel and may contain survivorship bias.
- Historical index membership, delistings, corporate actions, and suspension details are not fully modeled.
- Transaction costs and slippage are simplified bps assumptions.
- Monthly rebalancing ignores intraday execution paths and queue priority.
- The market-stress module is not a limit-order-book simulator.
- The test horizon contains only 40 monthly factor observations, limiting inference power.
- ML models are trained on a small, non-stationary panel; model stability should be treated as provisional.
- Hyperparameter search is deliberately lightweight and should not be mistaken for a full nonconvex bilevel optimization implementation.

These limitations are part of the research contribution. They make the project more credible by separating reproducible methodology from investment marketing.

## 11. Reproducibility Checklist

| Requirement | Status |
|---|---|
| Python package with CLI | Implemented |
| Config-driven runs | Implemented |
| Sample data for public smoke tests | Implemented |
| Full migrated local dataset support | Implemented |
| Unit and integration tests | Implemented |
| GitHub Actions smoke test | Implemented |
| Factor validation outputs | Implemented |
| ML walk-forward outputs | Implemented |
| Decision optimizer outputs | Implemented |
| Validation-only tuning outputs | Implemented |
| Market stress outputs | Implemented |
| Data card and model card | Implemented |

The main commands are:

```powershell
pytest
cfp run-all --skip-fetch --config configs/sample.yaml
cfp run-all --skip-fetch --config configs/default.yaml
```

## References

- Zharmagambetov, A., Amos, B., Ferber, A., Huang, T., Dilkina, B., and Tian, Y. LANCER: Landscape Surrogate: Learning Decision Losses for Mathematical Optimization Under Partial Information. arXiv:2307.08964.
- Liu, R., Liu, Z., Yao, W., Zeng, S., and Zhang, J. Moreau Envelope for Nonconvex Bi-Level Optimization: A Single-Loop and Hessian-Free Solution Strategy. ICML 2024. arXiv:2405.09927.
- Zhong, M., Yang, Z., Liu, Y., Tang, K., and Yang, P. EvoMarket: A High-Fidelity and Scalable Financial Market Simulator. arXiv:2604.18046, 2026.
- Fama, E. F., and MacBeth, J. D. Risk, Return, and Equilibrium: Empirical Tests. Journal of Political Economy, 1973.
- Grinold, R. C., and Kahn, R. N. Active Portfolio Management. McGraw-Hill, 1999.
- Pedregosa, F. et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 2011.
