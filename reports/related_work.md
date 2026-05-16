# Related Work and Adaptation Boundary

## LANCER

LANCER studies landscape surrogates and decision-focused learning for optimization problems, including portfolio optimization. The key lesson for this project is that prediction models should be evaluated by downstream decision quality, not only by statistical prediction loss.

Adaptation in this repository:

- independent mean-variance decision optimizer;
- factor-score and ML-score portfolio comparison;
- risk, turnover, and concentration penalties;
- no copied LANCER source code or dependency.

## MEHA

MEHA studies nonconvex bilevel optimization using a Moreau-envelope, single-loop, Hessian-free strategy. The key lesson for this project is that model selection and portfolio hyperparameter tuning can be framed as nested optimization.

Adaptation in this repository:

- chronological grid search as a lightweight bilevel proxy;
- validation-only parameter selection;
- held-out test reporting;
- no hypergradient or Moreau-envelope reimplementation.

## EvoMarket

EvoMarket is a high-fidelity, scalable market simulator with explicit trading mechanisms, microstructure observability, and intervention-oriented experiments. The key lesson for this project is that a strategy should be evaluated under market frictions and institutional constraints, not only under a frictionless backtest.

Adaptation in this repository:

- transaction-cost and high-cost regimes;
- liquidity-stress filtering;
- price-limit-aware tradability filter;
- T+1-style delayed rebalance approximation;
- no claim of full LOB simulation.

## Why This Boundary Matters

The project is intended for a data science/statistics graduate application. The implementation emphasizes reproducible empirical design, model comparison, and honest limitations rather than large-scale replication of specialized research systems.
