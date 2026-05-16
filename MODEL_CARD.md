# Model Card

## Models

The supervised learning layer compares linear regression, ridge, lasso, logistic classification, random forest, and gradient boosting using monthly factor features.

## Target

The default target is next-month stock return, `fwd_1m_ret`. Logistic classification predicts whether the next-month return is positive.

## Validation

The ML pipeline uses expanding-window walk-forward splits. Training months always precede test months, and the code does not randomly shuffle time-series labels.

## Metrics

- Regression: RMSE, MAE, rank IC.
- Classification/ranking: AUC, precision at top quantile.
- Portfolio translation: long-only, short-basket, long-short, and equal-weight returns from model scores.

## Intended Use

The models are used to demonstrate statistical learning workflow, validation discipline, and feature interpretation for graduate applications.

## Limitations

- Monthly cross-sectional returns are noisy.
- Hyperparameters are intentionally conservative and not extensively tuned.
- Predictive relationships may be unstable across market regimes.
- Performance tables are research diagnostics, not live trading evidence.
