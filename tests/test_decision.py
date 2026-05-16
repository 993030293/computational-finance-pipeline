from __future__ import annotations

import numpy as np
import pandas as pd

from cfpipeline.decision import optimize_long_only_weights, optimize_score_panel


def test_decision_weights_are_long_only_and_sum_to_one() -> None:
    expected = pd.Series({"a": 0.05, "b": 0.03, "c": 0.01})
    cov = np.eye(3) * 0.02
    weights = optimize_long_only_weights(expected, cov, max_weight=0.8)
    assert (weights >= -1e-10).all()
    assert abs(weights.sum() - 1.0) < 1e-8


def test_turnover_penalty_reduces_weight_change() -> None:
    expected = pd.Series({"a": 0.10, "b": 0.01})
    cov = np.eye(2) * 0.01
    previous = pd.Series({"a": 0.0, "b": 1.0})
    low_penalty = optimize_long_only_weights(expected, cov, previous, turnover_penalty=0.0, max_weight=1.0)
    high_penalty = optimize_long_only_weights(expected, cov, previous, turnover_penalty=10.0, max_weight=1.0)
    low_turnover = (low_penalty - previous).abs().sum() / 2
    high_turnover = (high_penalty - previous).abs().sum() / 2
    assert high_turnover <= low_turnover + 1e-8


def test_risk_aversion_reduces_variance() -> None:
    expected = pd.Series({"high_risk": 0.10, "low_risk": 0.03})
    cov = np.array([[0.50, 0.0], [0.0, 0.01]])
    low_risk_aversion = optimize_long_only_weights(expected, cov, risk_aversion=0.01, max_weight=1.0)
    high_risk_aversion = optimize_long_only_weights(expected, cov, risk_aversion=10.0, max_weight=1.0)
    low_var = float(low_risk_aversion.to_numpy() @ cov @ low_risk_aversion.to_numpy())
    high_var = float(high_risk_aversion.to_numpy() @ cov @ high_risk_aversion.to_numpy())
    assert high_var <= low_var + 1e-8


def test_optimize_score_panel_outputs_returns() -> None:
    idx = pd.to_datetime(["2022-01-31", "2022-02-28", "2022-03-31"])
    score = pd.DataFrame({"a": [0.1, 0.2, 0.3], "b": [0.2, 0.1, 0.0]}, index=idx)
    ret = pd.DataFrame({"a": [0.01, 0.02, 0.03], "b": [0.02, 0.01, 0.00]}, index=idx)
    weights, returns, turnover = optimize_score_panel(score, ret, lookback_months=2, max_weight=1.0)
    assert not weights.empty
    assert len(returns) == len(turnover)
