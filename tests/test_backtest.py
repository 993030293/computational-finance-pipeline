from __future__ import annotations

import pandas as pd
import pytest

from cfpipeline.backtest import (
    apply_transaction_costs,
    build_portfolio_weights,
    build_portfolios,
    calculate_turnover,
    performance_metrics,
)


def test_portfolio_returns_and_metrics() -> None:
    idx = pd.to_datetime(["2022-01-31", "2022-02-28", "2022-03-31"])
    score = pd.DataFrame(
        {
            "000001": [3.0, 1.0, 2.0],
            "000002": [2.0, 2.0, 1.0],
            "000003": [1.0, 3.0, 3.0],
        },
        index=idx,
    )
    ret = pd.DataFrame(
        {
            "000001": [0.10, 0.01, 0.02],
            "000002": [0.00, 0.02, 0.01],
            "000003": [-0.05, 0.03, 0.04],
        },
        index=idx,
    )
    portfolios = build_portfolios(score, ret, top_quantile=1 / 3)
    assert list(portfolios.columns) == ["long_only", "short_only", "long_short", "benchmark_ew"]
    assert portfolios.loc[idx[0], "long_short"] == pytest.approx(0.15)

    metrics = performance_metrics(portfolios["long_short"], benchmark=portfolios["benchmark_ew"])
    assert metrics["ann_vol"] > 0
    assert "sharpe" in metrics
    assert "information_ratio" in metrics


def test_turnover_and_transaction_costs_reduce_returns() -> None:
    idx = pd.to_datetime(["2022-01-31", "2022-02-28"])
    score = pd.DataFrame({"a": [2.0, 1.0], "b": [1.0, 2.0]}, index=idx)
    ret = pd.DataFrame({"a": [0.02, 0.03], "b": [0.01, 0.04]}, index=idx)
    weights = build_portfolio_weights(score, top_quantile=0.5)
    gross = build_portfolios(score, ret, top_quantile=0.5)
    turnover = calculate_turnover(weights)
    net = apply_transaction_costs(gross, turnover, transaction_cost_bps=10, slippage_bps=5)
    assert (turnover >= 0).all().all()
    assert (net <= gross).all().all()
