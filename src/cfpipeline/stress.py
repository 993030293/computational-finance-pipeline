from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import atomic_write_csv, atomic_write_text
from .backtest import (
    add_factor_score,
    apply_sign_flip,
    build_portfolio_weights,
    build_portfolios,
    calculate_turnover,
    factor_ic_signs,
    load_data,
    performance_metrics,
    pivot_panel,
)
from .paths import PipelinePaths


def base_score_panel(factors: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    zcols = [col for col in ["z_VALUE", "z_MOM_12_1", "z_QUALITY", "z_SIZE"] if col in factors.columns]
    data = factors.copy()
    if bool(cfg.get("sign_flip", True)):
        signs, _ = factor_ic_signs(data, zcols, ret_col="fwd_1m_ret")
        data = apply_sign_flip(data, signs)
    data = add_factor_score(data, weights=cfg.get("factor_weights"))
    return pivot_panel(data, "score")


def price_limit_mask(prices: pd.DataFrame, month_index: pd.Index, columns: pd.Index, limit_pct: float) -> pd.DataFrame:
    if "pct_chg" not in prices.columns:
        return pd.DataFrame(False, index=month_index, columns=columns)
    data = prices[["symbol", "date", "pct_chg"]].copy()
    data["month_end"] = data["date"].dt.to_period("M").dt.to_timestamp("M")
    last = data.sort_values("date").groupby(["symbol", "month_end"]).tail(1)
    locked = last.assign(is_locked=last["pct_chg"].abs() >= float(limit_pct)).pivot(
        index="month_end", columns="symbol", values="is_locked"
    )
    aligned = locked.reindex(index=month_index, columns=columns)
    return aligned.where(aligned.notna(), False).astype(bool)


def summarize_regimes(regime_returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime in regime_returns.columns:
        metrics = performance_metrics(regime_returns[regime].dropna())
        metrics["regime"] = regime
        rows.append(metrics)
    return pd.DataFrame(rows)


def run_stress(cfg: dict[str, Any]) -> dict[str, Path]:
    paths = PipelinePaths.from_config(cfg)
    paths.ensure_output_dirs()
    stress_cfg = cfg.get("stress", {})
    backtest_cfg = cfg.get("backtest", {})
    out_dir = paths.output_dir / "stress"
    out_dir.mkdir(parents=True, exist_ok=True)

    prices, factors, prices_path, factors_path = load_data(paths)
    score_panel = base_score_panel(factors, backtest_cfg)
    ret_panel = pivot_panel(factors, "fwd_1m_ret").reindex_like(score_panel)
    top_quantile = float(backtest_cfg.get("top_quantile", 0.2))
    weights = build_portfolio_weights(score_panel, top_quantile=top_quantile)
    gross = build_portfolios(score_panel, ret_panel, top_quantile=top_quantile)["long_short"]
    turnover = calculate_turnover(weights)["long_short"].reindex(gross.index).fillna(0.0)

    normal_cost = float(backtest_cfg.get("transaction_cost_bps", 10.0)) + float(backtest_cfg.get("slippage_bps", 5.0))
    high_cost = float(stress_cfg.get("high_cost_bps", 75.0))
    regime_returns = pd.DataFrame(index=gross.index)
    regime_returns["gross_return"] = gross
    regime_returns["net_return"] = gross - turnover * normal_cost / 10000.0
    regime_returns["high_cost_return"] = gross - turnover * high_cost / 10000.0

    liquidity_cut = float(stress_cfg.get("liquidity_drop_quantile", 0.2))
    size_panel = pivot_panel(factors, "SIZE").reindex_like(score_panel)
    liquid_score = score_panel.mask(size_panel.rank(axis=1, pct=True).le(liquidity_cut))
    regime_returns["liquidity_stress_return"] = build_portfolios(liquid_score, ret_panel, top_quantile=top_quantile)[
        "long_short"
    ]

    locked = price_limit_mask(
        prices, score_panel.index, score_panel.columns, float(stress_cfg.get("price_limit_pct", 9.5))
    )
    price_limit_score = score_panel.mask(locked)
    regime_returns["price_limit_filtered_return"] = build_portfolios(
        price_limit_score, ret_panel, top_quantile=top_quantile
    )["long_short"]

    # Approximate T+1 inventory constraint by delaying the rebalance signal one month.
    delayed_score = score_panel.shift(1)
    regime_returns["t_plus_one_delay_return"] = build_portfolios(delayed_score, ret_panel, top_quantile=top_quantile)[
        "long_short"
    ]

    metrics = summarize_regimes(regime_returns)
    returns_path = out_dir / "market_stress_returns.csv"
    metrics_path = out_dir / "market_stress_metrics.csv"
    report_path = out_dir / "market_stress_report.md"
    atomic_write_csv(regime_returns, returns_path, encoding="utf-8-sig")
    atomic_write_csv(metrics, metrics_path, index=False, encoding="utf-8-sig")
    lines = [
        "# EvoMarket-Inspired Market Mechanism Stress Report",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Prices: `{prices_path}`",
        f"- Factors: `{factors_path}`",
        "",
        "This is not a full limit-order-book simulator. It is a mechanism-aware sensitivity layer inspired by EvoMarket's emphasis on institutional rules, frictions, and counterfactual interventions.",
        "",
        "## Regimes",
        "- `gross_return`: baseline factor long-short return.",
        "- `net_return`: baseline after configured transaction cost and slippage.",
        "- `high_cost_return`: stress case with elevated cost assumptions.",
        "- `liquidity_stress_return`: excludes bottom-liquidity symbols by SIZE rank.",
        "- `price_limit_filtered_return`: excludes symbols near daily price-limit conditions using only information available at the decision month.",
        "- `t_plus_one_delay_return`: approximates T+1 restrictions by delaying rebalance signals one month.",
        "",
        "## Metrics",
        metrics.to_markdown(index=False, floatfmt=".6f"),
    ]
    atomic_write_text(report_path, "\n".join(lines), encoding="utf-8")
    return {
        "returns": returns_path,
        "metrics": metrics_path,
        "report": report_path,
    }
