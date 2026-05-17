from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import atomic_write_csv, atomic_write_json, atomic_write_text
from .cleaning import read_csv
from .paths import PipelinePaths, first_existing
from .research import robustness_by_quantile, summarize_by_split

DEFAULT_ZCOLS = ["z_VALUE", "z_MOM_12_1", "z_QUALITY", "z_SIZE"]


def price_candidates(paths: PipelinePaths) -> list[Path]:
    return [
        paths.output_processed_dir / "daily_price_panel.csv",
        paths.output_processed_dir / "daily_price_50.csv",
        paths.input_processed_dir / "daily_price_panel.csv",
        paths.input_processed_dir / "daily_price_50.csv",
        paths.input_processed_dir / "daily_price.csv",
    ]


def factor_candidates(paths: PipelinePaths) -> list[Path]:
    return [
        paths.output_project4_dir / "factors.csv",
        paths.input_project4_dir / "factors.csv",
    ]


def load_data(paths: PipelinePaths) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    prices_path = first_existing(price_candidates(paths))
    factors_path = first_existing(factor_candidates(paths))
    prices = read_csv(prices_path, parse_dates=["date"])
    factors = read_csv(factors_path, parse_dates=["month_end"])

    required_prices = {"symbol", "date", "close"}
    required_factors = {"symbol", "month_end", "fwd_1m_ret", *DEFAULT_ZCOLS}
    missing_prices = required_prices.difference(prices.columns)
    missing_factors = required_factors.difference(factors.columns)
    if missing_prices:
        raise KeyError(f"Prices file missing required columns: {sorted(missing_prices)}")
    if missing_factors:
        raise KeyError(f"Factors file missing required columns: {sorted(missing_factors)}")

    prices["symbol"] = prices["symbol"].astype("string")
    factors["symbol"] = factors["symbol"].astype("string")
    return (
        prices.sort_values(["symbol", "date"]).reset_index(drop=True),
        factors.sort_values(["symbol", "month_end"]).reset_index(drop=True),
        prices_path,
        factors_path,
    )


def add_factor_score(factors: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    zcols = [col for col in DEFAULT_ZCOLS if col in factors.columns]
    if weights is None:
        weights = {col: 1.0 for col in zcols}
    weights = {key: float(value) for key, value in weights.items() if key in zcols}
    if not weights:
        raise ValueError("No valid factor weights provided.")
    w = pd.Series(weights, index=zcols).fillna(0.0)
    if w.abs().sum() == 0:
        raise ValueError("All factor weights are zero.")
    w = w / w.abs().sum()
    out = factors.copy()
    out["score"] = out[zcols].dot(w.values)
    return out


def pivot_panel(factors: pd.DataFrame, col: str) -> pd.DataFrame:
    return factors.pivot(index="month_end", columns="symbol", values=col).sort_index().sort_index(axis=1)


def factor_ic_signs(
    factors: pd.DataFrame, zcols: list[str], ret_col: str = "fwd_1m_ret"
) -> tuple[dict[str, float], dict[str, float]]:
    signs: dict[str, float] = {}
    ic_mean: dict[str, float] = {}
    for zcol in zcols:
        factor_col = zcol
        ic_by_month = factors.groupby("month_end", group_keys=False)[[factor_col, ret_col]].apply(
            lambda frame, col=factor_col: frame[col].rank().corr(frame[ret_col].rank())
        )
        mean_ic = float(ic_by_month.mean(skipna=True))
        ic_mean[zcol] = mean_ic
        signs[zcol] = 1.0 if mean_ic >= 0 else -1.0
    return signs, ic_mean


def apply_sign_flip(factors: pd.DataFrame, signs: dict[str, float]) -> pd.DataFrame:
    out = factors.copy()
    for zcol, sign in signs.items():
        if zcol in out.columns:
            out[zcol] = out[zcol] * sign
    return out


def build_portfolio_weights(score_panel: pd.DataFrame, top_quantile: float = 0.2) -> dict[str, pd.DataFrame]:
    ranks = score_panel.rank(axis=1, ascending=False, method="first")
    n = ranks.notna().sum(axis=1)
    top_n = (n * top_quantile).round().astype(int).clip(lower=1)

    long_mask = ranks.le(top_n, axis=0)
    short_mask = ranks.gt(n - top_n, axis=0)
    ew_mask = ranks.notna()

    long_w = long_mask.div(long_mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    short_w = short_mask.div(short_mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    ew_w = ew_mask.div(ew_mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return {
        "long_only": long_w,
        "short_only": short_w,
        "long_short": long_w - short_w,
        "benchmark_ew": ew_w,
    }


def returns_from_weights(weights: dict[str, pd.DataFrame], ret_panel: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for name, weight in weights.items():
        aligned_ret = ret_panel.reindex_like(weight)
        if name == "short_only":
            # Compatibility with the original project: this is the return of
            # the short basket, not the P&L of a standalone short portfolio.
            rows[name] = (weight * aligned_ret).sum(axis=1)
        else:
            rows[name] = (weight * aligned_ret).sum(axis=1)
    return pd.DataFrame(rows)


def build_portfolios(score_panel: pd.DataFrame, ret_panel: pd.DataFrame, top_quantile: float = 0.2) -> pd.DataFrame:
    weights = build_portfolio_weights(score_panel, top_quantile=top_quantile)
    ret_panel = ret_panel.reindex_like(score_panel)
    long_ret = (weights["long_only"] * ret_panel).sum(axis=1)
    short_ret = (weights["short_only"] * ret_panel).sum(axis=1)
    benchmark_ret = (weights["benchmark_ew"] * ret_panel).sum(axis=1)
    return pd.DataFrame(
        {
            "long_only": long_ret,
            "short_only": short_ret,
            "long_short": long_ret - short_ret,
            "benchmark_ew": benchmark_ret,
        }
    )


def calculate_turnover(weights: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One-way turnover by strategy using portfolio weight changes."""
    out = {}
    for name, weight in weights.items():
        previous = weight.shift(1).fillna(0.0)
        out[name] = (weight - previous).abs().sum(axis=1) / 2.0
    return pd.DataFrame(out)


def apply_transaction_costs(
    returns: pd.DataFrame,
    turnover: pd.DataFrame,
    *,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> pd.DataFrame:
    total_bps = float(transaction_cost_bps) + float(slippage_bps)
    return returns - turnover.reindex_like(returns).fillna(0.0) * total_bps / 10000.0


def compute_drawdown(ret: pd.Series) -> pd.Series:
    nav = (1.0 + ret.fillna(0.0)).cumprod()
    return nav / nav.cummax() - 1.0


def rolling_metrics(ret: pd.Series, window: int = 12, freq: int = 12) -> pd.DataFrame:
    rolling_vol = ret.rolling(window).std(ddof=0) * np.sqrt(freq)
    rolling_sharpe = ret.rolling(window).mean() * np.sqrt(freq) / rolling_vol
    return pd.DataFrame({"rolling_vol": rolling_vol, "rolling_sharpe": rolling_sharpe})


def performance_metrics(
    ret: pd.Series,
    *,
    freq: int = 12,
    benchmark: pd.Series | None = None,
    rf: float = 0.0,
) -> dict[str, float]:
    ret = ret.dropna()
    if len(ret) == 0:
        return {}
    total_ret = (1.0 + ret).prod()
    ann_return = total_ret ** (freq / len(ret)) - 1.0
    ann_vol = ret.std(ddof=0) * np.sqrt(freq)
    sharpe = (ann_return - rf) / ann_vol if ann_vol > 0 else np.nan
    drawdown = compute_drawdown(ret)
    max_drawdown = drawdown.min()
    calmar = (ann_return - rf) / abs(max_drawdown) if max_drawdown < 0 else np.nan
    downside = ret[ret < 0]
    if len(downside) > 0:
        downside_vol = downside.std(ddof=0) * np.sqrt(freq)
        sortino = (ann_return - rf) / downside_vol if downside_vol > 0 else np.nan
    else:
        sortino = np.nan

    metrics = {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "sortino": sortino,
    }
    if benchmark is not None:
        bench = benchmark.loc[ret.index].dropna()
        common_idx = ret.index.intersection(bench.index)
        r = ret.loc[common_idx]
        b = bench.loc[common_idx]
        if len(b) > 1:
            cov = np.cov(r, b)[0, 1]
            var_b = np.var(b)
            beta = cov / var_b if var_b > 0 else np.nan
            alpha_ann = (r.mean() - beta * b.mean()) * freq
            diff = r - b
            diff_std = diff.std(ddof=0)
            information_ratio = diff.mean() * np.sqrt(freq) / diff_std if diff_std > 0 else np.nan
        else:
            beta = np.nan
            alpha_ann = np.nan
            information_ratio = np.nan
        metrics.update({"beta": beta, "alpha_ann": alpha_ann, "information_ratio": information_ratio})
    return {key: float(value) if pd.notna(value) else np.nan for key, value in metrics.items()}


def plot_cum_returns(port_ret: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    nav = (1.0 + port_ret).cumprod()
    plt.figure(figsize=(10, 6))
    for col in nav.columns:
        plt.plot(nav.index, nav[col], label=col)
    plt.title("Cumulative Returns")
    plt.xlabel("Date (month_end)")
    plt.ylabel("NAV")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_drawdown(ret: pd.Series, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    drawdown = compute_drawdown(ret)
    plt.figure(figsize=(10, 4))
    plt.plot(drawdown.index, drawdown)
    plt.title("Drawdown - Long Short")
    plt.xlabel("Date (month_end)")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_rolling_metrics(ret: pd.Series, out_path: Path, window: int = 12, freq: int = 12) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    metrics = rolling_metrics(ret, window=window, freq=freq)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(metrics.index, metrics["rolling_vol"])
    axes[0].set_ylabel("Rolling Volatility")
    axes[0].grid(True)
    axes[1].plot(metrics.index, metrics["rolling_sharpe"])
    axes[1].set_ylabel("Rolling Sharpe")
    axes[1].set_xlabel("Date (month_end)")
    axes[1].grid(True)
    fig.suptitle(f"Rolling {window}-month Volatility and Sharpe (Long-Short)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(out_path, dpi=160)
    plt.close()


def write_summary(
    out_dir: Path,
    port_ret: pd.DataFrame,
    metrics: pd.DataFrame,
    prices_path: Path,
    factors_path: Path,
    cfg: dict[str, Any],
    *,
    net_metrics: pd.DataFrame | None = None,
    turnover: pd.DataFrame | None = None,
) -> Path:
    nav = (1.0 + port_ret).cumprod()
    long_short = port_ret["long_short"]
    best = port_ret.loc[long_short.idxmax()]
    worst = port_ret.loc[long_short.idxmin()]
    summary = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prices_path": str(prices_path),
        "factors_path": str(factors_path),
        "config": cfg,
        "date_range": [str(port_ret.index.min().date()), str(port_ret.index.max().date())],
        "periods": int(len(port_ret)),
        "final_nav": {col: float(nav[col].iloc[-1]) for col in nav.columns},
        "long_short_best_month": {
            "month_end": str(best.name.date()),
            "return": float(best["long_short"]),
        },
        "long_short_worst_month": {
            "month_end": str(worst.name.date()),
            "return": float(worst["long_short"]),
        },
        "metrics": metrics.to_dict(orient="index"),
    }
    if net_metrics is not None:
        summary["net_metrics"] = net_metrics.to_dict(orient="index")
    if turnover is not None:
        summary["average_turnover"] = {col: float(turnover[col].mean()) for col in turnover.columns}
    json_path = out_dir / "backtest_summary.json"
    atomic_write_json(json_path, summary)

    md_path = out_dir / "BACKTEST_SUMMARY.md"
    lines = [
        "# Backtest Summary",
        "",
        f"- Generated: {summary['generated']}",
        f"- Prices: `{prices_path}`",
        f"- Factors: `{factors_path}`",
        f"- Periods: {summary['periods']}",
        f"- Date range: {summary['date_range'][0]} to {summary['date_range'][1]}",
        "",
        "## Final NAV",
        *[f"- {name}: {value:.6f}" for name, value in summary["final_nav"].items()],
        "",
        "## Long-Short Extremes",
        f"- Best month: {summary['long_short_best_month']['month_end']} ({summary['long_short_best_month']['return']:.6f})",
        f"- Worst month: {summary['long_short_worst_month']['month_end']} ({summary['long_short_worst_month']['return']:.6f})",
        "",
        "## Performance",
        metrics.to_markdown(floatfmt=".6f"),
    ]
    if net_metrics is not None:
        lines.extend(["", "## Net Performance After Costs", net_metrics.to_markdown(floatfmt=".6f")])
    if turnover is not None:
        lines.extend(
            [
                "",
                "## Average One-Way Turnover",
                *[f"- {col}: {turnover[col].mean():.6f}" for col in turnover.columns],
            ]
        )
    atomic_write_text(md_path, "\n".join(lines), encoding="utf-8")
    return md_path


def run_backtest(cfg: dict[str, Any]) -> dict[str, Path]:
    paths = PipelinePaths.from_config(cfg)
    paths.ensure_output_dirs()
    backtest_cfg = cfg.get("backtest", {})
    _, factors, prices_path, factors_path = load_data(paths)

    zcols = [col for col in DEFAULT_ZCOLS if col in factors.columns]
    if bool(backtest_cfg.get("sign_flip", True)):
        signs, ic_mean = factor_ic_signs(factors, zcols, ret_col="fwd_1m_ret")
        factors = apply_sign_flip(factors, signs)
    else:
        signs = {col: 1.0 for col in zcols}
        ic_mean = {col: np.nan for col in zcols}

    out_dir = paths.output_backtest_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(pd.Series(ic_mean, name="mean_ic"), out_dir / "ic_mean.csv")
    atomic_write_csv(pd.Series(signs, name="sign"), out_dir / "factor_signs.csv")

    scored = add_factor_score(factors, weights=backtest_cfg.get("factor_weights"))
    score_panel = pivot_panel(scored, "score")
    ret_panel = pivot_panel(scored, "fwd_1m_ret")
    top_quantile = float(backtest_cfg.get("top_quantile", 0.2))
    weights = build_portfolio_weights(score_panel, top_quantile=top_quantile)
    port_ret = build_portfolios(score_panel, ret_panel, top_quantile)
    atomic_write_csv(port_ret, out_dir / "portfolio_returns.csv", float_format="%.8f")
    turnover = calculate_turnover(weights)
    atomic_write_csv(turnover, out_dir / "turnover.csv", float_format="%.8f")
    net_ret = apply_transaction_costs(
        port_ret,
        turnover,
        transaction_cost_bps=float(backtest_cfg.get("transaction_cost_bps", 0.0)),
        slippage_bps=float(backtest_cfg.get("slippage_bps", 0.0)),
    )
    atomic_write_csv(net_ret, out_dir / "portfolio_returns_net.csv", float_format="%.8f")
    nav = (1.0 + port_ret).cumprod()
    atomic_write_csv(nav, out_dir / "portfolio_nav.csv", float_format="%.8f")
    net_nav = (1.0 + net_ret).cumprod()
    atomic_write_csv(net_nav, out_dir / "portfolio_nav_net.csv", float_format="%.8f")

    metrics_rows = []
    for col in port_ret.columns:
        row = performance_metrics(
            port_ret[col], freq=int(backtest_cfg.get("freq", 12)), benchmark=port_ret["benchmark_ew"]
        )
        row["strategy"] = col
        metrics_rows.append(row)
    metrics = pd.DataFrame(metrics_rows).set_index("strategy")
    atomic_write_csv(metrics, out_dir / "performance_metrics.csv", float_format="%.6f")
    net_metrics_rows = []
    for col in net_ret.columns:
        row = performance_metrics(
            net_ret[col], freq=int(backtest_cfg.get("freq", 12)), benchmark=net_ret["benchmark_ew"]
        )
        row["strategy"] = col
        net_metrics_rows.append(row)
    net_metrics = pd.DataFrame(net_metrics_rows).set_index("strategy")
    atomic_write_csv(net_metrics, out_dir / "performance_metrics_net.csv", float_format="%.6f")

    long_short = port_ret["long_short"]
    atomic_write_csv(compute_drawdown(long_short), out_dir / "drawdown_long_short.csv", header=["drawdown"])
    rolling_metrics(
        long_short,
        window=int(backtest_cfg.get("rolling_window", 12)),
        freq=int(backtest_cfg.get("freq", 12)),
    ).pipe(atomic_write_csv, out_dir / "rolling_long_short.csv", float_format="%.6f")
    split_metrics = summarize_by_split(port_ret, cfg.get("research", {}))
    atomic_write_csv(split_metrics, out_dir / "split_performance.csv", index=False, encoding="utf-8-sig")
    split_metrics_net = summarize_by_split(net_ret, cfg.get("research", {}))
    atomic_write_csv(split_metrics_net, out_dir / "split_performance_net.csv", index=False, encoding="utf-8-sig")
    robustness_quantiles = [float(x) for x in backtest_cfg.get("robustness_quantiles", [0.1, 0.2, 0.3])]
    robustness = robustness_by_quantile(score_panel, ret_panel, build_portfolios, robustness_quantiles)
    atomic_write_csv(robustness, out_dir / "robustness_quantiles.csv", index=False, encoding="utf-8-sig")

    strategy_name = str(backtest_cfg.get("strategy_name", "VALUE+SIZE")).replace(" ", "_")
    plot_cum_returns(port_ret, out_dir / f"cumret_curves__{strategy_name}.png")
    plot_drawdown(long_short, out_dir / f"drawdown_long_short__{strategy_name}.png")
    plot_rolling_metrics(
        long_short,
        out_dir / f"rolling_long_short__{strategy_name}.png",
        window=int(backtest_cfg.get("rolling_window", 12)),
        freq=int(backtest_cfg.get("freq", 12)),
    )
    summary_path = write_summary(
        out_dir,
        port_ret,
        metrics,
        prices_path,
        factors_path,
        backtest_cfg,
        net_metrics=net_metrics,
        turnover=turnover,
    )

    return {
        "portfolio_returns": out_dir / "portfolio_returns.csv",
        "portfolio_returns_net": out_dir / "portfolio_returns_net.csv",
        "portfolio_nav": out_dir / "portfolio_nav.csv",
        "portfolio_nav_net": out_dir / "portfolio_nav_net.csv",
        "performance_metrics": out_dir / "performance_metrics.csv",
        "performance_metrics_net": out_dir / "performance_metrics_net.csv",
        "turnover": out_dir / "turnover.csv",
        "split_performance": out_dir / "split_performance.csv",
        "robustness_quantiles": out_dir / "robustness_quantiles.csv",
        "drawdown": out_dir / "drawdown_long_short.csv",
        "rolling": out_dir / "rolling_long_short.csv",
        "summary": summary_path,
    }
