from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSplitConfig:
    train_end: str = "2022-12-31"
    valid_end: str = "2023-12-31"
    test_start: str = "2024-01-01"


def assign_time_split(dates: pd.Series, cfg: dict[str, Any] | None = None) -> pd.Series:
    """Assign train/validation/test labels without random shuffling."""
    cfg = cfg or {}
    split = TimeSplitConfig(
        train_end=str(cfg.get("train_end", "2022-12-31")),
        valid_end=str(cfg.get("valid_end", "2023-12-31")),
        test_start=str(cfg.get("test_start", "2024-01-01")),
    )
    values = pd.to_datetime(dates)
    labels = pd.Series("holdout", index=dates.index, dtype="string")
    labels.loc[values <= pd.Timestamp(split.train_end)] = "train"
    labels.loc[(values > pd.Timestamp(split.train_end)) & (values <= pd.Timestamp(split.valid_end))] = "validation"
    labels.loc[values >= pd.Timestamp(split.test_start)] = "test"
    return labels


def bootstrap_mean_ci(
    values: pd.Series,
    *,
    samples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    clean = values.dropna().astype(float).to_numpy()
    if len(clean) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = rng.choice(clean, size=(samples, len(clean)), replace=True).mean(axis=1)
    alpha = (1 - ci) / 2
    return float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))


def ic_significance_table(
    ic: pd.DataFrame,
    *,
    bootstrap_samples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for factor, group in ic.groupby("factor"):
        values = group["IC"].dropna().astype(float)
        n = len(values)
        mean = float(values.mean()) if n else np.nan
        std = float(values.std(ddof=1)) if n > 1 else np.nan
        t_stat = mean / (std / np.sqrt(n)) if n > 1 and std and not np.isnan(std) else np.nan
        ci_low, ci_high = bootstrap_mean_ci(values, samples=bootstrap_samples, ci=ci, seed=seed)
        rows.append(
            {
                "factor": factor,
                "mean_ic": mean,
                "std_ic": std,
                "count": n,
                "t_stat": t_stat,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
            }
        )
    return pd.DataFrame(rows).sort_values("factor").reset_index(drop=True)


def ic_decay(panel: pd.DataFrame, factor_cols: list[str], horizons: list[int]) -> pd.DataFrame:
    """Rank IC between factor at t and forward h-month returns."""
    data = panel.sort_values(["symbol", "month_end"]).copy()
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        ret_col = f"fwd_{horizon}m_ret"
        data[ret_col] = data.groupby("symbol")["close_me"].pct_change(periods=horizon).shift(-horizon)
        for month_end, group in data.groupby("month_end"):
            for factor in factor_cols:
                valid = group[factor].notna() & group[ret_col].notna()
                if valid.sum() < 3:
                    continue
                ic_value = group.loc[valid, factor].rank().corr(group.loc[valid, ret_col].rank())
                if pd.notna(ic_value):
                    rows.append(
                        {
                            "month_end": month_end,
                            "horizon_months": horizon,
                            "factor": factor,
                            "IC": float(ic_value),
                        }
                    )
    return pd.DataFrame(rows).sort_values(["factor", "horizon_months", "month_end"]).reset_index(drop=True)


def summarize_ic_decay(decay: pd.DataFrame) -> pd.DataFrame:
    if decay.empty:
        return pd.DataFrame(columns=["factor", "horizon_months", "mean", "std", "count"])
    return (
        decay.groupby(["factor", "horizon_months"])["IC"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["factor", "horizon_months"])
    )


def quantile_group_returns(
    panel: pd.DataFrame,
    factor_cols: list[str],
    *,
    quantiles: int = 5,
    ret_col: str = "fwd_1m_ret",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month_end, group in panel.groupby("month_end"):
        for factor in factor_cols:
            data = group[[factor, ret_col]].dropna()
            if len(data) < quantiles:
                continue
            try:
                buckets = pd.qcut(data[factor].rank(method="first"), quantiles, labels=False) + 1
            except ValueError:
                continue
            grouped = data.assign(quantile=buckets).groupby("quantile")[ret_col].mean()
            for quantile, value in grouped.items():
                rows.append(
                    {
                        "month_end": month_end,
                        "factor": factor,
                        "quantile": int(quantile),
                        "mean_return": float(value),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["month_end", "factor", "quantile", "mean_return"])
    return pd.DataFrame(rows).sort_values(["factor", "month_end", "quantile"]).reset_index(drop=True)


def summarize_group_returns(group_returns: pd.DataFrame) -> pd.DataFrame:
    if group_returns.empty:
        return pd.DataFrame(columns=["factor", "quantile", "mean_return", "count"])
    return (
        group_returns.groupby(["factor", "quantile"])["mean_return"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "mean_return"})
        .reset_index()
    )


def fama_macbeth_regression(panel: pd.DataFrame, factor_cols: list[str], ret_col: str = "fwd_1m_ret") -> pd.DataFrame:
    """Cross-sectional monthly OLS followed by time-series coefficient summary."""
    rows: list[dict[str, Any]] = []
    cols = [col for col in factor_cols if col in panel.columns]
    for month_end, group in panel.groupby("month_end"):
        data = group[[ret_col, *cols]].dropna()
        if len(data) <= len(cols) + 1:
            continue
        x = np.column_stack([np.ones(len(data)), data[cols].to_numpy(dtype=float)])
        y = data[ret_col].to_numpy(dtype=float)
        try:
            beta = np.linalg.lstsq(x, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        row = {"month_end": month_end, "intercept": float(beta[0])}
        row.update({col: float(value) for col, value in zip(cols, beta[1:])})
        rows.append(row)
    coef = pd.DataFrame(rows)
    if coef.empty:
        return pd.DataFrame(columns=["term", "mean_coef", "std_coef", "t_stat", "count"])
    out_rows = []
    for term in ["intercept", *cols]:
        values = coef[term].dropna()
        n = len(values)
        mean = float(values.mean()) if n else np.nan
        std = float(values.std(ddof=1)) if n > 1 else np.nan
        t_stat = mean / (std / np.sqrt(n)) if n > 1 and std and not np.isnan(std) else np.nan
        out_rows.append({"term": term, "mean_coef": mean, "std_coef": std, "t_stat": t_stat, "count": n})
    return pd.DataFrame(out_rows)


def summarize_by_split(returns: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = assign_time_split(pd.Series(returns.index, index=returns.index), cfg)
    for split_name, idx in labels.groupby(labels).groups.items():
        if split_name == "holdout":
            continue
        subset = returns.loc[list(idx)]
        for col in returns.columns:
            values = subset[col].dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "split": split_name,
                    "strategy": col,
                    "periods": int(len(values)),
                    "mean_return": float(values.mean()),
                    "volatility": float(values.std(ddof=0)),
                    "cumulative_return": float((1 + values).prod() - 1),
                }
            )
    return pd.DataFrame(rows).sort_values(["strategy", "split"]).reset_index(drop=True)


def robustness_by_quantile(
    score_panel: pd.DataFrame,
    ret_panel: pd.DataFrame,
    build_fn,
    quantiles: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for quantile in quantiles:
        returns = build_fn(score_panel, ret_panel, top_quantile=quantile)
        long_short = returns["long_short"].dropna()
        if long_short.empty:
            continue
        rows.append(
            {
                "top_quantile": quantile,
                "periods": int(len(long_short)),
                "final_nav": float((1 + long_short).prod()),
                "mean_monthly_return": float(long_short.mean()),
                "monthly_volatility": float(long_short.std(ddof=0)),
                "max_drawdown": float(((1 + long_short).cumprod() / (1 + long_short).cumprod().cummax() - 1).min()),
            }
        )
    return pd.DataFrame(rows)
