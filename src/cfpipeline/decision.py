from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .backtest import (
    DEFAULT_ZCOLS,
    add_factor_score,
    apply_sign_flip,
    factor_ic_signs,
    performance_metrics,
    pivot_panel,
)
from .cleaning import read_csv
from .paths import PipelinePaths, first_existing


def factor_candidates(paths: PipelinePaths) -> list[Path]:
    return [paths.output_project4_dir / "factors.csv", paths.input_project4_dir / "factors.csv"]


def ml_prediction_candidates(paths: PipelinePaths) -> list[Path]:
    return [
        paths.output_dir / "ml" / "ml_predictions.csv",
        paths.input_dir / "ml" / "ml_predictions.csv",
    ]


def load_factors(paths: PipelinePaths) -> tuple[pd.DataFrame, Path]:
    path = first_existing(factor_candidates(paths))
    df = read_csv(path, parse_dates=["month_end"])
    df["symbol"] = df["symbol"].astype("string")
    return df.sort_values(["month_end", "symbol"]).reset_index(drop=True), path


def load_ml_predictions(paths: PipelinePaths) -> tuple[pd.DataFrame | None, Path | None]:
    for path in ml_prediction_candidates(paths):
        if path.exists():
            df = read_csv(path, parse_dates=["month_end", "train_start", "train_end", "test_start", "test_end"])
            df["symbol"] = df["symbol"].astype("string")
            return df.sort_values(["month_end", "symbol"]).reset_index(drop=True), path
    return None, None


def make_factor_score(factors: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    zcols = [col for col in DEFAULT_ZCOLS if col in factors.columns]
    scored = factors.copy()
    if bool(cfg.get("sign_flip", True)):
        signs, _ = factor_ic_signs(scored, zcols, ret_col="fwd_1m_ret")
        scored = apply_sign_flip(scored, signs)
    scored = add_factor_score(scored, weights=cfg.get("factor_weights"))
    return scored.rename(columns={"score": "factor_score"})


def make_score_panels(
    factors: pd.DataFrame,
    ml_predictions: pd.DataFrame | None,
    cfg: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    scored = make_factor_score(factors, cfg)
    panels = {"factor_score": pivot_panel(scored, "factor_score")}
    if ml_predictions is not None and not ml_predictions.empty:
        model = str(cfg.get("ml_model", "logistic"))
        model_data = ml_predictions[ml_predictions["model"].eq(model)].copy()
        if not model_data.empty:
            model_data = model_data.rename(columns={"score": "ml_score"})
            panels["ml_score"] = pivot_panel(model_data, "ml_score")
    return panels


def covariance_for_month(ret_panel: pd.DataFrame, month: pd.Timestamp, symbols: list[str], lookback: int) -> np.ndarray:
    history = ret_panel.loc[ret_panel.index < month, symbols].tail(lookback).dropna(axis=1, how="all")
    if len(history) < 2:
        return np.eye(len(symbols)) * 0.05
    history = history.reindex(columns=symbols).fillna(0.0)
    cov = history.cov().to_numpy(dtype=float)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    # Numerical stabilizer for SLSQP and positive semi-definite edge cases.
    return cov + np.eye(len(symbols)) * 1e-6


def optimize_long_only_weights(
    expected_return: pd.Series,
    covariance: np.ndarray,
    previous_weights: pd.Series | None = None,
    *,
    risk_aversion: float = 5.0,
    turnover_penalty: float = 0.05,
    concentration_penalty: float = 0.01,
    max_weight: float = 0.25,
) -> pd.Series:
    mu = expected_return.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    symbols = list(mu.index.astype(str))
    n = len(symbols)
    if n == 0:
        return pd.Series(dtype=float)
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (n, n):
        cov = np.eye(n) * 0.05
    if previous_weights is None or previous_weights.empty:
        prev = np.zeros(n)
    else:
        prev = previous_weights.reindex(symbols).fillna(0.0).to_numpy(dtype=float)
    x0 = np.repeat(1.0 / n, n)
    upper = min(float(max_weight), 1.0)
    bounds = [(0.0, upper) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def objective(w: np.ndarray) -> float:
        return float(
            -np.dot(mu.to_numpy(dtype=float), w)
            + float(risk_aversion) * (w @ cov @ w)
            + float(turnover_penalty) * np.abs(w - prev).sum()
            + float(concentration_penalty) * np.square(w).sum()
        )

    result = minimize(objective, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 200})
    if not result.success or np.any(~np.isfinite(result.x)):
        weights = x0
    else:
        weights = np.clip(result.x, 0.0, upper)
        total = weights.sum()
        weights = weights / total if total > 0 else x0
    return pd.Series(weights, index=symbols, dtype=float)


def optimize_score_panel(
    score_panel: pd.DataFrame,
    ret_panel: pd.DataFrame,
    *,
    lookback_months: int = 12,
    risk_aversion: float = 5.0,
    turnover_penalty: float = 0.05,
    concentration_penalty: float = 0.01,
    max_weight: float = 0.25,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    weights_by_month: dict[pd.Timestamp, pd.Series] = {}
    returns: dict[pd.Timestamp, float] = {}
    turnover: dict[pd.Timestamp, float] = {}
    previous = pd.Series(dtype=float)
    aligned_returns = ret_panel.reindex_like(score_panel)
    for month in score_panel.index:
        scores = score_panel.loc[month].dropna()
        realized = aligned_returns.loc[month].dropna()
        symbols = sorted(set(scores.index.astype(str)).intersection(set(realized.index.astype(str))))
        if not symbols:
            continue
        expected = scores.reindex(symbols)
        cov = covariance_for_month(aligned_returns, month, symbols, lookback_months)
        weights = optimize_long_only_weights(
            expected,
            cov,
            previous,
            risk_aversion=risk_aversion,
            turnover_penalty=turnover_penalty,
            concentration_penalty=concentration_penalty,
            max_weight=max_weight,
        )
        returns[month] = float((weights * realized.reindex(weights.index).fillna(0.0)).sum())
        turnover[month] = float((weights - previous.reindex(weights.index).fillna(0.0)).abs().sum() / 2.0)
        weights_by_month[month] = weights
        previous = weights
    weights_df = pd.DataFrame(weights_by_month).T.sort_index().fillna(0.0)
    return weights_df, pd.Series(returns, name="return").sort_index(), pd.Series(turnover, name="turnover").sort_index()


def long_form_weights(weights: pd.DataFrame, source: str) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame(columns=["month_end", "symbol", "weight", "source"])
    out = weights.reset_index(names="month_end").melt(id_vars="month_end", var_name="symbol", value_name="weight")
    out = out[out["weight"].abs() > 1e-12].copy()
    out["source"] = source
    return out[["source", "month_end", "symbol", "weight"]]


def run_decision(cfg: dict[str, Any]) -> dict[str, Path]:
    paths = PipelinePaths.from_config(cfg)
    paths.ensure_output_dirs()
    decision_cfg = cfg.get("decision", {})
    out_dir = paths.output_dir / "decision"
    out_dir.mkdir(parents=True, exist_ok=True)

    factors, factor_path = load_factors(paths)
    ml_predictions, ml_path = load_ml_predictions(paths)
    ret_panel = pivot_panel(factors, "fwd_1m_ret")
    score_panels = make_score_panels(factors, ml_predictions, {**cfg.get("backtest", {}), **decision_cfg})

    returns_frames: list[pd.DataFrame] = []
    weight_frames: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, Any]] = []
    for source, score_panel in score_panels.items():
        weights, gross_ret, turnover = optimize_score_panel(
            score_panel,
            ret_panel,
            lookback_months=int(decision_cfg.get("lookback_months", 12)),
            risk_aversion=float(decision_cfg.get("risk_aversion", 5.0)),
            turnover_penalty=float(decision_cfg.get("turnover_penalty", 0.05)),
            concentration_penalty=float(decision_cfg.get("concentration_penalty", 0.01)),
            max_weight=float(decision_cfg.get("max_weight", 0.25)),
        )
        total_cost_bps = float(decision_cfg.get("transaction_cost_bps", 10.0)) + float(decision_cfg.get("slippage_bps", 5.0))
        net_ret = gross_ret - turnover.reindex(gross_ret.index).fillna(0.0) * total_cost_bps / 10000.0
        returns_frames.append(
            pd.DataFrame(
                {
                    "source": source,
                    "month_end": gross_ret.index,
                    "gross_return": gross_ret.values,
                    "net_return": net_ret.values,
                    "turnover": turnover.reindex(gross_ret.index).fillna(0.0).values,
                }
            )
        )
        weight_frames.append(long_form_weights(weights, source))
        for label, series in [("gross", gross_ret), ("net", net_ret)]:
            row = performance_metrics(series, freq=int(cfg.get("backtest", {}).get("freq", 12)))
            row["source"] = source
            row["return_type"] = label
            metrics_rows.append(row)

    returns_df = pd.concat(returns_frames, ignore_index=True) if returns_frames else pd.DataFrame()
    weights_df = pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame()
    metrics_df = pd.DataFrame(metrics_rows)

    returns_path = out_dir / "decision_returns.csv"
    weights_path = out_dir / "decision_weights.csv"
    metrics_path = out_dir / "decision_metrics.csv"
    metadata_path = out_dir / "decision_metadata.json"
    report_path = out_dir / "DECISION_REPORT.md"
    returns_df.to_csv(returns_path, index=False, encoding="utf-8-sig")
    weights_df.to_csv(weights_path, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    metadata_path.write_text(
        json.dumps(
            {
                "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "factor_path": str(factor_path),
                "ml_prediction_path": None if ml_path is None else str(ml_path),
                "config": decision_cfg,
                "sources": sorted(score_panels.keys()),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    lines = [
        "# LANCER-Inspired Decision Optimization Report",
        "",
        "This module translates prediction scores into long-only portfolio weights using an independent mean-variance objective.",
        "It is inspired by decision-focused learning but does not copy or depend on LANCER source code.",
        "",
        "## Metrics",
        metrics_df.to_markdown(index=False, floatfmt=".6f") if not metrics_df.empty else "No decision metrics generated.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "returns": returns_path,
        "weights": weights_path,
        "metrics": metrics_path,
        "report": report_path,
        "metadata": metadata_path,
    }
