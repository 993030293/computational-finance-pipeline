from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import atomic_write_csv, atomic_write_text
from .paths import PipelinePaths

REGISTRY_COLUMNS = [
    "benchmark",
    "category",
    "validation_method",
    "return_type",
    "model",
    "strategy",
    "ann_return",
    "sharpe",
    "max_drawdown",
    "turnover",
    "transaction_cost_bps",
    "auc",
    "rank_ic",
    "selection_role",
    "source_file",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _metric_value(row: pd.Series, name: str) -> float:
    value = row.get(name, np.nan)
    return float(value) if pd.notna(value) else np.nan


def _base_row(
    *,
    benchmark: str,
    category: str,
    source_file: Path,
    validation_method: str = "",
    return_type: str = "",
    model: str = "",
    strategy: str = "",
    ann_return: float = np.nan,
    sharpe: float = np.nan,
    max_drawdown: float = np.nan,
    turnover: float = np.nan,
    transaction_cost_bps: float = np.nan,
    auc: float = np.nan,
    rank_ic: float = np.nan,
    selection_role: str = "reporting_only",
) -> dict[str, Any]:
    return {
        "benchmark": benchmark,
        "category": category,
        "validation_method": validation_method,
        "return_type": return_type,
        "model": model,
        "strategy": strategy,
        "ann_return": ann_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "turnover": turnover,
        "transaction_cost_bps": transaction_cost_bps,
        "auc": auc,
        "rank_ic": rank_ic,
        "selection_role": selection_role,
        "source_file": str(source_file),
    }


def _turnover_lookup(paths: PipelinePaths) -> dict[str, float]:
    turnover = _read_csv(paths.output_backtest_dir / "turnover.csv")
    if turnover.empty:
        return {}
    return {
        col: float(pd.to_numeric(turnover[col], errors="coerce").mean())
        for col in turnover.columns
        if col != "month_end"
    }


def _add_backtest_rows(rows: list[dict[str, Any]], paths: PipelinePaths, cfg: dict[str, Any]) -> None:
    cost_bps = float(cfg.get("backtest", {}).get("transaction_cost_bps", 0.0)) + float(
        cfg.get("backtest", {}).get("slippage_bps", 0.0)
    )
    turnover = _turnover_lookup(paths)
    for return_type, filename, cost in [
        ("gross", "performance_metrics.csv", 0.0),
        ("net", "performance_metrics_net.csv", cost_bps),
    ]:
        path = paths.output_backtest_dir / filename
        metrics = _read_csv(path)
        if metrics.empty:
            continue
        for _, row in metrics.iterrows():
            strategy = str(row.get("strategy", ""))
            if strategy not in {"benchmark_ew", "long_short"}:
                continue
            benchmark = "equal_weight_baseline" if strategy == "benchmark_ew" else "factor_long_short_baseline"
            rows.append(
                _base_row(
                    benchmark=benchmark,
                    category="backtest",
                    return_type=return_type,
                    strategy=strategy,
                    source_file=path,
                    ann_return=_metric_value(row, "ann_return"),
                    sharpe=_metric_value(row, "sharpe"),
                    max_drawdown=_metric_value(row, "max_drawdown"),
                    turnover=turnover.get(strategy, np.nan),
                    transaction_cost_bps=cost,
                )
            )


def _add_ml_rows(rows: list[dict[str, Any]], paths: PipelinePaths) -> None:
    metrics_path = paths.output_dir / "ml" / "ml_model_metrics.csv"
    metrics = _read_csv(metrics_path)
    if not metrics.empty:
        for _, row in metrics.iterrows():
            rows.append(
                _base_row(
                    benchmark="ml_prediction_baseline",
                    category="ml",
                    validation_method=str(row.get("validation_method", "")),
                    return_type="prediction",
                    model=str(row.get("model", "")),
                    source_file=metrics_path,
                    auc=_metric_value(row, "auc"),
                    rank_ic=_metric_value(row, "rank_ic"),
                )
            )

    backtest_path = paths.output_dir / "ml" / "ml_backtest_metrics.csv"
    backtest = _read_csv(backtest_path)
    if not backtest.empty:
        for _, row in backtest[backtest["strategy"].eq("long_short")].iterrows():
            rows.append(
                _base_row(
                    benchmark="ml_portfolio_baseline",
                    category="ml",
                    validation_method=str(row.get("validation_method", "")),
                    return_type="gross",
                    model=str(row.get("model", "")),
                    strategy="long_short",
                    source_file=backtest_path,
                    ann_return=_metric_value(row, "ann_return"),
                    sharpe=_metric_value(row, "sharpe"),
                    max_drawdown=_metric_value(row, "max_drawdown"),
                )
            )


def _add_decision_rows(rows: list[dict[str, Any]], paths: PipelinePaths, cfg: dict[str, Any]) -> None:
    path = paths.output_dir / "decision" / "decision_metrics.csv"
    metrics = _read_csv(path)
    if metrics.empty:
        return
    cost_bps = float(cfg.get("decision", {}).get("transaction_cost_bps", 0.0)) + float(
        cfg.get("decision", {}).get("slippage_bps", 0.0)
    )
    returns = _read_csv(paths.output_dir / "decision" / "decision_returns.csv")
    turnover_by_source = {}
    if not returns.empty and {"source", "turnover"}.issubset(returns.columns):
        turnover_by_source = returns.groupby("source")["turnover"].mean().astype(float).to_dict()
    for _, row in metrics.iterrows():
        source = str(row.get("source", ""))
        return_type = str(row.get("return_type", ""))
        rows.append(
            _base_row(
                benchmark=f"decision_{source}",
                category="decision",
                return_type=return_type,
                strategy="optimized_long_only",
                source_file=path,
                ann_return=_metric_value(row, "ann_return"),
                sharpe=_metric_value(row, "sharpe"),
                max_drawdown=_metric_value(row, "max_drawdown"),
                turnover=turnover_by_source.get(source, np.nan),
                transaction_cost_bps=cost_bps if return_type == "net" else 0.0,
            )
        )


def _add_tuned_rows(rows: list[dict[str, Any]], paths: PipelinePaths) -> None:
    path = paths.output_dir / "tuning" / "test_performance.csv"
    test = _read_csv(path)
    if test.empty:
        return
    for _, row in test.iterrows():
        rows.append(
            _base_row(
                benchmark="tuned_strategy",
                category="tuning",
                return_type="net",
                strategy="long_short",
                source_file=path,
                ann_return=_metric_value(row, "ann_return"),
                sharpe=_metric_value(row, "sharpe"),
                max_drawdown=_metric_value(row, "max_drawdown"),
                turnover=_metric_value(row, "mean_turnover"),
                selection_role="test_reporting_only",
            )
        )


def build_benchmark_registry(paths: PipelinePaths, cfg: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    _add_backtest_rows(rows, paths, cfg)
    _add_ml_rows(rows, paths)
    _add_decision_rows(rows, paths, cfg)
    _add_tuned_rows(rows, paths)
    if not rows:
        return pd.DataFrame(columns=REGISTRY_COLUMNS)
    return pd.DataFrame(rows, columns=REGISTRY_COLUMNS)


def gross_net_comparison(registry: pd.DataFrame) -> pd.DataFrame:
    if registry.empty:
        return pd.DataFrame(
            columns=["benchmark", "strategy", "model", "gross_ann_return", "net_ann_return", "cost_drag"]
        )
    comparable = registry[registry["return_type"].isin(["gross", "net"])].copy()
    if comparable.empty:
        return pd.DataFrame(
            columns=["benchmark", "strategy", "model", "gross_ann_return", "net_ann_return", "cost_drag"]
        )
    index_cols = ["benchmark", "strategy", "model"]
    pivot = comparable.pivot_table(index=index_cols, columns="return_type", values="ann_return", aggfunc="first")
    pivot = pivot.reset_index().rename(columns={"gross": "gross_ann_return", "net": "net_ann_return"})
    if "gross_ann_return" not in pivot.columns:
        pivot["gross_ann_return"] = np.nan
    if "net_ann_return" not in pivot.columns:
        pivot["net_ann_return"] = np.nan
    pivot["cost_drag"] = pivot["gross_ann_return"] - pivot["net_ann_return"]
    return pivot[["benchmark", "strategy", "model", "gross_ann_return", "net_ann_return", "cost_drag"]]


def validation_method_comparison(paths: PipelinePaths) -> pd.DataFrame:
    path = paths.output_dir / "ml" / "ml_validation_comparison.csv"
    metrics = _read_csv(path)
    if metrics.empty:
        return pd.DataFrame(columns=["model", "metric", "expanding", "purged", "purged_minus_expanding"])
    rows: list[dict[str, Any]] = []
    for metric in ["auc", "rank_ic", "precision_at_top"]:
        pivot = metrics.pivot_table(index="model", columns="validation_method", values=metric, aggfunc="first")
        for model, values in pivot.iterrows():
            expanding = values.get("expanding", np.nan)
            purged = values.get("purged", np.nan)
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "expanding": expanding,
                    "purged": purged,
                    "purged_minus_expanding": purged - expanding
                    if pd.notna(expanding) and pd.notna(purged)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def write_stability_report(
    out_dir: Path,
    registry: pd.DataFrame,
    cost_comparison: pd.DataFrame,
    validation_comparison: pd.DataFrame,
    cfg: dict[str, Any],
) -> Path:
    path = out_dir / "STABILITY_REPORT.md"
    validation_cfg = cfg.get("validation", {})
    lines = [
        "# Stability and Benchmark Registry Report",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Primary validation method: `{validation_cfg.get('method', 'expanding')}`",
        f"- Embargo months: {validation_cfg.get('embargo_months', 0)}",
        "",
        "This report is a research diagnostic. Stricter validation can lower headline metrics, but it reduces leakage and overfitting risk.",
        "",
        "## Benchmark Registry",
        registry.to_markdown(index=False, floatfmt=".6f") if not registry.empty else "No benchmark rows generated.",
        "",
        "## Gross vs Net Cost Sensitivity",
        cost_comparison.to_markdown(index=False, floatfmt=".6f")
        if not cost_comparison.empty
        else "No gross/net pairs available.",
        "",
        "## Expanding vs Purged Validation",
        validation_comparison.to_markdown(index=False, floatfmt=".6f")
        if not validation_comparison.empty
        else "No validation comparison available.",
        "",
        "## Interpretation",
        "- Test metrics are reporting-only diagnostics and must not participate in hyperparameter selection.",
        "- Transaction costs and turnover are shown to separate gross signal quality from net implementability.",
        "- The registry is not a live trading performance claim.",
    ]
    atomic_write_text(path, "\n".join(lines), encoding="utf-8")
    return path


def run_benchmarks(cfg: dict[str, Any]) -> dict[str, Path]:
    paths = PipelinePaths.from_config(cfg)
    out_dir = paths.output_dir / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = build_benchmark_registry(paths, cfg)
    cost_comparison = gross_net_comparison(registry)
    validation_comparison = validation_method_comparison(paths)

    registry_path = out_dir / "benchmark_registry.csv"
    cost_path = out_dir / "gross_net_comparison.csv"
    validation_path = out_dir / "validation_method_comparison.csv"
    atomic_write_csv(registry, registry_path, index=False, encoding="utf-8-sig")
    atomic_write_csv(cost_comparison, cost_path, index=False, encoding="utf-8-sig")
    atomic_write_csv(validation_comparison, validation_path, index=False, encoding="utf-8-sig")
    report_path = write_stability_report(out_dir, registry, cost_comparison, validation_comparison, cfg)
    return {
        "registry": registry_path,
        "gross_net_comparison": cost_path,
        "validation_comparison": validation_path,
        "report": report_path,
    }
