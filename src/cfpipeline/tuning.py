from __future__ import annotations

import itertools
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import atomic_write_csv, atomic_write_json, atomic_write_text
from .backtest import (
    DEFAULT_ZCOLS,
    add_factor_score,
    apply_sign_flip,
    apply_transaction_costs,
    build_portfolio_weights,
    build_portfolios,
    calculate_turnover,
    factor_ic_signs,
    performance_metrics,
    pivot_panel,
)
from .decision import load_factors
from .paths import PipelinePaths
from .research import assign_time_split


def param_grid(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    weight_sets = cfg.get(
        "factor_weight_sets",
        [
            {"z_VALUE": 1.0, "z_SIZE": 1.0},
            {"z_VALUE": 1.0, "z_MOM_12_1": 1.0, "z_QUALITY": 1.0, "z_SIZE": 1.0},
            {"z_VALUE": 1.0, "z_REVERSAL_1M": 1.0, "z_VOL_1M": 1.0},
        ],
    )
    risk = [float(x) for x in cfg.get("risk_aversion", [1.0, 5.0, 10.0])]
    turnover = [float(x) for x in cfg.get("turnover_penalty", [0.0, 0.05, 0.15])]
    quantiles = [float(x) for x in cfg.get("top_quantile", [0.1, 0.2, 0.3])]
    rows = []
    for i, (weights, r, t, q) in enumerate(itertools.product(weight_sets, risk, turnover, quantiles)):
        rows.append(
            {"candidate_id": i, "factor_weights": weights, "risk_aversion": r, "turnover_penalty": t, "top_quantile": q}
        )
    return rows


def score_with_train_signs(
    factors: pd.DataFrame, weights: dict[str, float], research_cfg: dict[str, Any]
) -> pd.DataFrame:
    out = factors.copy()
    train_mask = pd.to_datetime(out["month_end"]) <= pd.Timestamp(str(research_cfg.get("train_end", "2022-12-31")))
    zcols = [col for col in DEFAULT_ZCOLS if col in out.columns]
    signs, _ = factor_ic_signs(out.loc[train_mask].copy(), zcols, ret_col="fwd_1m_ret")
    out = apply_sign_flip(out, signs)
    out = add_factor_score(out, weights=weights)
    return out


def evaluate_candidate(
    factors: pd.DataFrame,
    candidate: dict[str, Any],
    research_cfg: dict[str, Any],
    decision_cfg: dict[str, Any],
) -> dict[str, Any]:
    scored = score_with_train_signs(factors, candidate["factor_weights"], research_cfg)
    score_panel = pivot_panel(scored, "score")
    ret_panel = pivot_panel(scored, "fwd_1m_ret")
    # Fast validation proxy: penalize volatile names in the score, then tune
    # quantile and turnover costs chronologically. The heavier continuous
    # optimizer remains in `decision.py`; tuning must stay lightweight enough
    # for CI and repeated application-review runs.
    rolling_vol = ret_panel.rolling(int(decision_cfg.get("lookback_months", 12)), min_periods=3).std(ddof=0).shift(1)
    adjusted_score = score_panel - float(candidate["risk_aversion"]) * rolling_vol.reindex_like(score_panel).fillna(0.0)
    gross_panel = build_portfolios(adjusted_score, ret_panel, top_quantile=float(candidate["top_quantile"]))
    weights = build_portfolio_weights(adjusted_score, top_quantile=float(candidate["top_quantile"]))
    turnover_panel = calculate_turnover(weights)
    total_cost_bps = (
        float(decision_cfg.get("transaction_cost_bps", 10.0))
        + float(decision_cfg.get("slippage_bps", 5.0))
        + float(candidate["turnover_penalty"]) * 100.0
    )
    net_panel = apply_transaction_costs(
        gross_panel, turnover_panel, transaction_cost_bps=total_cost_bps, slippage_bps=0.0
    )
    net_ret = net_panel["long_short"]
    turnover = turnover_panel["long_short"]
    labels = assign_time_split(pd.Series(net_ret.index, index=net_ret.index), research_cfg)

    rows = []
    for split in ["train", "validation", "test"]:
        split_ret = net_ret.loc[labels[labels.eq(split)].index].dropna()
        metrics = performance_metrics(split_ret) if not split_ret.empty else {}
        rows.append(
            {
                **candidate,
                "factor_weights_json": json.dumps(candidate["factor_weights"], sort_keys=True),
                "split": split,
                "periods": int(len(split_ret)),
                "ann_return": metrics.get("ann_return", np.nan),
                "ann_vol": metrics.get("ann_vol", np.nan),
                "sharpe": metrics.get("sharpe", np.nan),
                "max_drawdown": metrics.get("max_drawdown", np.nan),
                "mean_turnover": float(turnover.loc[split_ret.index].mean()) if not split_ret.empty else np.nan,
            }
        )
    return {"rows": rows, "returns": net_ret}


def select_candidate_from_validation(results: pd.DataFrame) -> int:
    validation = results[results["split"].eq("validation")].copy()
    if validation.empty:
        return int(results["candidate_id"].iloc[0])
    selected = validation.sort_values(["sharpe", "ann_return"], ascending=False).iloc[0]
    return int(selected["candidate_id"])


def run_tuning(cfg: dict[str, Any]) -> dict[str, Path]:
    paths = PipelinePaths.from_config(cfg)
    paths.ensure_output_dirs()
    factors, factor_path = load_factors(paths)
    tuning_cfg = cfg.get("tuning", {})
    research_cfg = cfg.get("research", {})
    decision_cfg = cfg.get("decision", {})
    out_dir = paths.output_dir / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    returns_by_candidate: dict[int, pd.Series] = {}
    for candidate in param_grid(tuning_cfg):
        result = evaluate_candidate(factors, candidate, research_cfg, decision_cfg)
        all_rows.extend(result["rows"])
        returns_by_candidate[int(candidate["candidate_id"])] = result["returns"]
    results = pd.DataFrame(all_rows)
    selected_id = select_candidate_from_validation(results)
    selected = results[results["candidate_id"].eq(selected_id)].copy()
    test_performance = selected[selected["split"].eq("test")].copy()

    results_path = out_dir / "tuning_results.csv"
    selected_path = out_dir / "selected_params.csv"
    test_path = out_dir / "test_performance.csv"
    returns_path = out_dir / "selected_returns.csv"
    report_path = out_dir / "TUNING_REPORT.md"
    metadata_path = out_dir / "tuning_metadata.json"

    atomic_write_csv(results, results_path, index=False, encoding="utf-8-sig")
    atomic_write_csv(selected, selected_path, index=False, encoding="utf-8-sig")
    atomic_write_csv(test_performance, test_path, index=False, encoding="utf-8-sig")
    atomic_write_csv(returns_by_candidate[selected_id].rename("net_return"), returns_path, header=True)
    atomic_write_json(
        metadata_path,
        {
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "factor_path": str(factor_path),
            "selected_candidate_id": selected_id,
            "selection_rule": "highest validation Sharpe, test split held out",
            "research_config": research_cfg,
            "tuning_config": tuning_cfg,
        },
    )
    atomic_write_text(
        report_path,
        "\n".join(
            [
                "# MEHA-Inspired Bilevel Tuning Report",
                "",
                "This module uses chronological validation as a lightweight bilevel optimization proxy.",
                "The inner level builds scores and decision weights; the outer level selects hyperparameters using validation net Sharpe only.",
                "",
                f"- Selected candidate: {selected_id}",
                "",
                "## Selected Split Performance",
                selected.to_markdown(index=False, floatfmt=".6f"),
            ]
        ),
        encoding="utf-8",
    )
    return {
        "results": results_path,
        "selected_params": selected_path,
        "test_performance": test_path,
        "selected_returns": returns_path,
        "report": report_path,
        "metadata": metadata_path,
    }
