from __future__ import annotations

from pathlib import Path

import pandas as pd

from cfpipeline.cleaning import run_cleaning
from cfpipeline.decision import run_decision
from cfpipeline.factors import run_factors
from cfpipeline.ml import run_ml
from cfpipeline.stress import run_stress
from cfpipeline.tuning import run_tuning


def _cfg(tmp_path: Path, make_daily_sample) -> dict:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    processed = data_dir / "processed"
    processed.mkdir(parents=True)
    make_daily_sample(symbols=8, start="2019-01-01", end="2023-12-31").to_csv(
        processed / "daily_price_50.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return {
        "data": {"input_dir": str(data_dir), "output_dir": str(output_dir)},
        "cleaning": {"ts_plots_n": 1},
        "factors": {
            "factor_cols": ["VALUE", "MOM_12_1", "QUALITY", "SIZE"],
            "enhanced_factor_cols": ["REVERSAL_1M", "VOL_1M", "ILLIQUIDITY"],
            "include_enhanced": True,
            "winsor_p": 0.01,
            "ic_decay_horizons": [1, 2],
            "quantile_groups": 3,
        },
        "backtest": {
            "factor_weights": {"z_VALUE": 1.0, "z_SIZE": 1.0},
            "top_quantile": 0.2,
            "freq": 12,
            "sign_flip": True,
            "rolling_window": 3,
            "transaction_cost_bps": 10.0,
            "slippage_bps": 5.0,
        },
        "research": {
            "train_end": "2021-12-31",
            "valid_end": "2022-06-30",
            "test_start": "2022-07-01",
            "bootstrap_samples": 20,
            "random_seed": 42,
        },
        "ml": {
            "target": "fwd_1m_ret",
            "min_train_months": 12,
            "test_window_months": 4,
            "models": ["linear", "logistic"],
            "feature_cols": ["z_VALUE", "z_MOM_12_1", "z_QUALITY", "z_SIZE"],
            "top_quantile": 0.2,
            "random_seed": 42,
        },
        "decision": {
            "ml_model": "logistic",
            "factor_weights": {"z_VALUE": 1.0, "z_SIZE": 1.0},
            "lookback_months": 6,
            "risk_aversion": 3.0,
            "turnover_penalty": 0.05,
            "max_weight": 0.35,
            "transaction_cost_bps": 10.0,
            "slippage_bps": 5.0,
        },
        "tuning": {
            "factor_weight_sets": [{"z_VALUE": 1.0, "z_SIZE": 1.0}, {"z_VALUE": 1.0, "z_MOM_12_1": 1.0}],
            "risk_aversion": [1.0, 3.0],
            "turnover_penalty": [0.0, 0.05],
            "top_quantile": [0.2],
        },
        "stress": {"high_cost_bps": 75.0, "liquidity_drop_quantile": 0.2, "price_limit_pct": 9.5},
    }


def test_decision_tuning_stress_pipeline(tmp_path: Path, make_daily_sample) -> None:
    cfg = _cfg(tmp_path, make_daily_sample)
    run_cleaning(cfg)
    run_factors(cfg)
    run_ml(cfg)
    decision_outputs = run_decision(cfg)
    tuning_outputs = run_tuning(cfg)
    stress_outputs = run_stress(cfg)

    weights = pd.read_csv(decision_outputs["weights"])
    sums = weights.groupby(["source", "month_end"])["weight"].sum()
    assert ((sums - 1.0).abs() < 1e-6).all()

    tuning = pd.read_csv(tuning_outputs["results"])
    selected = pd.read_csv(tuning_outputs["selected_params"])
    validation = tuning[tuning["split"].eq("validation")]
    best_id = int(validation.sort_values(["sharpe", "ann_return"], ascending=False)["candidate_id"].iloc[0])
    assert int(selected["candidate_id"].iloc[0]) == best_id

    stress = pd.read_csv(stress_outputs["returns"])
    assert (stress["high_cost_return"] <= stress["net_return"] + 1e-12).all()
