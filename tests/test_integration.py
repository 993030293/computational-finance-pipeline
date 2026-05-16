from __future__ import annotations

from pathlib import Path

from cfpipeline.backtest import run_backtest
from cfpipeline.cleaning import run_cleaning
from cfpipeline.factors import run_factors
from cfpipeline.ml import run_ml


def test_clean_factors_backtest_pipeline(tmp_path: Path, make_daily_sample) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    processed = data_dir / "processed"
    processed.mkdir(parents=True)
    make_daily_sample(symbols=6).to_csv(processed / "daily_price_50.csv", index=False, encoding="utf-8-sig")

    cfg = {
        "data": {"input_dir": str(data_dir), "output_dir": str(output_dir)},
        "cleaning": {"ts_plots_n": 1},
        "factors": {
            "factor_cols": ["VALUE", "MOM_12_1", "QUALITY", "SIZE"],
            "enhanced_factor_cols": ["REVERSAL_1M", "VOL_1M", "ILLIQUIDITY"],
            "include_enhanced": True,
            "winsor_p": 0.01,
        },
        "backtest": {
            "factor_weights": {"z_VALUE": 1.0, "z_MOM_12_1": 0.0, "z_QUALITY": 0.0, "z_SIZE": 1.0},
            "top_quantile": 0.2,
            "freq": 12,
            "sign_flip": True,
            "rolling_window": 3,
            "strategy_name": "test",
            "transaction_cost_bps": 10.0,
            "slippage_bps": 5.0,
            "robustness_quantiles": [0.2],
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
            "models": ["linear", "ridge", "logistic"],
            "feature_cols": ["z_VALUE", "z_MOM_12_1", "z_QUALITY", "z_SIZE"],
            "top_quantile": 0.2,
            "random_seed": 42,
        },
    }
    clean_outputs = run_cleaning(cfg)
    factor_outputs = run_factors(cfg)
    backtest_outputs = run_backtest(cfg)
    ml_outputs = run_ml(cfg)

    assert clean_outputs["tech_indicators"].exists()
    assert factor_outputs["factors"].exists()
    assert factor_outputs["ic_significance"].exists()
    assert backtest_outputs["performance_metrics"].exists()
    assert backtest_outputs["performance_metrics_net"].exists()
    assert ml_outputs["model_metrics"].exists()
