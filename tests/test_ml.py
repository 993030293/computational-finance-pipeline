from __future__ import annotations

from pathlib import Path

from cfpipeline.cleaning import run_cleaning
from cfpipeline.factors import run_factors
from cfpipeline.ml import load_factor_dataset, prepare_ml_dataset, run_ml, run_walk_forward
from cfpipeline.paths import PipelinePaths


def test_ml_walk_forward_has_no_date_overlap(tmp_path: Path, make_daily_sample) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    processed = data_dir / "processed"
    processed.mkdir(parents=True)
    make_daily_sample(symbols=8, start="2019-01-01", end="2023-12-31").to_csv(
        processed / "daily_price_50.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cfg = {
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
        "research": {"bootstrap_samples": 20, "random_seed": 42},
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
    run_cleaning(cfg)
    run_factors(cfg)
    paths = PipelinePaths.from_config(cfg)
    factors, _ = load_factor_dataset(paths)
    dataset, feature_cols = prepare_ml_dataset(factors, cfg["ml"])
    predictions, _ = run_walk_forward(dataset, feature_cols, cfg["ml"])
    assert not predictions.empty
    for _, group in predictions.groupby("split_id"):
        assert group["train_end"].max() < group["test_start"].min()

    outputs = run_ml(cfg)
    assert outputs["model_metrics"].exists()
