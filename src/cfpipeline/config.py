from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "input_dir": "data",
        "output_dir": "outputs/latest",
    },
    "fetch": {
        "start_date": "20200101",
        "end_date": None,
        "probe_n": 5,
        "final_n": 300,
        "max_retries": 4,
        "base_delay": 1.0,
        "sleep_each": 0.5,
        "adjust": "qfq",
    },
    "cleaning": {
        "winsor_lower_q": 0.005,
        "winsor_upper_q": 0.995,
        "z_thresh": 3.0,
        "iqr_k": 1.5,
        "ma_short": 5,
        "ma_long": 20,
        "ema_fast": 12,
        "ema_slow": 26,
        "ema_signal": 9,
        "vol_win": 20,
        "rsi_period": 14,
        "ts_plots_n": 3,
        "rng_seed": 42,
    },
    "factors": {
        "factor_cols": ["VALUE", "MOM_12_1", "QUALITY", "SIZE"],
        "enhanced_factor_cols": ["REVERSAL_1M", "VOL_1M", "ILLIQUIDITY"],
        "include_enhanced": True,
        "winsor_p": 0.01,
        "ic_decay_horizons": [1, 2, 3, 6],
        "quantile_groups": 5,
    },
    "backtest": {
        "factor_weights": {
            "z_VALUE": 1.0,
            "z_MOM_12_1": 0.0,
            "z_QUALITY": 0.0,
            "z_SIZE": 1.0,
        },
        "top_quantile": 0.2,
        "freq": 12,
        "sign_flip": True,
        "rolling_window": 12,
        "strategy_name": "VALUE+SIZE",
        "transaction_cost_bps": 10.0,
        "slippage_bps": 5.0,
        "robustness_quantiles": [0.1, 0.2, 0.3],
    },
    "research": {
        "train_end": "2022-12-31",
        "valid_end": "2023-12-31",
        "test_start": "2024-01-01",
        "bootstrap_samples": 1000,
        "bootstrap_ci": 0.95,
        "random_seed": 42,
    },
    "ml": {
        "target": "fwd_1m_ret",
        "classification_threshold": 0.0,
        "min_train_months": 18,
        "test_window_months": 6,
        "models": ["linear", "ridge", "lasso", "logistic", "random_forest", "gradient_boosting"],
        "feature_cols": [
            "z_VALUE",
            "z_MOM_12_1",
            "z_QUALITY",
            "z_SIZE",
            "z_REVERSAL_1M",
            "z_VOL_1M",
            "z_ILLIQUIDITY",
        ],
        "top_quantile": 0.2,
        "random_seed": 42,
    },
    "decision": {
        "ml_model": "logistic",
        "sign_flip": True,
        "factor_weights": {
            "z_VALUE": 1.0,
            "z_MOM_12_1": 0.0,
            "z_QUALITY": 0.0,
            "z_SIZE": 1.0,
        },
        "lookback_months": 12,
        "risk_aversion": 5.0,
        "turnover_penalty": 0.05,
        "concentration_penalty": 0.01,
        "max_weight": 0.25,
        "transaction_cost_bps": 10.0,
        "slippage_bps": 5.0,
    },
    "tuning": {
        "factor_weight_sets": [
            {"z_VALUE": 1.0, "z_SIZE": 1.0},
            {"z_VALUE": 1.0, "z_MOM_12_1": 1.0, "z_QUALITY": 1.0, "z_SIZE": 1.0},
            {"z_VALUE": 1.0, "z_REVERSAL_1M": 1.0, "z_VOL_1M": 1.0},
        ],
        "risk_aversion": [1.0, 5.0, 10.0],
        "turnover_penalty": [0.0, 0.05, 0.15],
        "top_quantile": [0.1, 0.2, 0.3],
    },
    "stress": {
        "high_cost_bps": 75.0,
        "liquidity_drop_quantile": 0.2,
        "price_limit_pct": 9.5,
    },
}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config and merge it with project defaults."""
    cfg = deepcopy(DEFAULT_CONFIG)
    if path is None:
        return cfg

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load YAML config files.") from exc

    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must contain a mapping at top level: {config_path}")
    return deep_update(cfg, loaded)


def with_overrides(
    cfg: dict[str, Any],
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    result = deepcopy(cfg)
    if data_dir is not None:
        result.setdefault("data", {})["input_dir"] = str(data_dir)
    if output_dir is not None:
        result.setdefault("data", {})["output_dir"] = str(output_dir)
    return result
