from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_text

ALLOWED_ML_MODELS = {"linear", "ridge", "lasso", "logistic", "random_forest", "gradient_boosting"}
ALLOWED_VALIDATION_METHODS = {"expanding", "purged"}


class ConfigError(ValueError):
    """Raised when a config file is syntactically valid but semantically invalid."""


@dataclass(frozen=True)
class DataConfig:
    input_dir: str = "data"
    output_dir: str = "outputs/latest"


@dataclass(frozen=True)
class FetchConfig:
    start_date: str = "20200101"
    end_date: str | None = None
    probe_n: int = 5
    final_n: int = 300
    max_retries: int = 4
    base_delay: float = 1.0
    sleep_each: float = 0.5
    adjust: str = "qfq"
    use_cache: bool = True
    resume: bool = True
    max_workers: int = 1
    checkpoint_every: int = 25
    rate_limit_per_second: float | None = None
    cache_dir: str = "cache/daily_prices"


@dataclass(frozen=True)
class CleaningConfig:
    winsor_lower_q: float = 0.005
    winsor_upper_q: float = 0.995
    z_thresh: float = 3.0
    iqr_k: float = 1.5
    ma_short: int = 5
    ma_long: int = 20
    ema_fast: int = 12
    ema_slow: int = 26
    ema_signal: int = 9
    vol_win: int = 20
    rsi_period: int = 14
    ts_plots_n: int = 3
    rng_seed: int = 42


@dataclass(frozen=True)
class FactorsConfig:
    factor_cols: list[str] = field(default_factory=lambda: ["VALUE", "MOM_12_1", "QUALITY", "SIZE"])
    enhanced_factor_cols: list[str] = field(default_factory=lambda: ["REVERSAL_1M", "VOL_1M", "ILLIQUIDITY"])
    include_enhanced: bool = True
    winsor_p: float = 0.01
    ic_decay_horizons: list[int] = field(default_factory=lambda: [1, 2, 3, 6])
    quantile_groups: int = 5


@dataclass(frozen=True)
class BacktestConfig:
    factor_weights: dict[str, float] = field(
        default_factory=lambda: {
            "z_VALUE": 1.0,
            "z_MOM_12_1": 0.0,
            "z_QUALITY": 0.0,
            "z_SIZE": 1.0,
        }
    )
    top_quantile: float = 0.2
    freq: int = 12
    sign_flip: bool = True
    rolling_window: int = 12
    strategy_name: str = "VALUE+SIZE"
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0
    robustness_quantiles: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])


@dataclass(frozen=True)
class ResearchConfig:
    train_end: str = "2022-12-31"
    valid_end: str = "2023-12-31"
    test_start: str = "2024-01-01"
    bootstrap_samples: int = 1000
    bootstrap_ci: float = 0.95
    random_seed: int = 42


@dataclass(frozen=True)
class ValidationConfig:
    method: str = "purged"
    embargo_months: int = 1
    min_train_months: int = 18
    test_window_months: int = 6


@dataclass(frozen=True)
class MLConfig:
    target: str = "fwd_1m_ret"
    classification_threshold: float = 0.0
    min_train_months: int = 18
    test_window_months: int = 6
    models: list[str] = field(
        default_factory=lambda: ["linear", "ridge", "lasso", "logistic", "random_forest", "gradient_boosting"]
    )
    feature_cols: list[str] = field(
        default_factory=lambda: [
            "z_VALUE",
            "z_MOM_12_1",
            "z_QUALITY",
            "z_SIZE",
            "z_REVERSAL_1M",
            "z_VOL_1M",
            "z_ILLIQUIDITY",
        ]
    )
    top_quantile: float = 0.2
    random_seed: int = 42


@dataclass(frozen=True)
class DecisionConfig:
    ml_model: str = "logistic"
    sign_flip: bool = True
    factor_weights: dict[str, float] = field(
        default_factory=lambda: {
            "z_VALUE": 1.0,
            "z_MOM_12_1": 0.0,
            "z_QUALITY": 0.0,
            "z_SIZE": 1.0,
        }
    )
    lookback_months: int = 12
    risk_aversion: float = 5.0
    turnover_penalty: float = 0.05
    concentration_penalty: float = 0.01
    max_weight: float = 0.25
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0


@dataclass(frozen=True)
class TuningConfig:
    factor_weight_sets: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"z_VALUE": 1.0, "z_SIZE": 1.0},
            {"z_VALUE": 1.0, "z_MOM_12_1": 1.0, "z_QUALITY": 1.0, "z_SIZE": 1.0},
            {"z_VALUE": 1.0, "z_REVERSAL_1M": 1.0, "z_VOL_1M": 1.0},
        ]
    )
    risk_aversion: list[float] = field(default_factory=lambda: [1.0, 5.0, 10.0])
    turnover_penalty: list[float] = field(default_factory=lambda: [0.0, 0.05, 0.15])
    top_quantile: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])


@dataclass(frozen=True)
class StressConfig:
    high_cost_bps: float = 75.0
    liquidity_drop_quantile: float = 0.2
    price_limit_pct: float = 9.5


@dataclass(frozen=True)
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    fetch: FetchConfig = field(default_factory=FetchConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    factors: FactorsConfig = field(default_factory=FactorsConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    stress: StressConfig = field(default_factory=StressConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG: dict[str, Any] = PipelineConfig().to_dict()


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _require_mapping(cfg: dict[str, Any], section: str) -> dict[str, Any]:
    value = cfg.get(section)
    if not isinstance(value, dict):
        raise ConfigError(f"Config section `{section}` must be a mapping.")
    return value


def _require_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"Config `{name}` must be numeric.")
    return float(value)


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"Config `{name}` must be an integer.")
    return int(value)


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"Config `{name}` must be true or false.")
    return value


def _require_probability(value: Any, name: str) -> float:
    numeric = _require_number(value, name)
    if not 0.0 < numeric < 1.0:
        raise ConfigError(f"Config `{name}` must be between 0 and 1, exclusive.")
    return numeric


def _require_unit_interval(value: Any, name: str) -> float:
    numeric = _require_number(value, name)
    if not 0.0 < numeric <= 1.0:
        raise ConfigError(f"Config `{name}` must be in (0, 1].")
    return numeric


def _require_positive_percent(value: Any, name: str) -> float:
    numeric = _require_number(value, name)
    if not 0.0 < numeric <= 100.0:
        raise ConfigError(f"Config `{name}` must be in (0, 100].")
    return numeric


def _require_non_negative(value: Any, name: str) -> float:
    numeric = _require_number(value, name)
    if numeric < 0.0:
        raise ConfigError(f"Config `{name}` must be non-negative.")
    return numeric


def _require_positive_int(value: Any, name: str) -> int:
    numeric = _require_int(value, name)
    if numeric <= 0:
        raise ConfigError(f"Config `{name}` must be greater than 0.")
    return numeric


def _require_date_order(research_cfg: dict[str, Any]) -> None:
    import pandas as pd

    train_end = pd.Timestamp(str(research_cfg.get("train_end")))
    valid_end = pd.Timestamp(str(research_cfg.get("valid_end")))
    test_start = pd.Timestamp(str(research_cfg.get("test_start")))
    if not train_end < valid_end < test_start:
        raise ConfigError("Config dates must satisfy research.train_end < research.valid_end < research.test_start.")


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate merged runtime config and return a defensive copy."""
    result = deepcopy(cfg)

    for section in PipelineConfig.__dataclass_fields__:
        _require_mapping(result, section)

    fetch_cfg = result["fetch"]
    _require_positive_int(fetch_cfg.get("probe_n"), "fetch.probe_n")
    _require_positive_int(fetch_cfg.get("final_n"), "fetch.final_n")
    _require_positive_int(fetch_cfg.get("max_retries"), "fetch.max_retries")
    _require_non_negative(fetch_cfg.get("base_delay"), "fetch.base_delay")
    _require_non_negative(fetch_cfg.get("sleep_each"), "fetch.sleep_each")
    _require_bool(fetch_cfg.get("use_cache"), "fetch.use_cache")
    _require_bool(fetch_cfg.get("resume"), "fetch.resume")
    _require_positive_int(fetch_cfg.get("max_workers"), "fetch.max_workers")
    _require_positive_int(fetch_cfg.get("checkpoint_every"), "fetch.checkpoint_every")
    rate_limit = fetch_cfg.get("rate_limit_per_second")
    if rate_limit is not None and _require_number(rate_limit, "fetch.rate_limit_per_second") <= 0.0:
        raise ConfigError("Config `fetch.rate_limit_per_second` must be greater than 0 when set.")
    if not isinstance(fetch_cfg.get("cache_dir"), str) or not str(fetch_cfg.get("cache_dir")).strip():
        raise ConfigError("Config `fetch.cache_dir` must be a non-empty string.")

    cleaning_cfg = result["cleaning"]
    lower = _require_probability(cleaning_cfg.get("winsor_lower_q"), "cleaning.winsor_lower_q")
    upper = _require_probability(cleaning_cfg.get("winsor_upper_q"), "cleaning.winsor_upper_q")
    if not lower < upper:
        raise ConfigError("Config `cleaning.winsor_lower_q` must be lower than `cleaning.winsor_upper_q`.")
    for name in ["ma_short", "ma_long", "ema_fast", "ema_slow", "ema_signal", "vol_win", "rsi_period"]:
        _require_positive_int(cleaning_cfg.get(name), f"cleaning.{name}")

    factors_cfg = result["factors"]
    _require_probability(factors_cfg.get("winsor_p"), "factors.winsor_p")
    _require_positive_int(factors_cfg.get("quantile_groups"), "factors.quantile_groups")
    horizons = factors_cfg.get("ic_decay_horizons")
    if not isinstance(horizons, list) or not horizons:
        raise ConfigError("Config `factors.ic_decay_horizons` must be a non-empty list.")
    for idx, value in enumerate(horizons):
        _require_positive_int(value, f"factors.ic_decay_horizons[{idx}]")

    backtest_cfg = result["backtest"]
    _require_probability(backtest_cfg.get("top_quantile"), "backtest.top_quantile")
    _require_positive_int(backtest_cfg.get("freq"), "backtest.freq")
    _require_positive_int(backtest_cfg.get("rolling_window"), "backtest.rolling_window")
    _require_non_negative(backtest_cfg.get("transaction_cost_bps"), "backtest.transaction_cost_bps")
    _require_non_negative(backtest_cfg.get("slippage_bps"), "backtest.slippage_bps")
    robustness_quantiles = backtest_cfg.get("robustness_quantiles")
    if not isinstance(robustness_quantiles, list) or not robustness_quantiles:
        raise ConfigError("Config `backtest.robustness_quantiles` must be a non-empty list.")
    for idx, value in enumerate(robustness_quantiles):
        _require_probability(value, f"backtest.robustness_quantiles[{idx}]")

    research_cfg = result["research"]
    _require_date_order(research_cfg)
    _require_positive_int(research_cfg.get("bootstrap_samples"), "research.bootstrap_samples")
    bootstrap_ci = _require_probability(research_cfg.get("bootstrap_ci"), "research.bootstrap_ci")
    if not 0.5 <= bootstrap_ci < 1.0:
        raise ConfigError("Config `research.bootstrap_ci` must be in [0.5, 1.0).")

    validation_cfg = result["validation"]
    method = str(validation_cfg.get("method"))
    if method not in ALLOWED_VALIDATION_METHODS:
        allowed = ", ".join(sorted(ALLOWED_VALIDATION_METHODS))
        raise ConfigError(f"Config `validation.method` must be one of: {allowed}.")
    embargo = _require_int(validation_cfg.get("embargo_months"), "validation.embargo_months")
    if embargo < 0:
        raise ConfigError("Config `validation.embargo_months` must be non-negative.")
    _require_positive_int(validation_cfg.get("min_train_months"), "validation.min_train_months")
    _require_positive_int(validation_cfg.get("test_window_months"), "validation.test_window_months")

    ml_cfg = result["ml"]
    _require_positive_int(ml_cfg.get("min_train_months"), "ml.min_train_months")
    _require_positive_int(ml_cfg.get("test_window_months"), "ml.test_window_months")
    _require_probability(ml_cfg.get("top_quantile"), "ml.top_quantile")
    models = ml_cfg.get("models")
    if not isinstance(models, list) or not models:
        raise ConfigError("Config `ml.models` must be a non-empty list.")
    invalid_models = sorted(set(str(model) for model in models) - ALLOWED_ML_MODELS)
    if invalid_models:
        allowed = ", ".join(sorted(ALLOWED_ML_MODELS))
        raise ConfigError(f"Config `ml.models` contains unsupported models {invalid_models}. Allowed: {allowed}.")

    decision_cfg = result["decision"]
    if str(decision_cfg.get("ml_model")) not in ALLOWED_ML_MODELS:
        allowed = ", ".join(sorted(ALLOWED_ML_MODELS))
        raise ConfigError(f"Config `decision.ml_model` must be one of: {allowed}.")
    _require_positive_int(decision_cfg.get("lookback_months"), "decision.lookback_months")
    _require_non_negative(decision_cfg.get("risk_aversion"), "decision.risk_aversion")
    _require_non_negative(decision_cfg.get("turnover_penalty"), "decision.turnover_penalty")
    _require_non_negative(decision_cfg.get("concentration_penalty"), "decision.concentration_penalty")
    _require_unit_interval(decision_cfg.get("max_weight"), "decision.max_weight")
    _require_non_negative(decision_cfg.get("transaction_cost_bps"), "decision.transaction_cost_bps")
    _require_non_negative(decision_cfg.get("slippage_bps"), "decision.slippage_bps")

    tuning_cfg = result["tuning"]
    for list_name in ["risk_aversion", "turnover_penalty", "top_quantile"]:
        values = tuning_cfg.get(list_name)
        if not isinstance(values, list) or not values:
            raise ConfigError(f"Config `tuning.{list_name}` must be a non-empty list.")
    for idx, value in enumerate(tuning_cfg["risk_aversion"]):
        _require_non_negative(value, f"tuning.risk_aversion[{idx}]")
    for idx, value in enumerate(tuning_cfg["turnover_penalty"]):
        _require_non_negative(value, f"tuning.turnover_penalty[{idx}]")
    for idx, value in enumerate(tuning_cfg["top_quantile"]):
        _require_probability(value, f"tuning.top_quantile[{idx}]")

    stress_cfg = result["stress"]
    _require_non_negative(stress_cfg.get("high_cost_bps"), "stress.high_cost_bps")
    _require_probability(stress_cfg.get("liquidity_drop_quantile"), "stress.liquidity_drop_quantile")
    _require_positive_percent(stress_cfg.get("price_limit_pct"), "stress.price_limit_pct")

    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config, merge it with project defaults, and validate it."""
    cfg = deepcopy(DEFAULT_CONFIG)
    if path is None:
        return validate_config(cfg)

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
        raise ConfigError(f"Config must contain a mapping at top level: {config_path}")
    return validate_config(deep_update(cfg, loaded))


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
    return validate_config(result)


def write_resolved_config(cfg: dict[str, Any], output_dir: str | Path | None = None) -> Path:
    """Write the fully merged and validated config used by a CLI run."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to write resolved config files.") from exc

    resolved = validate_config(cfg)
    out_dir = Path(output_dir or resolved["data"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "resolved_config.yaml"
    atomic_write_text(path, yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path
