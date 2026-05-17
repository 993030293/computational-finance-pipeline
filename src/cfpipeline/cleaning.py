from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import atomic_write_csv, atomic_write_json, atomic_write_text
from .paths import PipelinePaths, first_existing

COLUMN_ALIASES = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "涨跌幅": "pct_chg",
    "涨跌额": "change",
    "换手率": "turnover_rate",
    "振幅": "amplitude",
    "代码": "symbol",
    "名称": "name",
    "Date": "date",
    "Open": "open",
    "Close": "close",
    "High": "high",
    "Low": "low",
    "Volume": "volume",
    "Amount": "amount",
    "Pct_chg": "pct_chg",
    "Change": "change",
    "Code": "symbol",
    "code": "symbol",
    "day": "date",
    "时间": "date",
    "最新价": "price",
    "今开": "open",
    "昨收": "pre_close",
}


DAILY_COLUMNS = [
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "pct_chg",
]


def read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    kwargs.setdefault("encoding", "utf-8-sig")
    kwargs.setdefault("dtype", {"symbol": "string", "code": "string"})
    return pd.read_csv(path, **kwargs)


def rename_any_columns(df: pd.DataFrame) -> pd.DataFrame:
    existing = {key: value for key, value in COLUMN_ALIASES.items() if key in df.columns}
    return df.rename(columns=existing, inplace=False)


def ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
    return out


def to_datetime_col(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def to_numeric_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def clean_daily_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize and clean daily OHLCV data."""
    out = rename_any_columns(df)
    out = ensure_columns(out, DAILY_COLUMNS)
    out["symbol"] = out["symbol"].astype("string").str.strip()
    out = to_datetime_col(out, "date")
    out = to_numeric_cols(out, ["open", "high", "low", "close", "volume", "amount", "pct_chg"])
    out = out.dropna(subset=["symbol", "date"])
    out = out.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last")

    bad_high_low = out["high"].notna() & out["low"].notna() & (out["high"] < out["low"])
    if bad_high_low.any():
        high = out.loc[bad_high_low, "high"].copy()
        low = out.loc[bad_high_low, "low"].copy()
        out.loc[bad_high_low, "high"] = low
        out.loc[bad_high_low, "low"] = high

    if out["pct_chg"].isna().mean() > 0.9:
        out["pct_chg"] = out.groupby("symbol")["close"].pct_change() * 100.0

    return out[DAILY_COLUMNS].reset_index(drop=True)


def winsorize_series(s: pd.Series, lower_q: float = 0.005, upper_q: float = 0.995) -> pd.Series:
    if s.dropna().empty:
        return s
    lower, upper = s.quantile(lower_q), s.quantile(upper_q)
    return s.clip(lower=lower, upper=upper)


def zscore_flags(s: pd.Series, thresh: float = 3.0) -> pd.Series:
    mean, std = s.mean(), s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(False, index=s.index)
    return ((s - mean) / std).abs() > thresh


def iqr_flags(s: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or np.isnan(iqr):
        return pd.Series(False, index=s.index)
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


def quality_profile(df: pd.DataFrame) -> dict[str, Any]:
    out = rename_any_columns(df)
    profile: dict[str, Any] = {
        "shape": [int(out.shape[0]), int(out.shape[1])],
        "duplicates": int(out.duplicated().sum()),
        "columns": list(out.columns),
    }
    if "symbol" in out.columns:
        profile["symbols"] = int(out["symbol"].nunique(dropna=True))
    if "date" in out.columns:
        dates = pd.to_datetime(out["date"], errors="coerce")
        profile["date_range"] = [
            None if pd.isna(dates.min()) else str(dates.min().date()),
            None if pd.isna(dates.max()) else str(dates.max().date()),
        ]
    if {"symbol", "date"}.issubset(out.columns):
        profile["duplicate_symbol_date"] = int(out.duplicated(["symbol", "date"]).sum())
    for col in ["open", "high", "low", "close"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    price_cols = [col for col in ["open", "high", "low", "close"] if col in out.columns]
    if price_cols:
        profile["negative_price_rows"] = int((out[price_cols] < 0).any(axis=1).sum())
    if {"high", "low"}.issubset(out.columns):
        profile["high_lt_low_rows"] = int((out["high"] < out["low"]).sum())
    profile["na_ratio"] = {col: float(val) for col, val in out.isna().mean().sort_values(ascending=False).items()}
    profile["dtypes"] = {col: str(dtype) for col, dtype in out.dtypes.items()}
    return profile


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def engineer_features(
    df: pd.DataFrame,
    *,
    winsorize: bool = True,
    cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean daily prices and add technical indicators."""
    cfg = cfg or {}
    before = quality_profile(df)
    out = clean_daily_prices(df)
    out["ret"] = out.groupby("symbol")["close"].pct_change()
    prev_close = out.groupby("symbol")["close"].shift(1)
    valid_log = (out["close"] > 0) & (prev_close > 0)
    out["log_ret"] = np.nan
    out.loc[valid_log, "log_ret"] = np.log(out.loc[valid_log, "close"] / prev_close.loc[valid_log])
    out["flag_z"] = zscore_flags(out["ret"], float(cfg.get("z_thresh", 3.0)))
    out["flag_iqr"] = iqr_flags(out["ret"], float(cfg.get("iqr_k", 1.5)))

    if winsorize:
        out["ret_w"] = winsorize_series(
            out["ret"],
            float(cfg.get("winsor_lower_q", 0.005)),
            float(cfg.get("winsor_upper_q", 0.995)),
        )
    else:
        out["ret_w"] = out["ret"]

    frames: list[pd.DataFrame] = []
    for _, group in out.groupby("symbol", sort=False):
        g = group.sort_values("date").copy()
        g["ma_5"] = g["close"].rolling(int(cfg.get("ma_short", 5)), min_periods=int(cfg.get("ma_short", 5))).mean()
        g["ma_20"] = g["close"].rolling(int(cfg.get("ma_long", 20)), min_periods=int(cfg.get("ma_long", 20))).mean()
        ema_fast = _ema(g["close"], int(cfg.get("ema_fast", 12)))
        ema_slow = _ema(g["close"], int(cfg.get("ema_slow", 26)))
        macd = ema_fast - ema_slow
        g["ema12"] = ema_fast
        g["ema26"] = ema_slow
        g["macd"] = macd
        g["macd_signal"] = _ema(macd, int(cfg.get("ema_signal", 9)))
        g["vol_20"] = g["log_ret"].rolling(int(cfg.get("vol_win", 20)), min_periods=int(cfg.get("vol_win", 20))).std(
            ddof=0
        ) * np.sqrt(252)
        g["rsi_14"] = _rsi(g["close"], int(cfg.get("rsi_period", 14)))
        g["abs_ret"] = g["ret_w"].abs()
        frames.append(g)

    final = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    stats = {
        "before": before,
        "after": quality_profile(final),
        "z_flags": int(final["flag_z"].sum()),
        "iqr_flags": int(final["flag_iqr"].sum()),
        "ret_stats": {key: float(val) for key, val in final["ret"].describe().items() if pd.notna(val)},
    }
    return final, stats


def plot_eda(df: pd.DataFrame, out_dir: Path, cfg: dict[str, Any]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    plots_n = int(cfg.get("ts_plots_n", 3))

    path = out_dir / "hist_returns.png"
    plt.figure(figsize=(7, 4))
    df["ret_w"].dropna().hist(bins=80)
    plt.title("Daily Returns")
    plt.xlabel("Return")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    paths.append(path)

    path = out_dir / "boxplot_returns.png"
    plt.figure(figsize=(5, 4))
    plt.boxplot(df["ret_w"].dropna(), orientation="vertical")
    plt.title("Daily Returns Boxplot")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    paths.append(path)

    for idx, symbol in enumerate(df["symbol"].dropna().drop_duplicates().head(plots_n), start=1):
        sample = df[df["symbol"] == symbol].sort_values("date")
        path = out_dir / f"price_series_{idx}.png"
        plt.figure(figsize=(8, 4))
        plt.plot(sample["date"], sample["close"])
        plt.title(f"Close Price - {symbol}")
        plt.xlabel("Date")
        plt.ylabel("Close")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(path)

    path = out_dir / "scatter_vol_return.png"
    sample = df[["vol_20", "ret_w"]].dropna()
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=int(cfg.get("rng_seed", 42)))
    plt.figure(figsize=(6, 4))
    plt.scatter(sample["vol_20"], sample["ret_w"], s=8, alpha=0.4)
    plt.title("Volatility vs Return")
    plt.xlabel("Annualized Volatility (20d)")
    plt.ylabel("Daily Return")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    paths.append(path)

    corr_cols = ["ret_w", "log_ret", "vol_20", "macd", "macd_signal", "ma_5", "ma_20", "rsi_14", "abs_ret"]
    corr_cols = [col for col in corr_cols if col in df.columns]
    corr = df[corr_cols].astype(float).corr(min_periods=100)
    path = out_dir / "corr_heatmap.png"
    plt.figure(figsize=(7, 6))
    image = plt.imshow(corr.values, vmin=-1, vmax=1)
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    paths.append(path)

    return paths


def write_quality_report(stats: dict[str, Any], source_path: Path, output_path: Path, cfg: dict[str, Any]) -> None:
    after = stats.get("after", {})
    before = stats.get("before", {})
    ret_stats = stats.get("ret_stats", {})
    lines = [
        "# Data Quality and Feature Engineering Report",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Source: `{source_path}`",
        "",
        "## Input",
        f"- Shape: {before.get('shape')}",
        f"- Symbols: {before.get('symbols')}",
        f"- Date range: {before.get('date_range')}",
        f"- Duplicate rows: {before.get('duplicates')}",
        f"- Duplicate symbol/date rows: {before.get('duplicate_symbol_date')}",
        f"- Negative price rows: {before.get('negative_price_rows')}",
        f"- high < low rows: {before.get('high_lt_low_rows')}",
        "",
        "## Output",
        f"- Shape: {after.get('shape')}",
        f"- Symbols: {after.get('symbols')}",
        f"- Date range: {after.get('date_range')}",
        f"- Z-score flags: {stats.get('z_flags')}",
        f"- IQR flags: {stats.get('iqr_flags')}",
        "",
        "## Return Statistics",
        f"- count: {ret_stats.get('count')}",
        f"- mean: {ret_stats.get('mean')}",
        f"- std: {ret_stats.get('std')}",
        f"- min: {ret_stats.get('min')}",
        f"- max: {ret_stats.get('max')}",
        "",
        "## Parameters",
        "```json",
        json.dumps(cfg, indent=2, ensure_ascii=False),
        "```",
    ]
    atomic_write_text(output_path, "\n".join(lines), encoding="utf-8")


def input_daily_candidates(paths: PipelinePaths) -> list[Path]:
    return [
        paths.output_processed_dir / "daily_price_panel.csv",
        paths.output_processed_dir / "daily_price_50.csv",
        paths.input_processed_dir / "daily_price_panel.csv",
        paths.input_processed_dir / "daily_price_50.csv",
        paths.input_processed_dir / "daily_price.csv",
        paths.input_raw_dir / "daily_price.csv",
    ]


def run_cleaning(cfg: dict[str, Any]) -> dict[str, Path]:
    paths = PipelinePaths.from_config(cfg)
    paths.ensure_output_dirs()
    cleaning_cfg = cfg.get("cleaning", {})
    source_path = first_existing(input_daily_candidates(paths))
    daily = read_csv(source_path)
    cleaned_daily = clean_daily_prices(daily)
    tech, stats = engineer_features(cleaned_daily, winsorize=True, cfg=cleaning_cfg)

    daily_panel_path = paths.output_processed_dir / "daily_price_panel.csv"
    compat_daily_path = paths.output_processed_dir / "daily_price_50.csv"
    tech_path = paths.output_processed_dir / "tech_indicators.csv"
    report_path = paths.output_dir / "QUALITY_REPORT.md"
    steps_path = paths.output_dir / "CLEANING_STEPS.json"

    atomic_write_csv(cleaned_daily, daily_panel_path, index=False, encoding="utf-8-sig")
    atomic_write_csv(cleaned_daily, compat_daily_path, index=False, encoding="utf-8-sig")
    atomic_write_csv(tech, tech_path, index=False, encoding="utf-8-sig")
    plot_eda(tech, paths.output_eda_dir, cleaning_cfg)
    write_quality_report(stats, source_path, report_path, cleaning_cfg)
    atomic_write_json(
        steps_path,
        {
            "config": cleaning_cfg,
            "source": str(source_path),
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    return {
        "daily_panel": daily_panel_path,
        "compat_daily": compat_daily_path,
        "tech_indicators": tech_path,
        "quality_report": report_path,
        "steps": steps_path,
    }
