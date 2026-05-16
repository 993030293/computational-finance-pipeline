from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from requests.exceptions import ConnectTimeout, ProxyError, ReadTimeout

from .cleaning import clean_daily_prices, rename_any_columns
from .paths import PipelinePaths


RETRY_EXCEPTIONS = (ProxyError, ReadTimeout, ConnectTimeout, ConnectionError, TimeoutError)


def disable_proxies() -> None:
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


def code_to_prefixed(symbol: str) -> str:
    value = str(symbol)
    if value.startswith(("000", "001", "002", "003", "300")):
        return "sz" + value
    if value.startswith(("600", "601", "603", "605", "688")):
        return "sh" + value
    return value if value[:2] in {"sh", "sz"} else value


def _load_akshare():
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("akshare is required for `cfp fetch`. Install with `pip install akshare`.") from exc
    return ak


def save_to_database(df: pd.DataFrame, db_path: Path, table_name: str) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    data = df.copy()
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(db_path) as conn:
        data.to_sql(table_name, conn, if_exists="replace", index=False, chunksize=1000)


def get_stock_universe(ak: Any, output_raw_dir: Path, db_path: Path) -> pd.DataFrame:
    df = ak.stock_info_a_code_name()
    if df is None or df.empty:
        raise RuntimeError("AkShare returned an empty stock universe.")

    df = rename_any_columns(df)
    if "symbol" not in df.columns and "code" in df.columns:
        df["symbol"] = df["code"].astype(str)
    if "code" not in df.columns:
        df["code"] = df["symbol"].astype(str)
    if "name" not in df.columns:
        df["name"] = ""
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["update_time"] = datetime.now()

    output_raw_dir.mkdir(parents=True, exist_ok=True)
    df[["code", "name", "symbol", "update_time"]].to_csv(
        output_raw_dir / "stock_universe.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_to_database(df[["code", "name", "symbol", "update_time"]], db_path, "stock_universe")
    return df[["symbol", "code", "name", "update_time"]]


def _fetch_daily_primary(ak: Any, symbol: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
    return ak.stock_zh_a_hist(symbol=symbol, start_date=start_date, end_date=end_date, adjust=adjust)


def _fetch_daily_fallback(ak: Any, symbol: str) -> pd.DataFrame:
    prefixed = code_to_prefixed(symbol)
    if hasattr(ak, "stock_zh_a_daily"):
        try:
            df = ak.stock_zh_a_daily(symbol=prefixed)
            if df is not None and not df.empty:
                return df.reset_index() if "date" not in df.columns else df
        except Exception:
            pass
    for func_name in ("stock_zh_a_daily_tx", "stock_zh_a_daily_qq"):
        if hasattr(ak, func_name):
            try:
                df = getattr(ak, func_name)(symbol=prefixed)
                if df is not None and not df.empty:
                    return df.reset_index() if "date" not in df.columns else df
            except Exception:
                pass
    return pd.DataFrame()


def fetch_daily_with_retries(
    ak: Any,
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    adjust: str = "qfq",
    max_retries: int = 4,
    base_delay: float = 1.0,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            df = _fetch_daily_primary(ak, symbol, start_date, end_date, adjust)
            if df is not None and not df.empty:
                out = rename_any_columns(df)
                out["symbol"] = str(symbol).zfill(6)
                return out
        except RETRY_EXCEPTIONS as exc:
            last_error = exc
            time.sleep(base_delay * (2**attempt))
        except Exception as exc:
            last_error = exc
            break

    fallback = _fetch_daily_fallback(ak, symbol)
    if fallback is not None and not fallback.empty:
        out = rename_any_columns(fallback)
        out["symbol"] = str(symbol).zfill(6)
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        return out[(out["date"] >= start) & (out["date"] <= end)]
    if last_error is not None:
        raise RuntimeError(f"Failed to fetch {symbol}: {last_error}") from last_error
    return pd.DataFrame()


def write_acquisition_report(
    output_dir: Path,
    db_path: Path,
    daily: pd.DataFrame,
    universe: pd.DataFrame,
    cfg: dict[str, Any],
) -> Path:
    report_path = output_dir / "REPORT.md"
    dmin = pd.to_datetime(daily["date"], errors="coerce").min()
    dmax = pd.to_datetime(daily["date"], errors="coerce").max()
    lines = [
        "# Data Acquisition Report",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Output dir: `{output_dir}`",
        f"- Database: `{db_path}`",
        "",
        "## Data Sources",
        "- A-share universe: `ak.stock_info_a_code_name()`",
        "- Daily quotes: `ak.stock_zh_a_hist(..., adjust='qfq')` with fallback daily endpoints",
        "",
        "## Methodology",
        "1. Disable proxies.",
        "2. Pull stock universe.",
        f"3. Select first {cfg.get('final_n', 300)} symbols.",
        "4. Download daily quotes with retries.",
        "5. Standardize columns, clean data, save CSV and SQLite.",
        "",
        "## Output Summary",
        f"- Rows: {len(daily)}",
        f"- Symbols: {daily['symbol'].nunique()}",
        f"- Date range: {dmin:%Y-%m-%d} to {dmax:%Y-%m-%d}",
        f"- Universe rows: {len(universe)}",
        "- Primary CSV: `processed/daily_price_panel.csv`",
        "- Compatibility CSV: `processed/daily_price_50.csv`",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_fetch(cfg: dict[str, Any]) -> dict[str, Path]:
    disable_proxies()
    ak = _load_akshare()
    paths = PipelinePaths.from_config(cfg)
    paths.ensure_output_dirs()
    fetch_cfg = cfg.get("fetch", {})

    end_date = fetch_cfg.get("end_date")
    if not end_date:
        end_date = datetime.now(tz=timezone(timedelta(hours=8))).strftime("%Y%m%d")

    db_path = paths.output_dir / "database" / "financial_data.db"
    universe = get_stock_universe(ak, paths.output_raw_dir, db_path)
    symbols = (
        universe["symbol"]
        .astype(str)
        .dropna()
        .drop_duplicates()
        .head(int(fetch_cfg.get("final_n", 300)))
        .tolist()
    )

    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        df = fetch_daily_with_retries(
            ak,
            symbol,
            str(fetch_cfg.get("start_date", "20200101")),
            str(end_date),
            adjust=str(fetch_cfg.get("adjust", "qfq")),
            max_retries=int(fetch_cfg.get("max_retries", 4)),
            base_delay=float(fetch_cfg.get("base_delay", 1.0)),
        )
        if not df.empty:
            frames.append(df)
        time.sleep(float(fetch_cfg.get("sleep_each", 0.5)))

    if not frames:
        raise RuntimeError("No daily price data was downloaded.")

    raw = pd.concat(frames, ignore_index=True)
    raw_path = paths.output_raw_dir / "daily_price.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    save_to_database(raw, db_path, "daily_price_raw")

    cleaned = clean_daily_prices(raw)
    panel_path = paths.output_processed_dir / "daily_price_panel.csv"
    compat_path = paths.output_processed_dir / "daily_price_50.csv"
    cleaned.to_csv(panel_path, index=False, encoding="utf-8-sig")
    cleaned.to_csv(compat_path, index=False, encoding="utf-8-sig")
    save_to_database(cleaned, db_path, "daily_price")

    report_path = write_acquisition_report(paths.output_dir, db_path, cleaned, universe, fetch_cfg)
    return {
        "raw_daily": raw_path,
        "daily_panel": panel_path,
        "compat_daily": compat_path,
        "database": db_path,
        "report": report_path,
    }
