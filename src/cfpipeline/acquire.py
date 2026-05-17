from __future__ import annotations

import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd
from requests.exceptions import ConnectTimeout, ProxyError, ReadTimeout

from .artifacts import atomic_write_csv, atomic_write_text
from .cleaning import clean_daily_prices, rename_any_columns
from .paths import PipelinePaths

RETRY_EXCEPTIONS = (ProxyError, ReadTimeout, ConnectTimeout, ConnectionError, TimeoutError)
CHECKPOINT_COLUMNS = ["symbol", "status", "error", "rows", "cache_path", "updated_at"]


@dataclass(frozen=True)
class SymbolFetchResult:
    symbol: str
    status: str
    frame: pd.DataFrame
    cache_hit: bool = False
    error: str = ""
    cache_path: Path | None = None
    elapsed_seconds: float = 0.0


class RateLimiter:
    """Thread-safe limiter that spaces external request starts."""

    def __init__(self, *, sleep_each: float = 0.0, rate_limit_per_second: float | None = None) -> None:
        if rate_limit_per_second is not None:
            self.interval = 1.0 / max(float(rate_limit_per_second), 1e-9)
        else:
            self.interval = max(float(sleep_each), 0.0)
        self._lock = Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait_until = max(now, self._next_allowed)
            self._next_allowed = wait_until + self.interval
        delay = wait_until - now
        if delay > 0:
            time.sleep(delay)


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
    atomic_write_csv(
        df[["code", "name", "symbol", "update_time"]],
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


def _normalize_symbol(symbol: str) -> str:
    return str(symbol).zfill(6)


def resolve_cache_dir(paths: PipelinePaths, fetch_cfg: dict[str, Any]) -> Path:
    configured = Path(str(fetch_cfg.get("cache_dir", "cache/daily_prices")))
    cache_dir = configured if configured.is_absolute() else paths.output_dir / configured
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def symbol_cache_path(cache_dir: Path, symbol: str) -> Path:
    safe_symbol = "".join(ch for ch in _normalize_symbol(symbol) if ch.isalnum() or ch in {"_", "-"})
    return cache_dir / f"{safe_symbol}.csv"


def read_symbol_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        cached = pd.read_csv(path)
    except Exception:
        return None
    if cached.empty:
        return None
    if "date" in cached.columns:
        cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
    if "symbol" in cached.columns:
        cached["symbol"] = cached["symbol"].astype(str).str.zfill(6)
    return cached


def write_symbol_cache(df: pd.DataFrame, path: Path) -> None:
    atomic_write_csv(df, path, index=False, encoding="utf-8-sig")


def load_fetch_checkpoint(path: Path, symbols: list[str]) -> dict[str, dict[str, str]]:
    checkpoint = {
        _normalize_symbol(symbol): {
            "symbol": _normalize_symbol(symbol),
            "status": "pending",
            "error": "",
            "rows": "0",
            "cache_path": "",
            "updated_at": "",
        }
        for symbol in symbols
    }
    if not path.exists():
        return checkpoint

    try:
        existing = pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return checkpoint

    for _, row in existing.iterrows():
        symbol = _normalize_symbol(str(row.get("symbol", "")))
        if symbol not in checkpoint:
            continue
        status = str(row.get("status", "pending"))
        checkpoint[symbol] = {
            "symbol": symbol,
            "status": "pending" if status == "running" else status,
            "error": str(row.get("error", "")),
            "rows": str(row.get("rows", "0")),
            "cache_path": str(row.get("cache_path", "")),
            "updated_at": str(row.get("updated_at", "")),
        }
    return checkpoint


def write_fetch_checkpoint(path: Path, checkpoint: dict[str, dict[str, str]], symbols: list[str]) -> None:
    rows = [checkpoint[_normalize_symbol(symbol)] for symbol in symbols]
    df = pd.DataFrame(rows, columns=CHECKPOINT_COLUMNS)
    atomic_write_csv(df, path, index=False, encoding="utf-8-sig")


def _now_text() -> str:
    return datetime.now(tz=timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S%z")


def _checkpoint_row(
    symbol: str,
    status: str,
    *,
    error: str = "",
    rows: int = 0,
    cache_path: Path | None = None,
) -> dict[str, str]:
    return {
        "symbol": _normalize_symbol(symbol),
        "status": status,
        "error": error,
        "rows": str(rows),
        "cache_path": str(cache_path or ""),
        "updated_at": _now_text(),
    }


def _fetch_one_symbol(
    ak: Any,
    symbol: str,
    fetch_cfg: dict[str, Any],
    cache_dir: Path,
    limiter: RateLimiter,
    fetch_func=fetch_daily_with_retries,
) -> SymbolFetchResult:
    started = time.monotonic()
    normalized = _normalize_symbol(symbol)
    cache_path = symbol_cache_path(cache_dir, normalized)
    use_cache = bool(fetch_cfg.get("use_cache", True))

    if use_cache:
        cached = read_symbol_cache(cache_path)
        if cached is not None:
            return SymbolFetchResult(
                symbol=normalized,
                status="succeeded",
                frame=cached,
                cache_hit=True,
                cache_path=cache_path,
                elapsed_seconds=time.monotonic() - started,
            )

    limiter.wait()
    try:
        df = fetch_func(
            ak,
            normalized,
            str(fetch_cfg.get("start_date", "20200101")),
            str(fetch_cfg.get("end_date")),
            adjust=str(fetch_cfg.get("adjust", "qfq")),
            max_retries=int(fetch_cfg.get("max_retries", 4)),
            base_delay=float(fetch_cfg.get("base_delay", 1.0)),
        )
        if df is None or df.empty:
            return SymbolFetchResult(
                symbol=normalized,
                status="failed",
                frame=pd.DataFrame(),
                error="empty data",
                cache_path=cache_path,
                elapsed_seconds=time.monotonic() - started,
            )
        out = rename_any_columns(df)
        out["symbol"] = normalized
        if use_cache:
            write_symbol_cache(out, cache_path)
        return SymbolFetchResult(
            symbol=normalized,
            status="succeeded",
            frame=out,
            cache_path=cache_path,
            elapsed_seconds=time.monotonic() - started,
        )
    except Exception as exc:
        return SymbolFetchResult(
            symbol=normalized,
            status="failed",
            frame=pd.DataFrame(),
            error=str(exc),
            cache_path=cache_path,
            elapsed_seconds=time.monotonic() - started,
        )


def fetch_symbol_data(
    ak: Any,
    symbols: list[str],
    fetch_cfg: dict[str, Any],
    cache_dir: Path,
    checkpoint_path: Path,
    *,
    fetch_func=fetch_daily_with_retries,
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    run_id = datetime.now(tz=timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    started = time.monotonic()
    normalized_symbols = [_normalize_symbol(symbol) for symbol in symbols]
    checkpoint = load_fetch_checkpoint(checkpoint_path, normalized_symbols)
    use_cache = bool(fetch_cfg.get("use_cache", True))
    resume = bool(fetch_cfg.get("resume", True))
    max_workers = max(1, int(fetch_cfg.get("max_workers", 1)))
    checkpoint_every = max(1, int(fetch_cfg.get("checkpoint_every", 25)))
    rate_limit = fetch_cfg.get("rate_limit_per_second")
    rate_limit_value = None if rate_limit is None else float(rate_limit)
    limiter = RateLimiter(sleep_each=float(fetch_cfg.get("sleep_each", 0.5)), rate_limit_per_second=rate_limit_value)

    frames: list[pd.DataFrame] = []
    failed_symbols: list[dict[str, str]] = []
    cache_hits = 0
    to_fetch: list[str] = []

    for symbol in normalized_symbols:
        row = checkpoint[symbol]
        cache_path = symbol_cache_path(cache_dir, symbol)
        if resume and row.get("status") == "succeeded" and use_cache:
            cached = read_symbol_cache(cache_path)
            if cached is not None:
                frames.append(cached)
                cache_hits += 1
                checkpoint[symbol] = _checkpoint_row(symbol, "succeeded", rows=len(cached), cache_path=cache_path)
                continue
        to_fetch.append(symbol)
        checkpoint[symbol] = _checkpoint_row(symbol, "running", cache_path=cache_path)

    write_fetch_checkpoint(checkpoint_path, checkpoint, normalized_symbols)

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_one_symbol, ak, symbol, fetch_cfg, cache_dir, limiter, fetch_func): symbol
            for symbol in to_fetch
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = SymbolFetchResult(
                    symbol=symbol,
                    status="failed",
                    frame=pd.DataFrame(),
                    error=str(exc),
                    cache_path=symbol_cache_path(cache_dir, symbol),
                )

            completed += 1
            if result.status == "succeeded":
                frames.append(result.frame)
                cache_hits += int(result.cache_hit)
                checkpoint[result.symbol] = _checkpoint_row(
                    result.symbol,
                    "succeeded",
                    rows=len(result.frame),
                    cache_path=result.cache_path,
                )
            else:
                failed_symbols.append({"symbol": result.symbol, "error": result.error})
                checkpoint[result.symbol] = _checkpoint_row(
                    result.symbol,
                    "failed",
                    error=result.error,
                    cache_path=result.cache_path,
                )

            if completed % checkpoint_every == 0:
                write_fetch_checkpoint(checkpoint_path, checkpoint, normalized_symbols)

    write_fetch_checkpoint(checkpoint_path, checkpoint, normalized_symbols)

    succeeded = sum(1 for row in checkpoint.values() if row.get("status") == "succeeded")
    failed = sum(1 for row in checkpoint.values() if row.get("status") == "failed")
    summary = {
        "run_id": run_id,
        "symbols_total": len(normalized_symbols),
        "symbols_requested": len(to_fetch),
        "succeeded": succeeded,
        "failed": failed,
        "cache_hits": cache_hits,
        "cache_hit_rate": cache_hits / len(normalized_symbols) if normalized_symbols else 0.0,
        "elapsed_seconds": time.monotonic() - started,
        "failed_symbols": failed_symbols,
        "max_workers": max_workers,
        "rate_limit_per_second": rate_limit_value,
        "sleep_each": float(fetch_cfg.get("sleep_each", 0.5)),
        "checkpoint_path": str(checkpoint_path),
        "cache_dir": str(cache_dir),
    }
    return frames, summary


def write_fetch_report(output_dir: Path, summary: dict[str, Any]) -> Path:
    report_path = output_dir / "FETCH_REPORT.md"
    failed_symbols = summary.get("failed_symbols", [])
    failed_lines = (
        [f"- `{item['symbol']}`: {item.get('error', '')}" for item in failed_symbols] if failed_symbols else ["- None"]
    )
    lines = [
        "# Fetch Report",
        "",
        f"- Run ID: `{summary.get('run_id')}`",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Symbol total: {summary.get('symbols_total')}",
        f"- Symbols requested from API/cache worker: {summary.get('symbols_requested')}",
        f"- Succeeded: {summary.get('succeeded')}",
        f"- Failed: {summary.get('failed')}",
        f"- Cache hits: {summary.get('cache_hits')}",
        f"- Cache hit rate: {float(summary.get('cache_hit_rate', 0.0)):.2%}",
        f"- Elapsed seconds: {float(summary.get('elapsed_seconds', 0.0)):.2f}",
        f"- Max workers: {summary.get('max_workers')}",
        f"- Rate limit per second: {summary.get('rate_limit_per_second')}",
        f"- Sleep each: {summary.get('sleep_each')}",
        f"- Cache dir: `{summary.get('cache_dir')}`",
        f"- Checkpoint: `{summary.get('checkpoint_path')}`",
        "",
        "## Failed Symbols",
        *failed_lines,
    ]
    atomic_write_text(report_path, "\n".join(lines), encoding="utf-8")
    return report_path


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
    atomic_write_text(report_path, "\n".join(lines), encoding="utf-8")
    return report_path


def run_fetch(cfg: dict[str, Any]) -> dict[str, Path]:
    disable_proxies()
    ak = _load_akshare()
    paths = PipelinePaths.from_config(cfg)
    paths.ensure_output_dirs()
    fetch_cfg = dict(cfg.get("fetch", {}))

    end_date = fetch_cfg.get("end_date")
    if not end_date:
        end_date = datetime.now(tz=timezone(timedelta(hours=8))).strftime("%Y%m%d")
    fetch_cfg["end_date"] = end_date

    db_path = paths.output_dir / "database" / "financial_data.db"
    universe = get_stock_universe(ak, paths.output_raw_dir, db_path)
    symbols = (
        universe["symbol"].astype(str).dropna().drop_duplicates().head(int(fetch_cfg.get("final_n", 300))).tolist()
    )

    cache_dir = resolve_cache_dir(paths, fetch_cfg)
    checkpoint_path = paths.output_dir / "fetch_checkpoint.csv"
    frames, fetch_summary = fetch_symbol_data(ak, symbols, fetch_cfg, cache_dir, checkpoint_path)
    fetch_report_path = write_fetch_report(paths.output_dir, fetch_summary)

    if not frames:
        raise RuntimeError("No daily price data was downloaded.")

    raw = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    raw_path = paths.output_raw_dir / "daily_price.csv"
    atomic_write_csv(raw, raw_path, index=False, encoding="utf-8-sig")
    save_to_database(raw, db_path, "daily_price_raw")

    cleaned = clean_daily_prices(raw)
    panel_path = paths.output_processed_dir / "daily_price_panel.csv"
    compat_path = paths.output_processed_dir / "daily_price_50.csv"
    atomic_write_csv(cleaned, panel_path, index=False, encoding="utf-8-sig")
    atomic_write_csv(cleaned, compat_path, index=False, encoding="utf-8-sig")
    save_to_database(cleaned, db_path, "daily_price")

    report_path = write_acquisition_report(paths.output_dir, db_path, cleaned, universe, fetch_cfg)
    return {
        "raw_daily": raw_path,
        "daily_panel": panel_path,
        "compat_daily": compat_path,
        "database": db_path,
        "report": report_path,
        "fetch_report": fetch_report_path,
        "checkpoint": checkpoint_path,
    }
