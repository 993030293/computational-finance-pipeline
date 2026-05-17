from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from cfpipeline.acquire import fetch_symbol_data, symbol_cache_path, write_fetch_report, write_symbol_cache
from cfpipeline.config import DEFAULT_CONFIG


def _fetch_cfg(**overrides: Any) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG["fetch"])
    cfg.update(
        {
            "start_date": "20200101",
            "end_date": "20200110",
            "sleep_each": 0.0,
            "rate_limit_per_second": None,
            "checkpoint_every": 1,
            "max_workers": 1,
        }
    )
    cfg.update(overrides)
    return cfg


def _daily_frame(symbol: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [symbol],
            "date": [pd.Timestamp("2020-01-02")],
            "open": [10.0],
            "high": [10.5],
            "low": [9.5],
            "close": [10.2],
            "volume": [1000],
            "amount": [10200],
            "pct_chg": [0.0],
        }
    )


def test_fetch_cache_hit_skips_external_fetch(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    checkpoint_path = tmp_path / "checkpoint.csv"
    write_symbol_cache(_daily_frame("000001"), symbol_cache_path(cache_dir, "000001"))

    def fetch_func(*args: Any, **kwargs: Any) -> pd.DataFrame:
        raise AssertionError("external fetch should not be called on cache hit")

    frames, summary = fetch_symbol_data(
        None,
        ["000001"],
        _fetch_cfg(use_cache=True, resume=False),
        cache_dir,
        checkpoint_path,
        fetch_func=fetch_func,
    )

    assert len(frames) == 1
    assert summary["cache_hits"] == 1
    assert summary["succeeded"] == 1
    checkpoint = pd.read_csv(checkpoint_path)
    assert checkpoint.loc[0, "status"] == "succeeded"


def test_single_symbol_failure_keeps_successes(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    checkpoint_path = tmp_path / "checkpoint.csv"

    def fetch_func(ak: Any, symbol: str, start_date: str, end_date: str, **kwargs: Any) -> pd.DataFrame:
        if symbol == "000002":
            raise RuntimeError("temporary upstream failure")
        return _daily_frame(symbol)

    frames, summary = fetch_symbol_data(
        None,
        ["000001", "000002"],
        _fetch_cfg(use_cache=False, resume=False),
        cache_dir,
        checkpoint_path,
        fetch_func=fetch_func,
    )

    assert len(frames) == 1
    assert summary["succeeded"] == 1
    assert summary["failed"] == 1
    assert summary["failed_symbols"] == [{"symbol": "000002", "error": "temporary upstream failure"}]
    checkpoint = pd.read_csv(checkpoint_path).set_index("symbol")
    assert checkpoint.loc[1, "status"] == "succeeded"
    assert checkpoint.loc[2, "status"] == "failed"


def test_resume_retries_only_failed_and_pending_symbols(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    checkpoint_path = tmp_path / "checkpoint.csv"
    write_symbol_cache(_daily_frame("000001"), symbol_cache_path(cache_dir, "000001"))
    pd.DataFrame(
        [
            {
                "symbol": "000001",
                "status": "succeeded",
                "error": "",
                "rows": "1",
                "cache_path": str(symbol_cache_path(cache_dir, "000001")),
                "updated_at": "2026-01-01 00:00:00+0800",
            },
            {
                "symbol": "000002",
                "status": "failed",
                "error": "old failure",
                "rows": "0",
                "cache_path": "",
                "updated_at": "2026-01-01 00:00:00+0800",
            },
        ]
    ).to_csv(checkpoint_path, index=False)
    calls: list[str] = []

    def fetch_func(ak: Any, symbol: str, start_date: str, end_date: str, **kwargs: Any) -> pd.DataFrame:
        calls.append(symbol)
        return _daily_frame(symbol)

    frames, summary = fetch_symbol_data(
        None,
        ["000001", "000002", "000003"],
        _fetch_cfg(use_cache=True, resume=True),
        cache_dir,
        checkpoint_path,
        fetch_func=fetch_func,
    )

    assert calls == ["000002", "000003"]
    assert len(frames) == 3
    assert summary["cache_hits"] == 1
    assert summary["succeeded"] == 3
    checkpoint = pd.read_csv(checkpoint_path).set_index("symbol")
    assert checkpoint["status"].to_dict() == {1: "succeeded", 2: "succeeded", 3: "succeeded"}


def test_max_workers_one_preserves_sequential_fetch_order(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    checkpoint_path = tmp_path / "checkpoint.csv"
    calls: list[str] = []

    def fetch_func(ak: Any, symbol: str, start_date: str, end_date: str, **kwargs: Any) -> pd.DataFrame:
        calls.append(symbol)
        return _daily_frame(symbol)

    frames, summary = fetch_symbol_data(
        None,
        ["000001", "000002", "000003"],
        _fetch_cfg(use_cache=False, resume=False, max_workers=1),
        cache_dir,
        checkpoint_path,
        fetch_func=fetch_func,
    )

    assert calls == ["000001", "000002", "000003"]
    assert len(frames) == 3
    assert summary["symbols_requested"] == 3
    assert summary["cache_hits"] == 0


def test_fetch_report_contains_audit_summary(tmp_path: Path) -> None:
    path = write_fetch_report(
        tmp_path,
        {
            "run_id": "20260517-120000",
            "symbols_total": 2,
            "symbols_requested": 1,
            "succeeded": 1,
            "failed": 1,
            "cache_hits": 1,
            "cache_hit_rate": 0.5,
            "elapsed_seconds": 1.25,
            "max_workers": 1,
            "rate_limit_per_second": None,
            "sleep_each": 0.0,
            "cache_dir": str(tmp_path / "cache"),
            "checkpoint_path": str(tmp_path / "checkpoint.csv"),
            "failed_symbols": [{"symbol": "000002", "error": "temporary upstream failure"}],
        },
    )

    text = path.read_text(encoding="utf-8")
    assert "Cache hit rate: 50.00%" in text
    assert "`000002`: temporary upstream failure" in text
