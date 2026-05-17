from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

PACKAGE_NAME = "computational-finance-pipeline"
RUN_ID_FORMAT = "%Y%m%d_%H%M%S"


def now_iso() -> str:
    return datetime.now(tz=timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def make_run_id() -> str:
    return datetime.now(tz=timezone(timedelta(hours=8))).strftime(RUN_ID_FORMAT)


def atomic_write(path: str | Path, writer: Callable[[Path], None]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        writer(tmp_path)
        tmp_path.replace(target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return target


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    def writer(tmp_path: Path) -> None:
        tmp_path.write_text(text, encoding=encoding)

    return atomic_write(path, writer)


def atomic_write_json(path: str | Path, data: Any, *, indent: int = 2) -> Path:
    text = json.dumps(data, indent=indent, ensure_ascii=False, default=str)
    return atomic_write_text(path, text + "\n", encoding="utf-8")


def atomic_write_csv(frame: pd.DataFrame | pd.Series, path: str | Path, **kwargs: Any) -> Path:
    return atomic_write(path, lambda tmp_path: frame.to_csv(tmp_path, **kwargs))


def package_version() -> str:
    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return "0.1.0"


def git_sha(cwd: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(cwd or "."),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def versioned_run_dir(configured_output_dir: str | Path, run_id: str) -> tuple[Path, Path]:
    configured = Path(configured_output_dir)
    root = configured.parent if configured.name else Path("outputs")
    if root == Path("."):
        root = Path("outputs")
    return root / "runs" / run_id, root / "latest"


def _copytree_atomic(source_dir: Path, latest_dir: Path) -> None:
    latest_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = latest_dir.with_name(f".{latest_dir.name}.tmp-{os.getpid()}-{time.time_ns()}")
    backup_dir = latest_dir.with_name(f".{latest_dir.name}.backup-{os.getpid()}-{time.time_ns()}")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    shutil.copytree(source_dir, tmp_dir)
    if latest_dir.exists() or latest_dir.is_symlink():
        latest_dir.replace(backup_dir) if latest_dir.is_file() or latest_dir.is_symlink() else latest_dir.rename(
            backup_dir
        )
    tmp_dir.rename(latest_dir)
    if backup_dir.exists():
        if backup_dir.is_dir():
            shutil.rmtree(backup_dir)
        else:
            backup_dir.unlink()


def publish_latest(run_dir: str | Path, latest_dir: str | Path) -> Path:
    """Publish a Windows-compatible latest directory by copying the finished run."""
    source = Path(run_dir)
    target = Path(latest_dir)
    _copytree_atomic(source, target)
    pointer = target.parent / "LATEST_RUN.json"
    atomic_write_json(
        pointer,
        {
            "run_id": source.name,
            "run_dir": str(source),
            "latest_dir": str(target),
            "updated_at": now_iso(),
        },
    )
    return target


def _path_info(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    }


def collect_output_files(outputs: dict[str, Any]) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for key, value in outputs.items():
        if isinstance(value, (str, Path)):
            path = Path(value)
            files.append({"key": key, **_path_info(path)})
    return files


def collect_input_files(cfg: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    output_dir = Path(cfg.get("data", {}).get("output_dir", "outputs/latest"))
    input_dir = Path(cfg.get("data", {}).get("input_dir", "data"))
    patterns_by_stage = {
        "fetch": [],
        "clean": [
            input_dir / "processed" / "daily_price_panel.csv",
            input_dir / "processed" / "daily_price_50.csv",
            input_dir / "processed" / "daily_price.csv",
            input_dir / "raw" / "daily_price.csv",
            output_dir / "raw" / "daily_price.csv",
        ],
        "factors": [
            output_dir / "processed" / "tech_indicators.csv",
            output_dir / "processed" / "daily_price_panel.csv",
            input_dir / "processed" / "tech_indicators.csv",
            input_dir / "processed" / "daily_price_panel.csv",
            input_dir / "processed" / "daily_price_50.csv",
        ],
        "backtest": [
            output_dir / "project4" / "factors.csv",
            input_dir / "project4" / "factors.csv",
            output_dir / "processed" / "daily_price_panel.csv",
            input_dir / "processed" / "daily_price_panel.csv",
        ],
        "ml": [
            output_dir / "project4" / "factors.csv",
            input_dir / "project4" / "factors.csv",
        ],
        "decision": [
            output_dir / "project4" / "factors.csv",
            output_dir / "ml" / "ml_predictions.csv",
        ],
        "tune": [
            output_dir / "project4" / "factors.csv",
        ],
        "stress": [
            output_dir / "project4" / "factors.csv",
            output_dir / "processed" / "daily_price_panel.csv",
        ],
        "benchmarks": [
            output_dir / "proj5_output" / "performance_metrics.csv",
            output_dir / "proj5_output" / "performance_metrics_net.csv",
            output_dir / "ml" / "ml_model_metrics.csv",
            output_dir / "ml" / "ml_validation_comparison.csv",
            output_dir / "decision" / "decision_metrics.csv",
            output_dir / "tuning" / "test_performance.csv",
            output_dir / "stress" / "market_stress_metrics.csv",
        ],
    }
    return [_path_info(path) for path in patterns_by_stage.get(stage, []) if path.exists()]


def summarize_metrics(outputs: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in outputs.items():
        if not isinstance(value, (str, Path)):
            continue
        path = Path(value)
        if not path.exists() or path.suffix.lower() != ".csv" or "metrics" not in path.name.lower():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        summary[key] = {
            "path": str(path),
            "rows": int(len(df)),
            "columns": list(df.columns),
        }
    return summary


def create_manifest(
    *,
    run_id: str,
    output_dir: str | Path,
    command: str,
    resolved_config_path: str | Path,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "started_at": now_iso(),
        "finished_at": None,
        "git_sha": git_sha(),
        "command": command,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "package_version": package_version(),
        "resolved_config_path": str(resolved_config_path),
        "input_files": [],
        "output_files": [],
        "stage_status": {},
        "metrics_summary": {},
        "output_dir": str(output_dir),
    }


def manifest_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "run_manifest.json"


def write_manifest(output_dir: str | Path, manifest: dict[str, Any]) -> Path:
    return atomic_write_json(manifest_path(output_dir), manifest)


def mark_stage_started(manifest: dict[str, Any], cfg: dict[str, Any], stage: str) -> None:
    manifest["stage_status"][stage] = {
        "status": "running",
        "started_at": now_iso(),
        "finished_at": None,
        "elapsed_seconds": None,
        "input_files": collect_input_files(cfg, stage),
        "output_files": [],
        "error": None,
    }


def mark_stage_finished(
    manifest: dict[str, Any],
    stage: str,
    outputs: dict[str, Any],
    *,
    elapsed_seconds: float,
    status: str = "succeeded",
    error: str | None = None,
) -> None:
    stage_entry = manifest["stage_status"].setdefault(stage, {})
    output_files = collect_output_files(outputs)
    stage_entry.update(
        {
            "status": status,
            "finished_at": now_iso(),
            "elapsed_seconds": float(elapsed_seconds),
            "output_files": output_files,
            "error": error,
        }
    )
    manifest["input_files"] = _dedupe_file_records(
        [record for entry in manifest["stage_status"].values() for record in entry.get("input_files", [])]
    )
    manifest["output_files"] = _dedupe_file_records(
        [record for entry in manifest["stage_status"].values() for record in entry.get("output_files", [])]
    )
    manifest["metrics_summary"].update(summarize_metrics(outputs))


def mark_stage_skipped(manifest: dict[str, Any], stage: str, reason: str) -> None:
    manifest["stage_status"][stage] = {
        "status": "skipped",
        "started_at": now_iso(),
        "finished_at": now_iso(),
        "elapsed_seconds": 0.0,
        "input_files": [],
        "output_files": [],
        "error": None,
        "reason": reason,
    }


def finalize_manifest(manifest: dict[str, Any], *, status: str = "succeeded") -> None:
    manifest["finished_at"] = now_iso()
    manifest["status"] = status


def _dedupe_file_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for record in records:
        path = str(record.get("path", ""))
        if not path or path in seen:
            continue
        seen.add(path)
        deduped.append(record)
    return deduped
