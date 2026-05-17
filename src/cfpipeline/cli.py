from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path

from .acquire import run_fetch
from .artifacts import (
    create_manifest,
    finalize_manifest,
    make_run_id,
    mark_stage_finished,
    mark_stage_skipped,
    mark_stage_started,
    publish_latest,
    versioned_run_dir,
    write_manifest,
)
from .backtest import run_backtest
from .benchmarks import run_benchmarks
from .cleaning import run_cleaning
from .config import ConfigError, load_config, with_overrides, write_resolved_config
from .decision import run_decision
from .factors import run_factors
from .ml import run_ml
from .stress import run_stress
from .tuning import run_tuning

Runner = Callable[[dict], dict]


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file. Defaults to configs/default.yaml when present, otherwise built-in defaults.",
    )
    parser.add_argument("--data-dir", default=None, help="Input data directory. Defaults to config data.input_dir.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to config data.output_dir.")


def _load(args: argparse.Namespace, *, write_config: bool = True) -> dict:
    try:
        if args.config is None:
            default_path = Path("configs/default.yaml")
            cfg = load_config(default_path if default_path.exists() else None)
        else:
            cfg = load_config(Path(args.config))
        resolved = with_overrides(cfg, args.data_dir, args.output_dir)
        if write_config:
            write_resolved_config(resolved)
        return resolved
    except FileNotFoundError:
        raise
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _print_outputs(outputs: dict) -> None:
    for key, value in outputs.items():
        print(f"{key}: {value}")


def _command_text(args: argparse.Namespace) -> str:
    return str(getattr(args, "_command_text", "cfp"))


def _run_manifested_stage(
    stage: str,
    runner: Runner,
    cfg: dict,
    manifest: dict,
) -> dict:
    mark_stage_started(manifest, cfg, stage)
    write_manifest(cfg["data"]["output_dir"], manifest)
    started = time.monotonic()
    try:
        outputs = runner(cfg)
    except Exception as exc:
        mark_stage_finished(
            manifest,
            stage,
            {},
            elapsed_seconds=time.monotonic() - started,
            status="failed",
            error=str(exc),
        )
        write_manifest(cfg["data"]["output_dir"], manifest)
        raise
    mark_stage_finished(manifest, stage, outputs, elapsed_seconds=time.monotonic() - started)
    write_manifest(cfg["data"]["output_dir"], manifest)
    return outputs


def _run_stage(args: argparse.Namespace, runner: Runner) -> int:
    cfg = _load(args)
    output_dir = Path(cfg["data"]["output_dir"])
    resolved_config_path = output_dir / "resolved_config.yaml"
    run_id = make_run_id()
    manifest = create_manifest(
        run_id=run_id,
        output_dir=output_dir,
        command=_command_text(args),
        resolved_config_path=resolved_config_path,
    )
    outputs = _run_manifested_stage(str(args.command), runner, cfg, manifest)
    finalize_manifest(manifest)
    write_manifest(output_dir, manifest)
    _print_outputs(outputs)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cfp", description="Computational finance research pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="Fetch A-share universe and daily prices.")
    _add_common_args(fetch_parser)
    fetch_parser.set_defaults(func=lambda args: _run_stage(args, run_fetch))

    clean_parser = subparsers.add_parser("clean", help="Clean daily prices and engineer technical indicators.")
    _add_common_args(clean_parser)
    clean_parser.set_defaults(func=lambda args: _run_stage(args, run_cleaning))

    factors_parser = subparsers.add_parser("factors", help="Build factor panel and IC analysis.")
    _add_common_args(factors_parser)
    factors_parser.set_defaults(func=lambda args: _run_stage(args, run_factors))

    backtest_parser = subparsers.add_parser("backtest", help="Run vectorized multi-factor backtest.")
    _add_common_args(backtest_parser)
    backtest_parser.set_defaults(func=lambda args: _run_stage(args, run_backtest))

    ml_parser = subparsers.add_parser("ml", help="Run supervised learning experiments on factor features.")
    _add_common_args(ml_parser)
    ml_parser.set_defaults(func=lambda args: _run_stage(args, run_ml))

    decision_parser = subparsers.add_parser("decision", help="Run decision-aware portfolio optimization.")
    _add_common_args(decision_parser)
    decision_parser.set_defaults(func=lambda args: _run_stage(args, run_decision))

    tune_parser = subparsers.add_parser("tune", help="Run chronological hyperparameter tuning.")
    _add_common_args(tune_parser)
    tune_parser.set_defaults(func=lambda args: _run_stage(args, run_tuning))

    stress_parser = subparsers.add_parser("stress", help="Run market mechanism stress tests.")
    _add_common_args(stress_parser)
    stress_parser.set_defaults(func=lambda args: _run_stage(args, run_stress))

    benchmarks_parser = subparsers.add_parser("benchmarks", help="Build benchmark registry and stability report.")
    _add_common_args(benchmarks_parser)
    benchmarks_parser.set_defaults(func=lambda args: _run_stage(args, run_benchmarks))

    run_all_parser = subparsers.add_parser("run-all", help="Run the full pipeline.")
    _add_common_args(run_all_parser)
    run_all_parser.add_argument(
        "--skip-fetch", action="store_true", help="Use existing data and skip AkShare download."
    )
    run_all_parser.set_defaults(func=run_all)
    return parser


def run_all(args: argparse.Namespace) -> int:
    base_cfg = _load(args, write_config=False)
    run_id = make_run_id()
    run_dir, latest_dir = versioned_run_dir(base_cfg["data"]["output_dir"], run_id)
    cfg = with_overrides(base_cfg, output_dir=run_dir)
    resolved_config_path = write_resolved_config(cfg)
    manifest = create_manifest(
        run_id=run_id,
        output_dir=run_dir,
        command=_command_text(args),
        resolved_config_path=resolved_config_path,
    )
    write_manifest(run_dir, manifest)
    print(f"run_id: {run_id}")
    print(f"run_dir: {run_dir}")
    try:
        if not args.skip_fetch:
            print("== fetch ==")
            _print_outputs(_run_manifested_stage("fetch", run_fetch, cfg, manifest))
        else:
            mark_stage_skipped(manifest, "fetch", "--skip-fetch was provided")
            write_manifest(run_dir, manifest)
        print("== clean ==")
        _print_outputs(_run_manifested_stage("clean", run_cleaning, cfg, manifest))
        print("== factors ==")
        _print_outputs(_run_manifested_stage("factors", run_factors, cfg, manifest))
        print("== backtest ==")
        _print_outputs(_run_manifested_stage("backtest", run_backtest, cfg, manifest))
        print("== ml ==")
        _print_outputs(_run_manifested_stage("ml", run_ml, cfg, manifest))
        print("== decision ==")
        _print_outputs(_run_manifested_stage("decision", run_decision, cfg, manifest))
        print("== tune ==")
        _print_outputs(_run_manifested_stage("tune", run_tuning, cfg, manifest))
        print("== stress ==")
        _print_outputs(_run_manifested_stage("stress", run_stress, cfg, manifest))
        print("== benchmarks ==")
        _print_outputs(_run_manifested_stage("benchmarks", run_benchmarks, cfg, manifest))
    except Exception:
        finalize_manifest(manifest, status="failed")
        write_manifest(run_dir, manifest)
        raise
    finalize_manifest(manifest)
    write_manifest(run_dir, manifest)
    publish_latest(run_dir, latest_dir)
    print(f"latest_dir: {latest_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_argv)
    args._command_text = "cfp " + " ".join(raw_argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ConfigError) as exc:
        parser.error(str(exc))
        return 2
