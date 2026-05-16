from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from .acquire import run_fetch
from .backtest import run_backtest
from .cleaning import run_cleaning
from .config import load_config, with_overrides
from .decision import run_decision
from .factors import run_factors
from .ml import run_ml
from .stress import run_stress
from .tuning import run_tuning


Runner = Callable[[dict], dict]


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file.")
    parser.add_argument("--data-dir", default=None, help="Input data directory. Defaults to config data.input_dir.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to config data.output_dir.")


def _load(args: argparse.Namespace) -> dict:
    config_path = Path(args.config)
    cfg = load_config(config_path if config_path.exists() else None)
    return with_overrides(cfg, args.data_dir, args.output_dir)


def _print_outputs(outputs: dict) -> None:
    for key, value in outputs.items():
        print(f"{key}: {value}")


def _run_stage(args: argparse.Namespace, runner: Runner) -> int:
    cfg = _load(args)
    outputs = runner(cfg)
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

    run_all_parser = subparsers.add_parser("run-all", help="Run the full pipeline.")
    _add_common_args(run_all_parser)
    run_all_parser.add_argument("--skip-fetch", action="store_true", help="Use existing data and skip AkShare download.")
    run_all_parser.set_defaults(func=run_all)
    return parser


def run_all(args: argparse.Namespace) -> int:
    cfg = _load(args)
    if not args.skip_fetch:
        print("== fetch ==")
        _print_outputs(run_fetch(cfg))
    print("== clean ==")
    _print_outputs(run_cleaning(cfg))
    print("== factors ==")
    _print_outputs(run_factors(cfg))
    print("== backtest ==")
    _print_outputs(run_backtest(cfg))
    print("== ml ==")
    _print_outputs(run_ml(cfg))
    print("== decision ==")
    _print_outputs(run_decision(cfg))
    print("== tune ==")
    _print_outputs(run_tuning(cfg))
    print("== stress ==")
    _print_outputs(run_stress(cfg))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
