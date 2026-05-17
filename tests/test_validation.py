from __future__ import annotations

from pathlib import Path

import pandas as pd

from cfpipeline.benchmarks import run_benchmarks
from cfpipeline.tuning import select_candidate_from_validation
from cfpipeline.validation import assert_no_split_leakage, month_gap, walk_forward_month_splits


def test_purged_walk_forward_enforces_embargo_gap() -> None:
    months = list(pd.date_range("2020-01-31", periods=24, freq="ME"))

    splits = walk_forward_month_splits(
        months,
        min_train_months=12,
        test_window_months=3,
        method="purged",
        embargo_months=2,
    )

    assert splits
    for split in splits:
        assert_no_split_leakage(split)
        assert split.train_end < split.test_start
        assert month_gap(split.train_end, split.test_start) >= 2


def test_expanding_walk_forward_has_no_train_test_overlap() -> None:
    months = list(pd.date_range("2020-01-31", periods=18, freq="ME"))

    splits = walk_forward_month_splits(
        months,
        min_train_months=6,
        test_window_months=4,
        method="expanding",
        embargo_months=0,
    )

    assert splits
    for split in splits:
        assert_no_split_leakage(split)
        assert set(split.train_months).isdisjoint(split.test_months)


def test_tuning_selection_uses_validation_not_test() -> None:
    results = pd.DataFrame(
        [
            {"candidate_id": 0, "split": "validation", "sharpe": 2.0, "ann_return": 0.2},
            {"candidate_id": 0, "split": "test", "sharpe": -5.0, "ann_return": -0.5},
            {"candidate_id": 1, "split": "validation", "sharpe": 1.0, "ann_return": 0.1},
            {"candidate_id": 1, "split": "test", "sharpe": 99.0, "ann_return": 9.9},
        ]
    )

    assert select_candidate_from_validation(results) == 0


def test_benchmark_registry_marks_test_metrics_reporting_only(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    tuning_dir = output_dir / "tuning"
    tuning_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "candidate_id": 1,
                "split": "test",
                "ann_return": 0.1,
                "sharpe": 1.2,
                "max_drawdown": -0.05,
                "mean_turnover": 0.3,
            }
        ]
    ).to_csv(tuning_dir / "test_performance.csv", index=False)
    cfg = {
        "data": {"input_dir": str(tmp_path / "data"), "output_dir": str(output_dir)},
        "validation": {"method": "purged", "embargo_months": 1},
    }

    outputs = run_benchmarks(cfg)
    registry = pd.read_csv(outputs["registry"])

    tuned = registry[registry["benchmark"].eq("tuned_strategy")].iloc[0]
    assert tuned["selection_role"] == "test_reporting_only"
    assert outputs["report"].exists()
