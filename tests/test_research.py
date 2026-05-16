from __future__ import annotations

import pandas as pd

from cfpipeline.research import (
    assign_time_split,
    bootstrap_mean_ci,
    fama_macbeth_regression,
    ic_significance_table,
)


def test_time_split_labels_are_chronological() -> None:
    dates = pd.Series(pd.to_datetime(["2021-01-31", "2023-06-30", "2024-02-29"]))
    labels = assign_time_split(dates, {"train_end": "2022-12-31", "valid_end": "2023-12-31", "test_start": "2024-01-01"})
    assert labels.tolist() == ["train", "validation", "test"]


def test_bootstrap_ci_and_fama_macbeth_run() -> None:
    low, high = bootstrap_mean_ci(pd.Series([0.1, 0.2, 0.3]), samples=50, seed=1)
    assert low <= high

    ic = pd.DataFrame({"factor": ["z_A", "z_A", "z_A"], "IC": [0.1, 0.2, 0.0]})
    sig = ic_significance_table(ic, bootstrap_samples=50)
    assert {"t_stat", "bootstrap_ci_low", "bootstrap_ci_high"}.issubset(sig.columns)

    panel = pd.DataFrame(
        {
            "month_end": pd.to_datetime(["2022-01-31"] * 5 + ["2022-02-28"] * 5),
            "fwd_1m_ret": [0.01, 0.02, 0.00, -0.01, 0.03, 0.02, 0.01, -0.02, 0.03, 0.04],
            "z_A": [1, 2, 0, -1, 3, 2, 1, -2, 3, 4],
            "z_B": [0, 1, 2, 3, 4, 1, 0, 2, 4, 3],
        }
    )
    fmb = fama_macbeth_regression(panel, ["z_A", "z_B"])
    assert set(["term", "mean_coef", "t_stat"]).issubset(fmb.columns)
