from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

VALIDATION_METHODS = {"expanding", "purged"}


@dataclass(frozen=True)
class WalkForwardSplit:
    split_id: int
    train_months: list[pd.Timestamp]
    test_months: list[pd.Timestamp]
    method: str
    embargo_months: int = 0

    @property
    def train_start(self) -> pd.Timestamp:
        return min(self.train_months)

    @property
    def train_end(self) -> pd.Timestamp:
        return max(self.train_months)

    @property
    def test_start(self) -> pd.Timestamp:
        return min(self.test_months)

    @property
    def test_end(self) -> pd.Timestamp:
        return max(self.test_months)


def walk_forward_month_splits(
    months: list[pd.Timestamp],
    *,
    min_train_months: int,
    test_window_months: int,
    method: str = "expanding",
    embargo_months: int = 0,
) -> list[WalkForwardSplit]:
    """Build chronological walk-forward splits with optional purge/embargo gap."""
    if method not in VALIDATION_METHODS:
        raise ValueError(f"validation method must be one of {sorted(VALIDATION_METHODS)}")
    ordered = sorted(pd.Timestamp(month) for month in months)
    splits: list[WalkForwardSplit] = []
    split_id = 0
    start = int(min_train_months)
    while start < len(ordered):
        train_end_idx = start
        if method == "purged":
            train_end_idx = max(0, start - int(embargo_months))
        train_months = ordered[:train_end_idx]
        test_months = ordered[start : start + int(test_window_months)]
        if not test_months:
            break
        if train_months:
            splits.append(
                WalkForwardSplit(
                    split_id=split_id,
                    train_months=train_months,
                    test_months=test_months,
                    method=method,
                    embargo_months=int(embargo_months) if method == "purged" else 0,
                )
            )
            split_id += 1
        start += int(test_window_months)
    return splits


def assert_no_split_leakage(split: WalkForwardSplit) -> None:
    train_set = set(split.train_months)
    test_set = set(split.test_months)
    overlap = train_set.intersection(test_set)
    if overlap:
        raise ValueError(f"walk-forward split {split.split_id} has overlapping train/test months: {sorted(overlap)}")
    if split.train_months and split.test_months and split.train_end >= split.test_start:
        raise ValueError(f"walk-forward split {split.split_id} violates chronological ordering")
    if split.method == "purged" and split.embargo_months > 0:
        gap = month_gap(split.train_end, split.test_start)
        if gap < split.embargo_months:
            raise ValueError(
                f"walk-forward split {split.split_id} expected embargo >= {split.embargo_months}, got {gap}"
            )


def month_gap(left: pd.Timestamp, right: pd.Timestamp) -> int:
    """Number of whole calendar-month steps between two timestamps."""
    left_period = pd.Timestamp(left).to_period("M")
    right_period = pd.Timestamp(right).to_period("M")
    return int(right_period.ordinal - left_period.ordinal)
