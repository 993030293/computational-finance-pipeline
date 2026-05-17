from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelinePaths:
    """Resolved input and output directories for one pipeline run."""

    input_dir: Path
    output_dir: Path

    @classmethod
    def from_config(cls, cfg: dict) -> PipelinePaths:
        data_cfg = cfg.get("data", {})
        return cls(
            input_dir=Path(data_cfg.get("input_dir", "data")),
            output_dir=Path(data_cfg.get("output_dir", "outputs/latest")),
        )

    @property
    def input_raw_dir(self) -> Path:
        return self.input_dir / "raw"

    @property
    def input_processed_dir(self) -> Path:
        return self.input_dir / "processed"

    @property
    def input_project4_dir(self) -> Path:
        return self.input_dir / "project4"

    @property
    def output_raw_dir(self) -> Path:
        return self.output_dir / "raw"

    @property
    def output_processed_dir(self) -> Path:
        return self.output_dir / "processed"

    @property
    def output_project4_dir(self) -> Path:
        return self.output_dir / "project4"

    @property
    def output_eda_dir(self) -> Path:
        return self.output_dir / "eda"

    @property
    def output_backtest_dir(self) -> Path:
        return self.output_dir / "proj5_output"

    def ensure_output_dirs(self) -> None:
        for path in (
            self.output_raw_dir,
            self.output_processed_dir,
            self.output_project4_dir,
            self.output_eda_dir,
            self.output_backtest_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    expected = ", ".join(str(p) for p in paths)
    raise FileNotFoundError(f"No input file found. Expected one of: {expected}")
