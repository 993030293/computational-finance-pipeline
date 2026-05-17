from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from cfpipeline.cli import _load, main
from cfpipeline.config import DEFAULT_CONFIG, deep_update, load_config, validate_config, write_resolved_config


def test_missing_explicit_config_path_fails(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    with pytest.raises(SystemExit) as exc:
        main(["clean", "--config", str(missing)])

    assert exc.value.code == 2
    assert "Config file not found" in capsys.readouterr().err


def test_no_config_uses_default(tmp_path: Path) -> None:
    args = SimpleNamespace(config=None, data_dir=tmp_path / "data", output_dir=tmp_path / "outputs")

    cfg = _load(args)

    assert cfg["data"]["input_dir"] == str(tmp_path / "data")
    assert cfg["data"]["output_dir"] == str(tmp_path / "outputs")
    assert cfg["backtest"]["top_quantile"] == 0.2
    assert (tmp_path / "outputs" / "resolved_config.yaml").exists()


def test_invalid_quantile_fails() -> None:
    cfg = deep_update(DEFAULT_CONFIG, {"backtest": {"top_quantile": 1.5}})

    with pytest.raises(ValueError, match="backtest.top_quantile"):
        validate_config(cfg)


def test_invalid_model_name_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad_model.yaml"
    path.write_text(
        yaml.safe_dump({"ml": {"models": ["linear", "made_up_model"]}}, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported models"):
        load_config(path)


def test_resolved_config_written(tmp_path: Path) -> None:
    cfg = deep_update(DEFAULT_CONFIG, {"data": {"output_dir": str(tmp_path)}})

    path = write_resolved_config(cfg)

    assert path == tmp_path / "resolved_config.yaml"
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert loaded["data"]["output_dir"] == str(tmp_path)
    assert loaded["ml"]["models"] == DEFAULT_CONFIG["ml"]["models"]
