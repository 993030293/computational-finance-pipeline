from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cfpipeline import cli
from cfpipeline.artifacts import atomic_write, atomic_write_text, create_manifest, write_manifest


def test_atomic_write_preserves_existing_target_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "result.txt"
    atomic_write_text(target, "old")

    def broken_writer(tmp_path: Path) -> None:
        tmp_path.write_text("partial", encoding="utf-8")
        raise RuntimeError("simulated write failure")

    with pytest.raises(RuntimeError, match="simulated write failure"):
        atomic_write(target, broken_writer)

    assert target.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".result.txt.tmp-*"))


def test_manifest_written_with_required_fields(tmp_path: Path) -> None:
    manifest = create_manifest(
        run_id="20260517_120000",
        output_dir=tmp_path,
        command="cfp run-all --skip-fetch",
        resolved_config_path=tmp_path / "resolved_config.yaml",
    )

    path = write_manifest(tmp_path, manifest)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["run_id"] == "20260517_120000"
    assert loaded["command"] == "cfp run-all --skip-fetch"
    assert "python_version" in loaded
    assert "stage_status" in loaded


def test_run_all_creates_versioned_run_manifest_and_latest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_runner(stage: str):
        def run(cfg: dict[str, Any]) -> dict[str, Path]:
            output_dir = Path(cfg["data"]["output_dir"])
            path = output_dir / stage / f"{stage}.txt"
            atomic_write_text(path, "ok")
            return {stage: path}

        return run

    monkeypatch.setattr(cli, "run_cleaning", fake_runner("clean"))
    monkeypatch.setattr(cli, "run_factors", fake_runner("factors"))
    monkeypatch.setattr(cli, "run_backtest", fake_runner("backtest"))
    monkeypatch.setattr(cli, "run_ml", fake_runner("ml"))
    monkeypatch.setattr(cli, "run_decision", fake_runner("decision"))
    monkeypatch.setattr(cli, "run_tuning", fake_runner("tune"))
    monkeypatch.setattr(cli, "run_stress", fake_runner("stress"))

    rc = cli.main(["run-all", "--skip-fetch", "--output-dir", str(tmp_path / "outputs")])

    assert rc == 0
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    manifest_path = run_dir / "run_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == run_dir.name
    assert manifest["stage_status"]["fetch"]["status"] == "skipped"
    assert manifest["stage_status"]["clean"]["status"] == "succeeded"
    assert (tmp_path / "latest" / "run_manifest.json").exists()
    assert (tmp_path / "latest" / "clean" / "clean.txt").exists()
