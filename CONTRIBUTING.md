# Contributing

This repository is a reproducible computational finance research pipeline. Contributions should improve reliability, auditability, validation discipline, or documentation clarity. Do not optimize headline returns by introducing data leakage or overfitting.

## Environment Setup

```powershell
git clone https://github.com/993030293/computational-finance-pipeline.git
cd computational-finance-pipeline
python -m pip install -e ".[dev]"
```

Optional pre-commit hooks:

```powershell
pre-commit install
```

## Local Quality Checks

Run the same checks used by CI:

```powershell
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m pytest --cov=cfpipeline --cov-report=term-missing --cov-fail-under=50
```

To let Ruff format files locally:

```powershell
python -m ruff format .
```

## Sample Pipeline

The repository includes a small sample dataset so contributors can run the full pipeline without the large local data directory.

```powershell
python scripts/create_sample_data.py
cfp run-all --skip-fetch --config configs/sample.yaml
```

The run writes versioned artifacts to `outputs/runs/<run_id>/` and publishes the latest successful run to `outputs/latest/`.

## Pull Request Workflow

1. Keep PRs small and focused.
2. Explain the research or engineering risk being addressed.
3. Include tests for behavior changes.
4. Update README or docs when commands, outputs, configs, or stage contracts change.
5. Do not commit large generated artifacts from `data/` or `outputs/`.
6. Report validation results in the PR description.

Suggested PR checklist:

```text
- [ ] Tests pass locally
- [ ] Ruff check and format check pass
- [ ] Mypy passes
- [ ] Sample pipeline still runs
- [ ] Docs updated if behavior changed
- [ ] No large data or generated outputs committed
```

## Adding a Pipeline Stage

New stages should follow the existing pattern:

1. Add stage logic under `src/cfpipeline/`.
2. Accept a merged config dictionary.
3. Resolve paths through `PipelinePaths.from_config`.
4. Write outputs with `atomic_write_csv`, `atomic_write_json`, or `atomic_write_text`.
5. Return a dictionary of output names to `Path` objects.
6. Add CLI entry in `src/cfpipeline/cli.py`.
7. Add the stage to `run-all` only if it is fast enough for the sample pipeline.
8. Add stage input/output records to `collect_input_files` in `src/cfpipeline/artifacts.py`.
9. Add tests and update `docs/stage_io_contract.md`.

## Adding Tests

Use small synthetic data or the sample dataset. Unit tests must not call external APIs. Mock AkShare or network-dependent functions when testing `fetch`.

Preferred test style:

- one behavior per test;
- explicit temporary directories via `tmp_path`;
- deterministic random seeds;
- no assumptions about local private data;
- assertions on files, schema, and leakage boundaries, not only "no exception".

## Data and Licensing Notes

The source code is MIT licensed. Market data fetched through AkShare or other upstream providers may have separate terms. Large migrated datasets are not committed to this repository. See `DATA_CARD.md` for data source, bias, and use-boundary notes.
