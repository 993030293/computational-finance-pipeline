# Architecture

This project is organized as a reproducible research pipeline. The CLI is the public entry point, configs define run behavior, and every full `run-all` execution writes versioned artifacts plus a manifest.

## Pipeline Graph

```mermaid
flowchart LR
    CFG[configs/*.yaml] --> CLI[cfp CLI]
    SAMPLE[examples/sample_data] --> CLEAN
    DATA[data/ local full data] --> CLEAN
    AK[AkShare API] --> FETCH

    CLI --> FETCH[fetch]
    CLI --> CLEAN[clean]
    FETCH --> RAW[(raw daily prices)]
    RAW --> CLEAN
    CLEAN --> FACTORS[factors]
    FACTORS --> BACKTEST[backtest]
    FACTORS --> ML[ml]
    FACTORS --> DECISION[decision]
    ML --> DECISION
    FACTORS --> TUNE[tune]
    FACTORS --> STRESS[stress]
    CLEAN --> STRESS
    BACKTEST --> BENCH[benchmarks]
    ML --> BENCH
    DECISION --> BENCH
    TUNE --> BENCH
    STRESS --> BENCH

    FETCH --> OUT[outputs/runs/run_id]
    CLEAN --> OUT
    FACTORS --> OUT
    BACKTEST --> OUT
    ML --> OUT
    DECISION --> OUT
    TUNE --> OUT
    STRESS --> OUT
    BENCH --> OUT
```

## Config, Manifest, and Outputs

```mermaid
flowchart TD
    USER[User command] --> LOAD[Load config]
    DEFAULTS[Built-in defaults] --> LOAD
    YAML[configs/default.yaml or sample.yaml] --> LOAD
    OVERRIDES[--data-dir / --output-dir] --> LOAD
    LOAD --> VALIDATE[Typed schema validation]
    VALIDATE --> RESOLVED[resolved_config.yaml]
    VALIDATE --> RUNID[Create run_id]
    RUNID --> RUNDIR[outputs/runs/run_id]
    RUNDIR --> MANIFEST[run_manifest.json]
    RUNDIR --> STAGES[stage outputs]
    STAGES --> MANIFEST
    MANIFEST --> LATEST[outputs/latest copy]
    RUNDIR --> POINTER[outputs/LATEST_RUN.json]
```

## Package Layout

```text
src/cfpipeline/
  acquire.py       fetch, cache, checkpoint, fetch report
  cleaning.py      data cleaning and technical indicators
  factors.py       factor construction and statistical validation
  backtest.py      portfolio returns, costs, NAV, performance
  ml.py            supervised learning with walk-forward validation
  decision.py      decision-aware portfolio optimization
  tuning.py        validation-only hyperparameter selection
  stress.py        market mechanism stress tests
  benchmarks.py    benchmark registry and stability report
  validation.py    expanding and purged walk-forward splits
  artifacts.py     atomic writes, run manifest, latest publishing
  config.py        typed config defaults and validation
  cli.py           cfp command-line interface
```

## Design Principles

- Keep all pipeline logic callable from Python and from the CLI.
- Make full runs reproducible through `resolved_config.yaml` and `run_manifest.json`.
- Keep generated data and large outputs out of git.
- Treat performance metrics as research diagnostics, not trading claims.
- Prefer explicit chronological validation over random train/test splits.
