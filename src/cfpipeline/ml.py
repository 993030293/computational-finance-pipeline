from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import atomic_write_csv, atomic_write_json, atomic_write_text
from .backtest import build_portfolios, performance_metrics, pivot_panel
from .cleaning import read_csv
from .paths import PipelinePaths, first_existing
from .validation import assert_no_split_leakage, walk_forward_month_splits


def factor_candidates(paths: PipelinePaths) -> list[Path]:
    return [paths.output_project4_dir / "factors.csv", paths.input_project4_dir / "factors.csv"]


def load_factor_dataset(paths: PipelinePaths) -> tuple[pd.DataFrame, Path]:
    path = first_existing(factor_candidates(paths))
    df = read_csv(path, parse_dates=["month_end"])
    df["symbol"] = df["symbol"].astype("string")
    return df.sort_values(["month_end", "symbol"]).reset_index(drop=True), path


def prepare_ml_dataset(factors: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = [col for col in cfg.get("feature_cols", []) if col in factors.columns]
    if not feature_cols:
        feature_cols = [col for col in factors.columns if col.startswith("z_")]
    target = str(cfg.get("target", "fwd_1m_ret"))
    required = ["symbol", "month_end", target, *feature_cols]
    missing = [col for col in required if col not in factors.columns]
    if missing:
        raise KeyError(f"ML dataset missing required columns: {missing}")
    data = factors[required].replace([np.inf, -np.inf], np.nan).dropna().copy()
    data["target_class"] = (data[target] > float(cfg.get("classification_threshold", 0.0))).astype(int)
    return data, feature_cols


def make_model(name: str, seed: int):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "linear":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=seed))
    if name == "lasso":
        return make_pipeline(StandardScaler(), Lasso(alpha=0.001, random_state=seed, max_iter=10000))
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=120, min_samples_leaf=5, random_state=seed, n_jobs=-1)
    if name == "gradient_boosting":
        return GradientBoostingRegressor(random_state=seed)
    raise ValueError(f"Unknown model: {name}")


def validation_runtime_config(ml_cfg: dict[str, Any], validation_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    validation_cfg = validation_cfg or {}
    return {
        "method": str(validation_cfg.get("method", "expanding")),
        "embargo_months": int(validation_cfg.get("embargo_months", 0)),
        "min_train_months": int(validation_cfg.get("min_train_months", ml_cfg.get("min_train_months", 18))),
        "test_window_months": int(validation_cfg.get("test_window_months", ml_cfg.get("test_window_months", 6))),
    }


def walk_forward_splits(
    months: list[pd.Timestamp],
    min_train_months: int,
    test_window_months: int,
    *,
    method: str = "expanding",
    embargo_months: int = 0,
):
    yield from walk_forward_month_splits(
        months,
        min_train_months=min_train_months,
        test_window_months=test_window_months,
        method=method,
        embargo_months=embargo_months,
    )


def predict_scores(model, model_name: str, x_test: pd.DataFrame) -> np.ndarray:
    if model_name == "logistic":
        return model.predict_proba(x_test)[:, 1]
    return model.predict(x_test)


def rank_ic(y_true: pd.Series, score: pd.Series) -> float:
    valid = y_true.notna() & score.notna()
    if valid.sum() < 3:
        return np.nan
    return float(y_true[valid].rank().corr(score[valid].rank()))


def precision_at_top(y_true_class: pd.Series, score: pd.Series, top_quantile: float) -> float:
    valid = y_true_class.notna() & score.notna()
    if valid.sum() == 0:
        return np.nan
    n = max(1, int(round(valid.sum() * top_quantile)))
    top_idx = score[valid].sort_values(ascending=False).head(n).index
    return float(y_true_class.loc[top_idx].mean())


def evaluate_predictions(predictions: pd.DataFrame, top_quantile: float) -> pd.DataFrame:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

    rows: list[dict[str, Any]] = []
    group_cols = ["validation_method", "model"] if "validation_method" in predictions.columns else ["model"]
    for group_key, group in predictions.groupby(group_cols):
        if isinstance(group_key, tuple):
            validation_method, model = group_key
        else:
            validation_method, model = "", group_key
        y = group["y_true"].astype(float)
        score = group["score"].astype(float)
        target_class = group["target_class"].astype(int)
        try:
            auc = roc_auc_score(target_class, score) if target_class.nunique() == 2 else np.nan
        except ValueError:
            auc = np.nan
        if model == "logistic":
            rmse = np.nan
            mae = np.nan
        else:
            rmse = float(np.sqrt(mean_squared_error(y, score)))
            mae = float(mean_absolute_error(y, score))
        row = {
            "model": str(model),
            "observations": int(len(group)),
            "rmse": rmse,
            "mae": mae,
            "rank_ic": rank_ic(y, score),
            "auc": float(auc) if pd.notna(auc) else np.nan,
            "precision_at_top": precision_at_top(target_class, score, top_quantile),
        }
        if validation_method:
            row["validation_method"] = str(validation_method)
        rows.append(row)
    sort_cols = ["validation_method", "model"] if rows and "validation_method" in rows[0] else ["model"]
    return pd.DataFrame(rows).sort_values(sort_cols).reset_index(drop=True)


def model_feature_importance(model, model_name: str, feature_cols: list[str]) -> pd.Series:
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(estimator, "coef_"):
        values = np.asarray(estimator.coef_).reshape(-1)
        if len(values) == len(feature_cols):
            return pd.Series(np.abs(values), index=feature_cols)
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_)
        if len(values) == len(feature_cols):
            return pd.Series(values, index=feature_cols)
    return pd.Series(np.nan, index=feature_cols)


def run_walk_forward(
    data: pd.DataFrame, feature_cols: list[str], cfg: dict[str, Any], validation_cfg: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = str(cfg.get("target", "fwd_1m_ret"))
    months = list(pd.Series(data["month_end"].drop_duplicates()).sort_values())
    seed = int(cfg.get("random_seed", 42))
    validation = validation_runtime_config(cfg, validation_cfg)
    predictions: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []

    for split in walk_forward_splits(
        months,
        validation["min_train_months"],
        validation["test_window_months"],
        method=validation["method"],
        embargo_months=validation["embargo_months"],
    ):
        assert_no_split_leakage(split)
        split_id = split.split_id
        train_months = split.train_months
        test_months = split.test_months
        train = data[data["month_end"].isin(train_months)]
        test = data[data["month_end"].isin(test_months)]
        if train.empty or test.empty:
            continue
        x_train = train[feature_cols]
        x_test = test[feature_cols]
        y_train_reg = train[target].astype(float)
        y_train_cls = train["target_class"].astype(int)

        for model_name in cfg.get("models", ["linear", "ridge", "lasso", "logistic"]):
            model = make_model(str(model_name), seed)
            if model_name == "logistic":
                if y_train_cls.nunique() < 2:
                    continue
                model.fit(x_train, y_train_cls)
            else:
                model.fit(x_train, y_train_reg)
            score = predict_scores(model, str(model_name), x_test)
            pred = test[["symbol", "month_end", target, "target_class"]].copy()
            pred = pred.rename(columns={target: "y_true"})
            pred["model"] = str(model_name)
            pred["split_id"] = split_id
            pred["validation_method"] = validation["method"]
            pred["embargo_months"] = validation["embargo_months"] if validation["method"] == "purged" else 0
            pred["train_start"] = split.train_start
            pred["train_end"] = split.train_end
            pred["test_start"] = split.test_start
            pred["test_end"] = split.test_end
            pred["score"] = score
            predictions.append(pred)

            imp = model_feature_importance(model, str(model_name), feature_cols).rename("importance").reset_index()
            imp = imp.rename(columns={"index": "feature"})
            imp["model"] = str(model_name)
            imp["split_id"] = split_id
            importances.append(imp)

    if not predictions:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(predictions, ignore_index=True), pd.concat(importances, ignore_index=True)


def backtest_ml_scores(predictions: pd.DataFrame, top_quantile: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    group_cols = ["validation_method", "model"] if "validation_method" in predictions.columns else ["model"]
    for group_key, group in predictions.groupby(group_cols):
        if isinstance(group_key, tuple):
            validation_method, model = group_key
        else:
            validation_method, model = "", group_key
        score_panel = pivot_panel(group.rename(columns={"score": "ml_score"}), "ml_score")
        ret_panel = pivot_panel(group.rename(columns={"y_true": "fwd_1m_ret"}), "fwd_1m_ret")
        returns = build_portfolios(score_panel, ret_panel, top_quantile=top_quantile)
        returns.insert(0, "model", model)
        if validation_method:
            returns.insert(0, "validation_method", validation_method)
        rows.append(returns.reset_index())
        for strategy in ["long_only", "short_only", "long_short", "benchmark_ew"]:
            metric = performance_metrics(returns[strategy], benchmark=returns["benchmark_ew"])
            metric["model"] = model
            if validation_method:
                metric["validation_method"] = validation_method
            metric["strategy"] = strategy
            metrics.append(metric)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(metrics)


def write_ml_report(
    out_dir: Path,
    dataset_path: Path,
    metrics: pd.DataFrame,
    backtest_metrics: pd.DataFrame,
    cfg: dict[str, Any],
) -> Path:
    path = out_dir / "ML_EXPERIMENT_REPORT.md"
    lines = [
        "# Supervised Learning Experiment Report",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Dataset: `{dataset_path}`",
        f"- Target: `{cfg.get('target', 'fwd_1m_ret')}`",
        f"- Validation protocol: `{cfg.get('validation_method', 'walk-forward')}` walk-forward split; no random shuffle.",
        f"- Embargo months: {cfg.get('embargo_months', 0)}",
        "",
        "## Prediction Metrics",
        metrics.to_markdown(index=False, floatfmt=".6f") if not metrics.empty else "No metrics generated.",
        "",
        "## Portfolio Metrics From ML Scores",
        backtest_metrics.to_markdown(index=False, floatfmt=".6f")
        if not backtest_metrics.empty
        else "No backtest metrics generated.",
        "",
        "## Limitations",
        "- Monthly labels are noisy and non-stationary.",
        "- The experiment is designed for statistical learning demonstration, not live trading deployment.",
        "- Hyperparameters are intentionally conservative to reduce overfitting in an admissions portfolio context.",
    ]
    atomic_write_text(path, "\n".join(lines), encoding="utf-8")
    return path


def run_ml(cfg: dict[str, Any]) -> dict[str, Path]:
    paths = PipelinePaths.from_config(cfg)
    paths.ensure_output_dirs()
    ml_cfg = cfg.get("ml", {})
    validation_cfg = cfg.get("validation", {})
    out_dir = paths.output_dir / "ml"
    out_dir.mkdir(parents=True, exist_ok=True)

    factors, dataset_path = load_factor_dataset(paths)
    dataset, feature_cols = prepare_ml_dataset(factors, ml_cfg)
    atomic_write_csv(dataset, out_dir / "ml_dataset.csv", index=False, encoding="utf-8-sig")
    primary_validation = validation_runtime_config(ml_cfg, validation_cfg)
    predictions, importances = run_walk_forward(dataset, feature_cols, ml_cfg, primary_validation)
    predictions_path = out_dir / "ml_predictions.csv"
    atomic_write_csv(predictions, predictions_path, index=False, encoding="utf-8-sig")

    top_quantile = float(ml_cfg.get("top_quantile", 0.2))
    metrics = evaluate_predictions(predictions, top_quantile) if not predictions.empty else pd.DataFrame()
    metrics_path = out_dir / "ml_model_metrics.csv"
    atomic_write_csv(metrics, metrics_path, index=False, encoding="utf-8-sig")

    comparison_frames: list[pd.DataFrame] = []
    for method in ["expanding", "purged"]:
        comparison_validation = {
            **primary_validation,
            "method": method,
            "embargo_months": primary_validation["embargo_months"] if method == "purged" else 0,
        }
        compare_predictions, _ = run_walk_forward(dataset, feature_cols, ml_cfg, comparison_validation)
        if not compare_predictions.empty:
            comparison_frames.append(evaluate_predictions(compare_predictions, top_quantile))
    validation_comparison = pd.concat(comparison_frames, ignore_index=True) if comparison_frames else pd.DataFrame()
    validation_comparison_path = out_dir / "ml_validation_comparison.csv"
    atomic_write_csv(validation_comparison, validation_comparison_path, index=False, encoding="utf-8-sig")

    importances_path = out_dir / "ml_feature_importance.csv"
    if not importances.empty:
        atomic_write_csv(
            importances.groupby(["model", "feature"], as_index=False)["importance"].mean(),
            importances_path,
            index=False,
            encoding="utf-8-sig",
        )
    else:
        atomic_write_csv(pd.DataFrame(columns=["model", "feature", "importance"]), importances_path, index=False)

    if predictions.empty:
        ml_returns = pd.DataFrame()
        ml_backtest_metrics = pd.DataFrame()
    else:
        ml_returns, ml_backtest_metrics = backtest_ml_scores(predictions, top_quantile)
    ml_returns_path = out_dir / "ml_portfolio_returns.csv"
    ml_backtest_path = out_dir / "ml_backtest_metrics.csv"
    atomic_write_csv(ml_returns, ml_returns_path, index=False, encoding="utf-8-sig")
    atomic_write_csv(ml_backtest_metrics, ml_backtest_path, index=False, encoding="utf-8-sig")

    metadata_path = out_dir / "ml_run_metadata.json"
    atomic_write_json(
        metadata_path,
        {
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_path": str(dataset_path),
            "rows": int(len(dataset)),
            "feature_cols": feature_cols,
            "config": ml_cfg,
            "validation": primary_validation,
            "validation_comparison_path": str(validation_comparison_path),
        },
    )
    report_cfg = {**ml_cfg, **primary_validation}
    report_path = write_ml_report(out_dir, dataset_path, metrics, ml_backtest_metrics, report_cfg)
    return {
        "dataset": out_dir / "ml_dataset.csv",
        "predictions": predictions_path,
        "model_metrics": metrics_path,
        "validation_comparison": validation_comparison_path,
        "feature_importance": importances_path,
        "portfolio_returns": ml_returns_path,
        "backtest_metrics": ml_backtest_path,
        "report": report_path,
        "metadata": metadata_path,
    }
