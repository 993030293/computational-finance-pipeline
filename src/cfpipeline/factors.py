from __future__ import annotations

import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cleaning import clean_daily_prices, read_csv, rename_any_columns
from .paths import PipelinePaths, first_existing
from .research import (
    fama_macbeth_regression,
    ic_decay,
    ic_significance_table,
    quantile_group_returns,
    summarize_group_returns,
    summarize_ic_decay,
)


BASE_FACTOR_COLS = ["VALUE", "MOM_12_1", "QUALITY", "SIZE"]
ENHANCED_FACTOR_COLS = ["REVERSAL_1M", "VOL_1M", "ILLIQUIDITY"]


def price_panel_candidates(paths: PipelinePaths) -> list[Path]:
    return [
        paths.output_processed_dir / "tech_indicators.csv",
        paths.output_processed_dir / "daily_price_panel.csv",
        paths.output_processed_dir / "daily_price_50.csv",
        paths.input_processed_dir / "tech_indicators.csv",
        paths.input_processed_dir / "daily_price_panel.csv",
        paths.input_processed_dir / "daily_price_50.csv",
        paths.input_processed_dir / "daily_price.csv",
        paths.input_raw_dir / "daily_price.csv",
    ]


def load_price_panel(paths: PipelinePaths) -> tuple[pd.DataFrame, Path]:
    source_path = first_existing(price_panel_candidates(paths))
    df = read_csv(source_path)
    df = rename_any_columns(df)
    daily = clean_daily_prices(df)
    return daily, source_path


def build_monthly_closes(daily: pd.DataFrame) -> pd.DataFrame:
    data = daily[["symbol", "date", "close", "volume", "amount"]].copy()
    data["ym"] = data["date"].dt.to_period("M")
    last = data.sort_values("date").groupby(["symbol", "ym"]).tail(1).copy()
    last["month_end"] = last["ym"].dt.to_timestamp("M")
    last = last.sort_values(["symbol", "month_end"])
    last["fwd_1m_ret"] = last.groupby("symbol")["close"].pct_change(periods=1).shift(-1)
    last["ret_1m"] = last.groupby("symbol")["close"].pct_change()
    last = last.rename(columns={"close": "close_me", "volume": "vol_me", "amount": "amt_me"})
    return last[["symbol", "month_end", "close_me", "vol_me", "amt_me", "fwd_1m_ret", "ret_1m"]]


def attach_daily_rolls(daily: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    data = daily[["symbol", "date", "close", "volume", "amount"]].copy().sort_values(["symbol", "date"])
    if "amount" in data.columns and data["amount"].notna().any():
        data["dollar_vol"] = data["amount"]
    else:
        data["dollar_vol"] = data["close"] * data["volume"]

    grouped = data.groupby("symbol", group_keys=False)
    data["avg_dv_21"] = grouped["dollar_vol"].rolling(21, min_periods=10).mean().reset_index(level=0, drop=True)
    data["ma_252"] = grouped["close"].rolling(252, min_periods=126).mean().reset_index(level=0, drop=True)
    data["ret_d"] = grouped["close"].pct_change()
    mu_63 = grouped["ret_d"].rolling(63, min_periods=40).mean().reset_index(level=0, drop=True)
    sd_63 = grouped["ret_d"].rolling(63, min_periods=40).std(ddof=0).reset_index(level=0, drop=True)
    data["sharpe_3m"] = mu_63 / sd_63.replace(0, np.nan)
    data["vol_21"] = grouped["ret_d"].rolling(21, min_periods=15).std(ddof=0).reset_index(level=0, drop=True)
    data["illiq_raw"] = data["ret_d"].abs() / data["dollar_vol"].replace(0, np.nan)
    data["illiq_21"] = grouped["illiq_raw"].rolling(21, min_periods=15).mean().reset_index(level=0, drop=True)

    # Preserve the original project behavior: factor values are evaluated only
    # when the calendar month end is an actual trading date in the daily panel.
    merged = monthly.merge(data, left_on=["symbol", "month_end"], right_on=["symbol", "date"], how="left")
    return merged.drop(columns=["date"])


def compute_momentum_12_1(monthly_close: pd.DataFrame) -> pd.Series:
    data = monthly_close.sort_values(["symbol", "month_end"]).copy()
    data["close_lag1"] = data.groupby("symbol")["close_me"].shift(1)
    data["close_lag12"] = data.groupby("symbol")["close_me"].shift(12)
    return data["close_lag1"] / data["close_lag12"] - 1.0


def compute_factors(
    daily: pd.DataFrame,
    *,
    include_enhanced: bool = True,
) -> pd.DataFrame:
    monthly = build_monthly_closes(daily)
    panel = attach_daily_rolls(daily, monthly)
    panel["SIZE"] = np.log(panel["avg_dv_21"].replace(0, np.nan))
    panel["VALUE"] = (panel["ma_252"] - panel["close_me"]) / panel["ma_252"]
    panel["MOM_12_1"] = compute_momentum_12_1(panel[["symbol", "month_end", "close_me"]])
    panel["QUALITY"] = panel["sharpe_3m"]

    keep = ["symbol", "month_end", "close_me", "fwd_1m_ret", *BASE_FACTOR_COLS]
    if include_enhanced:
        panel["REVERSAL_1M"] = -panel["ret_1m"]
        panel["VOL_1M"] = -panel["vol_21"]
        panel["ILLIQUIDITY"] = -np.log(panel["illiq_21"].replace(0, np.nan))
        keep.extend(ENHANCED_FACTOR_COLS)

    out = panel[keep].copy()
    out = out.dropna(subset=["fwd_1m_ret", *BASE_FACTOR_COLS], how="any")
    return out.sort_values(["symbol", "month_end"]).reset_index(drop=True)


def winsor_cross_section(s: pd.Series, p: float = 0.01) -> pd.Series:
    if s.dropna().empty:
        return s
    lower, upper = s.quantile(p), s.quantile(1 - p)
    return s.clip(lower=lower, upper=upper)


def zscore_cross_section(s: pd.Series) -> pd.Series:
    mean, std = s.mean(), s.std(ddof=0)
    if std == 0 or np.isnan(std):
        std = 1.0
    return (s - mean) / std


def standardize_cross_section(panel: pd.DataFrame, factor_cols: list[str], winsor_p: float = 0.01) -> pd.DataFrame:
    out = panel.copy()
    for factor in factor_cols:
        z_col = f"z_{factor}"
        out[z_col] = np.nan
        for _, idx in out.groupby("month_end").groups.items():
            values = winsor_cross_section(out.loc[idx, factor], winsor_p)
            out.loc[idx, z_col] = zscore_cross_section(values).values
    return out


def rank_ic_by_month(panel: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month_end, group in panel.groupby("month_end"):
        returns = group["fwd_1m_ret"]
        for factor in factor_cols:
            values = group[factor]
            valid = values.notna() & returns.notna()
            if valid.sum() < 3:
                continue
            ic = values[valid].rank().corr(returns[valid].rank())
            if pd.notna(ic):
                rows.append({"month_end": month_end, "factor": factor, "IC": float(ic)})
    if not rows:
        return pd.DataFrame(columns=["month_end", "factor", "IC"])
    return pd.DataFrame(rows).sort_values(["factor", "month_end"]).reset_index(drop=True)


def summarize_ic(ic: pd.DataFrame) -> pd.DataFrame:
    if ic.empty:
        return pd.DataFrame(columns=["factor", "mean", "std", "count", "ICIR"])
    summary = ic.groupby("factor")["IC"].agg(["mean", "std", "count"]).reset_index()
    summary["ICIR"] = summary["mean"] / summary["std"].replace(0, np.nan)
    return summary


def factor_corr_matrix(panel: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    z_cols = [f"z_{factor}" for factor in factor_cols if f"z_{factor}" in panel.columns]
    corr = panel[z_cols].corr()
    corr.index = [col.removeprefix("z_") for col in corr.index]
    corr.columns = [col.removeprefix("z_") for col in corr.columns]
    return corr


def plot_factor_outputs(ic: pd.DataFrame, corr: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    for factor, group in ic.groupby("factor"):
        path = out_dir / f"ic_hist_{factor}.png"
        plt.figure(figsize=(7, 4))
        group["IC"].hist(bins=40)
        plt.title(f"IC Histogram - {factor}")
        plt.xlabel("IC")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(path)

        path = out_dir / f"ic_ts_{factor}.png"
        ordered = group.sort_values("month_end").copy()
        ordered["cumIC"] = ordered["IC"].cumsum()
        plt.figure(figsize=(8, 3.6))
        plt.plot(ordered["month_end"], ordered["cumIC"])
        plt.title(f"Cumulative IC - {factor}")
        plt.xlabel("Month")
        plt.ylabel("Cumulative IC")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(path)

    path = out_dir / "factor_corr_heatmap.png"
    plt.figure(figsize=(6, 5))
    image = plt.imshow(corr.values, vmin=-1, vmax=1)
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(corr.shape[1]), corr.columns, rotation=45, ha="right")
    plt.yticks(range(corr.shape[0]), corr.index)
    plt.title("Factor Correlation")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    paths.append(path)
    return paths


def make_pdf_report(out_dir: Path, source_path: Path, ic_summary: pd.DataFrame, image_paths: list[Path]) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = out_dir / "Project4_Report.pdf"
    with PdfPages(pdf_path) as pdf:
        plt.figure(figsize=(8.27, 11.69))
        plt.axis("off")
        lines = [
            "Project 4 - Factor Construction and Validity Analysis",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Source: {source_path}",
            "",
            "Factors: VALUE, MOM_12_1, QUALITY, SIZE",
            "Enhanced factors: REVERSAL_1M, VOL_1M, ILLIQUIDITY",
            "Validity: monthly Spearman Rank IC vs forward 1M return.",
        ]
        y = 0.95
        for line in lines:
            plt.text(0.05, y, line, fontsize=12, va="top")
            y -= 0.04
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(8.27, 11.69))
        plt.axis("off")
        plt.text(0.05, 0.95, "IC Summary", fontsize=14, va="top")
        y = 0.90
        for _, row in ic_summary.iterrows():
            text = (
                f"{row['factor']:<16} mean={row['mean']:.4f} "
                f"std={row['std']:.4f} ICIR={row['ICIR']:.3f} n={int(row['count'])}"
            )
            plt.text(0.08, y, text, fontsize=11, va="top")
            y -= 0.035
        pdf.savefig()
        plt.close()

        for image_path in image_paths:
            if not image_path.exists():
                continue
            img = plt.imread(image_path)
            plt.figure(figsize=(8.27, 11.69))
            plt.imshow(img)
            plt.axis("off")
            pdf.savefig()
            plt.close()
    return pdf_path


def make_submission_zip(out_dir: Path) -> Path:
    zip_path = out_dir / "Project4_Submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in out_dir.iterdir():
            if path.name == zip_path.name:
                continue
            if path.is_file() and path.suffix.lower() in {".csv", ".png", ".pdf", ".json", ".md"}:
                archive.write(path, arcname=path.name)
    return zip_path


def write_factor_direction_report(ic_summary: pd.DataFrame, out_dir: Path) -> Path:
    path = out_dir / "factor_direction_report.md"
    lines = [
        "# Factor Direction Report",
        "",
        "Positive sign means the standardized factor is used as-is. Negative sign means the backtest should flip the factor before scoring.",
        "",
        "| Factor | Mean IC | Suggested sign |",
        "|---|---:|---:|",
    ]
    for _, row in ic_summary.iterrows():
        mean_ic = float(row["mean"])
        sign = 1.0 if mean_ic >= 0 else -1.0
        lines.append(f"| {row['factor']} | {mean_ic:.6f} | {sign:.1f} |")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_factors(cfg: dict[str, Any]) -> dict[str, Path]:
    paths = PipelinePaths.from_config(cfg)
    paths.ensure_output_dirs()
    factor_cfg = cfg.get("factors", {})
    research_cfg = cfg.get("research", {})

    daily, source_path = load_price_panel(paths)
    include_enhanced = bool(factor_cfg.get("include_enhanced", True))
    factor_cols = list(factor_cfg.get("factor_cols", BASE_FACTOR_COLS))
    if include_enhanced:
        factor_cols = factor_cols + [col for col in factor_cfg.get("enhanced_factor_cols", ENHANCED_FACTOR_COLS) if col not in factor_cols]

    panel = compute_factors(daily, include_enhanced=include_enhanced)
    standardizable = [col for col in factor_cols if col in panel.columns]
    panel = standardize_cross_section(panel, standardizable, float(factor_cfg.get("winsor_p", 0.01)))

    factors_path = paths.output_project4_dir / "factors.csv"
    ic_path = paths.output_project4_dir / "ic_series.csv"
    ic_summary_path = paths.output_project4_dir / "ic_summary.csv"
    corr_path = paths.output_project4_dir / "corr_matrix.csv"
    metadata_path = paths.output_project4_dir / "factor_run_metadata.json"

    panel.to_csv(factors_path, index=False, encoding="utf-8-sig")
    z_cols = [f"z_{col}" for col in standardizable if f"z_{col}" in panel.columns]
    ic = rank_ic_by_month(panel, z_cols)
    ic.to_csv(ic_path, index=False, encoding="utf-8-sig")
    ic_summary = summarize_ic(ic)
    ic_summary.to_csv(ic_summary_path, index=False, encoding="utf-8-sig")
    corr = factor_corr_matrix(panel, standardizable)
    corr.to_csv(corr_path, encoding="utf-8-sig")
    significance = ic_significance_table(
        ic,
        bootstrap_samples=int(research_cfg.get("bootstrap_samples", 1000)),
        ci=float(research_cfg.get("bootstrap_ci", 0.95)),
        seed=int(research_cfg.get("random_seed", 42)),
    )
    significance_path = paths.output_project4_dir / "ic_significance.csv"
    significance.to_csv(significance_path, index=False, encoding="utf-8-sig")

    horizons = [int(x) for x in factor_cfg.get("ic_decay_horizons", [1, 2, 3, 6])]
    decay = ic_decay(panel, z_cols, horizons)
    decay_path = paths.output_project4_dir / "ic_decay.csv"
    decay.to_csv(decay_path, index=False, encoding="utf-8-sig")
    decay_summary = summarize_ic_decay(decay)
    decay_summary_path = paths.output_project4_dir / "ic_decay_summary.csv"
    decay_summary.to_csv(decay_summary_path, index=False, encoding="utf-8-sig")

    group_returns = quantile_group_returns(
        panel,
        z_cols,
        quantiles=int(factor_cfg.get("quantile_groups", 5)),
        ret_col="fwd_1m_ret",
    )
    group_returns_path = paths.output_project4_dir / "factor_group_returns.csv"
    group_returns.to_csv(group_returns_path, index=False, encoding="utf-8-sig")
    group_summary = summarize_group_returns(group_returns)
    group_summary_path = paths.output_project4_dir / "factor_group_return_summary.csv"
    group_summary.to_csv(group_summary_path, index=False, encoding="utf-8-sig")

    fmb = fama_macbeth_regression(panel, z_cols, ret_col="fwd_1m_ret")
    fmb_path = paths.output_project4_dir / "fama_macbeth_summary.csv"
    fmb.to_csv(fmb_path, index=False, encoding="utf-8-sig")

    image_paths = plot_factor_outputs(ic, corr, paths.output_project4_dir)
    pdf_path = make_pdf_report(paths.output_project4_dir, source_path, ic_summary, image_paths)
    zip_path = make_submission_zip(paths.output_project4_dir)
    direction_path = write_factor_direction_report(ic_summary, paths.output_project4_dir)
    metadata_path.write_text(
        json.dumps(
            {
                "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": str(source_path),
                "factor_cols": factor_cols,
                "rows": int(len(panel)),
                "symbols": int(panel["symbol"].nunique()),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "factors": factors_path,
        "ic_series": ic_path,
        "ic_summary": ic_summary_path,
        "ic_significance": significance_path,
        "ic_decay": decay_path,
        "ic_decay_summary": decay_summary_path,
        "group_returns": group_returns_path,
        "group_return_summary": group_summary_path,
        "fama_macbeth": fmb_path,
        "corr_matrix": corr_path,
        "pdf_report": pdf_path,
        "submission_zip": zip_path,
        "direction_report": direction_path,
        "metadata": metadata_path,
    }
