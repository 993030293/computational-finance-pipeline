# -*- coding: utf-8 -*-
"""
Project4 - Factor Construction & Validity Analysis
Inputs (priority):
  1) ./data/processed/tech_indicators.csv
  2) ./data/processed/daily_price.csv or daily_price_50.csv
  3) ./data/raw/daily_price.csv

Outputs (under ./data/project4):
  - factors.csv
  - ic_series.csv, ic_summary.csv
  - corr_matrix.csv
  - ic_hist_*.png, ic_ts_*.png, factor_corr_heatmap.png
  - Project4_Report.pdf
  - Project4_Submission.zip

Notes:
- Matplotlib only; one chart per figure; no explicit colors.
- Factor set:
    Momentum (MOM_12_1): 12-month return excluding last month
    Size (SIZE): log(21d avg dollar volume) as proxy; if 'amount' missing, use close*volume
    Value (VALUE): proxy if no fundamentals -> (MA252 - close)/MA252
    Quality (QUALITY): rolling 3m Sharpe ratio of daily ret (mean/std)
- Rank IC: Spearman rank corr between factor_t and forward 1M return
"""

import os, json, math, zipfile
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# -------------------- PATHS --------------------
DATA_DIR = "./data"
PROC_DIR = f"{DATA_DIR}/processed"
RAW_DIR  = f"{DATA_DIR}/raw"
OUT_DIR  = f"{DATA_DIR}/project4"
os.makedirs(OUT_DIR, exist_ok=True)

# candidate inputs
CANDIDATES = [
    f"{PROC_DIR}/tech_indicators.csv",
    f"{PROC_DIR}/daily_price.csv",
    f"{PROC_DIR}/daily_price_50.csv",
    f"{RAW_DIR}/daily_price.csv",
]

# -------------------- UTILS --------------------
def _read_first_available(paths):
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, encoding="utf-8-sig")
                return df, p
            except Exception:
                continue
    raise FileNotFoundError("No input price file found. Expected one of: " + ", ".join(paths))

def _rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    m = {
        "日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low",
        "成交量":"volume","成交额":"amount","涨跌幅":"pct_chg","代码":"symbol",
        "Date":"date","Open":"open","Close":"close","High":"high","Low":"low",
        "Volume":"volume","Amount":"amount","Pct_chg":"pct_chg","Code":"symbol",
    }
    exist = {k:v for k,v in m.items() if k in df.columns}
    return df.rename(columns=exist, inplace=False)

def _to_dt(df, col="date"):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def _to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

def _winsor_cs(s: pd.Series, p=0.01):
    if s.dropna().empty:
        return s
    lo, hi = s.quantile(p), s.quantile(1-p)
    return s.clip(lower=lo, upper=hi)

def _zscore_cs(s: pd.Series):
    mu, sd = s.mean(), s.std(ddof=0)
    return (s - mu) / (sd if (sd and not np.isnan(sd) and sd!=0) else 1.0)

# -------------------- LOAD & PREPARE --------------------
def load_price_panel():
    df, used_path = _read_first_available(CANDIDATES)
    df = _rename_cols(df)
    need = ["symbol","date","open","high","low","close","volume","amount"]
    df = _ensure_cols(df, need)
    df = _to_dt(df, "date")
    df = _to_num(df, ["open","high","low","close","volume","amount"])
    df = df.dropna(subset=["symbol","date"]).sort_values(["symbol","date"])
    # fix high<low
    m = df["high"].notna() & df["low"].notna() & (df["high"] < df["low"])
    if m.any():
        hi, lo = df.loc[m,"high"].copy(), df.loc[m,"low"].copy()
        df.loc[m,"high"], df.loc[m,"low"] = lo, hi
    # pct_chg fallback
    if ("pct_chg" not in df.columns) or (df["pct_chg"].isna().mean() > 0.9):
        df["pct_chg"] = df.groupby("symbol")["close"].pct_change() * 100.0
    return df, used_path

def build_monthly_closes(daily: pd.DataFrame):
    """Last observation of each calendar month per symbol; also forward 1M return."""
    d = daily[["symbol","date","close","volume","amount"]].copy()
    d["ym"] = d["date"].dt.to_period("M")
    last = d.sort_values("date").groupby(["symbol","ym"]).tail(1).copy()
    last["month_end"] = last["ym"].dt.to_timestamp("M")
    last = last.sort_values(["symbol","month_end"])
    # forward 1M return (month-end to next month-end)
    last["fwd_1m_ret"] = last.groupby("symbol")["close"].pct_change(periods=1).shift(-1)
    # keep clean
    last = last.rename(columns={"close":"close_me","volume":"vol_me","amount":"amt_me"})
    return last[["symbol","month_end","close_me","vol_me","amt_me","fwd_1m_ret"]]

def attach_daily_rolls(daily: pd.DataFrame, monthly: pd.DataFrame):
    """Attach rolling daily features evaluated at month-end dates for each symbol."""
    x = daily[["symbol","date","close","volume","amount"]].copy().sort_values(["symbol","date"])
    # 21d avg dollar volume
    if "amount" in x.columns and x["amount"].notna().any():
        x["dollar_vol"] = x["amount"]
    else:
        x["dollar_vol"] = x["close"] * x["volume"]
    x["avg_dv_21"] = x.groupby("symbol")["dollar_vol"].rolling(21, min_periods=10).mean().reset_index(level=0, drop=True)

    # momentum: 12-1 using month-end closes -> compute on monthly frame later
    # value proxy: MA252
    x["ma_252"] = x.groupby("symbol")["close"].rolling(252, min_periods=126).mean().reset_index(level=0, drop=True)

    # quality proxy: rolling Sharpe over past 63d on daily returns
    x["ret_d"] = x.groupby("symbol")["close"].pct_change()
    # mean/std with min_periods to avoid early NaNs
    mu = x.groupby("symbol")["ret_d"].rolling(63, min_periods=40).mean().reset_index(level=0, drop=True)
    sd = x.groupby("symbol")["ret_d"].rolling(63, min_periods=40).std(ddof=0).reset_index(level=0, drop=True)
    x["sharpe_3m"] = mu / sd.replace(0, np.nan)

    # merge month-end rows
    m = monthly.merge(x, left_on=["symbol","month_end"], right_on=["symbol","date"], how="left")
    m = m.drop(columns=["date"])
    return m

def compute_momentum_12_1(monthly_close: pd.DataFrame):
    """Use monthly closes; MOM_12_1 = close(t-1)/close(t-12) - 1."""
    z = monthly_close.sort_values(["symbol","month_end"]).copy()
    z["close_lag1"]  = z.groupby("symbol")["close_me"].shift(1)
    z["close_lag12"] = z.groupby("symbol")["close_me"].shift(12)
    mom = z["close_lag1"] / z["close_lag12"] - 1.0
    return mom

def compute_factors(daily: pd.DataFrame):
    """Return monthly factor panel with VALUE, MOM_12_1, QUALITY, SIZE."""
    monthly = build_monthly_closes(daily)
    m = attach_daily_rolls(daily, monthly)

    # SIZE: log(21d avg dollar volume) at month-end
    m["SIZE"] = np.log(m["avg_dv_21"].replace(0, np.nan))

    # VALUE:
    # 1) If ma_252 exists: VALUE_proxy = (MA252 - close)/MA252  (cheap => positive)
    # 2) If you later add PE_TTM: prefer VALUE = 1/PE_TTM
    m["VALUE"] = (m["ma_252"] - m["close_me"]) / m["ma_252"]

    # MOM_12_1
    m["MOM_12_1"] = compute_momentum_12_1(m[["symbol","month_end","close_me"]])

    # QUALITY: rolling 3m Sharpe ratio (daily)
    m["QUALITY"] = m["sharpe_3m"]

    # keep essential
    keep = ["symbol","month_end","close_me","fwd_1m_ret","VALUE","MOM_12_1","QUALITY","SIZE"]
    panel = m[keep].copy()
    # Drop earliest periods with insufficient lookbacks
    panel = panel.dropna(subset=["fwd_1m_ret","MOM_12_1","SIZE","VALUE","QUALITY"], how="any")
    return panel

def standardize_cross_section(panel: pd.DataFrame, cols):
    """Cross-sectional winsor + zscore per date."""
    out = panel.copy()
    for d, g in out.groupby("month_end"):
        idx = g.index
        for c in cols:
            w = _winsor_cs(g[c], p=0.01)
            z = _zscore_cs(w)
            out.loc[idx, f"z_{c}"] = z.values
    return out

# -------------------- IC & CORR --------------------
def rank_ic_by_month(panel: pd.DataFrame, factor_cols):
    """Spearman rank corr between factor_t and fwd_1m_ret per month."""
    from scipy.stats import spearmanr
    rows = []
    for d, g in panel.groupby("month_end"):
        ret = g["fwd_1m_ret"]
        for f in factor_cols:
            x = g[f]
            ok = x.notna() & ret.notna()
            if ok.sum() >= 10:
                rho, _ = spearmanr(x[ok], ret[ok])
                rows.append({"month_end": d, "factor": f, "IC": float(rho)})
    ic = pd.DataFrame(rows).sort_values(["factor","month_end"])
    return ic

def summarize_ic(ic: pd.DataFrame):
    summ = ic.groupby("factor")["IC"].agg(["mean","std","count"]).reset_index()
    summ["ICIR"] = summ["mean"] / summ["std"].replace(0, np.nan)
    return summ

def factor_corr_matrix(panel: pd.DataFrame, factor_cols):
    """Pooled (all months) correlation on standardized factors."""
    zcols = [f"z_{c}" for c in factor_cols]
    c = panel[zcols].corr()
    c.index = factor_cols
    c.columns = factor_cols
    return c

# -------------------- PLOTS --------------------
def plot_ic_hist(ic: pd.DataFrame, out_dir: str):
    for f, g in ic.groupby("factor"):
        plt.figure(figsize=(7,4))
        g["IC"].hist(bins=40)
        plt.title(f"IC Histogram - {f}")
        plt.xlabel("IC")
        plt.ylabel("Frequency")
        plt.tight_layout()
        path = os.path.join(out_dir, f"ic_hist_{f}.png")
        plt.savefig(path, dpi=160)
        plt.close()

def plot_ic_timeseries(ic: pd.DataFrame, out_dir: str):
    for f, g in ic.groupby("factor"):
        g = g.sort_values("month_end")
        g["cumIC"] = g["IC"].cumsum()
        plt.figure(figsize=(8,3.6))
        plt.plot(g["month_end"], g["cumIC"])
        plt.title(f"Cumulative IC - {f}")
        plt.xlabel("Date")
        plt.ylabel("Cumulative IC")
        plt.tight_layout()
        path = os.path.join(out_dir, f"ic_ts_{f}.png")
        plt.savefig(path, dpi=160)
        plt.close()

def plot_corr_heatmap(C: pd.DataFrame, out_dir: str):
    plt.figure(figsize=(6,5))
    im = plt.imshow(C.values, vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(C.shape[1]), C.columns, rotation=45, ha="right")
    plt.yticks(range(C.shape[0]), C.index)
    plt.title("Factor Correlation Heatmap")
    plt.tight_layout()
    path = os.path.join(out_dir, "factor_corr_heatmap.png")
    plt.savefig(path, dpi=160)
    plt.close()


def make_pdf_report(out_dir: str, used_path: str, ic_summary: pd.DataFrame, img_paths: list):
    pdf_path = os.path.join(out_dir, "Project4_Report.pdf")
    with PdfPages(pdf_path) as pdf:
        # Page 1: Title & Overview
        plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        plt.axis("off")
        lines = [
            "Computational Finance (MA216) Project 4",
            "Factor Construction & Validity Analysis",
            "",
            f"Data source file: {used_path}",
            f"Generated at: {datetime.now():%Y-%m-%d %H:%M:%S}",
            "",
            "Factors:",
            "  - VALUE: (MA252 - Close)/MA252  [proxy when fundamentals unavailable]",
            "  - MOM_12_1: 12-month return excluding last month",
            "  - QUALITY: rolling 3m Sharpe ratio of daily returns",
            "  - SIZE: log(21d avg dollar volume)",
            "",
            "Validity Analysis:",
            "  - Rank IC per month: Spearman corr between factor_t and forward 1M return",
            "  - Summary metrics: mean IC, std, ICIR",
        ]
        y = 0.95
        for txt in lines:
            plt.text(0.05, y, txt, fontsize=12, va="top")
            y -= 0.04
        pdf.savefig(); plt.close()

        # Page 2: IC Summary Table
        plt.figure(figsize=(8.27, 11.69))
        plt.axis("off")
        plt.text(0.05, 0.95, "IC Summary (per factor)", fontsize=14, va="top")
        y = 0.90
        cols = ["factor","mean","std","ICIR","count"]
        df_show = ic_summary[cols].copy()
        df_show = df_show.round({"mean":4,"std":4,"ICIR":3})
        for _, r in df_show.iterrows():
            row = f"{r['factor']:<10} mean={r['mean']:.4f}  std={r['std']:.4f}  ICIR={r['ICIR']:.3f}  n={int(r['count'])}"
            plt.text(0.08, y, row, fontsize=12, va="top")
            y -= 0.04
        pdf.savefig(); plt.close()

        # Pages: images
        for p in img_paths:
            if os.path.exists(p):
                img = plt.imread(p)
                plt.figure(figsize=(8.27, 11.69))
                plt.imshow(img)
                plt.axis("off")
                pdf.savefig(); plt.close()
    return pdf_path

# -------------------- ZIP PACKAGE --------------------
def make_zip(out_dir: str):
    zip_path = os.path.join(out_dir, "Project4_Submission.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fn in ["factors.csv","ic_series.csv","ic_summary.csv","corr_matrix.csv","Project4_Report.pdf"]:
            fpath = os.path.join(out_dir, fn)
            if os.path.exists(fpath):
                z.write(fpath, arcname=fn)
        # images
        for p in glob(os.path.join(out_dir, "ic_hist_*.png")) + \
                 glob(os.path.join(out_dir, "ic_ts_*.png")) + \
                 glob(os.path.join(out_dir, "factor_corr_heatmap.png")):
            z.write(p, arcname=os.path.basename(p))
    return zip_path

# -------------------- MAIN --------------------
def main():
    daily, used_path = load_price_panel()
    panel = compute_factors(daily)
    # standardize cross-section
    factor_cols = ["VALUE","MOM_12_1","QUALITY","SIZE"]
    panel = standardize_cross_section(panel, factor_cols)

    # save factors
    factors_path = os.path.join(OUT_DIR, "factors.csv")
    panel.to_csv(factors_path, index=False, encoding="utf-8-sig")

    # IC
    ic = rank_ic_by_month(panel, [f"z_{c}" for c in factor_cols])
    ic_path = os.path.join(OUT_DIR, "ic_series.csv")
    ic.to_csv(ic_path, index=False, encoding="utf-8-sig")

    ic_sum = summarize_ic(ic)
    ic_sum_path = os.path.join(OUT_DIR, "ic_summary.csv")
    ic_sum.to_csv(ic_sum_path, index=False, encoding="utf-8-sig")

    # Corr
    C = factor_corr_matrix(panel, factor_cols)
    C_path = os.path.join(OUT_DIR, "corr_matrix.csv")
    C.to_csv(C_path, encoding="utf-8-sig")

    # Plots
    plot_ic_hist(ic, OUT_DIR)
    plot_ic_timeseries(ic, OUT_DIR)
    plot_corr_heatmap(C, OUT_DIR)

    # Report
    imgs = sorted(glob(os.path.join(OUT_DIR, "ic_hist_*.png"))) + \
           sorted(glob(os.path.join(OUT_DIR, "ic_ts_*.png"))) + \
           [os.path.join(OUT_DIR, "factor_corr_heatmap.png")]
    pdf_path = make_pdf_report(OUT_DIR, used_path, ic_sum, imgs)

    # Zip
    zip_path = make_zip(OUT_DIR)

    # Console summary
    print("[OK] Factors  ->", os.path.abspath(factors_path))
    print("[OK] IC series->", os.path.abspath(ic_path))
    print("[OK] IC summary->", os.path.abspath(ic_sum_path))
    print("[OK] Corr     ->", os.path.abspath(C_path))
    print("[OK] Report   ->", os.path.abspath(pdf_path))
    print("[OK] ZIP      ->", os.path.abspath(zip_path))

if __name__ == "__main__":
    main()
