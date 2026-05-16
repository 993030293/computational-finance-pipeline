# -*- coding: utf-8 -*-
"""
Project3 - 数据清洗与探索性分析 (与 Project2/实时链路兼容)
输出:
  - ./data/processed/tech_indicators.csv
  - ./data/QUALITY_REPORT.md
  - ./data/eda/*.png
  - ./data/CLEANING_STEPS.json        # 处理流程与参数快照，便于复现

输入优先级 (历史 + 可选实时)：
  1) ./data/processed/daily_price.csv
  2) ./data/raw/daily_price.csv
  3) ./data/realtime/minute_*.csv     # 若1/2都空，则由分钟线聚合为日线
  4) 兜底: /mnt/data/fixed_daily_price.csv, /mnt/data/fixed_minute_1m.csv (可选, 若在同环境存在)

图表仅用 matplotlib，无 seaborn；每个图单独成像；不指定配色。
"""

import os, json, math
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- 路径 --------------------
DATA_DIR = "./data"
RAW = f"{DATA_DIR}/raw"
PROC = f"{DATA_DIR}/processed"
RT = f"{DATA_DIR}/realtime"
EDA_DIR = f"{DATA_DIR}/eda"
os.makedirs(PROC, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)

DAILY_PROC = f"{PROC}/daily_price_50.csv"
DAILY_RAW = f"{RAW}/daily_price.csv"
OUTPUT_CSV = f"{PROC}/tech_indicators.csv"
QUALITY_MD = f"{DATA_DIR}/QUALITY_REPORT.md"
STEPS_JSON = f"{DATA_DIR}/CLEANING_STEPS.json"
INTRADAY_COMPARE_CSV = f"{PROC}/intraday_daily_compare.csv"

# -------------------- 配置（可改） --------------------
CFG = dict(
    winsor_lower_q = 0.005,
    winsor_upper_q = 0.995,
    z_thresh = 3.0,
    iqr_k = 1.5,
    ma_short = 5,
    ma_long = 20,
    ema_fast = 12,
    ema_slow = 26,
    ema_signal = 9,
    vol_win = 20,
    rsi_period = 14,
    ts_plots_n = 3,        # 画价格时间序列的股票数（前 N 个）
    rng_seed = 42,         # 抽样散点图复现实验
)

# -------------------- 工具 --------------------
def _rename_any_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        # 中文
        "日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low",
        "成交量":"volume","成交额":"amount","涨跌幅":"pct_chg","涨跌额":"change",
        "换手率":"turnover_rate","振幅":"amplitude","代码":"symbol","名称":"name",
        # 英文
        "Date":"date","Open":"open","Close":"close","High":"high","Low":"low",
        "Volume":"volume","Amount":"amount","Pct_chg":"pct_chg","Change":"change",
        # 分钟线/实时
        "day":"date","时间":"date","最新价":"price","今开":"open","昨收":"pre_close",
        "市盈率-动态":"pe","换手率":"turnover",
        "code":"symbol",  # 常见股票池 code→symbol
    }
    exist = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(columns=exist, inplace=False)

def _ensure_cols(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

def _to_datetime(df: pd.DataFrame, col="date") -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def _to_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _winsorize_series(s: pd.Series, lower_q=0.005, upper_q=0.995) -> pd.Series:
    if s.isna().all():
        return s
    lo, hi = s.quantile(lower_q), s.quantile(upper_q)
    return s.clip(lower=lo, upper=hi)

def _zscore_flags(s: pd.Series, thresh=3.0) -> pd.Series:
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(False, index=s.index)
    z = (s - mu) / sd
    return z.abs() > thresh

def _iqr_flags(s: pd.Series, k=1.5) -> pd.Series:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or np.isnan(iqr):
        return pd.Series(False, index=s.index)
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (s < lo) | (s > hi)

# -------------------- 读入数据 --------------------
def _read_csv_if(path):
    return pd.read_csv(path, encoding="utf-8-sig") if os.path.exists(path) else None

def read_minute_optional() -> pd.DataFrame:
    files = sorted(glob(os.path.join(RT, "minute_*m.csv")))
    if not files:
        # 兜底：同环境的固定文件（若存在）
        fallback = "/mnt/data/fixed_minute_1m.csv"
        files = [fallback] if os.path.exists(fallback) else []
    frames = []
    for p in files:
        try:
            x = pd.read_csv(p, encoding="utf-8-sig")
            x = _rename_any_columns(x)
            x = _ensure_cols(x, ["symbol","date","open","high","low","close","volume","period","ts_fetch"])
            x = _to_datetime(x, "date")
            x = _to_numeric(x, ["open","high","low","close","volume"])
            frames.append(x)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["symbol","date"]).sort_values(["symbol","date"])
    return df

def _rebuild_daily_from_minute(minute_df: pd.DataFrame) -> pd.DataFrame:
    if minute_df is None or minute_df.empty:
        return pd.DataFrame()
    m = minute_df.copy()
    m["d"] = pd.to_datetime(m["date"]).dt.normalize()
    need = ["symbol","date","open","high","low","close"]
    m = m.dropna(subset=need)
    def agg_one(g):
        g = g.sort_values("date")
        return pd.Series({
            "open":  g["open"].iloc[0],
            "high":  g["high"].max(),
            "low":   g["low"].min(),
            "close": g["close"].iloc[-1],
            "volume":g["volume"].sum(skipna=True),
            "amount":np.nan
        })
    d = m.groupby(["symbol","d"], as_index=False).apply(agg_one).rename(columns={"d":"date"})
    d = d.sort_values(["symbol","date"]).reset_index(drop=True)
    d["pct_chg"] = d.groupby("symbol")["close"].pct_change() * 100.0
    return d

def read_daily_main() -> tuple[pd.DataFrame, dict]:
    """
    返回: (daily_df, source_info)
    source_info: {'path': '...', 'note': 'processed/raw/rebuilt/fixed'}
    """
    # 1) processed
    df = _read_csv_if(DAILY_PROC)
    if df is not None and len(df) > 0:
        df = _rename_any_columns(df)
        return df, {"path": DAILY_PROC, "note": "processed"}

    # 2) raw
    df = _read_csv_if(DAILY_RAW)
    if df is not None and len(df) > 0:
        df = _rename_any_columns(df)
        return df, {"path": DAILY_RAW, "note": "raw"}

    # 3) minute -> rebuild
    minute = read_minute_optional()
    rebuilt = _rebuild_daily_from_minute(minute)
    if len(rebuilt) > 0:
        return rebuilt, {"path": "minute_*m.csv", "note": "rebuilt_from_minute"}

    # 4) fixed fallback
    fixed = _read_csv_if("/mnt/data/fixed_daily_price.csv")
    if fixed is not None and len(fixed) > 0:
        fixed = _rename_any_columns(fixed)
        return fixed, {"path": "/mnt/data/fixed_daily_price_50.csv", "note": "fixed_fallback"}

    raise FileNotFoundError("未找到可用的日线数据。请先完成 Project2 或提供分钟线。")

# -------------------- 质量检查 --------------------
def dtypes_profile(df: pd.DataFrame) -> dict:
    return {c: str(t) for c, t in df.dtypes.items()}

def quality_checks(df: pd.DataFrame) -> dict:
    out = {}
    out["shape"] = list(df.shape)
    out["n_symbols"] = int(df["symbol"].nunique()) if "symbol" in df.columns else 0
    try:
        dmin, dmax = df["date"].min(), df["date"].max()
        out["date_range"] = [None if pd.isna(dmin) else pd.to_datetime(dmin).strftime("%Y-%m-%d"),
                             None if pd.isna(dmax) else pd.to_datetime(dmax).strftime("%Y-%m-%d")]
    except Exception:
        out["date_range"] = [None, None]
    out["dup_rows"] = int(df.duplicated().sum())
    out["dup_symbol_date"] = int(df.duplicated(subset=["symbol","date"]).sum()) if {"symbol","date"}.issubset(df.columns) else None
    out["na_ratio"] = df.isna().mean().sort_values(ascending=False).to_dict()
    if set(["open","high","low","close"]).issubset(df.columns):
        out["neg_price_rows"] = int((df[["open","high","low","close"]] < 0).any(axis=1).sum())
        out["high_lt_low_rows"] = int((df["high"] < df["low"]).sum())
    else:
        out["neg_price_rows"] = None
        out["high_lt_low_rows"] = None
    out["dtypes"] = dtypes_profile(df)
    return out

# -------------------- 清洗与指标 --------------------
def engineer_features(df: pd.DataFrame,
                      winsorize=True,
                      cfg: dict = CFG) -> tuple[pd.DataFrame, dict]:
    # 标准化+基础清洗
    need = ["symbol","date","open","high","low","close","volume","amount","pct_chg"]
    x = _rename_any_columns(df)
    x = _ensure_cols(x, need)
    x = _to_datetime(x, "date")
    x = _to_numeric(x, ["open","high","low","close","volume","amount","pct_chg"])
    x = x.dropna(subset=["symbol","date"]).sort_values(["symbol","date"])
    x = x.drop_duplicates(subset=["symbol","date"], keep="last")

    # 修复 high<low
    m = x["high"].notna() & x["low"].notna() & (x["high"] < x["low"])
    if m.any():
        hi = x.loc[m, "high"].copy()
        lo = x.loc[m, "low"].copy()
        x.loc[m, "high"] = lo
        x.loc[m, "low"] = hi
    # pct_chg 兜底
    if x["pct_chg"].isna().mean() > 0.9:
        x["pct_chg"] = x.groupby("symbol")["close"].pct_change() * 100.0

    stats = {"before": quality_checks(x)}

    # 收益
    x["ret"] = x.groupby("symbol")["close"].pct_change()
    x["log_ret"] = np.log(x["close"]).groupby(x["symbol"]).diff()

    # 异常值标记（ret）
    x["flag_z"] = False
    x["flag_iqr"] = False
    for sym, g in x.groupby("symbol", sort=False):
        idx = g.index
        s = g["ret"]
        x.loc[idx, "flag_z"] = _zscore_flags(s, thresh=cfg["z_thresh"]).values
        x.loc[idx, "flag_iqr"] = _iqr_flags(s, k=cfg["iqr_k"]).values

    # winsorize
    x["ret_w"] = x.groupby("symbol", group_keys=False)["ret"].apply(
        lambda s: _winsorize_series(s, cfg["winsor_lower_q"], cfg["winsor_upper_q"])
    ) if winsorize else x["ret"]

    # 指标
    def _ema(s, span):
        return s.ewm(span=span, adjust=False, min_periods=span).mean()
    def _rsi(close, period=14):
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period, min_periods=period).mean()
        avg_loss = loss.rolling(period, min_periods=period).mean()
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        return 100 - (100 / (1 + rs))

    out_frames = []
    for sym, g in x.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        g["ma_5"] = g["close"].rolling(cfg["ma_short"], min_periods=cfg["ma_short"]).mean()
        g["ma_20"] = g["close"].rolling(cfg["ma_long"],  min_periods=cfg["ma_long"]).mean()
        ema12 = _ema(g["close"], cfg["ema_fast"])
        ema26 = _ema(g["close"], cfg["ema_slow"])
        macd = ema12 - ema26
        signal = _ema(macd, cfg["ema_signal"])
        g["ema12"], g["ema26"], g["macd"], g["macd_signal"] = ema12, ema26, macd, signal
        g["vol_20"] = g["log_ret"].rolling(cfg["vol_win"], min_periods=cfg["vol_win"]).std(ddof=0) * np.sqrt(252)
        g["rsi_14"] = _rsi(g["close"], cfg["rsi_period"])
        g["abs_ret"] = g["ret_w"].abs()
        out_frames.append(g)

    final = pd.concat(out_frames, ignore_index=True)

    # 相关性
    num_cols = ["ret_w","log_ret","vol_20","macd","macd_signal","ma_5","ma_20","rsi_14","abs_ret"]
    corr = final[num_cols].astype(float).corr(min_periods=100)

    # 汇总
    stats["after"] = {
        "shape": list(final.shape),
        "na_ratio": final.isna().mean().sort_values(ascending=False).to_dict(),
        "flag_z_count": int(final["flag_z"].sum()),
        "flag_iqr_count": int(final["flag_iqr"].sum()),
        "summary_ret": {
            "count": int(final["ret"].count()),
            "mean": float(final["ret"].mean()),
            "std": float(final["ret"].std(ddof=0)),
            "skew": float(final["ret"].skew()),
            "kurt": float(final["ret"].kurt())
        }
    }
    stats["corr_cols"] = num_cols
    stats["corr_matrix"] = corr.round(4).to_dict()
    return final, stats

# -------------------- 盘中 vs 当日对齐（可选一致性） --------------------
def intraday_daily_compare(daily_df: pd.DataFrame, minute_df: pd.DataFrame) -> pd.DataFrame:
    if minute_df is None or minute_df.empty:
        return pd.DataFrame()
    dfm = minute_df.copy()
    dfm["date_only"] = pd.to_datetime(dfm["date"]).dt.normalize()
    today = pd.Timestamp.today().normalize()
    m_today = dfm[dfm["date_only"] == today]
    if m_today.empty:
        return pd.DataFrame()
    last_min = m_today.sort_values("date").groupby("symbol").tail(1)[["symbol","close"]].rename(columns={"close":"close_intraday"})
    d_today = daily_df[daily_df["date"] == today][["symbol","close"]].rename(columns={"close":"close_daily"})
    comp = pd.merge(last_min, d_today, on="symbol", how="inner")
    if comp.empty:
        return pd.DataFrame()
    comp["abs_diff"] = (comp["close_intraday"] - comp["close_daily"]).abs()
    comp["rel_diff_pct"] = comp["abs_diff"] / comp["close_daily"].replace(0, np.nan) * 100
    comp = comp.sort_values("rel_diff_pct", ascending=False)
    comp.to_csv(INTRADAY_COMPARE_CSV, index=False, encoding="utf-8-sig")
    return comp

# -------------------- EDA 图 --------------------
def plot_eda(df: pd.DataFrame, comp_df: pd.DataFrame, cfg: dict = CFG):
    np.random.seed(cfg["rng_seed"])

    # 1) 收益直方图
    plt.figure(figsize=(7,4))
    df["ret_w"].dropna().hist(bins=80)
    plt.title("Histogram of Daily Returns (winsorized)")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{EDA_DIR}/hist_returns.png", dpi=160)
    plt.close()

    # 2) 收益箱形图（异常值可视）
    plt.figure(figsize=(6,4))
    plt.boxplot(df["ret_w"].dropna().values, vert=True, whis=1.5)
    plt.title("Boxplot of Daily Returns")
    plt.ylabel("Return")
    plt.tight_layout()
    plt.savefig(f"{EDA_DIR}/boxplot_returns.png", dpi=160)
    plt.close()

    # 3) 前 N 个股票的价格时间序列
    syms = list(df["symbol"].dropna().unique())[:cfg["ts_plots_n"]]
    for s in syms:
        g = df[df["symbol"] == s]
        plt.figure(figsize=(8,3.6))
        plt.plot(g["date"], g["close"])
        plt.title(f"Price Series - {s}")
        plt.xlabel("Date")
        plt.ylabel("Close")
        plt.tight_layout()
        plt.savefig(f"{EDA_DIR}/price_series_{s}.png", dpi=160)
        plt.close()

    # 4) 波动率-收益散点
    sample = df.dropna(subset=["vol_20","ret_w"])
    if len(sample) > 0:
        if len(sample) > 5000:
            sample = sample.sample(n=5000, random_state=cfg["rng_seed"])
        plt.figure(figsize=(6.2,4.6))
        plt.scatter(sample["vol_20"], sample["ret_w"], s=8, alpha=0.4)
        plt.title("Volatility vs Return")
        plt.xlabel("Annualized Vol (20d)")
        plt.ylabel("Daily Return")
        plt.tight_layout()
        plt.savefig(f"{EDA_DIR}/scatter_vol_return.png", dpi=160)
        plt.close()

    # 5) 相关性热力图
    cols = ["ret_w","log_ret","vol_20","macd","macd_signal","ma_5","ma_20","rsi_14","abs_ret"]
    c = df[cols].astype(float).corr(min_periods=100)
    plt.figure(figsize=(7,6))
    im = plt.imshow(c, vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{EDA_DIR}/corr_heatmap.png", dpi=160)
    plt.close()

    # 6) 盘中 vs 当日 close 散点（若有）
    if comp_df is not None and not comp_df.empty:
        plt.figure(figsize=(6.2,4.6))
        plt.scatter(comp_df["close_daily"], comp_df["close_intraday"], s=12, alpha=0.6)
        lims = [
            min(comp_df["close_daily"].min(), comp_df["close_intraday"].min()),
            max(comp_df["close_daily"].max(), comp_df["close_intraday"].max())
        ]
        plt.plot(lims, lims)
        plt.title("Intraday Last vs Daily Close (Today)")
        plt.xlabel("Daily Close")
        plt.ylabel("Intraday Last Close")
        plt.tight_layout()
        plt.savefig(f"{EDA_DIR}/intraday_vs_daily.png", dpi=160)
        plt.close()

# -------------------- 报告 --------------------
def write_quality_report(stats: dict,
                         daily_info: dict,
                         source_info: dict,
                         minute_rows: int,
                         comp_summary: dict,
                         cfg: dict,
                         path=QUALITY_MD):
    before = stats.get("before", {})
    after = stats.get("after", {})
    corr_cols = stats.get("corr_cols", [])
    corr_mat = stats.get("corr_matrix", {})

    md = []
    md.append("# 数据质量与探索性分析报告 (Project 3)")
    md.append("")
    md.append(f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}")
    md.append(f"- 主数据来源：`{source_info.get('path')}`  ({source_info.get('note')})")
    md.append(f"- 是否有分钟线：{'是' if minute_rows>0 else '否'}；盘中对齐股票数：{comp_summary.get('rows',0)}")
    md.append("")
    md.append("## 一、数据质量检查（原始概况）")
    md.append(f"- 形状：{before.get('shape')}")
    md.append(f"- 股票数：{before.get('n_symbols')}")
    md.append(f"- 时间范围：{before.get('date_range')}")
    md.append(f"- 重复行：{before.get('dup_rows')} ；重复键(symbol,date)：{before.get('dup_symbol_date')}")
    md.append(f"- 负价格行数：{before.get('neg_price_rows')} ；high<low 行数：{before.get('high_lt_low_rows')}")
    md.append("")
    md.append("**字段类型**")
    for k, v in before.get("dtypes", {}).items():
        md.append(f"- {k}: {v}")
    md.append("")
    md.append("**缺失率（Top 10）**")
    na_sorted = sorted(before.get("na_ratio", {}).items(), key=lambda x: x[1], reverse=True)[:10]
    for k, v in na_sorted:
        md.append(f"- {k}: {v:.2%}")

    md.append("")
    md.append("## 二、异常值处理与标记")
    md.append(f"- Z 分数阈值：|Z| > {CFG['z_thresh']} ；IQR：超出 {CFG['iqr_k']}×IQR 区间记为异常")
    md.append(f"- Z 异常标记数：{after.get('flag_z_count')} ；IQR 异常标记数：{after.get('flag_iqr_count')}")
    md.append(f"- Winsorize 分位：[{CFG['winsor_lower_q']:.3f}, {CFG['winsor_upper_q']:.3f}] 应用于日收益 ret")
    md.append("")
    md.append("## 三、清洗后概况与指标列")
    md.append(f"- 形状：{after.get('shape')}")
    md.append("**清洗后缺失率（Top 10）**")
    na2 = sorted(after.get("na_ratio", {}).items(), key=lambda x: x[1], reverse=True)[:10]
    for k, v in na2:
        md.append(f"- {k}: {v:.2%}")
    sr = after.get("summary_ret", {})
    md.append("")
    md.append("**日收益 ret 的统计**")
    md.append(f"- count={sr.get('count')}  mean={sr.get('mean'):.6f}  std={sr.get('std'):.6f}  skew={sr.get('skew'):.6f}  kurt={sr.get('kurt'):.6f}")

    md.append("")
    md.append("## 四、相关性矩阵（选列）")
    md.append(f"- 列：{', '.join(corr_cols)}")
    # 打印前 6 行 × 前 6 列
    keys = list(corr_mat.keys())[:6]
    md.append("```text")
    for r in keys:
        row = corr_mat[r]
        cols = list(row.keys())[:6]
        line = " ".join([f"{row[c]:>7.3f}" for c in cols])
        md.append(f"{r:<12} {line}")
    md.append("```")

    md.append("")
    md.append("## 五、EDA 图表清单")
    md.append("- hist_returns.png：日收益直方图")
    md.append("- boxplot_returns.png：日收益箱形图")
    md.append("- price_series_*.png：价格时间序列（前若干股票）")
    md.append("- scatter_vol_return.png：波动率 vs 收益散点")
    md.append("- corr_heatmap.png：相关性热力图")
    if comp_summary.get("rows", 0) > 0:
        md.append("- intraday_vs_daily.png：盘中 vs 当日收盘对齐散点")
    md.append("")
    md.append("## 六、盘中与当日一致性（若当日有分钟线）")
    if comp_summary.get("rows", 0) > 0:
        md.append(f"- 对齐股票数：{comp_summary['rows']}")
        md.append(f"- 中位相对误差：{comp_summary['med_rel']:.4f}%")
        md.append(f"- 95 分位相对误差：{comp_summary['p95_rel']:.4f}%")
        md.append(f"- 最大相对误差：{comp_summary['max_rel']:.4f}%")
        md.append(f"- 结果 CSV：`{INTRADAY_COMPARE_CSV}`")
    else:
        md.append("- 当日无可比分钟/日线或尚未开盘")

    md.append("")
    md.append("## 七、处理步骤与参数（可复现）")
    md.append("- 列名标准化、类型纠正、去重（key：symbol+date）")
    md.append("- high<low 纠正；pct_chg 缺失则由 close 计算")
    md.append("- 日收益 ret、对数收益 log_ret")
    md.append(f"- 异常检测：Z={CFG['z_thresh']}，IQR={CFG['iqr_k']}×IQR；Winsor[{CFG['winsor_lower_q']},{CFG['winsor_upper_q']}]")
    md.append(f"- 技术指标：MA({CFG['ma_short']},{CFG['ma_long']}), EMA({CFG['ema_fast']},{CFG['ema_slow']}), MACD+signal({CFG['ema_signal']}), 年化波动({CFG['vol_win']}d), RSI({CFG['rsi_period']})")
    md.append(f"- 随机种子：{CFG['rng_seed']}（影响散点抽样）")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

# -------------------- 主流程 --------------------
def main():
    # 读主数据
    daily_raw, source_info = read_daily_main()
    # 基础概况（原始）
    daily_info = quality_checks(_rename_any_columns(daily_raw))

    # 读分钟线（可选，用于一致性核对）
    minute = read_minute_optional()

    # 清洗 + 特征
    cleaned, stats = engineer_features(daily_raw, winsorize=True, cfg=CFG)
    cleaned.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # 盘中 vs 当日一致性
    comp = intraday_daily_compare(cleaned, minute)
    comp_summary = {"rows": 0, "med_rel": np.nan, "p95_rel": np.nan, "max_rel": np.nan}
    if comp is not None and not comp.empty:
        comp_summary = {
            "rows": len(comp),
            "med_rel": float(np.nanmedian(comp["rel_diff_pct"])),
            "p95_rel": float(np.nanpercentile(comp["rel_diff_pct"], 95)),
            "max_rel": float(np.nanmax(comp["rel_diff_pct"]))
        }

    # 图表
    plot_eda(cleaned, comp, cfg=CFG)

    # 报告
    write_quality_report(
        stats=stats,
        daily_info=daily_info,
        source_info=source_info,
        minute_rows=(0 if minute is None else len(minute)),
        comp_summary=comp_summary,
        cfg=CFG,
        path=QUALITY_MD,
    )

    # 保存流程参数
    steps = dict(config=CFG, source=source_info, generated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    with open(STEPS_JSON, "w", encoding="utf-8") as f:
        json.dump(steps, f, ensure_ascii=False, indent=2)

    # 控制台摘要
    print("[OK] 清洗后数据 ->", os.path.abspath(OUTPUT_CSV))
    print("[OK] 质量报告   ->", os.path.abspath(QUALITY_MD))
    print("[OK] 图表目录   ->", os.path.abspath(EDA_DIR))
    if comp_summary["rows"] > 0:
        print("[OK] 一致性对比 ->", os.path.abspath(INTRADAY_COMPARE_CSV))
    print("Rows:", len(cleaned), "Symbols:", cleaned["symbol"].nunique())
    print("Range:", pd.to_datetime(cleaned["date"]).min(), "~", pd.to_datetime(cleaned["date"]).max())

if __name__ == "__main__":
    main()
