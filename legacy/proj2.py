# -*- coding: utf-8 -*-
"""
Project2 - Financial Data Acquisition & Basic Processing (PDF-compliant)
目录结构:
  ./data/{raw,processed,database}
生成物:
  - data/processed/daily_price_50.csv   # 50支股票的单一CSV（满足PDF要求）
  - data/database/financial_data.db     # SQLite（可选，已实现）
  - data/REPORT.md                      # 数据源、方法、挑战（PDF要求）
  - README.md                           # 运行说明（PDF要求）

技术要点:
  - API采集(akshare) -> 基础清洗(缺失/类型/去重/修正high<low, 补pct_chg) -> 存储(CSV/SQLite)
  - 稳健后备: 主接口失败则切换全量接口; pct_chg缺失用close计算
"""

# ---------- 0) 禁用代理 ----------
import os
def _disable_proxies():
    for k in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"):
        os.environ.pop(k, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
_disable_proxies()

# ---------- 1) 依赖 ----------
import sys
import time
import sqlite3
from datetime import datetime, timezone, timedelta
from http.client import RemoteDisconnected
from urllib.error import URLError

import pandas as pd
import numpy as np
from requests.exceptions import ProxyError, ReadTimeout, ConnectTimeout
import akshare as ak

# ============================ 常量与路径 ============================
DATA_DIR = "./data"
RAW_DIR = f"{DATA_DIR}/raw"
PROC_DIR = f"{DATA_DIR}/processed"
DB_DIR = f"{DATA_DIR}/database"
DB_PATH = f"{DB_DIR}/financial_data.db"
CSV_OUT = f"{PROC_DIR}/daily_price_50.csv"     # 单一CSV（50支）
CSV_RAW = f"{RAW_DIR}/daily_price.csv"         # 拉取后原始合并
REPORT_MD = f"{DATA_DIR}/REPORT.md"
README_MD = "README.md"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# ============================ 工具函数 ============================
def _code_to_prefixed(symbol: str) -> str:
    s = str(symbol)
    if s.startswith(("000","001","002","003","300")):
        return "sz" + s
    if s.startswith(("600","601","603","605","688")):
        return "sh" + s
    return s if s[:2] in ("sh","sz") else s

def _rename_any_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        # 中文
        "日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low",
        "成交量":"volume","成交额":"amount","涨跌幅":"pct_chg","涨跌额":"change",
        "换手率":"turnover_rate","振幅":"amplitude","代码":"code","名称":"name",
        # 英文
        "Date":"date","Open":"open","Close":"close","High":"high","Low":"low",
        "Volume":"volume","Amount":"amount","Pct_chg":"pct_chg","Change":"change",
    }
    exist = {k:v for k,v in mapping.items() if k in df.columns}
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

def _clean_daily(df: pd.DataFrame) -> pd.DataFrame:
    """基础清洗：类型/去重/修正(high<low)/补pct_chg"""
    df = _rename_any_columns(df)
    req = ["symbol","date","open","high","low","close","volume","amount","pct_chg"]
    df = _ensure_cols(df, req)
    df = _to_datetime(df, "date")
    df = _to_numeric(df, ["open","high","low","close","volume","amount","pct_chg"])
    # 基础约束
    df = df.dropna(subset=["symbol","date"])
    df = df.sort_values(["symbol","date"]).drop_duplicates(subset=["symbol","date"], keep="last")
    # 修正 high<low
    m = df["high"].notna() & df["low"].notna() & (df["high"] < df["low"])
    if m.any():
        hi = df.loc[m,"high"].copy(); lo = df.loc[m,"low"].copy()
        df.loc[m,"high"] = lo; df.loc[m,"low"] = hi
    # pct_chg 缺失或几乎全空则用 close 计算
    if df["pct_chg"].isna().mean() > 0.9:
        df["pct_chg"] = df.groupby("symbol")["close"].pct_change() * 100.0
    return df[req]

# ============================ 采集器 ============================
class Project2Collector:
    def __init__(self,
                 start_date: str = "20200101",
                 end_date: str | None = None,
                 probe_n: int = 5,
                 final_n: int = 50,
                 max_retries: int = 4,
                 base_delay: float = 1.0,
                 sleep_each: float = 0.5):
        self.start_date = start_date
        self.end_date = end_date or datetime.now(tz=timezone(timedelta(hours=8))).strftime("%Y%m%d")  # Asia/Taipei
        self.probe_n = probe_n
        self.final_n = final_n
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.sleep_each = sleep_each

    # ---------- DB ----------
    def save_to_database(self, df: pd.DataFrame, table_name: str) -> None:
        try:
            x = df.copy()
            if "date" in x.columns:
                x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            with sqlite3.connect(DB_PATH) as conn:
                x.to_sql(table_name, conn, if_exists="replace", index=False, chunksize=1000)
            print(f"[OK] to_sql -> {table_name}")
        except Exception as e:
            print(f"[ERR] to_sql {table_name}: {e}")

    # ---------- 股票池 ----------
    def get_stock_universe(self) -> pd.DataFrame:
        print("[*] 拉取股票池 ...")
        try:
            df = ak.stock_info_a_code_name()
            if df is None or df.empty:
                print("[WARN] 股票池为空")
                return pd.DataFrame()
            df = _rename_any_columns(df)
            # 统一列: code, name, symbol
            if "symbol" not in df.columns:
                df["symbol"] = df["code"].astype(str)
            df["update_time"] = datetime.now()
            df.to_csv(f"{RAW_DIR}/stock_universe.csv", index=False, encoding="utf-8-sig")
            self.save_to_database(df, "stock_universe")
            print(f"[OK] 股票池 {len(df)} 条")
            return df[["symbol","code","name","update_time"]]
        except Exception as e:
            print(f"[ERR] 股票池: {e}")
            return pd.DataFrame()

    # ---------- 个股日线：主接口 + 备份 ----------
    def _fetch_daily_em(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        # 东财区间
        return ak.stock_zh_a_hist(symbol=symbol, start_date=start_date, end_date=end_date, adjust="qfq")

    def _fetch_daily_prefixed_full(self, symbol: str) -> pd.DataFrame:
        # 全量，后本地过滤
        pref = _code_to_prefixed(symbol)
        if hasattr(ak, "stock_zh_a_daily"):
            try:
                df = ak.stock_zh_a_daily(symbol=pref)
                if df is not None and not df.empty:
                    return df.reset_index() if "date" not in df.columns else df
            except Exception:
                pass
        for name in ("stock_zh_a_daily_tx","stock_zh_a_daily_qq"):
            if hasattr(ak, name):
                try:
                    fn = getattr(ak, name)
                    df = fn(symbol=pref)
                    if df is not None and not df.empty:
                        return df.reset_index() if "date" not in df.columns else df
                except Exception:
                    pass
        raise RuntimeError("no_secondary_backend")

    def download_daily_data(self, symbols) -> pd.DataFrame:
        print(f"[*] 日线下载: {len(symbols)} 支, {self.start_date}~{self.end_date}")
        start_dt = pd.to_datetime(self.start_date); end_dt = pd.to_datetime(self.end_date)
        frames = []

        for i, symbol in enumerate(symbols, 1):
            retries, delay = 0, self.base_delay
            df = None
            while retries < self.max_retries and df is None:
                try:
                    df = self._fetch_daily_em(symbol, self.start_date, self.end_date)
                except Exception as e_em:
                    if isinstance(e_em, (RemoteDisconnected, ConnectionResetError, URLError,
                                         TimeoutError, ProxyError, ReadTimeout, ConnectTimeout)):
                        _disable_proxies()
                        retries += 1
                        print(f"  - [{i}/{len(symbols)}] {symbol} EM {retries}/{self.max_retries} NET: {type(e_em).__name__} -> {delay:.1f}s")
                        time.sleep(delay); delay *= 2
                        continue
                    try:
                        df = self._fetch_daily_prefixed_full(symbol)
                    except Exception as e2:
                        print(f"  - [{i}/{len(symbols)}] {symbol} 后端全失败: {e2}")
                        break

            if df is None or df.empty:
                print(f"  - [{i}/{len(symbols)}] {symbol} 空数据，跳过")
                continue

            df = _rename_any_columns(df)
            df = _to_datetime(df, "date")
            df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].copy()
            if df.empty:
                print(f"  - [{i}/{len(symbols)}] {symbol} 区间无数据，跳过")
                continue

            df["symbol"] = str(symbol)
            req = ["symbol","date","open","high","low","close","volume","amount","pct_chg"]
            df = _ensure_cols(df, req)
            df = _to_numeric(df, ["open","high","low","close","volume","amount","pct_chg"])
            df = df.sort_values("date")
            if df["pct_chg"].isna().mean() > 0.9 and "close" in df.columns:
                df["pct_chg"] = df["close"].pct_change() * 100.0
            df = df[req].drop_duplicates(subset=["symbol","date"])
            frames.append(df)
            time.sleep(self.sleep_each)

        if not frames:
            print("[WARN] 本轮无成功数据")
            return pd.DataFrame()

        all_df = pd.concat(frames, ignore_index=True)
        # 原始与处理后各存一份
        all_df.to_csv(CSV_RAW, index=False, encoding="utf-8-sig")
        self.save_to_database(all_df, "daily_price_raw")
        return _clean_daily(all_df)

    # ---------- 报告 ----------
    def generate_report(self, daily_df: pd.DataFrame, universe_df: pd.DataFrame) -> None:
        try:
            ak_ver = getattr(ak, "__version__", "unknown")
        except Exception:
            ak_ver = "unknown"

        # 文件树
        file_lines = []
        for root, _, files in os.walk(DATA_DIR):
            level = root.replace(DATA_DIR, "").count(os.sep)
            indent = "  " * level
            file_lines.append(f"{indent}{os.path.basename(root)}/")
            for f in files:
                file_lines.append(f"{indent}  {f}")
        file_tree = "```\n" + "\n".join(file_lines) + "\n```"

        # DB 概览
        tables_block = []
        try:
            with sqlite3.connect(DB_PATH) as conn:
                t = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;", conn)
                names = t["name"].tolist()
                for name in names:
                    cnt = pd.read_sql(f"SELECT COUNT(*) AS c FROM {name};", conn)["c"].iloc[0]
                    head = pd.read_sql(f"SELECT * FROM {name} LIMIT 5;", conn)
                    tables_block.append(f"### {name}\n- rows: {cnt}\n\n```text\n{head.to_string(index=False)}\n```")
        except Exception as e:
            tables_block.append(f"_DB scan error: {e}_")

        def _sum_na(df, title):
            if df is None or len(getattr(df, "index", [])) == 0:
                return f"### {title}\n- EMPTY\n\n**NA ratio**:\nEMPTY"
            dmin = pd.to_datetime(df["date"]).min() if "date" in df.columns else None
            dmax = pd.to_datetime(df["date"]).max() if "date" in df.columns else None
            syms = df["symbol"].nunique() if "symbol" in df.columns else 0
            na = df.isna().mean().sort_values(ascending=False)
            na_txt = "\n".join([f"- {k}: {v:.2%}" for k, v in na.items()])
            return f"### {title}\n- Rows: {len(df)} | Symbols: {syms} | Range: {dmin:%Y-%m-%d} ~ {dmax:%Y-%m-%d}\n\n**NA ratio**:\n{na_txt}"

        sec_daily = _sum_na(daily_df, "Daily Price (cleaned)")
        sec_uni = f"### Universe\n- Rows: {len(universe_df)}\n- Unique symbols: {universe_df['symbol'].nunique()}"

        md = []
        md.append("# Data Acquisition Report")
        md.append("")
        md.append("**Data sources, methodology, cleaning steps, and challenges.**")
        md.append("")
        md.append("## Overview")
        md.append(f"- Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
        md.append(f"- Data dir: `{DATA_DIR}` | DB: `{DB_PATH}`")
        md.append(f"- Python: {sys.version.split()[0]} | pandas: {pd.__version__} | akshare: {ak_ver}")
        md.append("")
        md.append("## Data Sources")
        md.append("- A-share universe: `ak.stock_info_a_code_name()`")
        md.append("- Daily quotes primary: `ak.stock_zh_a_hist(..., adjust='qfq')`")
        md.append("- Fallback full-history: `ak.stock_zh_a_daily('shXXXXXX'/'szXXXXXX')` and variants")
        md.append("")
        md.append("## Methodology")
        md.append("1) Disable proxies; 2) Pull universe; 3) Select 50 symbols; 4) Download daily quotes with retries;")
        md.append("5) Standardize columns; 6) Basic cleaning: dtype, duplicates, fix(high<low), pct_chg fallback;")
        md.append("7) Save CSV + SQLite; 8) Generate report + README.")
        md.append("")
        md.append("## Outputs")
        md.append("- CSV: `data/processed/daily_price_50.csv` (single CSV, 50 stocks)  ← PDF 要求")
        md.append("- SQLite: `data/database/financial_data.db` (optional)  ← PDF 可选项")
        md.append("- Report: `data/REPORT.md`")
        md.append("- README: `README.md`")
        md.append("")
        md.append("## Data Quality")
        md.append(sec_daily)
        md.append("")
        md.append(sec_uni)
        md.append("")
        md.append("## Database Preview")
        md.append("\n".join(tables_block))
        md.append("")
        md.append("## File Tree")
        md.append(file_tree)

        with open(REPORT_MD, "w", encoding="utf-8") as f:
            f.write("\n".join(md))
        print("[OK] REPORT.md ->", os.path.abspath(REPORT_MD))

    def write_readme(self, script_name="project2_acquire_clean.py") -> None:
        try:
            import platform as _pf
            os_info = f"{_pf.system()} {_pf.release()}"
        except Exception:
            os_info = os.name

        lines = [
            "# Code and Documentation",
            "",
            "This repository contains the code for **Project 2: Financial Data Acquisition and Basic Processing**.",
            "",
            "## Environment",
            f"- OS: {os_info}",
            f"- Python: {sys.version.split()[0]}",
            f"- pandas: {pd.__version__}",
            f"- akshare: {getattr(ak, '__version__', 'unknown')}",
            "",
            "## Installation",
            "```bash",
            "pip install -U akshare pandas numpy requests",
            "```",
            "",
            "## How to Run",
            "```bash",
            f"python {script_name}",
            "```",
            "",
            "## Outputs",
            "```text",
            f"{DATA_DIR}/",
            "  raw/",
            "    stock_universe.csv",
            "    daily_price.csv",
            "  processed/",
            "    daily_price_50.csv",
            "  database/",
            "    financial_data.db",
            "  REPORT.md",
            "```",
            "",
            "## Notes",
            "- Proxies are disabled by default to reduce network errors.",
            "- `pct_chg` is computed from `close` when missing.",
            "- Exactly 50 symbols are included in the single CSV as required by the PDF.",
            ""
        ]
        with open(README_MD, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print("[OK] README.md ->", os.path.abspath(README_MD))

# ============================ 主流程 ============================
def main():
    collector = Project2Collector(
        start_date="20200101",
        end_date=None,          # 默认=今天(Asia/Taipei)
        probe_n=5,
        final_n=300,
        max_retries=4,
        base_delay=1.0,
        sleep_each=0.5,
    )

    # 1) 股票池
    uni = collector.get_stock_universe()
    if uni.empty:
        print("[FATAL] 无法获取股票池"); return

    # 2) 选择 50 支（去重后取前 50）
    top50 = uni["symbol"].astype(str).dropna().drop_duplicates().head(collector.final_n).tolist()
    if len(top50) < collector.final_n:
        print(f"[WARN] 股票池不足 {collector.final_n}，实际 {len(top50)}")

    # 3) 下载日线
    raw_clean = collector.download_daily_data(symbols=top50)
    if raw_clean.empty:
        print("[FATAL] 50支日线数据全部为空"); return

    # 只保留所选 50 支
    df_50 = raw_clean[raw_clean["symbol"].isin(top50)].copy()
    # 落盘 CSV（PDF要求：单一CSV）
    df_50.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    # 同步写库（可选）
    collector.save_to_database(df_50, "daily_price")

    # 4) 报告 + README
    collector.generate_report(daily_df=df_50, universe_df=uni)
    collector.write_readme()

    # 控制台摘要
    dmin = pd.to_datetime(df_50["date"]).min()
    dmax = pd.to_datetime(df_50["date"]).max()
    print("[OK] CSV(50)  ->", os.path.abspath(CSV_OUT))
    print("[OK] DB       ->", os.path.abspath(DB_PATH))
    print("[OK] REPORT   ->", os.path.abspath(REPORT_MD))
    print("[OK] README   ->", os.path.abspath(README_MD))
    print(f"Rows: {len(df_50)}  Symbols: {df_50['symbol'].nunique()}  Range: {dmin:%Y-%m-%d} ~ {dmax:%Y-%m-%d}")

if __name__ == "__main__":
    main()
