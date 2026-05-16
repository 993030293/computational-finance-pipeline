"""
Project 5 - Multi-factor Vectorized Backtester

使用数据：
- daily_price_50.csv  (Project2/3 产物，日频价格)
- factors.csv         (Project4 产物，月末因子暴露 + 下一期 1M 收益 fwd_1m_ret)

输出：
- proj5_output/portfolio_returns.csv      : 各策略月度收益序列
- proj5_output/performance_metrics.csv    : 各策略绩效指标（年化收益、波动、Sharpe 等）
- proj5_output/drawdown_long_short.csv    : 多空组合回撤序列
- proj5_output/rolling_long_short.csv     : 多空组合滚动波动率 / Sharpe
- proj5_output/cumret_curves.png          : 各策略累计收益曲线
- proj5_output/drawdown_long_short.png    : 多空组合回撤图
- proj5_output/rolling_long_short.png     : 多空组合滚动指标图
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. 数据加载
# ----------------------------------------------------------------------

def load_data(
    prices_path: str = "daily_price_50.csv",
    factors_path: str = "factors.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    读取日频价格和月度因子数据。

    prices.csv 至少包含: ['symbol', 'date', 'close']
    factors.csv 必须包含: ['symbol', 'month_end',
                            'z_VALUE', 'z_MOM_12_1', 'z_QUALITY', 'z_SIZE',
                            'fwd_1m_ret']
    """
    prices = pd.read_csv(prices_path, parse_dates=["date"])
    factors = pd.read_csv(factors_path, parse_dates=["month_end"])

    required_prices_cols = {"symbol", "date", "close"}
    required_factors_cols = {
        "symbol",
        "month_end",
        "z_VALUE",
        "z_MOM_12_1",
        "z_QUALITY",
        "z_SIZE",
        "fwd_1m_ret",
    }

    missing_p = required_prices_cols.difference(prices.columns)
    missing_f = required_factors_cols.difference(factors.columns)

    if missing_p:
        raise KeyError(
            f"Prices file missing required columns: {sorted(missing_p)}. "
            f"Columns found: {list(prices.columns)}"
        )
    if missing_f:
        raise KeyError(
            f"Factors file missing required columns: {sorted(missing_f)}. "
            f"Columns found: {list(factors.columns)}"
        )

    prices = prices.sort_values(["symbol", "date"]).reset_index(drop=True)
    factors = factors.sort_values(["symbol", "month_end"]).reset_index(drop=True)
    return prices, factors


# ----------------------------------------------------------------------
# 2. 多因子打分 & pivot
# ----------------------------------------------------------------------

def add_factor_score(
    factors: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    根据标准化因子暴露加权，生成多因子综合得分 'score'。

    weights 的 key 必须属于 ['z_VALUE','z_MOM_12_1','z_QUALITY','z_SIZE']。
    若为 None，默认各因子等权。
    """
    zcols = ["z_VALUE", "z_MOM_12_1", "z_QUALITY", "z_SIZE"]
    if weights is None:
        weights = {col: 1.0 for col in zcols}
    weights = {k: v for k, v in weights.items() if k in zcols}
    if not weights:
        raise ValueError("No valid factor weights provided.")

    w_series = pd.Series(weights, index=zcols).fillna(0.0)
    if w_series.abs().sum() == 0:
        raise ValueError("All factor weights are zero.")
    # 归一化，方便解释
    w_series = w_series / w_series.abs().sum()

    factors = factors.copy()
    factors["score"] = factors[zcols].dot(w_series.values)
    return factors


def pivot_panel(factors: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    将 (month_end, symbol, col) 转成二维面板：index=month_end, columns=symbol。
    """
    panel = (
        factors.pivot(index="month_end", columns="symbol", values=col)
        .sort_index()
        .sort_index(axis=1)
    )
    return panel

def factor_ic_signs(factors: pd.DataFrame, zcols, ret_col="fwd_1m_ret"):
    signs = {}
    ic_mean = {}

    for z in zcols:
        ic_t = (factors
                .groupby("month_end", group_keys=False)[[z, ret_col]]
                .apply(lambda d: d[z].corr(d[ret_col], method="spearman")))
        m = float(ic_t.mean(skipna=True))
        ic_mean[z] = m
        signs[z] = 1.0 if m >= 0 else -1.0

    return signs, ic_mean


def apply_sign_flip(factors: pd.DataFrame, signs: dict) -> pd.DataFrame:
    f = factors.copy()
    for z, sgn in signs.items():
        f[z] = f[z] * sgn
    return f
# ----------------------------------------------------------------------
# 3. 向量化投资组合构建
# ----------------------------------------------------------------------

def build_portfolios(
    score_panel: pd.DataFrame,
    ret_panel: pd.DataFrame,
    top_quantile: float = 0.2,
) -> pd.DataFrame:
    """
    基于因子得分和下一期 1M 收益构建四种策略：

    - long_only    : 因子得分前 top_quantile，等权多头
    - short_only   : 因子得分后 top_quantile，等权空头
    - long_short   : long_only - short_only（多空对冲，名义多头 = 空头 = 1）
    - benchmark_ew : 所有股票等权基准

    返回：各策略月度收益 DataFrame (T x 4)
    """
    # 对齐
    score_panel, ret_panel = score_panel.align(ret_panel, join="inner", axis=None)
    score_panel = score_panel.sort_index().sort_index(axis=1)
    ret_panel = ret_panel.sort_index().sort_index(axis=1)

    # cross-section 排名：分数越大 rank 越靠前（1 为最好）
    ranks = score_panel.rank(axis=1, ascending=False, method="first")
    n = ranks.notna().sum(axis=1)              # 每月可交易股票数
    top_n = (n * top_quantile).round().astype(int).clip(lower=1)

    # top 组合
    long_mask = ranks.le(top_n, axis=0)

    # bottom 组合
    bottom_n = top_n
    cut = n - bottom_n + 1                     # 倒数 bottom_n 名
    short_mask = ranks.ge(cut, axis=0)

    long_w = long_mask.astype(float)
    long_w = long_w.div(long_w.sum(axis=1), axis=0).fillna(0.0)

    short_w = short_mask.astype(float)
    short_w = short_w.div(short_w.sum(axis=1), axis=0).fillna(0.0)

    # 等权基准
    ew_w = ret_panel.notna().astype(float)
    ew_w = ew_w.div(ew_w.sum(axis=1), axis=0).fillna(0.0)

    long_ret = (long_w * ret_panel).sum(axis=1)
    short_ret = (short_w * ret_panel).sum(axis=1)
    long_short_ret = long_ret - short_ret
    bench_ret = (ew_w * ret_panel).sum(axis=1)

    portfolio_returns = pd.DataFrame(
        {
            "long_only": long_ret,
            "short_only": short_ret,
            "long_short": long_short_ret,
            "benchmark_ew": bench_ret,
        }
    )
    return portfolio_returns


# ----------------------------------------------------------------------
# 4. 绩效指标与风险分析
# ----------------------------------------------------------------------

def compute_drawdown(ret: pd.Series) -> pd.Series:
    """
    根据收益序列计算回撤。
    """
    cum = (1.0 + ret).cumprod()
    running_max = cum.cummax()
    dd = cum / running_max - 1.0
    return dd


def rolling_metrics(
    ret: pd.Series,
    window: int = 12,
    freq: int = 12,
) -> pd.DataFrame:
    """
    计算滚动波动率和滚动 Sharpe（rf=0）。

    ret    : 月度收益
    window : 窗口长度（单位：月）
    freq   : 年化频率（12 表示月度数据）
    """
    rolling_vol = ret.rolling(window).std(ddof=0) * np.sqrt(freq)
    rolling_sharpe = ret.rolling(window).mean() * np.sqrt(freq) / rolling_vol
    return pd.DataFrame(
        {"rolling_vol": rolling_vol, "rolling_sharpe": rolling_sharpe}
    )


def performance_metrics(
    ret: pd.Series,
    freq: int = 12,
    benchmark: Optional[pd.Series] = None,
    rf: float = 0.0,
) -> Dict[str, float]:
    """
    计算策略常用绩效指标。
    """
    ret = ret.dropna()
    if len(ret) == 0:
        return {}

    # 年化收益（几何）
    total_ret = (1.0 + ret).prod()
    ann_return = total_ret ** (freq / len(ret)) - 1.0

    # 年化波动
    ann_vol = ret.std(ddof=0) * np.sqrt(freq)
    sharpe = (ann_return - rf) / ann_vol if ann_vol > 0 else np.nan

    # 回撤相关
    dd = compute_drawdown(ret)
    max_dd = dd.min()
    calmar = (ann_return - rf) / abs(max_dd) if max_dd < 0 else np.nan

    # Sortino
    downside = ret[ret < 0]
    if len(downside) > 0:
        down_vol = downside.std(ddof=0) * np.sqrt(freq)
        sortino = (ann_return - rf) / down_vol if down_vol > 0 else np.nan
    else:
        sortino = np.nan

    metrics: Dict[str, float] = {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "sortino": sortino,
    }

    # 相对基准的 Alpha / Beta / IR
    if benchmark is not None:
        bench = benchmark.loc[ret.index].dropna()
        common_idx = ret.index.intersection(bench.index)
        r = ret.loc[common_idx]
        b = bench.loc[common_idx]

        if len(b) > 1:
            cov = np.cov(r, b)[0, 1]
            var_b = np.var(b)
            beta = cov / var_b if var_b > 0 else np.nan

            # 月度超额收益（rf=0），得到月度 alpha，再年化
            alpha_monthly = r.mean() - beta * b.mean()
            alpha_ann = alpha_monthly * freq

            diff = r - b
            diff_std = diff.std(ddof=0)
            ir = (
                diff.mean() * np.sqrt(freq) / diff_std
                if diff_std > 0
                else np.nan
            )
        else:
            beta = np.nan
            alpha_ann = np.nan
            ir = np.nan

        metrics.update(
            {
                "beta": beta,
                "alpha_ann": alpha_ann,
                "information_ratio": ir,
            }
        )

    return metrics


# ----------------------------------------------------------------------
# 5. 绘图
# ----------------------------------------------------------------------

def plot_cum_returns(
    port_ret: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    绘制各策略累计收益曲线。
    """
    cum = (1.0 + port_ret).cumprod()

    plt.figure(figsize=(10, 6))
    for col in port_ret.columns:
        plt.plot(cum.index, cum[col], label=col)
    plt.xlabel("Date (month_end)")
    plt.ylabel("Cumulative growth of 1 unit")
    plt.title("Cumulative Returns - Multi-factor Strategies")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_drawdown(
    ret: pd.Series,
    out_path: Path,
    title: str = "Drawdown - Long-Short Strategy",
) -> None:
    dd = compute_drawdown(ret)
    plt.figure(figsize=(10, 4))
    plt.plot(dd.index, dd.values)
    plt.xlabel("Date (month_end)")
    plt.ylabel("Drawdown")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_rolling_metrics(
    ret: pd.Series,
    out_path: Path,
    window: int = 12,
    freq: int = 12,
) -> None:
    rm = rolling_metrics(ret, window=window, freq=freq)
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax[0].plot(rm.index, rm["rolling_vol"])
    ax[0].set_ylabel("Rolling Volatility")
    ax[0].grid(True)

    ax[1].plot(rm.index, rm["rolling_sharpe"])
    ax[1].set_ylabel("Rolling Sharpe")
    ax[1].set_xlabel("Date (month_end)")
    ax[1].grid(True)

    fig.suptitle(f"Rolling {window}-month Volatility & Sharpe (Long-Short)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(out_path)
    plt.close()


# ----------------------------------------------------------------------
# 6. 主函数
# ----------------------------------------------------------------------

def run_backtest(
    prices_path: str = "daily_price_50.csv",
    factors_path: str = "factors.csv",
    out_dir: str = "proj5_output",
    factor_weights: Optional[Dict[str, float]] = None,  # 兼容旧调用
    top_quantile: float = 0.2,
) -> None:
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    prices, factors = load_data(prices_path, factors_path)

    zcols = ["z_VALUE", "z_MOM_12_1", "z_QUALITY", "z_SIZE"]
    signs, ic_mean = factor_ic_signs(factors, zcols, ret_col="fwd_1m_ret")
    factors = apply_sign_flip(factors, signs)

    pd.Series(ic_mean, name="mean_ic").to_csv(out_dir_path / "ic_mean.csv")
    pd.Series(signs, name="sign").to_csv(out_dir_path / "factor_signs.csv")

    # 最终默认权重（若外部传入则用外部）
    final_weights = {"z_VALUE": 1, "z_MOM_12_1": 0, "z_QUALITY": 0, "z_SIZE": 1}
    w = final_weights if factor_weights is None else factor_weights

    f_final = add_factor_score(factors, weights=w)
    score_panel = pivot_panel(f_final, "score")
    ret_panel   = pivot_panel(f_final, "fwd_1m_ret")

    port_ret = build_portfolios(score_panel, ret_panel, top_quantile=top_quantile)
    port_ret.to_csv(out_dir_path / "portfolio_returns.csv", float_format="%.8f")

    nav = (1.0 + port_ret).cumprod()
    nav.to_csv(out_dir_path / "portfolio_nav.csv", float_format="%.8f")

    metrics_list = []
    for col in port_ret.columns:
        m = performance_metrics(port_ret[col], freq=12, benchmark=port_ret["benchmark_ew"])
        m["strategy"] = col
        metrics_list.append(m)
    pd.DataFrame(metrics_list).set_index("strategy").to_csv(
        out_dir_path / "performance_metrics.csv", float_format="%.6f"
    )

    ls_ret = port_ret["long_short"]
    compute_drawdown(ls_ret).to_csv(out_dir_path / "drawdown_long_short.csv", header=["drawdown"])
    rolling_metrics(ls_ret, window=12, freq=12).to_csv(
        out_dir_path / "rolling_long_short.csv", float_format="%.6f"
    )

    plot_cum_returns(port_ret, out_dir_path / "cumret_curves.png")
    plot_drawdown(ls_ret, out_dir_path / "drawdown_long_short.png")
    plot_rolling_metrics(ls_ret, out_dir_path / "rolling_long_short.png")


if __name__ == "__main__":
    # 根据自己的路径修改这两个参数
    run_backtest(
        prices_path="./data/processed/daily_price_50.csv",
        factors_path="./data/project4/factors.csv",
        factor_weights=None,
        top_quantile=0.2,
    )

