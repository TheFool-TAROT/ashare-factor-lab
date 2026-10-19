"""组合层面的回测指标：净值、年化、波动、Sharpe、最大回撤、Calmar、胜率。

为什么必须有这一层：IC 是"平均意义上的排序能力"，它不告诉你资金曲线的形状。
一个 IC 0.02 的策略可能因为波动和回撤而不可持有，也可能因为换手而必亏。
所有指标都由逐日收益序列算出，且**同一套函数同时用于策略与基准** —— 避免"策略算一套、
基准算另一套"的隐性美化。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equity_curve(daily_returns: pd.Series) -> pd.Series:
    return (1 + daily_returns.fillna(0.0)).cumprod()


def max_drawdown(daily_returns: pd.Series) -> float:
    curve = equity_curve(daily_returns)
    if curve.empty:
        return 0.0
    peak = curve.cummax()
    return float((curve / peak - 1).min())


def annualized_return(daily_returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = daily_returns.dropna()
    if r.empty:
        return float("nan")
    return float((1 + r.mean()) ** periods - 1)


def annualized_vol(daily_returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = daily_returns.dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(periods))


def sharpe_ratio(daily_returns: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    r = daily_returns.dropna()
    if len(r) < 2:
        return float("nan")
    excess = r - rf / periods
    std = excess.std(ddof=1)
    if not np.isfinite(std) or std < 1e-12:      # 常数序列的 std 可能只是浮点残差，不能当成"高 Sharpe"
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods))


def calmar_ratio(daily_returns: pd.Series) -> float:
    mdd = abs(max_drawdown(daily_returns))
    if mdd < 1e-12:
        return float("nan")
    return annualized_return(daily_returns) / mdd


def hit_rate(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna()
    return float((r > 0).mean()) if len(r) else float("nan")


def summarize(daily_returns: pd.Series, label: str = "strategy") -> dict:
    """一份指标口径，策略与基准都走这里。"""
    r = daily_returns.dropna()
    return {
        "label": label,
        "days": int(len(r)),
        "total_return": float(equity_curve(r).iloc[-1] - 1) if len(r) else float("nan"),
        "annualized_return": annualized_return(r),
        "annualized_vol": annualized_vol(r),
        "sharpe": sharpe_ratio(r),
        "max_drawdown": max_drawdown(r),
        "calmar": calmar_ratio(r),
        "hit_rate": hit_rate(r),
        "mean_daily": float(r.mean()) if len(r) else float("nan"),
    }


def compare_table(strategies: dict[str, pd.Series]) -> pd.DataFrame:
    """把多个收益序列（策略 + 基准）汇总成一张表，直接进报告。"""
    keys = ["days", "total_return", "annualized_return", "annualized_vol", "sharpe", "max_drawdown", "calmar", "hit_rate"]
    rows = {}
    for name, series in strategies.items():
        stats = summarize(series, label=name)
        rows[name] = {k: stats[k] for k in keys}
    return pd.DataFrame(rows).T