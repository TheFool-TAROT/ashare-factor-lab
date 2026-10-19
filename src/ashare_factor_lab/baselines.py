"""基准与随机对照：回答"你的模型比瞎选强吗"。

三件事必须做，缺一不可：

1. **同宇宙、同过滤条件的等权基准**：策略只买可交易的股票，基准也必须只算可交易股票，
   否则就是把"策略赚了可交易性过滤的钱"误记成"模型的能力"。原版的基准是全市场等权，
   而策略宇宙做过流动性/次新/ST/涨跌停过滤，这个比较是不公平的 —— 这里统一口径。
2. **随机选股分布**：每天从同一宇宙里随机抽同样数量的股票，重复 N 次，
   得到"瞎选"的超额收益分布。策略超额落在这个分布的哪个分位，直接给出 p 值。
   这比 IC 的 t 检验更贴近"组合能不能赚钱"这个真问题。
3. **单因子基准**：不加模型，直接按某个已存在的因子排序取头部（如 20 日动量）。
   如果模型打不过单因子，说明模型没创造价值，只是把因子重新组合了一遍。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import realized_turnover_series


@dataclass
class RandomPickResult:
    n_simulations: int
    mean_excess: float
    std_excess: float
    p_value: float
    strategy_excess: float
    percentile: float
    quantiles: dict


def universe_benchmark(data: pd.DataFrame, label: str = "fwd_ret") -> pd.Series:
    """可交易宇宙的每日等权收益（与策略完全相同的宇宙）。"""
    return data.groupby("date")[label].mean().sort_index()


def pick_top_by_score(data: pd.DataFrame, score_col: str, top_quantile: float = 0.1) -> dict:
    """按分数取头部组，返回 {日期: 股票集合}。"""
    picks: dict[pd.Timestamp, set] = {}
    for date, group in data.groupby("date"):
        clean = group.dropna(subset=[score_col, "fwd_ret"])
        if clean.empty:
            continue
        k = max(1, int(len(clean) * top_quantile))
        picks[date] = set(clean.nlargest(k, score_col)["code"])
    return picks


def top_group_returns(data: pd.DataFrame, score_col: str, top_quantile: float = 0.1) -> pd.Series:
    """头部组每日等权收益序列。"""
    out = {}
    for date, group in data.groupby("date"):
        clean = group.dropna(subset=[score_col, "fwd_ret"])
        if clean.empty:
            continue
        k = max(1, int(len(clean) * top_quantile))
        out[date] = float(clean.nlargest(k, score_col)["fwd_ret"].mean())
    return pd.Series(out).sort_index()


def random_pick_excess(
    data: pd.DataFrame,
    strategy_picks: dict,
    label: str = "fwd_ret",
    n_simulations: int = 500,
    seed: int = 42,
) -> RandomPickResult:
    """随机选股对照：每天随机抽同样数量的股票，重复 n_simulations 次。

    p 值 = 随机策略的超额 ≥ 真实策略超额的频率。p 值大 = 模型的挑选和瞎选没有区别。
    """
    # 必须写成推导式：`dict(groupby)` 会走映射协议，先读 `.keys` 属性 ——
    # 而 pandas 的 groupby.keys 是分组列名（字符串），于是 dict() 去"调用"一个字符串并报错。
    # ruff 的 C416 会把它"优化"成 dict(...)，这里显式豁免。
    by_date = {date: group for date, group in data.groupby("date")}  # noqa: C416
    sizes = {date: len(codes) for date, codes in strategy_picks.items() if date in by_date}
    if not sizes:
        return RandomPickResult(n_simulations, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), {})

    universe_daily = universe_benchmark(data, label)
    strategy_excess = _excess_of_picks(by_date, strategy_picks, universe_daily, label)

    rng = np.random.default_rng(seed)
    sim_excess = np.empty(n_simulations)
    for sim in range(n_simulations):
        picks = {}
        for date, group in by_date.items():
            k = min(sizes.get(date, 0), len(group))
            if k <= 0:
                continue
            idx = rng.choice(group.index.to_numpy(), size=k, replace=False)
            picks[date] = set(group.loc[idx, "code"])
        sim_excess[sim] = _excess_of_picks(by_date, picks, universe_daily, label)

    sim_excess = sim_excess[np.isfinite(sim_excess)]
    if sim_excess.size == 0:
        return RandomPickResult(n_simulations, float("nan"), float("nan"), float("nan"),
                                strategy_excess, float("nan"), {})
    p_value = float((sim_excess >= strategy_excess).mean())
    percentile = float((sim_excess < strategy_excess).mean())
    return RandomPickResult(
        n_simulations=int(sim_excess.size),
        mean_excess=float(sim_excess.mean()),
        std_excess=float(sim_excess.std(ddof=1)) if sim_excess.size > 1 else float("nan"),
        p_value=p_value,
        strategy_excess=strategy_excess,
        percentile=percentile,
        quantiles={
            "q05": float(np.quantile(sim_excess, 0.05)),
            "q50": float(np.quantile(sim_excess, 0.50)),
            "q95": float(np.quantile(sim_excess, 0.95)),
        },
    )


def _excess_of_picks(
    by_date: dict, picks: dict, universe_daily: pd.Series, label: str
) -> float:
    """给定每日选股集合，算「头部组 - 宇宙等权」的日频均值超额。"""
    excess = {}
    for date, codes in picks.items():
        group = by_date.get(date)
        if group is None or not codes:
            continue
        picked = group[group["code"].isin(codes)][label].dropna()
        if picked.empty:
            continue
        bench = universe_daily.get(date, np.nan)
        if pd.isna(bench):
            continue
        excess[date] = float(picked.mean() - bench)
    return float(pd.Series(excess).mean()) if excess else float("nan")


def momentum_baseline(data: pd.DataFrame, top_quantile: float = 0.1) -> dict:
    """单因子基准：按 20 日动量排序取头部（不含模型）。"""
    frame = data.copy()
    score = "mom20_z" if "mom20_z" in frame.columns else "mom20"
    picks = pick_top_by_score(frame, score, top_quantile)
    returns = top_group_returns(frame, score, top_quantile)
    bench = universe_benchmark(frame)
    return {
        "name": f"单因子基准({score})",
        "picks": picks,
        "returns": returns,
        "excess": returns.sub(bench).dropna(),
        "turnover": realized_turnover_series(picks),
    }