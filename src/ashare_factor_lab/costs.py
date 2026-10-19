"""交易成本模型：把「信号」翻译成「钱」的那一步。

A 股日频调仓的成本结构（2024 年后的实际水平）：
    佣金      0.025%/边（双边）
    过户费    0.001%/边（双边）
    印花税    0.05%（**仅卖出**）
    滑点      半价差 × 1/边 —— 无法拿到盘口数据，用 5/10/20bp 三档做敏感性分析

一次「换仓」= 卖出被替换掉的仓位 + 买入新仓位，所以换手率 t 对应的成本是：
    (佣金 + 过户费 + 半价差) × 2 + 印花税，再乘以 t

这也是本项目最重要的结论来源：换手 68%/天 × 每次 0.2~0.5% ≈ 每天 0.14~0.34% 的成本，
而信号毛超额只有 0.094%/天 —— 成本是毛收益的 1.5~3.6 倍，策略因此必亏。
「算清楚成本」比「把 IC 做高」对这个项目重要得多。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    commission: float = 0.00025      # 佣金/边
    transfer_fee: float = 0.00001    # 过户费/边
    stamp_duty: float = 0.0005       # 印花税（仅卖出）
    half_spread: float = 0.0010      # 半价差（滑点），10bp 基准假设
    min_commission: float = 5.0      # 单笔最低佣金（元），组合层面影响很小，保留以便扩展到小资金

    def per_turnover_cost(self) -> float:
        """换手率 100%（整仓换一遍）时的成本比例。"""
        one_side = self.commission + self.transfer_fee + self.half_spread
        return one_side * 2 + self.stamp_duty

    def daily_cost(self, turnover: float) -> float:
        return float(turnover) * self.per_turnover_cost()

    def break_even_gross_excess(self, turnover: float) -> float:
        """给定换手率，毛超额要多少才刚好覆盖成本（即"多强的信号才值得做"）。"""
        return self.daily_cost(turnover)

    def sensitivity_table(self, half_spreads=(0.0005, 0.0010, 0.0020)) -> pd.DataFrame:
        rows = []
        for hs in half_spreads:
            model = CostModel(self.commission, self.transfer_fee, self.stamp_duty, hs)
            rows.append(
                {
                    "half_spread_bp": hs * 1e4,
                    "per_turnover_cost": model.per_turnover_cost(),
                }
            )
        return pd.DataFrame(rows)


def turnover_rate(current: set, previous: set | None) -> float:
    """单边换手率 = 新进名单占当前名单的比例（0=完全没变，1=整仓换掉）。"""
    if previous is None or not current:
        return float("nan")
    return len(current - previous) / len(current)


def realized_turnover_series(picks_by_day: dict) -> pd.Series:
    """从「每天选中的股票集合」算出换手率序列。"""
    out, prev = {}, None
    for date in sorted(picks_by_day):
        current = picks_by_day[date]
        out[date] = turnover_rate(current, prev)
        prev = current
    return pd.Series(out).dropna()


def net_excess_curve(
    gross_excess: pd.Series, turnover: pd.Series, cost: CostModel
) -> pd.DataFrame:
    """逐日扣费：毛超额 - 换手率 × 单次换仓成本，并给出累乘净值。"""
    frame = pd.DataFrame({"gross_excess": gross_excess, "turnover": turnover}).dropna()
    frame["cost"] = frame["turnover"].map(cost.daily_cost)
    frame["net_excess"] = frame["gross_excess"] - frame["cost"]
    frame["equity_gross"] = (1 + frame["gross_excess"]).cumprod()
    frame["equity_net"] = (1 + frame["net_excess"]).cumprod()
    return frame


def annualize(daily_return: float, periods: int = 252) -> float:
    return float((1 + daily_return) ** periods - 1)


def summarize_costs(frame: pd.DataFrame) -> dict:
    net = frame["net_excess"]
    return {
        "days": int(len(frame)),
        "mean_gross_excess": float(frame["gross_excess"].mean()),
        "mean_cost": float(frame["cost"].mean()),
        "mean_net_excess": float(net.mean()),
        "mean_turnover": float(frame["turnover"].mean()),
        "break_even_turnover": float(frame["gross_excess"].mean() / frame["cost"].div(frame["turnover"]).replace(0, np.nan).mean())
        if frame["turnover"].mean() > 0
        else float("nan"),
        "annualized_net_excess": annualize(float(net.mean())),
        "equity_net_final": float(frame["equity_net"].iloc[-1]) if len(frame) else float("nan"),
    }