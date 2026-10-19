"""特征工程：全部使用**向后看**的窗口，任何一行特征只依赖该行及更早的数据。

因果性有两道保障：
1. 代码层面只用 `shift(+n)` / `rolling(...)` / `ewm(...)`（均为 trailing），
   全仓库禁止 `shift(-n)` 参与特征计算（标签模块 labels.py 是唯一使用 `shift(-n)` 的地方，
   且它产出的列名统一以 `fwd_` 前缀标记，便于 code review 与测试断言）；
2. `tests/test_causality.py` 会**篡改未来数据**再重建面板，逐列断言第 t 行特征不变 ——
   这是可自动化的"无未来函数"证明，而不是靠人眼审查。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .panel import PanelConfig

# 个股特征（截面标准化前）
STOCK_FEATURES = [
    # 收益结构
    "ret1", "gap", "oc", "hl_pos",
    # 动量 / 均线
    "mom5", "mom20", "mom60", "d_ma20", "d_ma60", "dd20", "skew20",
    # 波动
    "vol5", "vol20", "vol60", "ampl20",
    # 流动性 / 换手
    "to5", "to20", "volr", "amtr20", "amihud20",
    # 资金流代理（无逐笔数据时用「成交额 × 方向」构造，口径写在 docstring 里而不是口口相传）
    "mf5", "mf20", "act5", "act20", "pv_corr20",
    # 相对强度
    "ex_mom20",
]

# 市场级特征（按日期只有一行，用扩张窗口标准化）
MARKET_FEATURES = ["mkt_ret", "mkt_vol", "mkt_mom5", "mkt_mom20"]

ALL_FEATURES = STOCK_FEATURES + MARKET_FEATURES


def compute_features(df: pd.DataFrame, cfg: PanelConfig) -> pd.DataFrame:
    """在标准长表上计算特征。输入需按 (code, date) 排好序（panel.normalize_panel 已保证）。"""
    out = df.sort_values(["code", "date"]).reset_index(drop=True).copy()
    g = out.groupby("code", group_keys=False)

    prev_close = g["close"].shift(1)
    out["ret1"] = g["close"].pct_change()
    out["gap"] = out["open"] / prev_close - 1
    out["oc"] = out["close"] / out["open"] - 1
    rng_spread = (out["high"] - out["low"]).replace(0, np.nan)
    out["hl_pos"] = (out["close"] - out["low"]) / rng_spread
    out["ampl"] = (out["high"] - out["low"]) / prev_close

    def mean_of(col: str, window: int) -> pd.Series:
        return g[col].transform(lambda s, w=window: s.rolling(w, min_periods=max(3, w // 2)).mean())

    def std_of(col: str, window: int) -> pd.Series:
        return g[col].transform(lambda s, w=window: s.rolling(w, min_periods=max(3, w // 2)).std())

    out["ma20"] = mean_of("close", 20)
    out["ma60"] = mean_of("close", 60)
    for window in (5, 20, 60):
        out[f"mom{window}"] = out["close"] / g["close"].shift(window) - 1
    out["vol5"] = std_of("ret1", 5)
    out["vol20"] = std_of("ret1", 20)
    out["vol60"] = std_of("ret1", 60)
    out["to5"] = mean_of("turnover", 5)
    out["to20"] = mean_of("turnover", 20)
    out["volr"] = out["volume"] / mean_of("volume", 20).replace(0, np.nan)
    out["ampl20"] = mean_of("ampl", 20)
    out["amtr20"] = mean_of("amount", 20)
    out["d_ma20"] = out["close"] / out["ma20"] - 1
    out["d_ma60"] = out["close"] / out["ma60"] - 1
    out["skew20"] = g["ret1"].transform(lambda s: s.rolling(20, min_periods=10).skew())
    out["dd20"] = out["close"] / g["close"].transform(lambda s: s.rolling(20, min_periods=10).max()) - 1

    # —— 资金流代理 ——
    # 东财逐笔资金流接口对非登录 IP 不稳定，这里用「成交额 × 当日涨跌方向」的净额占比做代理。
    # 它是代理不是真值：口径弱点写在 README 的局限清单里，并在结论中按其保守方向解读。
    direction = np.sign(out["ret1"].fillna(0.0))
    out["_signed_amt"] = out["amount"] * direction
    out["_act"] = np.clip((out["close"] - out["open"]) / rng_spread, -1, 1)
    out["_act_amt"] = out["amount"] * out["_act"].fillna(0.0)
    for window in (5, 20):
        amt_sum = g["amount"].transform(lambda s, w=window: s.rolling(w, min_periods=w // 2).sum()).replace(0, np.nan)
        out[f"mf{window}"] = g["_signed_amt"].transform(
            lambda s, w=window: s.rolling(w, min_periods=w // 2).sum()
        ) / amt_sum
        out[f"act{window}"] = g["_act_amt"].transform(
            lambda s, w=window: s.rolling(w, min_periods=w // 2).sum()
        ) / amt_sum

    out["_amihud"] = out["ret1"].abs() / out["amount"].replace(0, np.nan)
    out["amihud20"] = g["_amihud"].transform(lambda s: s.rolling(20, min_periods=10).mean()) * 1e9

    # 量价相关的展开式实现（比 rolling.apply(corr) 快一个数量级，且截图/复算都容易核对）
    out["_rv"] = out["ret1"] * out["volume"]
    mean_r = mean_of("ret1", 20)
    mean_v = mean_of("volume", 20)
    cov = mean_of("_rv", 20) - mean_r * mean_v
    out["pv_corr20"] = cov / (std_of("ret1", 20) * std_of("volume", 20)).replace(0, np.nan)

    # —— 市场状态（按日期取截面均值，仅用当日及历史）——
    daily_ret = out.groupby("date")["ret1"].mean().sort_index()
    out["mkt_ret"] = out["date"].map(daily_ret)
    out["mkt_vol"] = out["date"].map(out.groupby("date")["ret1"].std())
    out["mkt_mom5"] = out["date"].map(daily_ret.rolling(5, min_periods=3).mean())
    out["mkt_mom20"] = out["date"].map(daily_ret.rolling(20, min_periods=10).mean())
    out["ex_mom20"] = out["mom20"] - out["mkt_mom20"]

    return out.drop(columns=["_signed_amt", "_act", "_act_amt", "_amihud", "_rv"])


def cross_sectional_zscore(df: pd.DataFrame, cols: list[str], suffix: str = "_z") -> pd.DataFrame:
    """按**日期截面**标准化。

    这里必须按日期分组：如果用全样本均值/方差，就是把未来（以及过去）的截面信息泄漏进每一行，
    属于最隐蔽也最常见的一类泄漏。市场级特征只有一个序列，改用扩张窗口标准化（expanding），
    因为它们没有截面可分组。
    """
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        if col in MARKET_FEATURES:
            series = out.groupby("date")[col].first().sort_index()
            expanding_mean = series.expanding(min_periods=60).mean()
            expanding_std = series.expanding(min_periods=60).std()
            out[col + suffix] = out["date"].map((series - expanding_mean) / expanding_std.replace(0, np.nan))
        else:
            by_date = out.groupby("date")[col]
            mean = by_date.transform("mean")
            std = by_date.transform("std").replace(0, np.nan)
            out[col + suffix] = (out[col] - mean) / std
    return out


def feature_columns() -> list[str]:
    return [c + "_z" for c in ALL_FEATURES]