"""标签与可交易性：把「信号能不能真的变成成交」写死成规则。

标签口径（A 股 T+1 下唯一可执行的口径）：
    t 日收盘后出信号  →  t+1 开盘买入  →  t+2 开盘卖出
即 y = open[t+2] / open[t+1] - 1。不能用 t 日收盘价买入（收盘价出现在信号之后）。

相比常见做法，这里额外做了两件事：
1. **t+1 / t+2 必须是"下一个交易日"**，而不只是"存在下一行"。中途停牌的股票，
   它的下一行可能在几天之后 —— 那种情况下"t+1 开盘买入"是不可执行的。原实现只用
   `shift(-1).notna()` 判断，会把跨停牌的样本当成可交易（实测占比 0.07%，方向上是
   高估可交易性），这里用市场交易日历严格判定。
2. **涨跌停判定逐日重算**：t+1 开盘价触及涨停 → 买不进；t+2 开盘价触及跌停 → 卖不出。
   跌停价用 t+1 收盘价算，涨停价用 t 日收盘价算，全部写成显式列以便复核。

所有引用未来数据的列名一律以 `fwd_` 开头，方便 grep 审计。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import compute_features
from .panel import PanelConfig, attach_listing_dates

FWD_COLS = ["fwd_open_t1", "fwd_date_t1", "fwd_open_t2", "fwd_date_t2", "fwd_ret"]

# add_labels 依赖这些列（来自 features.compute_features），缺失时直接报错而不是静默算错
REQUIRED_FEATURE_COLS = ["ret1", "amtr20"]


def build_labeled_panel(
    panel: pd.DataFrame,
    cfg: PanelConfig,
    *,
    calendar: pd.Series | None = None,
    holding_days: int = 1,
) -> pd.DataFrame:
    """唯一的「从裸行情到可训练面板」入口：特征 → 标签 → 可交易性。

    存在的意义是消除调用顺序这个坑：`compute_features` 与 `add_labels` 谁先谁后
    不是调用方该操心的事，写错顺序会得到看起来正常但数值错误的结果。
    """
    return add_labels(compute_features(panel, cfg), cfg, calendar=calendar, holding_days=holding_days)


def trading_calendar(panel: pd.DataFrame, calendar: pd.Series | None = None) -> pd.Series:
    """交易日历：日期 -> 下一个交易日。

    必须来自**全市场**日期集合。单只股票的日期集合不构成日历 —— 只有它一只票时，
    它的停牌日会从日历里消失，"下一行"和"下一交易日"就变成同一件事，
    停牌检测静默失效（这个坑在上游项目里真的存在过）。所以这里允许外部传入权威日历。
    """
    if calendar is not None:
        return calendar
    dates = pd.Index(sorted(panel["date"].unique()))
    nxt = pd.Series(dates[1:].to_numpy(), index=dates[:-1])
    nxt.loc[dates[-1]] = pd.NaT
    return nxt


def round_half_up(series: pd.Series, digits: int = 2) -> pd.Series:
    """四舍五入到分。不能用 Python/pandas 的 round()：它是银行家舍入（0.5 进偶数位），
    而交易所的涨跌停价是"四舍五入"（0.5 进位），差 1 分钱在边界上就会判错涨跌停。"""
    factor = 10**digits
    return np.floor(series.astype(float) * factor + 0.5) / factor


def limit_pct(codes: pd.Series, is_st: pd.Series, cfg: PanelConfig) -> pd.Series:
    """涨跌停幅度：创业板/科创板 20%，ST 5%，其余 10%。"""
    growth = codes.str.startswith(("300", "301", "688", "689"))
    pct = pd.Series(cfg.limit_main, index=codes.index, dtype=float)
    pct[growth] = cfg.limit_growth
    pct[is_st.fillna(False)] = cfg.limit_st
    return pct


def detect_st_proxy(df: pd.DataFrame, cfg: PanelConfig) -> pd.Series:
    """用「60 日内多次触及 ~5% 但从未超过 5.2%」推断 ST，作为公告口径的代理。

    为什么不用「近 20 日最大振幅 ≤ 5.5%」：那会把所有低波动股票全判成 ST。
    """
    abs_ret = df["ret1"].abs()
    max_abs = abs_ret.groupby(df["code"]).transform(
        lambda s: s.rolling(cfg.st_lookback, min_periods=20).max()
    )
    hits = (abs_ret >= 0.045).groupby(df["code"]).transform(
        lambda s: s.rolling(cfg.st_lookback, min_periods=20).sum()
    )
    return (max_abs <= cfg.limit_st + 0.002) & (hits >= 3)


def missing_reason_report(df: pd.DataFrame) -> pd.DataFrame:
    """标签缺失原因分布：单独把「退市/停牌导致没有后续报价」列出来。

    这一项就是"用未来可得性筛样本"的暴露面：这些行不是没有信号，而是**没有价格可观测**，
    通常对应急跌或缺流动性。直接删掉它们 = 系统性高估收益、低估回撤。
    正确做法（本项目已实现接口、真实数据需自行接入）：把 no_quote_after 的行按保守假设
    （如持有至退市清算价或 -100%）入账，并单列行数，而不是静默 dropna。
    """
    counts = df["fwd_missing_reason"].value_counts()
    labels = {
        "ok": "标签可用",
        "panel_end": "样本末尾（日历尽头，天然无未来行情）",
        "no_quote_after": "退市/长期停牌：无后续报价（**删掉会高估收益**）",
        "halted": "中途停牌：下一行不是下一交易日，不可执行",
    }
    total = len(df)
    return pd.DataFrame(
        [
            {
                "缺失原因": labels.get(k, k),
                "行数": int(v),
                "占比": f"{v / total * 100:.2f}%" if total else "—",
            }
            for k, v in counts.items()
        ]
        + [{"缺失原因": "合计", "行数": total, "占比": "100.00%"}]
    )


def add_labels(
    df: pd.DataFrame,
    cfg: PanelConfig,
    *,
    calendar: pd.Series | None = None,
    holding_days: int = 1,
) -> pd.DataFrame:
    """产出 `fwd_*` 标签列与可交易性列。输入必须先过 `features.compute_features`。

    `holding_days=h` 表示 t+1 开盘买入、持有 h 个交易日后在 t+1+h 开盘卖出（默认 h=1，即 t+2）。
    注意卖出侧的跌停校验必须落在**真正的卖出日** t+1+h 上 —— 上游项目在持有期实验里
    仍然用 h=1 的掩码，等于 K=20 时卖出日从未被校验，净收益因此偏乐观。
    """
    missing = [c for c in REQUIRED_FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"add_labels 需要先计算特征，缺少列 {missing} —— 请改用 build_labeled_panel()"
        )
    out = df.sort_values(["code", "date"]).reset_index(drop=True).copy()
    if "days_listed" not in out.columns:
        # 允许直接传裸行情面板：缺上市天数时按"无真实上市日期"退化处理，而不是静默用错口径。
        out = attach_listing_dates(out)
    g = out.groupby("code", group_keys=False)
    cal = trading_calendar(out, calendar)
    h = int(holding_days)
    if h < 1:
        raise ValueError("holding_days 必须 >= 1（A 股 T+1，最早也只能 t+1 开盘买、t+2 开盘卖）")

    out["fwd_open_t1"] = g["open"].shift(-1)
    out["fwd_date_t1"] = g["date"].shift(-1)
    out["fwd_open_t2"] = g["open"].shift(-(1 + h))
    out["fwd_date_t2"] = g["date"].shift(-(1 + h))

    # 交易日历连推 h+1 步：买入日是 t+1，卖出日是 t+1+h
    next_cal = out["date"].map(cal)
    exit_cal = next_cal
    for _ in range(h):
        exit_cal = exit_cal.map(cal)
    out["fwd_next_cal_date"] = next_cal
    out["fwd_next2_cal_date"] = exit_cal

    # 严格判定：下一行必须**就是**下一个交易日（中途停牌 → 下一行在几天之后 → 不可执行）
    date_ok = out["fwd_date_t1"].eq(next_cal) & out["fwd_date_t2"].eq(exit_cal)
    out["has_t1"] = out["fwd_date_t1"].eq(next_cal) & out["fwd_open_t1"].notna()
    out["has_t2"] = out["fwd_date_t2"].eq(exit_cal) & out["fwd_open_t2"].notna()
    out["fwd_ret"] = np.where(
        out["has_t1"] & out["has_t2"], out["fwd_open_t2"] / out["fwd_open_t1"] - 1, np.nan
    )

    # 缺失原因分级：把"样本末尾"和"退市/长期停牌导致没有后续报价"分开计数。
    # 这件事很关键：直接 dropna 会把退市股票的最后一截（通常是急跌）整段删掉，
    # 等于用"未来可得性"筛样本，系统性地美化绝对收益与尾部风险。
    out["fwd_missing_reason"] = np.select(
        [date_ok, out["fwd_date_t1"].isna() & next_cal.isna(), out["fwd_date_t1"].isna()],
        ["ok", "panel_end", "no_quote_after"],
        default="halted",
    )

    out["st_proxy"] = detect_st_proxy(out, cfg)
    out["limit_pct"] = limit_pct(out["code"], out["st_proxy"], cfg)
    out["prev_close_1"] = g["close"].shift(1)
    # 卖出日的跌停价参照「卖出日前一交易日收盘」= close[t+h]
    out["fwd_close_before_exit"] = g["close"].shift(-h)

    # 涨停价由 t 日收盘决定；跌停价由卖出日前收盘决定。
    # 必须用**不复权**价算限价：用后复权价算涨停价，在除权日会因复权因子跳变而判错。
    out["fwd_limit_up_t1"] = round_half_up(out["close"] * (1 + out["limit_pct"]))
    out["fwd_limit_dn_t2"] = round_half_up(out["fwd_close_before_exit"] * (1 - out["limit_pct"]))

    out["cant_buy"] = (out["fwd_open_t1"] >= out["fwd_limit_up_t1"] - cfg.tick).fillna(True)
    out["cant_sell"] = (out["fwd_open_t2"] <= out["fwd_limit_dn_t2"] + cfg.tick).fillna(True)

    out["liq_ok"] = out["amtr20"] > cfg.liquidity_min
    out["mature"] = out["days_listed"] >= cfg.min_list_days
    out["tradable"] = (
        out["liq_ok"]
        & out["mature"]
        & (~out["st_proxy"].fillna(True))
        & out["has_t1"]
        & out["has_t2"]
        & (~out["cant_buy"])
        & (~out["cant_sell"])
    )
    return out


def tradability_report(df: pd.DataFrame) -> pd.DataFrame:
    """逐条过滤的剔除行数，用于报告里解释"为什么样本只剩这么点"。"""
    checks = {
        "流动性不足(20日均额<下限)": ~df["liq_ok"],
        "次新(上市不足 N 日)": ~df["mature"],
        "ST 代理(5% 停板)": df["st_proxy"].fillna(True),
        "缺 t+1/t+2 交易日行情(含停牌)": ~(df["has_t1"] & df["has_t2"]),
        "t+1 开盘涨停买不进": df["cant_buy"],
        "t+2 开盘跌停卖不出": df["cant_sell"],
    }
    rows = [{"过滤项": name, "剔除行数": int(mask.sum())} for name, mask in checks.items()]
    total = len(df)
    kept = int(df["tradable"].sum())
    rows.append({"过滤项": "最终可交易", "剔除行数": kept})
    rows.append({"过滤项": "样本总行数", "剔除行数": total})
    return pd.DataFrame(rows)