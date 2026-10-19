"""标签与可交易性：用手工构造的小面板断言具体数值，避免"看起来对"。"""

from __future__ import annotations

import pandas as pd

from ashare_factor_lab.labels import tradability_report
from ashare_factor_lab.panel import PanelConfig
from tests.conftest import labeled, toy_panel


def _five_days(opens, closes, code="600000", dates=("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08")):
    return toy_panel(
        [
            {"code": code, "date": d, "open": o, "close": c}
            for d, o, c in zip(dates, opens, closes, strict=False)
        ]
    )


def test_label_is_t1_open_to_t2_open(panel_cfg):
    panel = _five_days([10.0, 11.0, 13.0, 14.0, 15.0], [10.5, 11.5, 12.5, 14.5, 15.5])
    out = labeled(panel, PanelConfig(liquidity_min=0, min_list_days=0))
    row = out[out["date"] == "2024-01-02"].iloc[0]

    assert row["fwd_open_t1"] == 11.0        # t+1 开盘买入
    assert row["fwd_open_t2"] == 13.0        # t+2 开盘卖出
    assert abs(row["fwd_ret"] - (13.0 / 11.0 - 1)) < 1e-12
    assert bool(row["has_t1"]) and bool(row["has_t2"])

    last = out[out["date"] == "2024-01-08"].iloc[0]
    assert pd.isna(last["fwd_ret"])          # 样本末尾没有未来行情


def test_suspension_makes_label_unavailable(panel_cfg):
    """中途停牌：下一行不是"下一个交易日"，因此 t+1 开盘买入不可执行。"""
    rows = [
        {"code": "600000", "date": "2024-01-02", "open": 10.0, "close": 10.1},
        {"code": "600000", "date": "2024-01-03", "open": 10.2, "close": 10.3},
        {"code": "600000", "date": "2024-01-08", "open": 10.4, "close": 10.5},  # 停了 01-04、01-05
        {"code": "600000", "date": "2024-01-09", "open": 10.6, "close": 10.7},
    ]
    # 必须有一只"从不缺席"的参照股：交易日历取自全市场日期集合，
    # 只有停牌股自己时，它的停牌日会从日历里消失，停牌检测会静默失效。
    for d, px in zip(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"],
        [20.0, 20.1, 20.2, 20.3, 20.4, 20.5], strict=False,
    ):
        rows.append({"code": "600519", "date": d, "open": px, "close": px * 1.001})
    panel = toy_panel(rows)
    out = labeled(panel, PanelConfig(liquidity_min=0, min_list_days=0))
    row = out[out["date"] == "2024-01-03"].iloc[0]

    assert row["fwd_date_t1"] == pd.Timestamp("2024-01-08")
    assert row["fwd_next_cal_date"] == pd.Timestamp("2024-01-04")
    assert not bool(row["has_t1"]), "停牌后的下一行不该被当成 t+1"
    assert pd.isna(row["fwd_ret"])
    assert not bool(row["tradable"])


def test_cannot_buy_when_t1_opens_at_limit_up(panel_cfg):
    """t+1 开盘直接涨停（= 昨收 ×1.1，四舍五入到分）→ 买不进。"""
    panel = _five_days([10.0, 11.0, 11.5, 11.8, 12.0], [10.0, 11.0, 11.4, 11.9, 12.1])
    out = labeled(panel, PanelConfig(liquidity_min=0, min_list_days=0))
    row = out[out["date"] == "2024-01-02"].iloc[0]

    assert row["fwd_limit_up_t1"] == 11.0    # 10.0 * 1.1
    assert bool(row["cant_buy"])
    assert not bool(row["tradable"])


def test_cannot_sell_when_t2_opens_at_limit_down(panel_cfg):
    """t+2 开盘跌停（= t+1 收盘 ×0.9）→ T+1 卖不出。"""
    panel = toy_panel(
        [
            {"code": "600000", "date": "2024-01-02", "open": 20.0, "close": 20.0},
            {"code": "600000", "date": "2024-01-03", "open": 19.5, "close": 18.0},   # t+1 收盘 18.0
            {"code": "600000", "date": "2024-01-04", "open": 16.2, "close": 16.5},   # 18.0*0.9 = 16.2
            {"code": "600000", "date": "2024-01-05", "open": 16.0, "close": 16.0},
        ]
    )
    out = labeled(panel, PanelConfig(liquidity_min=0, min_list_days=0))
    row = out[out["date"] == "2024-01-02"].iloc[0]

    assert row["fwd_limit_dn_t2"] == 16.2
    assert bool(row["cant_sell"])
    assert not bool(row["tradable"])


def test_boards_have_different_limits(panel_cfg):
    main = _five_days([10.0, 10.2, 10.3, 10.4, 10.5], [10.1, 10.2, 10.3, 10.4, 10.5], code="600000")
    growth = _five_days([10.0, 10.2, 10.3, 10.4, 10.5], [10.1, 10.2, 10.3, 10.4, 10.5], code="300001")
    cfg = PanelConfig(liquidity_min=0, min_list_days=0)
    a = labeled(main, cfg).iloc[0]
    b = labeled(growth, cfg).iloc[0]
    assert a["limit_pct"] == 0.10
    assert b["limit_pct"] == 0.20


def test_liquidity_filter_is_conservative_when_history_is_short(panel_cfg):
    """20 日均额算不出来（历史不足）时必须保守判为不可交易 —— 而不是当成"流动性无限好"。"""
    panel = _five_days([10.0, 10.2, 10.3, 10.4, 10.5], [10.1, 10.2, 10.3, 10.4, 10.5])
    out = labeled(panel, PanelConfig(liquidity_min=0.0, min_list_days=0))
    assert out["amtr20"].isna().all()
    assert not bool(out["liq_ok"].any())
    assert not bool(out["tradable"].any())


def test_listing_days_filter_uses_config_threshold(panel_cfg):
    panel = _five_days([10.0, 10.2, 10.3, 10.4, 10.5], [10.1, 10.2, 10.3, 10.4, 10.5])
    strict = labeled(panel, PanelConfig(liquidity_min=1e12, min_list_days=10_000))
    assert not bool(strict["liq_ok"].any())
    assert not bool(strict["mature"].any())
    assert not bool(strict["tradable"].any())

    relaxed = labeled(panel, PanelConfig(liquidity_min=1e12, min_list_days=0))
    assert bool(relaxed["mature"].any())          # 上市天数这一条放开了
    assert not bool(relaxed["tradable"].any())    # 但流动性这一条仍然拦住


def test_tradable_rows_exist_on_a_long_panel(panel_cfg, synthetic_panel):
    out = labeled(synthetic_panel, panel_cfg)
    assert bool(out["tradable"].any())
    assert 0 < int(out["tradable"].sum()) < len(out)   # 过滤确实在起作用，不是全通也不是全拦


def test_tradability_report_counts_lines(panel_cfg, synthetic_panel):

    out = labeled(synthetic_panel, panel_cfg)
    report = tradability_report(out)
    lookup = dict(zip(report["过滤项"], report["剔除行数"], strict=False))

    assert lookup["样本总行数"] == len(out)
    assert lookup["最终可交易"] == int(out["tradable"].sum())
    assert lookup["最终可交易"] <= lookup["样本总行数"]
    assert set(lookup) >= {
        "流动性不足(20日均额<下限)", "次新(上市不足 N 日)", "ST 代理(5% 停板)",
        "缺 t+1/t+2 交易日行情(含停牌)", "t+1 开盘涨停买不进", "t+2 开盘跌停卖不出",
    }
    assert set(report.columns) == {"过滤项", "剔除行数"}