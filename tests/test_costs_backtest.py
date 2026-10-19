"""交易成本与组合回测指标的数学断言。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ashare_factor_lab import backtest
from ashare_factor_lab.costs import (
    CostModel,
    net_excess_curve,
    realized_turnover_series,
    summarize_costs,
    turnover_rate,
)


def test_per_turnover_cost_matches_hand_calculation():
    cost = CostModel(commission=0.00025, transfer_fee=0.00001, stamp_duty=0.0005, half_spread=0.0010)
    # (0.00025 + 0.00001 + 0.0010) * 2 + 0.0005 = 0.00302
    assert cost.per_turnover_cost() == pytest.approx(0.00302, abs=1e-12)
    assert cost.daily_cost(0.68) == pytest.approx(0.68 * 0.00302, abs=1e-12)
    assert cost.daily_cost(0.0) == 0.0


def test_stamp_duty_only_charged_on_sell_side():
    """印花税只在卖出收：把佣金/过户费/滑点置零后，整仓换一遍的成本应恰好等于印花税。"""
    cost = CostModel(commission=0.0, transfer_fee=0.0, stamp_duty=0.0005, half_spread=0.0)
    assert cost.per_turnover_cost() == pytest.approx(0.0005, abs=1e-12)


def test_sensitivity_table_covers_three_levels():
    table = CostModel().sensitivity_table()
    assert list(table["half_spread_bp"]) == [5.0, 10.0, 20.0]
    assert table["per_turnover_cost"].is_monotonic_increasing


def test_break_even_matches_cost():
    cost = CostModel()
    turnover = 0.35
    assert cost.break_even_gross_excess(turnover) == pytest.approx(cost.daily_cost(turnover))


def test_turnover_rate_definition():
    assert turnover_rate({"a", "b", "c"}, None) is not np.nan
    assert np.isnan(turnover_rate({"a"}, None))
    assert turnover_rate({"a", "b", "c"}, {"a", "b", "c"}) == 0.0
    assert turnover_rate({"a", "b", "c"}, {"a"}) == pytest.approx(2 / 3)


def test_realized_turnover_series_is_first_day_nan():
    series = realized_turnover_series(
        {pd.Timestamp("2024-01-02"): {"a"}, pd.Timestamp("2024-01-03"): {"b"}}
    )
    assert len(series) == 1
    assert series.iloc[0] == 1.0


def test_net_excess_curve_subtracts_cost_and_compounds():
    idx = pd.bdate_range("2024-01-01", periods=5)
    gross = pd.Series(0.01, index=idx)
    turnover = pd.Series(0.5, index=idx)
    cost = CostModel()
    frame = net_excess_curve(gross, turnover, cost)

    expected_cost = 0.5 * cost.per_turnover_cost()
    assert frame["cost"].iloc[0] == pytest.approx(expected_cost)
    assert frame["net_excess"].iloc[0] == pytest.approx(0.01 - expected_cost)
    assert frame["equity_net"].iloc[-1] < frame["equity_gross"].iloc[-1]
    assert frame["equity_gross"].iloc[-1] == pytest.approx(1.01**5)


def test_cost_summary_reports_break_even_direction():
    idx = pd.bdate_range("2024-01-01", periods=200)
    gross = pd.Series(0.0009, index=idx)          # 每日毛超额 9bp
    turnover = pd.Series(0.68, index=idx)         # 每日换手 68%
    frame = net_excess_curve(gross, turnover, CostModel(half_spread=0.0010))
    summary = summarize_costs(frame)

    assert summary["mean_gross_excess"] == pytest.approx(0.0009)
    assert summary["mean_net_excess"] < 0, "成本高于毛收益时净超额必须为负"
    assert summary["annualized_net_excess"] < 0
    assert summary["equity_net_final"] < 1.0


def test_equity_curve_and_drawdown():
    returns = pd.Series([0.10, -0.20, 0.05])
    curve = backtest.equity_curve(returns)
    assert curve.iloc[-1] == pytest.approx(1.1 * 0.8 * 1.05)
    assert backtest.max_drawdown(returns) == pytest.approx(0.88 / 1.1 - 1)


def test_summarize_produces_all_risk_fields():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0005, 0.01, size=500))
    stats = backtest.summarize(returns, "strategy")
    assert stats["days"] == 500
    assert set(stats) >= {"total_return", "annualized_return", "annualized_vol", "sharpe", "max_drawdown", "calmar", "hit_rate"}
    assert stats["sharpe"] == pytest.approx(returns.mean() / returns.std(ddof=1) * np.sqrt(252), rel=1e-9)


def test_sharpe_is_nan_for_constant_returns():
    assert np.isnan(backtest.sharpe_ratio(pd.Series([0.001] * 10)))


def test_compare_table_rows_match_inputs():
    idx = pd.bdate_range("2024-01-01", periods=100)
    table = backtest.compare_table(
        {"a": pd.Series(0.001, index=idx), "b": pd.Series(-0.001, index=idx)}
    )
    assert list(table.index) == ["a", "b"]
    assert table.loc["a", "total_return"] > 0 > table.loc["b", "total_return"]