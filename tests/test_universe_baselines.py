"""宇宙构造与基准对照：这两块直接决定结论可信度。

生存偏差（只用"今天的成分股"回溯）会把 IC 抬高一半以上，所以必须能证明
「退市股真的被纳入了」；而基准必须与策略同宇宙同过滤，否则比较不公平。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ashare_factor_lab import baselines
from ashare_factor_lab.features import compute_features
from ashare_factor_lab.labels import add_labels
from ashare_factor_lab.panel import (
    PanelConfig,
    attach_listing_dates,
    build_panel,
    normalize_panel,
    synthesize,
)


def test_synthetic_panel_contains_delisted_stocks():
    panel = normalize_panel(synthesize(n_stocks=50, n_days=300, seed=1, delist_ratio=0.2))
    assert int(panel["is_delisted"].sum()) > 0
    delisted_codes = panel.loc[panel["is_delisted"] == 1, "code"].unique()
    assert len(delisted_codes) > 0
    # 退市股的最后一个交易日必须早于面板最后一天（否则"退市"没有意义）
    last = panel.groupby("code")["date"].max()
    panel_end = panel["date"].max()
    assert all(last[c] < panel_end for c in delisted_codes)


def test_days_listed_prefers_the_panel_own_listing_date():
    """合成/真实面板若自带上市日期，必须用它 —— 而不是用行号（上游项目的真实缺陷）。"""
    panel = normalize_panel(synthesize(n_stocks=3, n_days=200, seed=2))
    assert "listing_date" in panel.columns

    out = attach_listing_dates(panel)
    assert set(out["days_listed_source"]) == {"listing_date"}
    old_stock = out[out["code"] == out["code"].iloc[0]].iloc[0]
    assert old_stock["days_listed"] > 100          # 样本开始前上市，不该被判成次新


def test_days_listed_falls_back_to_row_index_without_listing_date():
    """拿不到上市日期时必须显式退化，并在列里留下痕迹以便审计。"""
    panel = normalize_panel(synthesize(n_stocks=3, n_days=200, seed=2)).drop(columns=["listing_date"])
    out = attach_listing_dates(panel)

    assert set(out["days_listed_source"]) == {"row_index_approx"}
    assert out[out["code"] == out["code"].iloc[0]]["days_listed"].iloc[0] == 0


def test_listing_csv_overrides_panel_column(tmp_path):
    panel = normalize_panel(synthesize(n_stocks=3, n_days=200, seed=2))
    listing = tmp_path / "listing_dates.csv"
    listing.write_text("证券代码,证券简称,上市日期\n100000,AAA,2020-01-01\n", encoding="utf-8")

    out = attach_listing_dates(panel, listing)
    row = out[out["code"] == "100000"].iloc[0]
    assert row["days_listed_source"] == "listing_date"
    assert row["listing_date"] == pd.Timestamp("2020-01-01")


def test_build_panel_merges_main_and_delisted_dirs(tmp_path):
    bars = tmp_path / "bars"
    delisted = tmp_path / "bars_delist"
    bars.mkdir()
    delisted.mkdir()

    main = synthesize(n_stocks=4, n_days=120, seed=3, delist_ratio=0.0)
    for code, group in main.groupby("code"):
        group.to_csv(bars / f"{code}.csv", index=False)

    dead = main[main["code"] == main["code"].iloc[0]].copy()
    dead["code"] = "999999"                      # 退市股用独立代码存到另一个目录
    dead.to_csv(delisted / "999999.csv", index=False)

    panel = build_panel(tmp_path)
    assert set(panel["is_delisted"].unique()) == {0, 1}
    assert 999999 in set(panel["code"].astype(int))


def test_universe_benchmark_is_equal_weight_of_tradable_universe():
    panel = normalize_panel(synthesize(n_stocks=20, n_days=200, seed=4))
    cfg = PanelConfig(liquidity_min=0, min_list_days=0)
    data = add_labels(compute_features(panel, cfg), cfg)
    data["fwd_ret"] = data["code"].astype(int) % 3 / 100.0     # 造一个确定性的标签

    bench = baselines.universe_benchmark(data[data["tradable"]])
    one_day = data[(data["tradable"]) & (data["date"] == bench.index[0])]
    assert bench.iloc[0] == pytest.approx(one_day["fwd_ret"].mean())


def test_random_control_detects_a_real_signal():
    """把"按未来收益排序选股"当作完美策略：它必须显著优于随机选股（p 很小）。"""
    panel = normalize_panel(synthesize(n_stocks=40, n_days=220, seed=5))
    cfg = PanelConfig(liquidity_min=0, min_list_days=0)
    data = add_labels(compute_features(panel, cfg), cfg)
    tradable = data[data["tradable"]].copy()
    test_days = sorted(tradable["date"].unique())[120:]
    tradable = tradable[tradable["date"].isin(test_days)]

    perfect = baselines.pick_top_by_score(tradable, "fwd_ret", 0.1)
    result = baselines.random_pick_excess(tradable, perfect, n_simulations=80, seed=0)

    assert result.p_value < 0.05
    assert result.percentile > 0.9
    assert result.strategy_excess > result.mean_excess


def test_random_control_p_value_is_uniform_for_random_picks():
    """随机的"策略"落在随机分布中部，p 值不该显著。"""
    panel = normalize_panel(synthesize(n_stocks=40, n_days=220, seed=6))
    cfg = PanelConfig(liquidity_min=0, min_list_days=0)
    data = add_labels(compute_features(panel, cfg), cfg)
    tradable = data[data["tradable"]].copy()
    test_days = sorted(tradable["date"].unique())[120:]
    tradable = tradable[tradable["date"].isin(test_days)]

    rng = np.random.default_rng(0)
    tradable = tradable.assign(noise=rng.normal(size=len(tradable)))
    random_picks = baselines.pick_top_by_score(tradable, "noise", 0.1)
    result = baselines.random_pick_excess(tradable, random_picks, n_simulations=80, seed=1)
    assert 0.0 <= result.p_value <= 1.0
    assert 0.1 < result.percentile < 0.9


def test_momentum_baseline_shapes():
    panel = normalize_panel(synthesize(n_stocks=25, n_days=240, seed=7))
    cfg = PanelConfig(liquidity_min=0, min_list_days=0)
    data = add_labels(compute_features(panel, cfg), cfg)
    tradable = data[data["tradable"]]
    baseline = baselines.momentum_baseline(tradable)
    assert set(baseline) == {"name", "picks", "returns", "excess", "turnover"}
    assert len(baseline["returns"]) > 50
    assert baseline["name"].startswith("单因子基准")