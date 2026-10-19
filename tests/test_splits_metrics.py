"""切分与统计指标的数学断言。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ashare_factor_lab import metrics
from ashare_factor_lab.splits import Fold, SplitConfig, assert_no_leakage, walk_forward

# ─────────────────────────── walk-forward ───────────────────────────

def test_folds_do_not_overlap_and_respect_embargo():
    dates = pd.bdate_range("2021-01-04", periods=1200)
    cfg = SplitConfig(train_days=500, test_days=60, step_days=60, embargo_days=2)
    folds = list(walk_forward(dates, cfg))

    assert len(folds) >= 8
    assert_no_leakage(folds, required_embargo=2)
    for fold in folds:
        assert fold.embargo_days >= 2
        assert fold.train_end < fold.test_start
        assert len(fold.train_dates) == cfg.train_days - cfg.embargo_days
        assert len(fold.test_dates) == cfg.test_days


def test_embargo_actually_removes_tail_days():
    """训练集末尾必须被砍掉 embargo 天 —— 否则标签会看到测试期前两天的开盘价。"""
    dates = pd.bdate_range("2021-01-04", periods=800)
    no_embargo = list(walk_forward(dates, SplitConfig(500, 60, 60, embargo_days=0)))[0]
    with_embargo = list(walk_forward(dates, SplitConfig(500, 60, 60, embargo_days=2)))[0]

    assert with_embargo.train_dates[-1] != no_embargo.train_dates[-1]
    assert with_embargo.train_end < no_embargo.train_end
    assert with_embargo.test_dates[0] == no_embargo.test_dates[0]


def test_insufficient_history_yields_no_fold():
    dates = pd.bdate_range("2024-01-01", periods=100)
    assert list(walk_forward(dates, SplitConfig(500, 60, 60, 2))) == []


def test_assert_no_leakage_catches_bad_fold():
    dates = pd.bdate_range("2021-01-04", periods=20)
    bad = Fold(index=0, train_dates=dates[:10].to_numpy(), test_dates=dates[8:].to_numpy())
    with pytest.raises(AssertionError, match="重叠"):
        assert_no_leakage([bad], required_embargo=0)


def test_assert_no_leakage_catches_missing_embargo():
    dates = pd.bdate_range("2021-01-04", periods=20)
    fold = Fold(index=0, train_dates=dates[:10].to_numpy(), test_dates=dates[10:].to_numpy())
    with pytest.raises(AssertionError, match="embargo"):
        assert_no_leakage([fold], required_embargo=2)


# ─────────────────────────── 指标 ───────────────────────────

def test_rank_ic_is_one_for_perfect_ranking():
    rng = np.random.default_rng(0)
    dates = pd.Series(pd.bdate_range("2024-01-01", periods=5).repeat(60))
    label = pd.Series(rng.normal(size=300))
    pred = label * 3 + 1                      # 单调变换，秩相关仍为 1
    ic = metrics.rank_ic_by_day(pred, label, dates)
    assert len(ic) == 5
    assert float(ic.min()) > 0.999


def test_rank_ic_is_zero_for_random_ranking():
    rng = np.random.default_rng(1)
    dates = pd.Series(pd.bdate_range("2024-01-01", periods=200).repeat(60))
    label = pd.Series(rng.normal(size=200 * 60))
    pred = pd.Series(rng.normal(size=200 * 60))
    ic = metrics.rank_ic_by_day(pred, label, dates)
    assert abs(float(ic.mean())) < 0.02


def test_newey_west_matches_ols_when_no_autocorrelation():
    """无自相关时，NW t 值应接近普通 t 值（修正项不该凭空放大或缩小）。"""
    rng = np.random.default_rng(2)
    series = pd.Series(rng.normal(0.002, 0.01, size=1000))
    plain = metrics.ic_tstat(series)
    nw = metrics.newey_west_tstat(series, lags=1)
    assert abs(nw - plain) < 0.4


def test_newey_west_penalizes_positive_autocorrelation():
    """重叠收益（自相关为正）会使普通 t 值虚高；NW 必须把它压下来。"""
    rng = np.random.default_rng(3)
    base = rng.normal(0.002, 0.01, size=800)
    overlapped = pd.Series(np.convolve(base, np.ones(5) / 5, mode="valid"))
    plain = metrics.ic_tstat(overlapped)
    nw = metrics.newey_west_tstat(overlapped)
    assert nw < plain, "NW 没有压掉重叠带来的虚高显著性"


def test_bootstrap_ci_brackets_mean_and_detects_positive():
    rng = np.random.default_rng(4)
    series = pd.Series(rng.normal(0.005, 0.01, size=500))
    out = metrics.bootstrap_mean_ci(series, n_boot=2000)
    assert out["ci_low"] < out["mean"] < out["ci_high"]
    assert out["p_value"] < 0.05


def test_bootstrap_p_value_is_large_for_noise():
    rng = np.random.default_rng(5)
    series = pd.Series(rng.normal(0.0, 0.01, size=500))
    out = metrics.bootstrap_mean_ci(series, n_boot=2000)
    assert out["p_value"] > 0.2


def test_deflated_sharpe_punishes_more_trials():
    rng = np.random.default_rng(6)
    returns = pd.Series(rng.normal(0.0006, 0.01, size=600))
    few = metrics.deflated_sharpe_ratio(returns, n_trials=1)
    many = metrics.deflated_sharpe_ratio(returns, n_trials=200)
    assert many["p_value"] < few["p_value"], "试错次数增加后 DSR 应该更严格"
    assert 0.0 <= many["p_value"] <= 1.0
    assert many["sr0_annual"] > few["sr0_annual"]


def test_summarize_ic_contains_all_fields():
    rng = np.random.default_rng(7)
    ic = pd.Series(rng.normal(0.02, 0.08, size=300))
    out = metrics.summarize_ic(ic, overlap_k=2)
    assert out["days"] == 300
    assert set(out) == {"days", "rank_ic_mean", "rank_ic_std", "ic_ir", "ic_tstat", "nw_tstat", "ic_positive_ratio"}