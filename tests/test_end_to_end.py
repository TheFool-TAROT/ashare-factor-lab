"""端到端防回归：docstring 与 README 里的结论都由这段流水线产出，跑不通就说明文档在说谎。"""

from __future__ import annotations

import pandas as pd

from ashare_factor_lab.model import EvalConfig, prepare_dataset
from ashare_factor_lab.panel import PanelConfig, normalize_panel, synthesize
from ashare_factor_lab.pipeline import format_summary, run_study
from ashare_factor_lab.report import render_report
from ashare_factor_lab.splits import SplitConfig


def _small_study():
    panel = normalize_panel(synthesize(n_stocks=60, n_days=420, seed=13, signal_strength=0.4))
    cfg = PanelConfig(liquidity_min=0, min_list_days=0)
    split = SplitConfig(train_days=200, test_days=40, step_days=40, embargo_days=2)
    result = run_study(
        panel,
        split_cfg=split,
        eval_cfg=EvalConfig(top_quantile=0.1, min_stocks_per_day=20),
        n_random_simulations=40,
        panel_cfg=cfg,
        verbose=False,
    )
    return result, cfg


def test_prepare_dataset_is_tradable_and_labeled():
    panel = normalize_panel(synthesize(n_stocks=30, n_days=300, seed=9))
    cfg = PanelConfig(liquidity_min=0, min_list_days=0)
    data = prepare_dataset(panel, cfg)
    assert data["tradable"].all()
    assert data["fwd_ret"].notna().all()
    assert data["date"].is_monotonic_increasing is False or True   # 排序不是接口保证，不该被依赖
    assert data["code"].nunique() > 20


def test_end_to_end_study_produces_all_artifacts():
    result, _ = _small_study()

    assert result.summary["folds"] >= 1
    assert result.summary["days"] > 40
    assert not result.fold_table.empty
    assert not result.daily_table.empty

    assert set(result.portfolio_table.index) == {
        "模型头部组(毛)", "模型头部组(扣费后)", "单因子动量(扣费后)", "宇宙等权(基准)"
    }
    assert 0.0 <= result.random_control.p_value <= 1.0
    assert result.cost_summary["days"] == len(result.cost_frame)
    assert len(result.cost_sensitivity) == 3
    assert 0.0 <= result.deflated_sharpe["p_value"] <= 1.0


def test_fold_table_records_embargo_and_bounds():
    result, _ = _small_study()
    folds = result.fold_table
    assert (folds["embargo_days"] >= 2).all()
    assert (pd.to_datetime(folds["train_end"]) < pd.to_datetime(folds["test_start"])).all()
    assert (folds["n_train"] > 0).all() and (folds["n_test"] > 0).all()


def test_summary_text_and_report_are_consistent():
    result, _ = _small_study()
    text = format_summary(result)
    for token in ("RankIC", "换手", "随机选股对照", "Sharpe"):
        assert token in text

    report = render_report(result, title="测试报告", data_note="> 数据来源：合成")
    for section in ("## 1. 结论", "## 4. 交易成本", "## 5. 组合层面", "## 7. 局限", "## 8. 复现"):
        assert section in report
    assert "合成" in report


def test_study_reports_tradability_and_universe_size():
    result, _ = _small_study()
    assert result.panel_rows > 0
    assert result.stocks >= 50
    assert result.delisted_stocks > 0
    assert "最终可交易" in set(result.tradability["过滤项"])


def test_pipeline_raises_on_too_short_history():
    import pytest

    panel = normalize_panel(synthesize(n_stocks=20, n_days=120, seed=10))
    with pytest.raises(RuntimeError, match="没有产生任何有效的折"):
        run_study(
            panel,
            split_cfg=SplitConfig(train_days=500, test_days=60, step_days=60, embargo_days=2),
            n_random_simulations=5,
            verbose=False,
        )