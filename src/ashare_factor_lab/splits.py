"""walk-forward 切分 + embargo。

为什么必须滚动而不能只切一次训练/测试：
单次切分只能得到一个样本外的"运气值"，无法回答"这个信号是不是只在某一段时间有效"。
滚动 10 折 + 每折 60 个交易日，能直接看出折与折之间的离散度（有几折是负的）。

为什么必须有 embargo：
标签跨越 t+1、t+2 两个交易日，训练集最后一天的标签用到了测试集开头的数据。
如果训练集直接接到测试集前面，模型在训练时就已经看到了测试期最初两天的开盘价 ——
这是**标签泄漏**，不是特征泄漏，肉眼审查特征列是查不出来的。
所以训练集末尾要砍掉 embargo 天（默认等于标签跨度 2 天）。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    index: int
    train_dates: np.ndarray
    test_dates: np.ndarray
    embargo_days: int = 0          # 训练集末尾与测试集开头之间隔了几个交易日（由全市场日历数出）

    @property
    def train_end(self) -> pd.Timestamp:
        return pd.Timestamp(self.train_dates[-1])

    @property
    def test_start(self) -> pd.Timestamp:
        return pd.Timestamp(self.test_dates[0])


@dataclass(frozen=True)
class SplitConfig:
    train_days: int = 750
    test_days: int = 60
    step_days: int = 60
    embargo_days: int = 2


def walk_forward(dates: np.ndarray | pd.Index, cfg: SplitConfig) -> Iterator[Fold]:
    """按交易日滚动切分。`dates` 必须是已排序的全市场交易日数组。"""
    calendar = np.array(sorted(pd.Index(dates).unique()))
    span = cfg.train_days + cfg.test_days
    if len(calendar) < span:
        return

    fold = 0
    for start in range(0, len(calendar) - span + 1, cfg.step_days):
        train = calendar[start : start + cfg.train_days - cfg.embargo_days]
        test = calendar[start + cfg.train_days : start + span]
        if len(train) < 2 or len(test) < 2:
            continue
        # embargo 必须用**全市场日历**数，不能用「训练∪测试」的日期集合：
        # 后者根本不包含被跳过的那几天，会恒等于 0（这个坑踩过一次，见 tests）。
        gap = int(np.sum((calendar > train[-1]) & (calendar < test[0])))
        yield Fold(index=fold, train_dates=train, test_dates=test, embargo_days=gap)
        fold += 1


def assert_no_leakage(folds: list[Fold], required_embargo: int) -> None:
    """切分层面的自检：训练/测试不重叠，且 embargo 足够。任何一条不满足就直接抛异常。"""
    for fold in folds:
        train = set(fold.train_dates)
        test = set(fold.test_dates)
        overlap = train & test
        if overlap:
            raise AssertionError(f"fold {fold.index}: 训练集与测试集重叠 {len(overlap)} 天")
        if fold.embargo_days < required_embargo:
            raise AssertionError(
                f"fold {fold.index}: embargo 只有 {fold.embargo_days} 天，"
                f"而标签跨度需要 {required_embargo} 天 —— 训练集末尾的标签会看到测试期数据"
            )
        if fold.train_end >= fold.test_start:
            raise AssertionError(f"fold {fold.index}: 训练集末尾不早于测试集开头")