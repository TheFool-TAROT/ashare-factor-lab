"""测试公共夹具：所有测试都在合成数据上跑，不联网、不依赖真实行情。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ashare_factor_lab.labels import build_labeled_panel
from ashare_factor_lab.panel import PanelConfig, normalize_panel, synthesize


@pytest.fixture
def panel_cfg() -> PanelConfig:
    """测试用的宽松阈值：小样本下不希望被流动性/次新过滤掉太多行。"""
    return PanelConfig(liquidity_min=1e4, min_list_days=0)


@pytest.fixture(scope="session")
def synthetic_panel() -> pd.DataFrame:
    return normalize_panel(synthesize(n_stocks=40, n_days=420, seed=3))


@pytest.fixture(scope="session")
def synthetic_cost_panel() -> pd.DataFrame:
    """带交易成本水平的合成面板（成交额更大，便于测试流动性过滤）。"""
    df = normalize_panel(synthesize(n_stocks=30, n_days=300, seed=11))
    df["amount"] = df["amount"] * 10
    return df


def _toy_panel(rows: list[dict]) -> pd.DataFrame:
    """手工构造的小面板，用于断言具体的标签数值。"""
    frame = pd.DataFrame(rows)
    frame["code"] = frame["code"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"])
    for col in ("open", "close", "high", "low", "volume", "amount", "turnover"):
        if col not in frame.columns:
            frame[col] = np.nan
    frame["volume"] = frame["volume"].fillna(1e6)
    frame["amount"] = frame["amount"].fillna(1e8)
    frame["turnover"] = frame["turnover"].fillna(0.02)
    frame["high"] = frame["high"].fillna(frame[["open", "close"]].max(axis=1) * 1.01)
    frame["low"] = frame["low"].fillna(frame[["open", "close"]].min(axis=1) * 0.99)
    return normalize_panel(frame)


def _labeled(panel: pd.DataFrame, cfg: PanelConfig) -> pd.DataFrame:
    return build_labeled_panel(panel, cfg)


# 以 fixture 形式暴露辅助函数：测试里写成函数参数即可，不需要 `from tests.conftest import ...`
# —— 后者在 CI 上会因为 rootdir 不在 sys.path 而 ImportError（踩过一次）。
@pytest.fixture
def toy_panel():
    return _toy_panel


@pytest.fixture
def labeled():
    return _labeled