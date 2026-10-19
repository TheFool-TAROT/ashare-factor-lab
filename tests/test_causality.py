"""因果性测试 —— 整个仓库最重要的一组断言。

思路：把第 t 天之后的所有行情**整体篡改**（乘系数 + 随机扰动），
重新跑一遍特征工程，然后断言"第 t 天及以前"的所有特征逐列完全不变。

为什么这个测试比人眼审查强：
- 它不依赖审查者是否记得每个指标的定义；
- 任何一行偷偷使用了未来数据的代码（shift(-n)、整段标准化、rolling 中心对齐、
  用全样本 mean/std 做填充）都会让它立刻变红；
- 它同时覆盖特征工程与截面标准化两条路径。

注意：标签模块 labels.py **会**使用未来数据（这是它的职责），所以本测试
只比对特征列，并把 fwd_* 列排除在外。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_factor_lab.features import ALL_FEATURES, compute_features
from ashare_factor_lab.panel import PanelConfig, normalize_panel, synthesize


def _mutate_future(panel: pd.DataFrame, cut: pd.Timestamp, seed: int = 0) -> pd.DataFrame:
    """把 cut 之后（严格大于 cut）的行情整体改掉，cut 当天不动。"""
    mutated = panel.copy()
    future = mutated["date"] > cut
    rng = np.random.default_rng(seed)
    for col in ("open", "close", "high", "low", "volume", "amount"):
        mutated.loc[future, col] = mutated.loc[future, col] * rng.uniform(0.2, 5.0, size=int(future.sum()))
    # 保证 high >= low 之类的内部一致性，避免制造出不合法的行情而掩盖真实结论
    future_mask = future.to_numpy()
    idx = mutated.index[future_mask]
    mutated.loc[idx, "high"] = mutated.loc[idx, ["open", "close", "high"]].max(axis=1) * 1.05
    mutated.loc[idx, "low"] = mutated.loc[idx, ["open", "close", "low"]].min(axis=1) * 0.95
    return mutated


def test_features_immune_to_future_mutation():
    panel = normalize_panel(synthesize(n_stocks=30, n_days=360, seed=1))
    dates = sorted(panel["date"].unique())
    cut = dates[280]
    cfg = PanelConfig()

    base = compute_features(panel, cfg)
    perturbed = compute_features(_mutate_future(panel, cut), cfg)

    cols = [c for c in ALL_FEATURES if c in base.columns]
    left = base[base["date"] <= cut][["code", "date", *cols]].reset_index(drop=True)
    right = perturbed[perturbed["date"] <= cut][["code", "date", *cols]].reset_index(drop=True)

    assert len(left) > 0 and len(left) == len(right)
    diff = (left[cols] - right[cols]).abs()
    worst = float(np.nanmax(diff.to_numpy())) if diff.size else 0.0
    assert worst < 1e-9, f"特征引用了未来数据，最大差异 {worst:.3e}：{list(diff.columns[(diff.max() > 1e-9)])}"


def test_cross_sectional_zscore_is_per_date_only():
    """截面标准化必须按日期分组：篡改未来日期不能影响历史日期的 z 分数。"""
    from ashare_factor_lab.features import cross_sectional_zscore

    panel = normalize_panel(synthesize(n_stocks=25, n_days=200, seed=2))
    cfg = PanelConfig()
    base = compute_features(panel, cfg)
    mutated = compute_features(_mutate_future(panel, sorted(panel["date"].unique())[150]), cfg)

    cols = ["mom5", "mom20", "vol20"]
    a = cross_sectional_zscore(base, cols)
    b = cross_sectional_zscore(mutated, cols)

    cut = sorted(panel["date"].unique())[150]
    za = a[a["date"] <= cut][["code", "date", *[c + "_z" for c in cols]]].reset_index(drop=True)
    zb = b[b["date"] <= cut][["code", "date", *[c + "_z" for c in cols]]].reset_index(drop=True)
    assert za.shape == zb.shape
    assert float((za[[c + "_z" for c in cols]] - zb[[c + "_z" for c in cols]]).abs().max().max()) < 1e-9


def test_market_features_are_contemporaneous_not_forward():
    """市场特征只允许用当日截面（以及更早）：篡改 t+1 之后的行情不能改变 t 日的 mkt_*。"""
    panel = normalize_panel(synthesize(n_stocks=20, n_days=260, seed=5))
    cfg = PanelConfig()
    cut = sorted(panel["date"].unique())[200]

    base = compute_features(panel, cfg)
    perturbed = compute_features(_mutate_future(panel, cut), cfg)

    mkt = [c for c in ALL_FEATURES if c.startswith("mkt_")] + ["ex_mom20"]
    left = base[base["date"] <= cut][mkt].reset_index(drop=True)
    right = perturbed[perturbed["date"] <= cut][mkt].reset_index(drop=True)
    assert float((left - right).abs().max().max()) < 1e-9


def test_no_forward_looking_calls_outside_labels():
    """静态检查（AST 层面，不做字符串匹配，避免把文档里的示例误判成违规）：

    - 除 labels.py 外，禁止 `shift(-n)` / `shift(periods=-n)`
    - 全仓库禁止 `rolling(center=True)`（中心对齐窗口会把未来数据卷进当行）

    用 AST 而不是 grep：文档里正当地提到 `shift(-n)` 不该被算作违规。
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "ashare_factor_lab"
    offenders: list[str] = []

    def _is_negative(node: ast.AST) -> bool:
        return isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)

    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", None)
            if name == "shift" and path.name != "labels.py":
                if node.args and _is_negative(node.args[0]):
                    offenders.append(f"{path.name}:{node.lineno} 使用了 shift(负数)")
                for kw in node.keywords:
                    if kw.arg == "periods" and _is_negative(kw.value):
                        offenders.append(f"{path.name}:{node.lineno} 使用了 shift(periods=负数)")
            if name == "rolling":
                for kw in node.keywords:
                    if kw.arg == "center" and getattr(kw.value, "value", False) is True:
                        offenders.append(f"{path.name}:{node.lineno} 使用了 rolling(center=True)")

    assert not offenders, f"发现疑似使用未来数据的代码：{offenders}"