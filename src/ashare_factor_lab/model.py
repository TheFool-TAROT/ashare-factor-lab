"""模型与滚动评估主循环：把面板、切分、模型、指标串起来。

模型选择上刻意用 sklearn 的 HistGradientBoostingRegressor 而不是 LightGBM：
功能上等价（都是直方图分裂的 GBDT），但它是 sklearn 的一部分，**零额外依赖**，
CI 和 demo 不需要装任何编译型包就能跑。私有版本用 LightGBM 得到的结论一致 —
"用哪个 GBDT"从来不是这个项目里重要的那个变量。

每折的流程：
    训练集（砍掉 embargo 天）拟合 -> 测试集逐日打分 -> 逐日算 RankIC / 头部超额 / 换手
折间不做任何参数选择，测试集只出现一次；所有汇总指标都是**外样本**的。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import ALL_FEATURES, cross_sectional_zscore, feature_columns
from .labels import build_labeled_panel
from .splits import Fold, SplitConfig, assert_no_leakage, walk_forward


@dataclass
class EvalConfig:
    top_quantile: float = 0.1          # 头部组取前 10%
    min_stocks_per_day: int = 50       # 截面股票太少不算指标
    label_col: str = "fwd_ret"
    standardize_label: bool = True     # 训练用截面 z 分数，排序不受影响但收敛更稳
    random_state: int = 42
    model_params: dict = field(default_factory=lambda: {
        "max_iter": 300,
        "learning_rate": 0.03,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 100,
        "l2_regularization": 1.0,
        "early_stopping": False,
    })


def prepare_dataset(
    panel: pd.DataFrame,
    cfg=None,
    *,
    use_market_features: bool = True,
    tradable_only: bool = True,
    label_col: str = "fwd_ret",
    holding_days: int = 1,
) -> pd.DataFrame:
    """从原始面板一路算到「特征 + 标签 + 可交易性」，返回可训练的宽表。"""
    from .panel import PanelConfig

    panel_cfg = cfg or PanelConfig()
    data = build_labeled_panel(panel, panel_cfg, holding_days=holding_days)
    data = cross_sectional_zscore(data, [c for c in ALL_FEATURES if c in data.columns])

    if not use_market_features:
        data = data.drop(columns=[c for c in feature_columns() if c.startswith("mkt_")], errors="ignore")
    if tradable_only:
        data = data[data["tradable"]].copy()

    stock_feats = [c for c in feature_columns() if c in data.columns and not c.startswith("mkt_")]
    data = data.dropna(subset=stock_feats)
    market_feats = [c for c in feature_columns() if c in data.columns and c.startswith("mkt_")]
    data[market_feats] = data[market_feats].fillna(0.0)     # 披露中断等缺失按历史均值处理
    data = data.dropna(subset=[label_col])
    return data.reset_index(drop=True)


def zscore_label(data: pd.DataFrame, label_col: str = "fwd_ret") -> pd.DataFrame:
    out = data.copy()
    by_date = out.groupby("date")[label_col]
    out[label_col + "_z"] = (out[label_col] - by_date.transform("mean")) / by_date.transform("std").replace(0, np.nan)
    return out


def _fit_predict(
    train: pd.DataFrame, test: pd.DataFrame, feats: list[str], cfg: EvalConfig, label_col: str
) -> np.ndarray:
    target = label_col + "_z" if cfg.standardize_label else label_col
    model = HistGradientBoostingRegressor(random_state=cfg.random_state, **cfg.model_params)
    model.fit(train[feats], train[target])
    return model.predict(test[feats])


def run_walk_forward(
    data: pd.DataFrame,
    split_cfg: SplitConfig | None = None,
    eval_cfg: EvalConfig | None = None,
    *,
    feats: list[str] | None = None,
    tag: str = "model",
    verbose: bool = False,
    label_overlap_days: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """滚动评估。返回 (每折指标表, 逐日明细表) —— 两张表都只含外样本数据。"""
    split_cfg = split_cfg or SplitConfig()
    eval_cfg = eval_cfg or EvalConfig()
    if eval_cfg.standardize_label:
        data = zscore_label(data, eval_cfg.label_col)
    feats = feats or [c for c in feature_columns() if c in data.columns]
    label_col = eval_cfg.label_col

    folds: list[Fold] = list(walk_forward(sorted(data["date"].unique()), split_cfg))
    assert_no_leakage(folds, required_embargo=label_overlap_days)

    fold_rows, daily_rows = [], []
    for fold in folds:
        train = data[data["date"].isin(fold.train_dates)]
        test = data[data["date"].isin(fold.test_dates)]
        if len(train) < 1000 or len(test) < 200:
            continue

        test = test.assign(pred=_fit_predict(train, test, feats, eval_cfg, label_col))
        prev_picks: set | None = None
        fold_ics, fold_top, fold_bench, fold_turn = [], [], [], []

        for date, day in test.groupby("date"):
            day = day.dropna(subset=["pred", label_col])
            if len(day) < eval_cfg.min_stocks_per_day:
                continue
            rho = day["pred"].rank().corr(day[label_col].rank())
            if pd.notna(rho):
                fold_ics.append(float(rho))

            k = max(1, int(len(day) * eval_cfg.top_quantile))
            top = day.nlargest(k, "pred")
            bench = float(day[label_col].mean())
            top_ret = float(top[label_col].mean())
            picks = set(top["code"])
            turn = np.nan
            if prev_picks is not None:
                turn = len(picks - prev_picks) / len(picks)
                fold_turn.append(turn)
            prev_picks = picks

            fold_top.append(top_ret)
            fold_bench.append(bench)
            daily_rows.append(
                {
                    "tag": tag,
                    "fold": fold.index,
                    "date": date,
                    "rank_ic": float(rho) if pd.notna(rho) else np.nan,
                    "top_ret": top_ret,
                    "bench_ret": bench,
                    "excess": top_ret - bench,
                    "turnover": turn,
                    "n_stocks": int(len(day)),
                    "picks": ",".join(sorted(picks)),
                }
            )

        if fold_ics:
            fold_rows.append(
                {
                    "tag": tag,
                    "fold": fold.index,
                    "train_start": fold.train_dates[0],
                    "train_end": fold.train_end,
                    "test_start": fold.test_dates[0],
                    "test_end": fold.test_dates[-1],
                    "embargo_days": fold.embargo_days,
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "days": len(fold_ics),
                    "rank_ic": float(np.mean(fold_ics)),
                    "ic_ir": float(np.mean(fold_ics) / (np.std(fold_ics, ddof=1) + 1e-12)),
                    "top_ret": float(np.mean(fold_top)),
                    "bench_ret": float(np.mean(fold_bench)),
                    "excess": float(np.mean(fold_top) - np.mean(fold_bench)),
                    "turnover": float(np.mean(fold_turn)) if fold_turn else np.nan,
                }
            )
            if verbose:
                print(
                    f"  fold {fold.index}: test {fold.test_dates[0]}~{fold.test_dates[-1]}"
                    f"  RankIC {np.mean(fold_ics) * 100:.3f}%  超额 {(np.mean(fold_top) - np.mean(fold_bench)) * 100:.4f}%",
                    flush=True,
                )

    return pd.DataFrame(fold_rows), pd.DataFrame(daily_rows)


def aggregate(fold_table: pd.DataFrame) -> dict:
    """折间汇总：均值 + 折的稳定性（几折为正、最差折）。"""
    if fold_table.empty:
        return {}
    return {
        "folds": int(len(fold_table)),
        "days": int(fold_table["days"].sum()),
        "rank_ic": float(fold_table["rank_ic"].mean()),
        "ic_ir": float(fold_table["ic_ir"].mean()),
        "top_ret": float(fold_table["top_ret"].mean()),
        "bench_ret": float(fold_table["bench_ret"].mean()),
        "excess": float(fold_table["excess"].mean()),
        "turnover": float(fold_table["turnover"].mean()),
        "positive_folds": int((fold_table["rank_ic"] > 0).sum()),
        "worst_fold_ic": float(fold_table["rank_ic"].min()),
        "best_fold_ic": float(fold_table["rank_ic"].max()),
    }