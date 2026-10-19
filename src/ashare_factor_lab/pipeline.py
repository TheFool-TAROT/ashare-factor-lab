"""完整研究流水线：一条函数跑完「特征 → 滚动评估 → 基准对照 → 成本 → 统计检验」。

被 cli.py（命令行）与 tests/test_end_to_end.py（防回归）共用 —— 这样"文档里写的结论"
与"测试跑的东西"是同一段代码，不会漂。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import backtest, baselines, costs, metrics
from .costs import CostModel
from .labels import missing_reason_report, tradability_report
from .model import EvalConfig, aggregate, prepare_dataset, run_walk_forward
from .splits import SplitConfig


@dataclass
class StudyResult:
    panel_rows: int
    stocks: int
    delisted_stocks: int
    date_range: tuple
    tradability: pd.DataFrame
    missing_reasons: pd.DataFrame
    fold_table: pd.DataFrame
    daily_table: pd.DataFrame
    summary: dict
    ic_stats: dict
    bootstrap: dict
    random_control: baselines.RandomPickResult
    momentum: dict
    cost_frame: pd.DataFrame
    cost_summary: dict
    cost_sensitivity: pd.DataFrame
    portfolio_table: pd.DataFrame
    deflated_sharpe: dict
    meta: dict = field(default_factory=dict)


def run_study(
    panel: pd.DataFrame,
    *,
    split_cfg: SplitConfig | None = None,
    eval_cfg: EvalConfig | None = None,
    cost_model: CostModel | None = None,
    panel_cfg=None,
    holding_days: int = 1,
    n_random_simulations: int = 300,
    n_trials: int = 6,
    random_seed: int = 42,
    verbose: bool = True,
) -> StudyResult:
    split_cfg = split_cfg or SplitConfig()
    eval_cfg = eval_cfg or EvalConfig()
    cost_model = cost_model or CostModel()

    data = prepare_dataset(panel, panel_cfg, holding_days=holding_days)
    if verbose:
        print(f"可训练样本 {len(data):,} 行 · 股票 {data['code'].nunique()} 只 · "
              f"{data['date'].min().date()} ~ {data['date'].max().date()}", flush=True)

    fold_table, daily_table = run_walk_forward(
        data, split_cfg, eval_cfg, tag="model", verbose=verbose,
        label_overlap_days=holding_days + 1,
    )
    if fold_table.empty:
        raise RuntimeError("没有产生任何有效的折 —— 检查数据长度与 SplitConfig（train+test 是否超过样本期）")

    summary = aggregate(fold_table)
    ic_series = daily_table.set_index("date")["rank_ic"].dropna()
    ic_stats = metrics.summarize_ic(ic_series, overlap_k=2)

    gross = daily_table.set_index("date")["top_ret"].sort_index()
    bench = daily_table.set_index("date")["bench_ret"].sort_index()
    turnover = daily_table.set_index("date")["turnover"].sort_index().fillna(0.0)
    excess = (gross - bench).dropna()

    bootstrap = metrics.bootstrap_mean_ci(excess)
    cost_frame = costs.net_excess_curve(excess, turnover.reindex(excess.index).fillna(0.0), cost_model)
    cost_summary = costs.summarize_costs(cost_frame)
    dsr = metrics.deflated_sharpe_ratio(cost_frame["net_excess"], n_trials=n_trials)

    # —— 基准对照：全部在同一宇宙、同一测试日上计算 ——
    test_data = data[data["date"].isin(daily_table["date"].unique())]
    random_control = baselines.random_pick_excess(
        test_data,
        _picks_from_daily(daily_table),
        n_simulations=n_random_simulations,
        seed=random_seed,
    )
    momentum = baselines.momentum_baseline(test_data, eval_cfg.top_quantile)

    portfolio_table = backtest.compare_table(
        {
            "模型头部组(毛)": gross,
            "模型头部组(扣费后)": cost_frame["net_excess"],
            "单因子动量(扣费后)": _net_of(momentum["excess"], momentum["turnover"], cost_model),
            "宇宙等权(基准)": bench,
        }
    )

    return StudyResult(
        panel_rows=int(len(panel)),
        stocks=int(panel["code"].nunique()),
        delisted_stocks=int(panel.loc[panel.get("is_delisted", 0) == 1, "code"].nunique())
        if "is_delisted" in panel.columns else 0,
        date_range=(panel["date"].min(), panel["date"].max()),
        tradability=tradability_report(_labeled(panel, panel_cfg)),
        missing_reasons=missing_reason_report(_labeled(panel, panel_cfg)),
        fold_table=fold_table,
        daily_table=daily_table,
        summary=summary,
        ic_stats=ic_stats,
        bootstrap=bootstrap,
        random_control=random_control,
        momentum=momentum,
        cost_frame=cost_frame,
        cost_summary=cost_summary,
        cost_sensitivity=cost_model.sensitivity_table(),
        portfolio_table=portfolio_table,
        deflated_sharpe=dsr,
        meta={
            "split": split_cfg.__dict__.copy(),
            "eval": {"top_quantile": eval_cfg.top_quantile, "min_stocks_per_day": eval_cfg.min_stocks_per_day},
            "cost": cost_model.__dict__.copy(),
            "n_random_simulations": n_random_simulations,
            "n_trials": n_trials,
            "holding_days": holding_days,
        },
    )


def _labeled(panel: pd.DataFrame, panel_cfg=None) -> pd.DataFrame:
    """报告可交易性统计需要标签列（amt20/上市天数等），这里按需补算一次。"""
    from .labels import build_labeled_panel
    from .panel import PanelConfig

    return build_labeled_panel(panel, panel_cfg or PanelConfig())


def _picks_from_daily(daily_table: pd.DataFrame) -> dict:
    picks = {}
    for row in daily_table.itertuples():
        picks[row.date] = set(str(row.picks).split(",")) if row.picks else set()
    return picks


def _net_of(excess: pd.Series, turnover: pd.Series, cost_model: CostModel) -> pd.Series:
    frame = pd.DataFrame({"g": excess, "t": turnover}).dropna()
    return (frame["g"] - frame["t"].map(cost_model.daily_cost)).sort_index()


def format_summary(result: StudyResult) -> str:
    """终端输出的核心结论（与报告同源，避免两处各算一遍）。"""
    s = result.summary
    lines = [
        f"外样本天数 {s['days']}  折数 {s['folds']}（正 IC 折 {s['positive_folds']}/{s['folds']}）",
        f"RankIC 均值 {s['rank_ic'] * 100:.3f}%   IC_IR {s['ic_ir']:.3f}"
        f"   NW t {result.ic_stats['nw_tstat']:.2f}",
        f"头部组日均超额(毛) {s['excess'] * 100:.4f}%   Bootstrap 95% CI "
        f"[{result.bootstrap['ci_low'] * 100:.4f}%, {result.bootstrap['ci_high'] * 100:.4f}%]"
        f"   p={result.bootstrap['p_value']:.3f}",
        f"日均换手 {s['turnover'] * 100:.1f}%   每次换仓成本 {result.cost_summary['mean_cost'] / max(s['turnover'], 1e-9) * 100:.3f}%"
        f"   日均成本 {result.cost_summary['mean_cost'] * 100:.4f}%",
        f"扣费后日均超额 {result.cost_summary['mean_net_excess'] * 100:.4f}%"
        f"   年化净超额 {result.cost_summary['annualized_net_excess'] * 100:.1f}%",
        f"随机选股对照：随机超额均值 {result.random_control.mean_excess * 100:.4f}%"
        f"  策略分位 {result.random_control.percentile * 100:.1f}%  p={result.random_control.p_value:.3f}",
        f"扣费后 Sharpe {result.deflated_sharpe['sharpe_annual']:.2f}"
        f"（考虑 {result.deflated_sharpe['n_trials']} 次试验后 DSR p={result.deflated_sharpe['p_value']:.3f}）",
    ]
    return "\n".join(lines)