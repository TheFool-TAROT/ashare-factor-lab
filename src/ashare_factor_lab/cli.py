"""命令行入口。

    afl demo                          合成数据跑完整流水线（默认，无需网络/真实数据）
    afl eval --data DIR               真实数据评估，可选 --listing-csv
    afl leakage-check                 因果性自检：篡改未来数据，断言历史特征不变
    afl report --data DIR --out FILE  生成 Markdown 报告

设计上刻意让 `demo` 与 `eval` 共用同一条 `pipeline.run_study`：
如果 demo 能跑、eval 不能跑，说明问题出在数据，而不是代码。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .costs import CostModel
from .model import EvalConfig
from .panel import PanelConfig, build_panel
from .pipeline import format_summary, run_study
from .report import render_report
from .splits import SplitConfig


def _demo_panel(args) -> tuple:
    panel = build_panel(
        synthetic=True,
        n_stocks=args.stocks,
        n_days=args.days,
        seed=args.seed,
        signal_strength=args.signal_strength,
    )
    split = SplitConfig(train_days=500, test_days=60, step_days=60, embargo_days=2)
    return panel, split, "合成数据（带已知反向动量信号，用于验证流水线能把它检出来）"


def _real_panel(args) -> tuple:
    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"数据目录不存在: {data_dir}", file=sys.stderr)
        raise SystemExit(2)
    panel = build_panel(data_dir, listing_csv=args.listing_csv)
    split = SplitConfig(train_days=args.train_days, test_days=args.test_days,
                        step_days=args.step_days, embargo_days=2)
    return panel, split, f"真实数据：{data_dir}"


def _run(args, panel, split, note: str) -> int:
    panel_cfg = PanelConfig(liquidity_min=args.liquidity_min, min_list_days=args.min_list_days)
    eval_cfg = EvalConfig(top_quantile=args.top_quantile, min_stocks_per_day=args.min_stocks)
    result = run_study(
        panel,
        split_cfg=split,
        eval_cfg=eval_cfg,
        panel_cfg=panel_cfg,
        holding_days=args.holding_days,
        cost_model=CostModel(half_spread=args.half_spread / 1e4),
        n_random_simulations=args.simulations,
        n_trials=args.trials,
        verbose=not args.quiet,
    )
    # 让 PanelConfig 进入 meta，报告里能显示过滤阈值
    result.meta["panel"] = panel_cfg.__dict__.copy()

    print("\n" + "=" * 78)
    print(format_summary(result))
    print("=" * 78 + "\n")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_report(result, data_note=f"> 数据来源：{note}"), encoding="utf-8")
        print(f"报告已写入 {out}")
    if args.save_daily:
        daily = Path(args.save_daily)
        daily.parent.mkdir(parents=True, exist_ok=True)
        result.daily_table.to_csv(daily, index=False, encoding="utf-8-sig")
        print(f"逐日明细已写入 {daily}")
    return 0


def cmd_demo(args) -> int:
    panel, split, note = _demo_panel(args)
    return _run(args, panel, split, note)


def cmd_eval(args) -> int:
    panel, split, note = _real_panel(args)
    return _run(args, panel, split, note)


def cmd_leakage_check(args) -> int:
    """因果性自检：把 t 之后的所有行情整体篡改，第 t 行之前的特征必须逐列不变。"""
    import numpy as np
    import pandas as pd

    from .features import ALL_FEATURES, compute_features
    from .panel import normalize_panel, synthesize

    panel = normalize_panel(synthesize(n_stocks=40, n_days=400, seed=args.seed))
    cut = sorted(panel["date"].unique())[300]

    mutated = panel.copy()
    future = mutated["date"] > cut
    for col in ("open", "close", "high", "low", "volume", "amount"):
        mutated.loc[future, col] = mutated.loc[future, col] * float(args.multiplier)
    rng = np.random.default_rng(0)
    mutated.loc[future, ["open", "close", "high", "low"]] *= rng.normal(
        1.0, 0.05, size=(int(future.sum()), 4)
    )

    cfg = PanelConfig()
    base = compute_features(panel, cfg)
    pert = compute_features(normalize_panel(mutated), cfg)

    cols = [c for c in ALL_FEATURES if c in base.columns]
    b = base[base["date"] <= cut][["code", "date", *cols]].reset_index(drop=True)
    p = pert[pert["date"] <= cut][["code", "date", *cols]].reset_index(drop=True)
    diff = (b[cols] - p[cols]).abs()
    worst = float(np.nanmax(diff.to_numpy())) if len(diff) else 0.0

    print(f"[因果性自检] 截断日 {pd.Timestamp(cut).date()}  对比行数 {len(b):,}  特征数 {len(cols)}")
    print(f"  最大差异 {worst:.3e}")
    if worst > 1e-9:
        bad = diff.columns[(diff.max() > 1e-9).to_numpy()]
        print(f"  ❌ 有 {len(bad)} 个特征引用了未来数据：{list(bad)}")
        return 1
    print("  ✅ 历史特征对未来数据完全免疫（无未来函数）")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="afl", description="A 股日频因子评估框架")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--top-quantile", type=float, default=0.1, help="头部组比例，默认 0.1")
        p.add_argument("--min-stocks", type=int, default=50, help="单日最少股票数")
        p.add_argument("--liquidity-min", type=float, default=5e7, help="20 日均成交额下限")
        p.add_argument("--min-list-days", type=int, default=250, help="上市天数下限")
        p.add_argument("--half-spread", type=float, default=10.0, help="半价差假设（bp），默认 10")
        p.add_argument("--holding-days", type=int, default=1, help="持有交易日数，默认 1（t+2 开盘卖出）")
        p.add_argument("--simulations", type=int, default=300, help="随机对照模拟次数")
        p.add_argument("--trials", type=int, default=6, help="多重检验的试验次数（DSR 用）")
        p.add_argument("--out", default=None, help="Markdown 报告输出路径")
        p.add_argument("--save-daily", default=None, help="逐日明细 CSV 输出路径")
        p.add_argument("--quiet", action="store_true")

    demo = sub.add_parser("demo", help="合成数据跑完整流水线")
    demo.add_argument("--stocks", type=int, default=160)
    demo.add_argument("--days", type=int, default=1100)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--signal-strength", type=float, default=0.02)
    common(demo)
    demo.set_defaults(func=cmd_demo, out="output/demo_report.md")

    ev = sub.add_parser("eval", help="真实数据评估")
    ev.add_argument("--data", required=True, help="数据目录（内含 bars/ 与 bars_delist/）")
    ev.add_argument("--listing-csv", default=None)
    ev.add_argument("--train-days", type=int, default=750)
    ev.add_argument("--test-days", type=int, default=60)
    ev.add_argument("--step-days", type=int, default=60)
    common(ev)
    ev.set_defaults(func=cmd_eval)

    lc = sub.add_parser("leakage-check", help="因果性自检")
    lc.add_argument("--seed", type=int, default=7)
    lc.add_argument("--multiplier", type=float, default=3.0)
    lc.set_defaults(func=cmd_leakage_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())