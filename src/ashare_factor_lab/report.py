"""Markdown 报告渲染：结论先行，证据在后。

报告结构按"读者愿意读下去的强度"排序：
    1. 结论（3 行以内，含正负号）
    2. 方法（口径写清楚，否则数字没有意义）
    3. 结果表（外样本）
    4. 成本敏感性（这个项目里最决定性的一张表）
    5. 随机对照与多重检验（"是不是运气"）
    6. 局限清单（不写这节的量化项目不值得信）
"""

from __future__ import annotations

import pandas as pd

from .pipeline import StudyResult


def pandas_float(series: pd.Series) -> bool:
    try:
        return pd.api.types.is_float_dtype(series)
    except TypeError:                     # object 列混了时间戳与浮点
        return False


def _pct(x: float, digits: int = 4) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:.{digits}f}%"


def _num(x: float, digits: int = 3) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.{digits}f}"


def _md_table(frame: pd.DataFrame, floatfmt: str = ".4f", index_name: str | None = None) -> str:
    """渲染 Markdown 表。带名字的索引（如组合对照表的策略名）必须渲染出来，
    否则表格会变成一堆没有行标签的数字 —— 读的人根本不知道哪一行是策略、哪一行是基准。"""
    if frame.empty:
        return "_(无数据)_"
    styled = frame.copy()
    if index_name is not None or not isinstance(styled.index, pd.RangeIndex):
        old_name = styled.index.name or "index"
        styled = styled.reset_index()
        if index_name and old_name in styled.columns:
            styled = styled.rename(columns={old_name: index_name})
    for col in styled.columns:
        if pd.api.types.is_datetime64_any_dtype(styled[col]):
            styled[col] = pd.to_datetime(styled[col]).dt.strftime("%Y-%m-%d")
        elif pandas_float(styled[col]):
            styled[col] = styled[col].map(lambda v: "—" if pd.isna(v) else format(v, floatfmt))
    header = "| " + " | ".join(str(c) for c in styled.columns) + " |"
    sep = "|" + "|".join("---" for _ in styled.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in styled.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def render_report(result: StudyResult, title: str = "日频横截面选股因子评估", data_note: str = "") -> str:
    s = result.summary
    net = result.cost_summary
    rc = result.random_control
    dsr = result.deflated_sharpe
    verdict = _verdict(result)
    holding = result.meta.get("holding_days", 1)

    parts = [
        f"# {title}",
        "",
        f"> 自动生成 · 外样本 {s['days']} 个交易日 / {s['folds']} 折 · "
        f"样本 {result.panel_rows:,} 行 / {result.stocks} 只股票（含退市 {result.delisted_stocks} 只）",
        "",
        "## 1. 结论",
        "",
        verdict,
        "",
        "## 2. 方法（口径先行）",
        "",
        "| 项 | 设置 |",
        "|---|---|",
        f"| 标签 | t 日收盘出信号 → **t+1 开盘买 → t+1+{holding} 开盘卖**"
        f"（`fwd_ret = open[t+1+{holding}]/open[t+1]-1`，默认持有 1 日） |",
        f"| 切分 | walk-forward，训练 {result.meta['split']['train_days']} 日 / 测试 {result.meta['split']['test_days']} 日 / "
        f"步长 {result.meta['split']['step_days']} 日 / embargo {result.meta['split']['embargo_days']} 日 |",
        "| 可交易性 | 流动性 + 上市天数 + ST 代理 + 交易日连续性 + t+1 涨停买不进 + t+2 跌停卖不出 |",
        f"| 头部组 | 每日按预测分取前 {result.meta['eval']['top_quantile'] * 100:.0f}% |",
        f"| 成本 | 佣金 {result.meta['cost']['commission'] * 1e4:.2f}bp/边 + 过户费 "
        f"{result.meta['cost']['transfer_fee'] * 1e4:.2f}bp/边 + 印花税 {result.meta['cost']['stamp_duty'] * 1e4:.0f}bp(卖出) "
        f"+ 半价差 {result.meta['cost']['half_spread'] * 1e4:.0f}bp/边 |",
        f"| 随机对照 | 同宇宙同规模随机选股 {result.meta['n_random_simulations']} 次 |",
        "",
        "## 3. 外样本结果",
        "",
        _md_table(
            pd.DataFrame(
                {
                    "指标": ["RankIC 均值", "IC_IR", "IC 普通 t", "IC Newey-West t", "正 IC 折数",
                             "头部组日均超额(毛)", "Bootstrap 95% CI 下界", "Bootstrap p 值", "日均换手"],
                    "值": [
                        _pct(s["rank_ic"]),
                        _num(s["ic_ir"]),
                        _num(result.ic_stats["ic_tstat"], 2),
                        _num(result.ic_stats["nw_tstat"], 2),
                        f"{s['positive_folds']}/{s['folds']}",
                        _pct(s["excess"]),
                        _pct(result.bootstrap["ci_low"]),
                        _num(result.bootstrap["p_value"]),
                        _pct(s["turnover"], 1),
                    ],
                }
            )
        ),
        "",
        "### 逐折稳定性（看最差折，而不是只看均值）",
        "",
        _md_table(
            result.fold_table[
                ["fold", "train_start", "train_end", "test_start", "test_end", "embargo_days",
                 "rank_ic", "excess", "turnover"]
            ],
            floatfmt=".4f",
        ),
        "",
        "外样本天数合计 " + str(int(result.fold_table["days"].sum())) + " 天；"
        "换手率是**单边替换率**（每天新进名单占当日的比例），成本按它折算。",
        "",
        "## 4. 交易成本：这个项目里最重要的一张表",
        "",
        _md_table(
            pd.DataFrame(
                {
                    "半价差(bp)": [f"{v:.0f}" for v in result.cost_sensitivity["half_spread_bp"]],
                    "整仓换一遍成本": [f"{v * 100:.3f}%" for v in result.cost_sensitivity["per_turnover_cost"]],
                    "日均成本": [f"{v * s['turnover'] * 100:.4f}%" for v in result.cost_sensitivity["per_turnover_cost"]],
                    "扣费后年化超额": [
                        _pct((1 + s['excess'] - v * s['turnover']) ** 252 - 1, 1)
                        for v in result.cost_sensitivity["per_turnover_cost"]
                    ],
                }
            )
        ),
        "",
        f"**毛超额 {_pct(s['excess'])} / 日 vs 成本 {_pct(net['mean_cost'])} / 日 → 成本是毛收益的 "
        f"{(net['mean_cost'] / s['excess']):.2f} 倍。**" if s["excess"] and s["excess"] > 0 else
        f"**毛超额 {_pct(s['excess'])} / 日，成本 {_pct(net['mean_cost'])} / 日。**",
        "",
        "## 5. 组合层面（含基准，全部同宇宙同测试日）",
        "",
        _md_table(result.portfolio_table, floatfmt=".4f", index_name="组合"),
        "",
        "### 随机选股对照（回答「是不是瞎选」）",
        "",
        _md_table(
            pd.DataFrame(
                {
                    "项": ["随机选股超额均值", "随机分布 5% / 50% / 95% 分位", "策略超额", "策略所处分位", "p 值"],
                    "值": [
                        _pct(rc.mean_excess),
                        f"{_pct(rc.quantiles.get('q05'))} / {_pct(rc.quantiles.get('q50'))} / {_pct(rc.quantiles.get('q95'))}",
                        _pct(rc.strategy_excess),
                        f"{rc.percentile * 100:.1f}%",
                        _num(rc.p_value),
                    ],
                }
            )
        ),
        "",
        "### 多重检验修正（Deflated Sharpe）",
        "",
        _md_table(
            pd.DataFrame(
                {
                    "项": ["扣费后年化 Sharpe", "试验次数", "多次试验下的期望最大 Sharpe",
                            "真实 Sharpe>0 的概率（已经多重检验修正）"],
                    "值": [
                        _num(dsr["sharpe_annual"], 2),
                        str(dsr["n_trials"]),
                        _num(dsr.get("sr0_annual", float("nan")), 2),
                        _num(dsr["p_value"]),
                    ],
                }
            )
        ),
        "",
        "## 6. 可交易性过滤明细",
        "",
        _md_table(result.tradability),
        "",
        "### 标签缺失原因分布（这里藏着一个容易犯的错）",
        "",
        _md_table(result.missing_reasons),
        "",
        "注意 `no_quote_after` 一行：这些行不是没有信号，而是**没有价格可观测**（退市、长期停牌），",
        "通常对应急跌。上游版本用 `dropna` 把它们整行删掉，等于用「未来可得性」筛样本，",
        "会系统性高估收益、低估回撤。本框架把这类行单独计数并暴露出来，",
        "绝对收益口径下应按保守假设（持有至清算价）入账，而不是静默丢弃。",
        "",
        "## 7. 局限（不写这一节的研究不值得信）",
        "",
        "1. **时点成分股不可得**：宇宙是「当前成分股 + 样本期内退市股」，不是真正的历史成分。",
        "2. **ST 是代理不是公告口径**：用 5% 停板特征推断，会有漏判与误判。",
        "3. **滑点是假设**：没有盘口/逐笔数据，用 5/10/20bp 三档做敏感性，真实值取决于资金量与冲击成本。",
        "4. **未做行业/市值中性化**：模型可能部分在赌行业轮动，这部分收益不稳健。",
        "5. **未考虑涨停板排队**：开盘价成交的假设在极端行情下偏乐观。",
        "6. **多重检验**：持有期等超参是扫过的，DSR 已做修正，但不排除仍有未记录的试错。",
        "",
        "## 8. 复现",
        "",
        "```bash",
        "make demo        # 合成数据，约 3 分钟跑完整条流水线（无需网络与真实数据）",
        "make test        # 含因果性扰动测试（篡改未来数据，断言第 t 行特征不变）",
        "afl eval --data <真实数据目录> --listing-csv <上市日期.csv>",
        "```",
        "",
    ]
    if data_note:
        parts.insert(3, data_note)
    return "\n".join(parts)


def _verdict(result: StudyResult) -> str:
    """把结论压成 3 行以内 —— 读者只读这一段也要能拿到完整判断。"""
    s = result.summary
    net = result.cost_summary
    rc = result.random_control
    lines = []

    if s["excess"] > 0 and net["mean_net_excess"] > 0:
        lines.append(
            f"信号存在（RankIC {_pct(s['rank_ic'])}，NW t {_num(result.ic_stats['nw_tstat'], 2)}），"
            f"且扣费后仍有 {_pct(net['mean_net_excess'])} / 日。"
        )
    elif s["excess"] > 0:
        lines.append(
            f"信号存在但被成本淹没：毛超额 {_pct(s['excess'])} / 日，成本 {_pct(net['mean_cost'])} / 日，"
            f"扣费后 {_pct(net['mean_net_excess'])} / 日。"
        )
    else:
        lines.append(f"外样本未检出可用的正向信号：头部组日均超额 {_pct(s['excess'])}。")

    if pd.notna(rc.p_value):
        lines.append(
            f"随机选股对照 p={_num(rc.p_value)}，策略落在随机分布的 {rc.percentile * 100:.1f}% 分位"
            + ("（与瞎选无显著差异）。" if rc.p_value > 0.05 else "（显著优于瞎选）。")
        )
    lines.append(
        f"稳定性：{s['positive_folds']}/{s['folds']} 折为正，最差折 RankIC {_pct(s['worst_fold_ic'])}；"
        f"换手 {_pct(s['turnover'], 1)} / 日 —— 换手是这个策略的第一约束。"
    )
    return "\n".join("> " + line for line in lines) if lines else "_(无结论)_"