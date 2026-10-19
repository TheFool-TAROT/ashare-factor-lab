"""评估指标：从"相关系数"到"能不能下结论"，每一层都给出对应的统计检验。

只报 RankIC 是不够的，IC 高不代表策略能赚钱；只报净值也不够，净值高不代表不是运气。
这里按组合告警的强度分层：

    rank_ic / ic_ir      —— 信号与次日收益的截面秩相关及其稳定性
    ic_tstat             —— IC 序列的 t 值（假设独立同分布，会**高估**显著性）
    newey_west_tstat     —— 修正重叠收益自相关的 t 值（这才是该报的那个）
    bootstrap_top_excess —— 头部组超额收益的 Bootstrap 置信区间与 p 值
    deflated_sharpe      —— 考虑"试了多少个变体"后的 Sharpe（多重检验修正）

为什么 IC 的普通 t 值会高估：IC 序列本身是横截面相关、且相邻日重叠（日频调仓），
自相关不为零，Newey-West 用截断核把它修掉。迟滞阶数默认取 ⌊4(n/100)^(2/9)⌋ 经验公式。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def rank_ic_by_day(pred: pd.Series, label: pd.Series, dates: pd.Series, min_stocks: int = 30) -> pd.Series:
    """逐日横截面 Spearman 秩相关。样本数太少的截面直接跳过（统计量没有意义）。"""
    frame = pd.DataFrame({"pred": pred, "label": label, "date": dates})
    ics = {}
    for date, group in frame.groupby("date"):
        clean = group.dropna()
        if len(clean) < min_stocks:
            continue
        rho = clean["pred"].rank().corr(clean["label"].rank())
        if pd.notna(rho):
            ics[date] = float(rho)
    return pd.Series(ics).sort_index()


def ic_ir(ic_series: pd.Series) -> float:
    """IC 的信息比：均值 / 标准差（不年化，避免给人虚假的"年化"错觉）。"""
    if len(ic_series) < 2 or ic_series.std(ddof=1) == 0:
        return 0.0
    return float(ic_series.mean() / ic_series.std(ddof=1))


def ic_tstat(ic_series: pd.Series) -> float:
    if len(ic_series) < 2:
        return 0.0
    t, _ = stats.ttest_1samp(ic_series, 0.0)
    return float(t) if np.isfinite(t) else 0.0


def newey_west_tstat(series: pd.Series, lags: int | None = None) -> float:
    """Newey-West 调整 t 值：HAC 标准误下的 H0: mean = 0。

    日频调仓时，K 日持有收益高度重叠，普通 t 值会明显虚高。默认滞后阶数用
    Newey-West 的经验公式 floor(4*(n/100)^(2/9))。
    """
    x = series.dropna().to_numpy(dtype=float)
    n = x.size
    if n < 10:
        return 0.0
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
        lags = max(1, min(lags, n - 2))

    demeaned = x - x.mean()
    gamma0 = float(demeaned @ demeaned) / n
    hac = gamma0
    for k in range(1, lags + 1):
        weight = 1.0 - k / (lags + 1)          # Bartlett 核
        cov = float(demeaned[k:] @ demeaned[:-k]) / n
        hac += 2.0 * weight * cov
    if hac <= 0:
        return 0.0
    se = np.sqrt(hac / n)
    return float(x.mean() / se) if se > 0 else 0.0


def bootstrap_mean_ci(
    values: pd.Series, n_boot: int = 5000, alpha: float = 0.05, seed: int = 42
) -> dict:
    """对均值做 Bootstrap：返回置信区间、单边 p 值与样本量。

    比 t 检验适合金融数据（厚尾、非正态）：它不假设分布形状，只假设样本可重抽。
    注意它**不能**修正"策略是筛出来"的偏差 —— 那是 deflated_sharpe 的活。
    """
    x = values.dropna().to_numpy(dtype=float)
    n = x.size
    if n < 10:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "p_value": float("nan"), "n": n, "n_boot": 0}
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    p_value = float((draws <= 0).mean()) if x.mean() > 0 else float((draws >= 0).mean())
    return {
        "mean": float(x.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": p_value,
        "n": int(n),
        "n_boot": int(n_boot),
    }


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int, periods_per_year: int = 252) -> dict:
    """Deflated Sharpe Ratio（Bailey & López de Prado, 2014）。

    我们在 K 个变体里挑了最好的那个 —— 挑出来的 Sharpe 天生偏高。DSR 把
    "试了多少次"折算成对 Sharpe 的期望最大值修正，再给出真实 Sharpe > 0 的概率。

    返回 p_value 的含义：在考虑 n_trials 次筛选后，真实 Sharpe 仍大于 0 的概率。
    """
    r = returns.dropna().to_numpy(dtype=float)
    n = r.size
    if n < 20 or n_trials < 1:
        return {"sharpe_annual": float("nan"), "sr0": float("nan"), "p_value": float("nan"), "n_trials": n_trials}

    std = r.std(ddof=1)
    if std == 0:
        return {"sharpe_annual": float("nan"), "sr0": float("nan"), "p_value": float("nan"), "n_trials": n_trials}

    sr = r.mean() / std                                  # 单期 Sharpe
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    euler = 0.5772156649
    z = stats.norm.ppf(1 - 1 / max(n_trials, 2)) if n_trials > 1 else 0.0
    z2 = stats.norm.ppf(1 - 1 / (max(n_trials, 2) * np.e)) if n_trials > 1 else 0.0
    expected_max_sr = (1 - euler) * z + euler * z2       # 多次试验下 Sharpe 的期望最大值
    sr_variance = (1 - skew * sr + (kurt - 1) / 4 * sr**2) / (n - 1)
    sr0 = np.sqrt(max(sr_variance, 0)) * expected_max_sr
    denom = np.sqrt(max(sr_variance, 1e-18))
    dsr = float(stats.norm.cdf((sr - sr0) / denom))
    return {
        "sharpe_annual": float(sr * np.sqrt(periods_per_year)),
        "sr0_annual": float(sr0 * np.sqrt(periods_per_year)),
        "p_value": dsr,
        "n_trials": int(n_trials),
        "skew": skew,
        "kurtosis": kurt,
    }


def summarize_ic(ic_series: pd.Series, overlap_k: int = 1) -> dict:
    """把 IC 序列的常用统计一次性算齐，报告与简历都从这里取数，避免两处各算一遍。"""
    return {
        "days": int(len(ic_series)),
        "rank_ic_mean": float(ic_series.mean()) if len(ic_series) else float("nan"),
        "rank_ic_std": float(ic_series.std(ddof=1)) if len(ic_series) > 1 else float("nan"),
        "ic_ir": ic_ir(ic_series),
        "ic_tstat": ic_tstat(ic_series),
        "nw_tstat": newey_west_tstat(ic_series, lags=max(1, int(overlap_k))),
        "ic_positive_ratio": float((ic_series > 0).mean()) if len(ic_series) else float("nan"),
    }