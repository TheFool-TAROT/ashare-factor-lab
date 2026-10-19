"""面板构造：真实 CSV 与合成数据两条来源，上层代码不需要区分两者。

真实数据格式（每只股票一个 CSV，列名与 akshare / 腾讯源一致）：
    日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 换手率
可选附带 `listing_dates.csv`（真实上市日期）与 `delisted.csv`（退市名单），
用于构造「含退市股」的宇宙 —— 这是消除生存偏差的关键，见 universe.py。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_COLS = ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "换手率"]

REQUIRED_COLS = {"code", "date", "open", "close", "high", "low", "volume", "amount"}


@dataclass
class PanelConfig:
    """面板构建参数，全部显式化，避免散落在各处的魔法数字。"""

    liquidity_min: float = 5e7        # 20 日均成交额下限（元）
    min_list_days: int = 250          # 上市天数下限（真的按上市日算，不是按数据起始行数）
    limit_main: float = 0.10          # 主板涨跌停幅度
    limit_growth: float = 0.20        # 创业板/科创板涨跌停幅度
    limit_st: float = 0.05            # ST 股涨跌停幅度
    st_lookback: int = 60             # ST 代理的观察窗口
    tick: float = 0.005               # 判定「开盘即涨/跌停」的容差（元）


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_bars(folder: str | Path, min_rows: int = 60) -> pd.DataFrame:
    """读取一个目录下每只股票一个 CSV 的行情，返回长表面板。"""
    frames = []
    for path in sorted(Path(folder).glob("*.csv")):
        df = pd.read_csv(path, dtype={"code": str})
        if len(df) >= min_rows:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=sorted(REQUIRED_COLS))
    return pd.concat(frames, ignore_index=True)


def normalize_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """把「中文列名的原始行情」规范成标准长表：code/date/open/close/high/low/volume/amount/turnover。"""
    if raw.empty:
        return pd.DataFrame(columns=sorted(REQUIRED_COLS))

    df = raw.copy()
    rename = {
        "日期": "date", "时间": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount", "换手率": "turnover",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "code" not in df.columns:
        raise ValueError("panel 缺少 code 列")

    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    df = _to_numeric(df, ["open", "close", "high", "low", "volume", "amount", "turnover"])

    for col in ("is_delisted",):
        if col not in df.columns:
            df[col] = 0
    if "turnover" not in df.columns:
        df["turnover"] = np.nan

    keep = ["code", "date", "open", "close", "high", "low", "volume", "amount", "turnover", "is_delisted"]
    keep += [c for c in ("listing_date",) if c in df.columns]
    df = df[keep].sort_values(["code", "date"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["code", "date"], keep="first")
    return df


def attach_listing_dates(panel: pd.DataFrame, listing_csv: str | Path | None = None) -> pd.DataFrame:
    """挂上真实上市日期，并据此计算 `days_listed`。

    原版实现用「数据起始行号」（groupby.cumcount）冒充上市天数，后果有三：
      1. 数据开始前就已上市的老股票，前 250 行被当成"次新"整片剔除（对应约 22% 的样本损失）；
      2. 换一个数据起始日重跑，样本集整体平移 → 结论随数据切片漂移；
      3. "剔除次新"这个说法在代码里根本不成立，它实际是"前 250 行预热"。
    这里改为优先消费真实上市日期，并把退化路径显式写进 `days_listed_source` 列以便审计。
    """
    df = panel.copy()

    if listing_csv is not None and Path(listing_csv).exists():
        lst = pd.read_csv(listing_csv, dtype=str)
        code_col = next((c for c in lst.columns if c in ("证券代码", "code", "股票代码")), None)
        date_col = next((c for c in lst.columns if "上市" in c and "日" in c), None)
        if code_col and date_col:
            lst = lst[[code_col, date_col]].rename(columns={code_col: "code", date_col: "listing_date"})
            lst["code"] = lst["code"].str.zfill(6)
            lst["listing_date"] = pd.to_datetime(lst["listing_date"], errors="coerce")
            df = df.drop(columns=["listing_date"], errors="ignore").merge(lst, on="code", how="left")

    if "listing_date" in df.columns:
        df["listing_date"] = pd.to_datetime(df["listing_date"], errors="coerce")
    else:
        df["listing_date"] = pd.NaT

    has_date = df["listing_date"].notna()
    # 自然日 → 交易日近似（×5/7），与逐日交易日计数的误差在 5 天以内，对 250 日门槛无实质影响
    from_list = ((df["date"] - df["listing_date"]).dt.days * 5 / 7).round()
    from_rows = df.groupby("code").cumcount()
    df["days_listed"] = np.where(has_date, from_list, from_rows)
    df["days_listed_source"] = np.where(has_date, "listing_date", "row_index_approx")
    return df


def _synthetic_ohlc(close: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    prev = np.concatenate([[close[0]], close[:-1]])
    open_ = prev * (1 + rng.normal(0, 0.004, size=close.size))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, size=close.size)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, size=close.size)))
    return open_, high, low


def synthesize(
    n_stocks: int = 160,
    n_days: int = 900,
    seed: int = 7,
    signal_strength: float = 0.02,
    delist_ratio: float = 0.12,
    new_listing_ratio: float = 0.10,
) -> pd.DataFrame:
    """生成带**已知弱信号**的合成面板，用于让 demo 与测试完全脱离网络。

    信号构造是刻意的「反向动量」：t+1 收益被 t 日为止的 5 日动量反向解释，
    系数 signal_strength 决定信噪比。这样做的目的不是"造一个赚钱策略"，
    而是验证整条流水线**能把它检出来** —— 检出已知信号是评估框架的最低要求。
    另外按 delist_ratio 让一部分股票中途退市（退市前呈下跌趋势），
    用来验证宇宙构造是否真的在防生存偏差。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    n_delist = int(n_stocks * delist_ratio)
    delist_codes = set(rng.choice(n_stocks, size=n_delist, replace=False).tolist())

    rows = []
    new_listing = set(rng.choice(n_stocks, size=int(n_stocks * new_listing_ratio), replace=False).tolist())
    for i in range(n_stocks):
        code = f"{100000 + i:06d}"
        # 真实上市日期：多数股票在样本开始前很久就上市；少数是样本期内上市的次新股
        if i in new_listing:
            listing_date = dates[int(rng.integers(int(n_days * 0.55), n_days - 30))]
        else:
            listing_date = dates[0] - pd.Timedelta(int(rng.integers(400, 4000)), "D")
        # 退市股：在随机一天之后停止交易，且末段收益持续为负（真实退市的典型形态）
        end = n_days
        if i in delist_codes:
            end = int(rng.integers(int(n_days * 0.45), n_days - 5))
        price = float(rng.uniform(6, 60))
        closes: list[float] = []
        for t in range(n_days):
            mom5 = closes[-1] / closes[-6] - 1 if len(closes) >= 6 else 0.0
            alpha = -signal_strength * mom5                      # 反向动量
            drift = -0.0012 if i in delist_codes and t > end - 60 else 0.0   # 退市前阴跌
            ret = alpha + drift + rng.normal(0, 0.022)
            price = max(1.0, price * (1 + ret))
            closes.append(price)

        close = np.array(closes[:end])
        open_, high, low = _synthetic_ohlc(close, rng)
        volume = rng.lognormal(mean=15.5, sigma=0.5, size=close.size)
        amount = volume * close
        turnover = rng.uniform(0.5, 8.0, size=close.size) / 100
        rows.append(
            pd.DataFrame(
                {
                    "code": code,
                    "date": dates[:end],
                    "open": open_,
                    "close": close,
                    "high": high,
                    "low": low,
                    "volume": volume,
                    "amount": amount,
                    "turnover": turnover,
                    "is_delisted": int(i in delist_codes),
                    "listing_date": listing_date,
                }
            )
        )

    return normalize_panel(pd.concat(rows, ignore_index=True))


def build_panel(
    data_dir: str | Path | None = None,
    *,
    synthetic: bool = False,
    listing_csv: str | Path | None = None,
    **synth_kwargs,
) -> pd.DataFrame:
    """统一入口：`synthetic=True` 走合成数据，否则从目录读真实 CSV（含 bars_delist）。"""
    if synthetic or data_dir is None:
        return attach_listing_dates(normalize_panel(synthesize(**synth_kwargs)))

    root = Path(data_dir)
    main = normalize_panel(load_bars(root / "bars"))
    main["is_delisted"] = 0
    delisted_dir = root / "bars_delist"
    delisted = normalize_panel(load_bars(delisted_dir)) if delisted_dir.exists() else pd.DataFrame()
    if not delisted.empty:
        delisted["is_delisted"] = 1
        main = pd.concat([main, delisted], ignore_index=True)
    return attach_listing_dates(main, listing_csv or root / "alt" / "listing_dates.csv")