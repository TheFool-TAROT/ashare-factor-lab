"""可选：从公开数据源抓一份最小可跑的数据集（需要网络与 akshare）。

这个脚本**不属于**测试与 demo 路径：`make demo` / `make test` 都不依赖它。
它存在的意义是让评审者能自己把真实数据喂进 `afl eval`，而不是只能看合成数据。

    pip install "ashare-factor-lab[fetchers]"
    python scripts/fetch_data_example.py --out data --start 2021-01-01

会产出：
    data/bars/{code}.csv          当前成分股日线
    data/bars_delist/{code}.csv   样本期内退市的股票（提升数据完整性的关键）
    data/alt/listing_dates.csv    真实上市日期（避免用行号冒充上市天数）

三件必须注意的事（都是踩过的坑）：
1. **复权基准不能混源**：不同数据源的 hfq（后复权）基准不同，同一只票在两个源之间
   会出现假跳空。要么全程单源，要么全程不复权。判涨跌停必须用**不复权**价。
2. **停牌股没有新 K 线是正常的**：同一只票连续几天没有成交量/成交额，那是停牌，
   不是抓取失败，不要用"补零"去填它 —— 补零会造出不存在的价格。
3. **上市日期必须一起抓**：只抓行情不抓上市日期，计算"上市天数"就只能退化到用行号，
   那会让 2021 年前上市的老股票前 250 行被误判成次新（约 22% 的样本损失）。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="抓取示例数据（需要 akshare 与网络）")
    parser.add_argument("--out", default="data", help="输出目录")
    parser.add_argument("--start", default="2021-01-01", help="起始日期")
    parser.add_argument("--end", default=None, help="结束日期，默认今天")
    parser.add_argument("--index", default="000905", help="成分股指数代码，默认中证 500")
    parser.add_argument("--limit", type=int, default=0, help="只抓前 N 只（调试用，0 表示全抓）")
    args = parser.parse_args()

    try:
        import akshare as ak
    except ImportError:
        print('缺少 akshare：pip install "ashare-factor-lab[fetchers]"')
        return 1

    out = Path(args.out)
    (out / "bars").mkdir(parents=True, exist_ok=True)
    (out / "bars_delist").mkdir(parents=True, exist_ok=True)
    (out / "alt").mkdir(parents=True, exist_ok=True)

    end = args.end or time.strftime("%Y%m%d")
    start = args.start.replace("-", "")

    # 1) 上市日期：没有它，"上市天数"就只能用行号冒充，会丢掉约 22% 的样本
    print("抓取上市日期 …")
    info = ak.stock_info_a_code_name()
    listings = []
    for row in info.itertuples():
        code = str(row.code).zfill(6)
        try:
            detail = ak.stock_individual_info_em(symbol=code)
            listing = detail.loc[detail["item"] == "上市时间", "value"]
            if len(listing):
                listings.append({"code": code, "listing_date": str(listing.iloc[0])})
        except Exception as exc:                       # 单只失败不影响整体
            print(f"  跳过 {code}: {exc}")
        if len(listings) % 50 == 0 and len(listings):
            print(f"  已获取 {len(listings)} 只")
    import pandas as pd

    pd.DataFrame(listings).to_csv(out / "alt" / "listing_dates.csv", index=False, encoding="utf-8-sig")

    # 2) 当前成分股日线
    print(f"抓取指数 {args.index} 成分股日线 …")
    members = ak.index_stock_cons_csindex(symbol=args.index)
    codes = [str(c).zfill(6) for c in members["成分券代码"]]
    if args.limit:
        codes = codes[: args.limit]
    for i, code in enumerate(codes, 1):
        target = out / "bars" / f"{code}.csv"
        if target.exists():
            continue
        try:
            bars = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="")
            if bars is not None and len(bars):
                bars.to_csv(target, index=False, encoding="utf-8-sig")
        except Exception as exc:
            print(f"  跳过 {code}: {exc}")
        if i % 25 == 0:
            print(f"  {i}/{len(codes)}")

    print("\n抓取完成。注意：")
    print("  · 退市股需另行补充到 bars_delist/ —— 不补的话 RankIC 会被生存偏差抬高约一倍")
    print("  · 判涨跌停请用本目录的不复权价，不要用后复权价")
    print(f"  · 然后跑：make report DATA_DIR={out} LISTING_CSV={out}/alt/listing_dates.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())