"""ashare-factor-lab：把「日频横截面选股因子到底有没有用」这个问题做成可复现的实验。

模块划分（每个模块都能单独测，不依赖网络、不依赖真实数据）：

    panel.py      面板构造：真实 CSV / 合成数据两种来源
    universe.py   宇宙与可交易性：退市股、真实上市天数、流动性、涨跌停、ST 代理
    features.py   特征工程（严格的 trailing 窗口，禁止未来函数）
    labels.py     标签：t 日收盘出信号 → t+1 开盘买 → t+2 开盘卖（A 股 T+1 可执行口径）
    splits.py     walk-forward 切分 + embargo
    model.py      训练/预测 + 滚动评估主循环
    metrics.py    RankIC / IC_IR / Newey-West t / Bootstrap / Deflated Sharpe
    costs.py      交易成本模型（佣金 / 过户费 / 印花税 / 滑点）
    backtest.py   组合层面：净值、年化、波动、Sharpe、最大回撤、Calmar
    baselines.py  同宇宙、同过滤条件下的基准（等权、随机选股分布、动量）
    report.py     生成 Markdown 报告
    cli.py        命令行入口

设计原则：任何可能引入未来信息的计算都必须能通过 `tests/test_causality.py` 的扰动测试。
"""

__version__ = "1.0.0"