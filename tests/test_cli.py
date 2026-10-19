"""命令行入口测试：保证 README 里写的那几条命令真的能跑。"""

from __future__ import annotations

import pandas as pd

from ashare_factor_lab import cli


def test_parser_builds_all_subcommands():
    parser = cli.build_parser()
    for command, extra in (("demo", []), ("eval", ["--data", "/tmp/none"]), ("leakage-check", [])):
        args = parser.parse_args([command, *extra])
        assert callable(args.func)


def test_leakage_check_command_passes_on_clean_code(capsys):
    assert cli.main(["leakage-check", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "因果性自检" in out
    assert "无未来函数" in out


def test_demo_writes_report_and_daily_csv(tmp_path, capsys):
    out = tmp_path / "report.md"
    daily = tmp_path / "daily.csv"
    code = cli.main(
        [
            "demo",
            "--stocks", "30",
            "--days", "900",
            "--simulations", "10",
            "--min-list-days", "0",
            "--liquidity-min", "0",
            "--min-stocks", "10",
            "--out", str(out),
            "--save-daily", str(daily),
            "--quiet",
        ]
    )
    assert code == 0
    assert out.exists() and "## 1. 结论" in out.read_text(encoding="utf-8")

    saved = pd.read_csv(daily)
    assert {"date", "rank_ic", "top_ret", "bench_ret", "excess", "turnover"} <= set(saved.columns)
    assert len(saved) > 10
    assert "报告已写入" in capsys.readouterr().out


def test_eval_errors_on_missing_directory(capsys):
    try:
        cli.main(["eval", "--data", "/tmp/definitely-not-here-12345"])
    except SystemExit as exc:
        assert exc.code == 2
    else:                                     # pragma: no cover - 防御性分支
        raise AssertionError("数据目录不存在时应该以退出码 2 结束")
    assert "不存在" in capsys.readouterr().err


def test_eval_reads_a_real_directory(tmp_path):
    """用合成数据落成 CSV 目录，走一遍真实数据的读取路径（bars/ 与 bars_delist/）。"""
    from ashare_factor_lab.panel import synthesize

    panel = synthesize(n_stocks=25, n_days=420, seed=21, delist_ratio=0.0)
    bars = tmp_path / "bars"
    bars.mkdir()
    for code, group in panel.groupby("code"):
        group.to_csv(bars / f"{code}.csv", index=False)
    main_panel = panel[panel["code"] == "100000"].copy()
    main_panel["code"] = "900001"
    main_panel["is_delisted"] = 1
    (tmp_path / "bars_delist").mkdir()
    main_panel.to_csv(tmp_path / "bars_delist" / "900001.csv", index=False)

    code = cli.main(
        [
            "eval", "--data", str(tmp_path),
            "--train-days", "200", "--test-days", "40", "--step-days", "40",
            "--min-list-days", "0", "--liquidity-min", "0", "--min-stocks", "10",
            "--simulations", "10", "--quiet",
        ]
    )
    assert code == 0