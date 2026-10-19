# Makefile：所有常用动作都有一条短命令，评审者不需要读文档也能跑起来。
# 约定：`make demo` 绝不依赖网络与真实行情数据。

PY ?= python3
export PYTHONPATH := src
PKG := ashare_factor_lab

.PHONY: help install demo demo-quick test lint fmt cov leakage report clean

help:
	@echo "make demo      - 合成数据跑完整流水线（约 3 分钟，无需网络）"
	@echo "make test      - 跑测试套件（含因果性扰动测试）"
	@echo "make cov       - 测试 + 覆盖率报告"
	@echo "make lint      - ruff 静态检查"
	@echo "make leakage   - 只跑"无未来函数"自检"
	@echo "make report    - 用真实数据生成 Markdown 报告（需 DATA_DIR）"
	@echo "make clean     - 清掉缓存与产物"

install:
	$(PY) -m pip install -e ".[dev]"

demo:
	$(PY) -m $(PKG).cli demo --out output/demo_report.md --save-daily output/demo_daily.csv

demo-quick:
	$(PY) -m $(PKG).cli demo --stocks 60 --days 900 --simulations 50 --min-list-days 0 --liquidity-min 0 --min-stocks 10 \
		--out output/demo_quick_report.md

test:
	$(PY) -m pytest

cov:
	$(PY) -m pytest --cov=$(PKG) --cov-report=term-missing

lint:
	ruff check .

fmt:
	ruff check --fix .

leakage:
	$(PY) -m $(PKG).cli leakage-check

report:
	@test -n "$(DATA_DIR)" || (echo "用法: make report DATA_DIR=path/to/data [LISTING_CSV=path/to/listing_dates.csv]"; exit 1)
	$(PY) -m $(PKG).cli eval --data "$(DATA_DIR)" $(if $(LISTING_CSV),--listing-csv "$(LISTING_CSV)",) \
		--out output/real_data_report.md --save-daily output/real_data_daily.csv

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	rm -rf output/