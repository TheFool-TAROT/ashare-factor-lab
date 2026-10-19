# ashare-factor-lab — Leakage-Safe Factor Evaluation for A-Share Equities

**An honest negative result, plus an engineering harness that turns the three most common ways to fool yourself into automated checks.**

[![CI](https://github.com/TheFool-TAROT/ashare-factor-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/TheFool-TAROT/ashare-factor-lab/actions/workflows/ci.yml)
![tests](https://img.shields.io/badge/tests-58%20passed-brightgreen)
![coverage](https://img.shields.io/badge/coverage-94%25-brightgreen)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-blue)

---

## TL;DR

| Claim | Evidence (all out-of-sample) |
|---|---|
| **The signal is real** — not a backtest artefact | RankIC **1.69%**, IC_IR 0.096, 8/8 walk-forward folds share the sign, random-pick control at the 100th percentile |
| **But it cannot pay for itself** | Gross excess **0.094%/day** vs trading cost **0.206%/day** (10bp half-spread) — cost is **2.2×** the signal |
| **So the correct answer is "don't trade it"** | Annualised net excess **−24.5%**; same-universe equal-weight benchmark ends at 1.821 vs the strategy's 0.896 |

> The valuable part of this project is not "AI predicts stock prices". It is a harness that can **prove it doesn't**:
> look-ahead bias, survivorship bias and cost denial are all made testable.
> Roughly 90% of wrong factor-research conclusions die at those three points — not from weak models.

**One command reproduces every number below** (no market data required):

```bash
git clone https://github.com/TheFool-TAROT/ashare-factor-lab && cd ashare-factor-lab
pip install -e ".[dev]"
make demo     # full pipeline on synthetic data, ~3 min
make test     # 58 tests, including the causality test that mutates future data
```

---

## What this repo actually solves

| Failure mode | Symptom | Countermeasure in this repo |
|---|---|---|
| **Look-ahead bias** | Features leak post-`t` information; backtest is beautiful, live trading is flat | Only trailing windows; a perturbation test rewrites *all* future prices and asserts row `t`'s features are bit-identical (`tests/test_causality.py`) |
| **Survivorship bias** | Universe is built from *today's* index members, delisted names silently vanish, IC roughly doubles | Universe carries explicit delisting flags and true listing dates; same features, only the universe toggles, so the effect is measured |
| **Cost denial** | Only gross excess is reported; daily churn and slippage are ignored | Cost model itemised per A-share rules (commission / transfer fee / stamp duty sell-side only / 5-10-20bp half-spread) plus break-even turnover |

Plus the two things usually skipped: **statistical significance** (Newey-West t, bootstrap CI, deflated Sharpe) and **fair baselines** (same universe, same filter, same test days, plus a random-pick null distribution).

---

## Method (conventions first — otherwise the numbers mean nothing)

| Item | Setting | Why it must be this |
|---|---|---|
| Label | Signal at close of `t` → **buy at open of `t+1` → sell at open of `t+1+h`** (h=1 ⇒ `t+2`) | A-shares are T+1; buying at the close of `t` is not executable |
| `t+1`/`t+2` check | Must be the **next trading day on the market calendar**, not merely "the next row" | A halted stock's next row may be days later — that open is not tradeable |
| Splits | Walk-forward: 750 train / 60 test / 60 step / **2-day embargo** | The label spans 2 trading days; an unembargoed train tail leaks label information (invisible when reviewing feature columns) |
| Tradability | Liquidity + true listing age + ST proxy + calendar continuity + cannot buy at limit-up on `t+1` + cannot sell at limit-down on the exit day | Rows with signal but no executable price are excluded — otherwise the backtest is paper profit |
| Costs | 2.5bp commission/side + 0.1bp transfer/side + 5bp stamp duty (**sell only**) + 5/10/20bp half-spread | Stamp duty is sell-side only; without order-book data, slippage can only be a sensitivity study |
| Cross-sectional standardisation | Grouped **by date**; market features use expanding windows | Whole-sample standardisation leaks the future cross-section into every row — the most insidious class of leakage |
| Metrics | RankIC / IC_IR / plain t / **Newey-West t** / bootstrap CI / deflated Sharpe / equity, drawdown, Sharpe, Calmar | Overlapping daily labels inflate plain t-statistics; picking the best of many variants is multiple testing and must be corrected |

---

## Results

### Real data (CSI 500 + names delisted inside the window)

698 stocks (500 current constituents + 198 delisted) · 803,540 daily rows · 2021-01 → 2026-09. Prices are not shipped. **All figures out-of-sample.**

| Variant | RankIC | IC_IR | Gross excess/day | Turnover/day | Net equity |
|---|---|---|---|---|---|
| A old features, old conventions | 1.355% | 0.090 | +0.0776% | 71.2% | 0.757 |
| B old features, strict tradability | 1.105% | — | +0.0686% | — | 0.772 |
| **C new features, strict conventions** | **1.688%** | **0.096** | **+0.0942%** | **68.3%** | **0.896** |
| Universe equal-weight (benchmark) | — | — | — | — | **1.821** |

| Half-spread assumption | Cost per full turnover | Cost/day | Annualised net excess |
|---|---|---|---|
| 5bp | 0.202% | 0.138% | **−10.4%** |
| 10bp | 0.302% | 0.206% | **−24.5%** |
| 20bp | 0.502% | 0.343% | **−46.4%** |

68% daily turnover × 0.2–0.5% per turnover = 0.14–0.34% per day of cost against 0.094% per day of gross signal. **Cost is 1.5–3.6× the signal, negative under every assumption tested.**

| Universe construction | RankIC |
|---|---|
| Current CSI 500 constituents only (common shortcut) | **3.25%** |
| Including the 198 names delisted inside the window | **1.36%** |

Same features, same filters — the only difference is whether dead stocks are allowed to speak. **Half of that IC was survivorship bias.**

### Synthetic demo (runs in CI on every push)

The demo plants a known weak reversal signal (tunable signal-to-noise) to prove the harness *detects it* — that is the minimum bar for any evaluation framework. Default settings (160 stocks × 1100 days):

| Metric | Value |
|---|---|
| Out-of-sample days / folds | 480 / 8 (8/8 folds rank-IC positive) |
| RankIC | 2.636% (plain t 6.06 / **Newey-West t 6.02**) |
| Top-decile gross excess | 0.0797%/day, bootstrap 95% CI [0.0217%, 0.1395%], p=0.003 |
| Turnover / cost per day | 69.1% / 0.2053% |
| **Net excess/day** | **−0.1256%** (annualised −27.1%, Sharpe −2.95) |
| Random-pick control | random mean 0.0004%; strategy at the 100th percentile, p<0.001 |

The synthetic run reproduces the *shape* of the real finding exactly: a detectable, significant signal that daily churn destroys. Full report: [`docs/demo_report.md`](docs/demo_report.md).

---

## Audit: 11 defects found and fixed in my own earlier implementation

An earlier version of this study (Windows scripts, manual data staging) did produce "conclusions". I audited it line by line with an independent process, found 11 defects, and handled every one of them here. **This section is the strongest evidence of capability in the repo: being able to attack your own results beats ten more features.**

| # | Defect | Consequence | Fix |
|---|---|---|---|
| 1 | `days_listed` used `groupby.cumcount()`, and the scraped `上市日期` column was never consumed | The first 250 rows of every stock listed before the sample start were dropped as "new listings" (174k rows / 22%); **re-running from a different start date shifted the whole sample — conclusions drifted with the data slice** | `attach_listing_dates()` consumes true listing dates; the fallback path is recorded in a `days_listed_source` column for auditing; 3 dedicated tests |
| 2 | Universe = today's constituents + all delisted names | Two opposite biases mixed; matches no investable portfolio | Universe is defined by panel-carried `is_delisted` / `listing_date`; the residual bias needs point-in-time constituents and is declared in Limitations |
| 3 | Sample screened with `fwd_ret.notna()` | Rows with no observable future quote (delisting / long halt) were deleted rather than booked as losses — **screening by future availability**, inflating returns and hiding tail risk | `fwd_missing_reason` classifies `panel_end` / `halted` / `no_quote_after` and reports the counts separately |
| 4 | Main result reported point estimates only; IC_IR uncorrected for overlap | 2-day labels ⇒ autocorrelated daily IC ⇒ inflated significance | Plain t and Newey-West t reported side by side; a test asserts NW < plain on overlapped series |
| 5 | Limit prices computed from **back-adjusted** prices, with banker's rounding | Wrong limit-up/down calls on ex-dividend days and at 1-cent boundaries | `round_half_up()` (the exchange rounds half up, not half to even) + documented requirement to use unadjusted prices |
| 6 | Candidate list filtered 3 conditions, the backtest 6 | Reported candidate counts still included names that could not be bought | A single `tradable` definition used by report and backtest alike |
| 7 | Holding-period experiments validated only the `t+2` exit | With K=20 the actual exit day's limit-down was never checked — overstated net returns | `holding_days=h` places the limit-down check on the real exit day `t+1+h`; the embargo requirement follows as `h+1` |
| 8 | "t = 1.5–2.0" hard-coded in six files' prose | Numbers decoupled from code; fix one, forget the other | Metrics have exactly one producer (`metrics.py` + `pipeline.py`); reports are generated by `report.py` — no hand-copied numbers |
| 9 | No dependency pinning / no CI / no tests / 69 hard-coded absolute paths / import-time file writes | Unrunnable on another machine; nothing was provable | `pyproject.toml` + GitHub Actions + 58 tests at 94% coverage + zero absolute paths + no module-level side effects |
| 10 | `has_t1`/`has_t2` only checked "a next row exists" | Samples crossing a trading halt were treated as executable (~0.07%, biased optimistic) | Strict market-calendar check; tests construct a halted stock and record the trap that *a single stock's date set is not a calendar* |
| 11 | Three duplicate feature implementations, self-check compared only a subset of columns | Training and daily scoring conventions drift apart silently | Single `build_labeled_panel()` entry point; an AST-level check forbids `shift(-n)` and `rolling(center=True)` outside `labels.py` |

---

## Repository layout

```
src/ashare_factor_lab/
├── panel.py      panel construction: real CSVs / synthetic (listing dates, delisting flags)
├── features.py   30 trailing-window features
├── labels.py     executable T+1 labels, tradability, missing-reason grading
├── splits.py     walk-forward + embargo (+ assert_no_leakage self-check)
├── model.py      GBDT training and the rolling evaluation loop
├── metrics.py    RankIC / IC_IR / Newey-West t / bootstrap / deflated Sharpe
├── costs.py      cost model, turnover, break-even turnover
├── backtest.py   equity, annualised return, vol, Sharpe, max drawdown, Calmar
├── baselines.py  same-universe benchmark, random-pick distribution, single-factor baseline
├── pipeline.py   end-to-end pipeline (shared by CLI and tests)
├── report.py     Markdown report rendering
└── cli.py        afl demo / eval / leakage-check
tests/            58 tests, 94% coverage
docs/demo_report.md   checked-in sample report
```

---

## Limitations

1. **Point-in-time constituents are unavailable**: delisted names are included, but historical index membership needs a paid feed; the universe remains "current members + delisted".
2. **ST is a proxy**: inferred from 5% price-limit behaviour, so it has false positives and negatives.
3. **Slippage is an assumption**: no order-book data; 5/10/20bp is a sensitivity study, not a reproduction.
4. **No industry/size neutralisation**: part of the return may be a size or sector bet.
5. **No queueing or market-impact modelling**: open-price fills are optimistic in extreme conditions.
6. **Synthetic data is not the market**: the demo's signal is planted; it validates the harness, not the strategy.
7. **No portfolio optimisation**: equal-weight decile, no risk model, no position sizing. Those would change the magnitude, not the sign, of "cost > signal".

---

## Running it on real data

```bash
# data/
# ├── bars/{code}.csv          constituents
# ├── bars_delist/{code}.csv   delisted inside the window
# └── alt/listing_dates.csv    listing dates (strongly recommended)

make report DATA_DIR=data LISTING_CSV=data/alt/listing_dates.csv
afl leakage-check                                    # standalone look-ahead self-check
afl eval --data data --holding-days 10 --half-spread 5
```

## License

MIT © [TheFool-TAROT](https://github.com/TheFool-TAROT)

> Research and engineering only — not investment advice. Every figure here is out-of-sample, and **the headline result is negative**: under the documented conventions and cost assumptions, this daily-frequency strategy does not work.