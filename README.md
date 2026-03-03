# Nigeria PMI & NGX Equity Returns — Analysis Code

**`analysis_v3_final.py`**  
Author: Toba Kudehinbu  
Version: V3 (Final) — February 2026  
Companion to: *The PMI Contrarian Signal: Can the Purchasing Managers' Index Predict Nigerian Equity Returns?* (March 2026)

---

## Overview

This script runs the complete statistical analysis examining whether the Stanbic IBTC Nigeria Purchasing Managers' Index (PMI) predicts GDP growth and NGX equity returns across 3, 6, 9, and 12-month horizons.

All statistical functions — OLS regression, matrix inversion, Newey-West HAC standard errors, normal CDF — are implemented from scratch using only Python's standard library. No `statsmodels`, `scipy`, or `sklearn` dependencies.

Data isn't included. You'll have to source that on your own and place in a data folder in your repo.

---

## Requirements

- Python 3.7+
- No third-party packages required
- Standard library only: `csv`, `math`, `datetime`, `collections`

---

## Data Requirements

The script expects 11 CSV files, each with a `Date` column in `YYYY-MM-DD` format and one value column. Two directories are used:

**`/data/` (primary data)**

| File | Value Column | Description | Source |
|---|---|---|---|
| `pmi_nsa.csv` | `PMI_NSA` | Stanbic IBTC Nigeria PMI (Non-Seasonally Adjusted) | Bloomberg |
| `gdp.csv` | `GDP_YoY` | Nigeria Real GDP YoY (%, quarterly) | Bloomberg / NBS |
| `ngx_asi.csv` | `ASI` | NGX All-Share Index (price level) | Bloomberg |
| `ngx30.csv` | `NGX30` | NGX 30 Index (price level) | Bloomberg |
| `ngx_banking.csv` | `Banking` | NGX Banking Index (price level) | Bloomberg |
| `ngx_consgoods.csv` | `ConsGoods` | NGX Consumer Goods Index (price level) | Bloomberg |
| `cpi.csv` | `CPI_YoY` | Nigeria CPI All Items YoY (%) | Bloomberg / NBS |
| `brent.csv` | `Brent` | Brent Crude Futures (USD/bbl) | Bloomberg |
| `nafex.csv` | `NAFEX` | FMDQ OTC NAFEX Index (NGN/USD), Apr 2017 onwards | Bloomberg |

**`/data/` (supplementary data)**

| File | Value Column | Description | Source |
|---|---|---|---|
| `mpr_actual.csv` | `MPR` | CBN Monetary Policy Rate (%) — 21 MPC rate changes | CBN MPC decisions |
| `parallel_fx.csv` | `ParallelRate` | Black market NGN/USD rate, Jan 2016 – Feb 2023 | Various |

> **Note on the spliced FX series:** The script automatically constructs a market-rate USD series by combining `parallel_fx.csv` (pre-April 2017) with `nafex.csv` (April 2017 onwards). The official pre-2016 rate of ₦197/$ was an artificial peg and is excluded from USD return calculations.

> **Note on `parallel_fx.csv`:** This file is optional. If missing, the script continues without it and the spliced FX series will be NAFEX-only from April 2017.

**Sample period:** January 2014 – January 2026 (145 months of PMI data)

---

## Usage

```bash
python3 analysis_v3_final.py
```

All results print to stdout. A backtest CSV is also written to `/backtest_final.csv`.

To save output to a file:

```bash
python3 analysis_v3_final.py > results.txt
```

---

## What the Script Does

The script runs 9 distinct analyses, printed as labelled sections A through I.

### A — Predictive Regressions (Nominal, Real, USD)
Univariate OLS regression of h-month-ahead ASI returns on contemporaneous PMI level, at horizons of 3, 6, 9, and 12 months. Runs separately for nominal returns, CPI-deflated real returns, and USD-denominated returns (via spliced FX). Reports beta coefficient, Newey-West standard error, t-statistic, p-value, and R².

### B — Sector-Level Regressions
Same predictive regression run separately on the NGX Banking Index and NGX Consumer Goods Index, for nominal and real returns at 6 and 12-month horizons.

### C — Regime Analysis
Compares average forward returns in PMI expansion months (PMI > 50) versus contraction months (PMI ≤ 50) using Welch's unequal-variance t-test. Covers ASI nominal, real, USD, Banking, and Consumer Goods at 6 and 12-month horizons.

### D — Crisis Exclusion Robustness
Re-estimates the core predictive regression under four sample configurations: full sample, excluding 2016 (all 12 months), excluding COVID (March–June 2020), and excluding both. Tests whether results are driven by crisis clustering.

### E — PMI Level vs Momentum (ΔPMI)
Tests whether the month-over-month change in PMI (ΔPMI) adds predictive power beyond the level. Runs univariate regressions for level only, delta only, and a bivariate specification at 6 and 12-month horizons.

### F — Out-of-Sample Test with Brent Benchmark
Trains all models on January 2014 – December 2019, then evaluates predictions on January 2020 – January 2026. Compares PMI model against three benchmarks: naive historical mean, Brent crude price level model, and Brent YoY change model. Reports RMSE, RMSE ratio vs naive, and directional accuracy.

### G — Backtest with Four Configurations
Constructs a rules-based contrarian strategy: invest in NGX ASI when PMI ≤ 50, otherwise hold cash at the CBN Monetary Policy Rate. Runs four versions comparing old vs new cash rate methodology and with/without a one-month execution lag. Reports nominal and real terminal values, maximum drawdown, and percentage of months in equities. Saves final backtest to CSV.

### H — PMI Threshold Optimisation
Tests PMI thresholds from 46 to 52, comparing average 12-month real returns above and below each threshold. Reports spread, observation count below threshold, t-statistic, and p-value.

### I — Structural Break Analysis
Splits the sample at January 2023 and estimates the predictive regression separately for the pre-reform era (2014–2022) and the reform era (2023–2026). Tests whether the June 2023 FX unification and subsequent structural changes altered the PMI-return relationship.

---

## V3 Corrections

This version incorporates three fixes applied following peer review of V2:

**FIX 1 — Newey-West finite-sample correction**  
The variance-covariance matrix is now multiplied by n/(n−k) before extracting standard errors. Standard econometrics packages (statsmodels, Stata, EViews) apply this correction by default. Without it, standard errors are slightly understated for small samples, overstating t-statistics by approximately 0.5–1% at n=145, k=2.

**FIX 2 — Actual CBN MPR with geometric compounding**  
Cash returns in the backtest previously used approximate proxy rates with simple monthly division (rate/12). V3 uses actual CBN Monetary Policy Rate data sourced from 21 MPC decisions (2014–2025) and converts to monthly geometric yields: (1 + annual_rate)^(1/12) − 1. This avoids a small but systematic understatement of cash returns, particularly at the elevated rates seen in 2023–2024.

**FIX 3 — Brent benchmark uses price level, not log return**  
The out-of-sample Brent benchmark previously used the 1-month log return of Brent crude, making it an apples-to-oranges comparison with the PMI level model. V3 uses the Brent price level as the predictor (consistent with PMI level), and separately tests Brent YoY change as an alternative.

---

## Methodology Notes

**OLS implementation:** All regressions use the normal equations solved via Gaussian elimination with partial pivoting. The `ols_regression()` function returns coefficients, residuals, R², observation count, and number of parameters.

**Newey-West HAC standard errors:** Implemented in `newey_west_se()`. The lag truncation parameter is set at 1.5 × the return horizon (e.g. 18 lags for 12-month returns), consistent with the convention used in the report. This is essential because overlapping monthly return windows (e.g. 12-month returns measured each month) induce mechanical serial correlation in regression residuals.

**Overlapping returns and R²:** The 12-month predictive regressions use overlapping return windows. The reported R² figures are mechanically inflated relative to what a non-overlapping design would produce. The Newey-West correction addresses serial correlation in standard errors, but does not correct R². Treat R² figures with appropriate scepticism — see Section 5.4 of the report.

**Real returns:** Deflated by subtracting average annualised CPI over the holding period: nominal return minus (average CPI × h/12). This is an approximation; a more precise deflation would use the cumulative price level over the period.

**Welch's t-test:** Used in regime analysis because expansion and contraction subsamples have materially different variances. Standard equal-variance t-tests would be inappropriate here.

**p-values:** Computed using a normal CDF approximation (Abramowitz & Stegun 26.2.17) for large samples (df > 30) and a correction factor for smaller samples. Stars denote: \* p<0.10, \*\* p<0.05, \*\*\* p<0.01.

---

## Output Files

| File | Location | Description |
|---|---|---|
| `backtest_final.csv` | root | Month-by-month backtest equity curve (nominal and real) for Strategy, Buy-and-Hold, and Cash, using Actual MPR + geometric yields + 1-month lag |

---

## Key Limitations

- **No external dependencies** means matrix operations and statistical functions are implemented manually. These have been validated against statsmodels outputs but have not been independently audited.
- **Small effective sample size:** At the 12-month horizon, approximately 12 non-overlapping observations exist per year, yielding roughly 12 truly independent data points over the full sample despite 145 monthly observations.
- **No transaction costs:** The backtest assumes costless execution at month-end closing prices. NGX bid-ask spreads and market impact would reduce returns in practice.
- **CPI discontinuity:** NBS rebased the CPI from a 2009 to a 2024 base year in January 2025. The Bloomberg CPI series reflects old methodology through December 2024 and new methodology from January 2025, creating a level discontinuity in the real return calculations.
- **Parallel FX measurement noise:** Pre-2017 FX data sourced from informal parallel market reporting carries measurement error that affects USD return calculations.

---

## Citation

If you use this code or the accompanying analysis, please cite:

Kudehinbu, T. (2026). *The PMI Contrarian Signal: Can the Purchasing Managers' Index Predict Nigerian Equity Returns?* Independent research report, March 2026.

---

## Disclaimer

This code and the accompanying report are for informational and research purposes only and do not constitute investment advice. Past performance is not indicative of future results. Data sourced from Bloomberg Terminal, S&P Global (Stanbic IBTC Nigeria PMI), NGX Group, NBS, CBN, and FMDQ. All analysis and conclusions are the author's own. Not affiliated with, sponsored by, or endorsed by any data provider.
