#!/usr/bin/env python3
"""
=============================================================================
Nigerian PMI as a Leading Indicator for GDP Growth and Equity Returns
Final Analysis (V3) — February 2026
Author: Toba Kudehinbu
=============================================================================

PURPOSE:
    This script runs the complete statistical analysis examining whether the
    Stanbic IBTC Nigeria PMI predicts GDP growth and equity returns (NGX ASI,
    NGX 30, NGX Banking, NGX Consumer Goods) across 3, 6, 9, and 12-month
    horizons.

ANALYSES:
    1. CPI-deflated real returns
    2. USD-denominated returns (via spliced FX: parallel + NAFEX)
    3. Crisis-exclusion robustness (2016 recession, COVID-19)
    4. Out-of-sample predictability (train: 2014-2019, test: 2020-2026)
    5. Contrarian PMI backtest with equity curve
    6. Sector-level analysis (Banking vs Consumer Goods)
    7. PMI threshold optimisation (46-52)
    8. Structural break analysis (pre-reform vs reform era)
    9. Delta PMI (level vs momentum)
   10. Brent crude level benchmark for OOS comparison

V3 CORRECTIONS (from peer review):
    FIX 1: Newey-West finite-sample correction n/(n-k) applied to
           variance-covariance matrix before extracting standard errors.
    FIX 2: Cash returns in backtest use (a) actual CBN MPR data (21 MPC
           rate changes, sourced from cbn.gov.ng) and (b) geometric monthly
           compounding: (1 + annual_rate)^(1/12) - 1, not rate/12.
    FIX 3: Brent OOS benchmark uses price LEVEL (apples-to-apples with PMI
           level), not 1-month log return. Also tests Brent YoY change.

DATA SOURCES:
    All data sourced from Bloomberg Terminal unless otherwise noted.
    - PMI: Stanbic IBTC Nigeria PMI (NSA), Jan 2014 - Jan 2026
    - GDP: Nigeria Real GDP YoY (quarterly, NBS), Q1 2014 - Q3 2025
    - ASI: NGX All-Share Index, Jan 2014 - Jan 2026
    - NGX30: NGX 30 Index, Jan 2014 - Jan 2026
    - Banking: NGX Banking Index, Jan 2014 - Jan 2026
    - ConsGoods: NGX Consumer Goods Index, Jan 2014 - Jan 2026
    - CPI: Nigeria CPI All Items YoY (%), Jan 2014 - Jan 2026
    - Brent: Brent Crude Futures (USD/bbl), Jan 2014 - Jan 2026
    - NAFEX: FMDQ OTC NAFEX Index (NGN/USD), Apr 2017 - Jan 2026
    - Parallel FX: Black market NGN/USD rate, Jan 2016 - Feb 2023
    - MPR: CBN Monetary Policy Rate (from MPC decisions), Jan 2014 - Jan 2026

METHODOLOGY:
    - OLS regression with Newey-West (1987) HAC standard errors
    - Finite-sample correction factor n/(n-k) applied to NW covariance
    - All statistical functions implemented from scratch (no statsmodels)

USAGE:
    python3 analysis_v3_final.py

    Assumes CSV files are present in the paths specified below.
    Outputs all results to stdout.

=============================================================================
"""

import csv
import math
from datetime import datetime
from collections import OrderedDict


# =============================================================================
# SECTION 1: DATA LOADING
# =============================================================================

def load_csv(path, val_col):
    """
    Load a two-column CSV into an OrderedDict keyed by (year, month).
    """
    data = OrderedDict()
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt = datetime.strptime(row['Date'], '%Y-%m-%d')
            data[(dt.year, dt.month)] = float(row[val_col])
    return data


DATA_DIR = 'data'
DATA_DIR2 = 'data'

pmi       = load_csv(f'{DATA_DIR}/pmi_nsa.csv',      'PMI_NSA')
gdp_data  = load_csv(f'{DATA_DIR}/gdp.csv',          'GDP_YoY')
asi       = load_csv(f'{DATA_DIR}/ngx_asi.csv',       'ASI')
ngx30     = load_csv(f'{DATA_DIR}/ngx30.csv',         'NGX30')
cpi       = load_csv(f'{DATA_DIR}/cpi.csv',           'CPI_YoY')
brent     = load_csv(f'{DATA_DIR}/brent.csv',         'Brent')
banking   = load_csv(f'{DATA_DIR}/ngx_banking.csv',   'Banking')
consgoods = load_csv(f'{DATA_DIR}/ngx_consgoods.csv', 'ConsGoods')
nafex     = load_csv(f'{DATA_DIR}/nafex.csv',         'NAFEX')

# V3: Actual MPR from CBN MPC decisions
mpr_data  = load_csv(f'{DATA_DIR2}/mpr_actual.csv',   'MPR')

# Parallel FX for spliced USD series
try:
    parallel_fx = load_csv(f'{DATA_DIR2}/parallel_fx.csv', 'ParallelRate')
except:
    parallel_fx = OrderedDict()

months = sorted(pmi.keys())
print(f"Loaded {len(months)} months of PMI data: {months[0]} to {months[-1]}")


# =============================================================================
# SECTION 2: TIME-SERIES UTILITY FUNCTIONS
# =============================================================================

def shift_month(ym, k):
    """Shift a (year, month) tuple forward by k months."""
    y, m = ym
    total = (y * 12 + m - 1) + k
    return (total // 12, total % 12 + 1)


def forward_returns(prices, months, horizons=[3, 6, 9, 12]):
    """Compute h-month-ahead NOMINAL percentage returns."""
    results = {}
    for h in horizons:
        rets = {}
        for ym in months:
            future = shift_month(ym, h)
            if ym in prices and future in prices:
                rets[ym] = (prices[future] / prices[ym] - 1) * 100
        results[h] = rets
    return results


def forward_real_returns(prices, cpi_data, months, horizons=[3, 6, 9, 12]):
    """
    Compute h-month-ahead CPI-DEFLATED REAL percentage returns.
    Real return = nominal return - average annualised CPI * (h/12).
    """
    results = {}
    for h in horizons:
        rets = {}
        for ym in months:
            future = shift_month(ym, h)
            if ym in prices and future in prices:
                nom = (prices[future] / prices[ym] - 1) * 100
                cpi_vals = []
                for k in range(h + 1):
                    mk = shift_month(ym, k)
                    if mk in cpi_data:
                        cpi_vals.append(cpi_data[mk])
                if len(cpi_vals) >= max(1, h // 2):
                    avg_cpi = sum(cpi_vals) / len(cpi_vals)
                    rets[ym] = nom - avg_cpi * h / 12
        results[h] = rets
    return results


def forward_usd_returns(prices_ngn, fx_rate, months, horizons=[3, 6, 9, 12]):
    """Compute h-month-ahead USD-denominated percentage returns."""
    results = {}
    for h in horizons:
        rets = {}
        for ym in months:
            future = shift_month(ym, h)
            if (ym in prices_ngn and future in prices_ngn and
                ym in fx_rate and future in fx_rate):
                usd_now = prices_ngn[ym] / fx_rate[ym]
                usd_future = prices_ngn[future] / fx_rate[future]
                rets[ym] = (usd_future / usd_now - 1) * 100
        results[h] = rets
    return results


# =============================================================================
# SECTION 3: STATISTICAL FUNCTIONS
# =============================================================================

def ols_regression(y, X):
    """
    OLS regression via normal equations with Gaussian elimination.
    Returns (coefficients, residuals, R-squared, n, k) or (None,...) if singular.
    """
    n = len(y)
    k = len(X[0])
    XtX = [[0.0] * k for _ in range(k)]
    for i in range(n):
        for j in range(k):
            for l in range(k):
                XtX[j][l] += X[i][j] * X[i][l]
    Xty = [0.0] * k
    for i in range(n):
        for j in range(k):
            Xty[j] += X[i][j] * y[i]
    aug = [XtX[i][:] + [Xty[i]] for i in range(k)]
    for col in range(k):
        max_row = col
        for row in range(col + 1, k):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]
        if abs(aug[col][col]) < 1e-12:
            return None, None, None, n, k
        for row in range(k):
            if row != col:
                factor = aug[row][col] / aug[col][col]
                for j in range(k + 1):
                    aug[row][j] -= factor * aug[col][j]
    coeffs = [aug[i][k] / aug[i][i] for i in range(k)]
    y_hat = [sum(X[i][j] * coeffs[j] for j in range(k)) for i in range(n)]
    residuals = [y[i] - y_hat[i] for i in range(n)]
    y_mean = sum(y) / n
    ss_res = sum(r ** 2 for r in residuals)
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return coeffs, residuals, R2, n, k


def invert_matrix(M, k):
    """Invert a k x k matrix via Gauss-Jordan elimination."""
    aug = [M[i][:] + [1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
    for col in range(k):
        max_row = col
        for row in range(col + 1, k):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]
        if abs(aug[col][col]) < 1e-15:
            return None
        pivot = aug[col][col]
        for j in range(2 * k):
            aug[col][j] /= pivot
        for row in range(k):
            if row != col:
                factor = aug[row][col]
                for j in range(2 * k):
                    aug[row][j] -= factor * aug[col][j]
    return [[aug[i][k + j] for j in range(k)] for i in range(k)]


def mat_mult(A, B, k):
    """Multiply two k x k matrices."""
    C = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            for l in range(k):
                C[i][j] += A[i][l] * B[l][j]
    return C


def newey_west_se(X, residuals, n, k, max_lag=None):
    """
    Newey-West (1987) HAC standard errors.

    V3 FIX 1: Includes finite-sample correction n/(n-k).
    Standard econometrics packages (statsmodels, Stata, EViews) apply
    this correction. Without it, SEs are ~0.5-1% too small for n=145, k=2.
    """
    if max_lag is None:
        max_lag = max(int(4 * (n / 100) ** (2 / 9)), 1)

    XtX = [[0.0] * k for _ in range(k)]
    for i in range(n):
        for j in range(k):
            for l in range(k):
                XtX[j][l] += X[i][j] * X[i][l]
    inv_XtX = invert_matrix(XtX, k)
    if inv_XtX is None:
        return [float('nan')] * k

    scores = [[X[i][j] * residuals[i] for j in range(k)] for i in range(n)]
    S = [[0.0] * k for _ in range(k)]
    for i in range(n):
        for j in range(k):
            for l in range(k):
                S[j][l] += scores[i][j] * scores[i][l]
    for lag in range(1, max_lag + 1):
        weight = 1 - lag / (max_lag + 1)
        for i in range(lag, n):
            for j in range(k):
                for l in range(k):
                    val = scores[i][j] * scores[i - lag][l]
                    S[j][l] += weight * val
                    S[l][j] += weight * val
    temp = mat_mult(S, inv_XtX, k)
    V = mat_mult(inv_XtX, temp, k)

    # FIX 1: Apply finite-sample correction n / (n - k)
    correction = n / (n - k) if n > k else 1.0
    for i in range(k):
        for j in range(k):
            V[i][j] *= correction

    se = [math.sqrt(max(V[j][j], 0)) for j in range(k)]
    return se


def norm_cdf(x):
    """Standard normal CDF (Abramowitz & Stegun 26.2.17)."""
    if x < -8: return 0.0
    if x > 8: return 1.0
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x / 2)
    return 0.5 * (1.0 + sign * y)


def t_to_p(t_stat, df):
    """Two-tailed p-value from t-statistic."""
    t = abs(t_stat)
    if df > 30:
        p = 2 * (1 - norm_cdf(t))
    else:
        p = 2 * (1 - norm_cdf(t * (1 - 1 / (4 * df))))
    return max(p, 1e-10)


def stars(p):
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''


# =============================================================================
# SECTION 4: COMPUTE ALL RETURN SERIES
# =============================================================================

print("\nComputing forward returns...")

# Spliced FX: parallel (pre-Apr 2017) + NAFEX (Apr 2017+)
spliced_fx = OrderedDict()
for ym in sorted(set(list(parallel_fx.keys()) + list(nafex.keys()))):
    if ym in nafex:
        spliced_fx[ym] = nafex[ym]
    elif ym in parallel_fx:
        spliced_fx[ym] = parallel_fx[ym]

# Nominal returns
asi_fwd       = forward_returns(asi, months)
ngx30_fwd     = forward_returns(ngx30, months)
banking_fwd   = forward_returns(banking, months)
consgoods_fwd = forward_returns(consgoods, months)

# Real (CPI-deflated)
asi_real_fwd       = forward_real_returns(asi, cpi, months)
banking_real_fwd   = forward_real_returns(banking, cpi, months)
consgoods_real_fwd = forward_real_returns(consgoods, cpi, months)

# USD (spliced FX)
asi_usd_fwd = forward_usd_returns(asi, spliced_fx, months)

# Delta PMI
delta_pmi = OrderedDict()
for i in range(1, len(months)):
    delta_pmi[months[i]] = pmi[months[i]] - pmi[months[i - 1]]

# Brent YoY (FIX 3: use level or YoY, not 1m log return)
brent_yoy = OrderedDict()
for ym in sorted(brent.keys()):
    prev = shift_month(ym, -12)
    if prev in brent:
        brent_yoy[ym] = (brent[ym] / brent[prev] - 1) * 100

print("  Done. Sample sizes at 12m horizon:")
print(f"    ASI nominal: {len(asi_fwd[12])}")
print(f"    ASI real:    {len(asi_real_fwd[12])}")
print(f"    ASI USD:     {len(asi_usd_fwd[12])}")


# =============================================================================
# SECTION 5: REGRESSION RUNNER
# =============================================================================

def run_regression(pmi_data, return_data, horizon, label="",
                   exclude_months=None, restrict_range=None):
    """
    Run univariate predictive regression:
        Return(t, t+h) = alpha + beta * PMI(t) + epsilon(t)
    with Newey-West HAC standard errors (including n/(n-k) correction).
    """
    y_vals, x_vals = [], []
    for ym in sorted(return_data.keys()):
        if ym not in pmi_data: continue
        if exclude_months and ym in exclude_months: continue
        if restrict_range:
            if ym < restrict_range[0] or ym > restrict_range[1]: continue
        y_vals.append(return_data[ym])
        x_vals.append(pmi_data[ym])
    n = len(y_vals)
    if n < 10:
        return {'n': n, 'valid': False, 'label': label}
    X = [[1.0, x] for x in x_vals]
    result = ols_regression(y_vals, X)
    if result[0] is None:
        return {'n': n, 'valid': False, 'label': label}
    coeffs, residuals, R2, n_obs, k = result
    nw_lag = max(int(horizon * 1.5), 1)
    se = newey_west_se(X, residuals, n, k, max_lag=nw_lag)
    t_stat = coeffs[1] / se[1] if se[1] > 0 else 0
    p_val = t_to_p(t_stat, n - k)
    return {
        'label': label, 'n': n, 'valid': True,
        'alpha': coeffs[0], 'beta': coeffs[1],
        'se_beta': se[1], 't_stat': t_stat, 'p_val': p_val,
        'R2': R2, 'horizon': horizon, 'nw_lag': nw_lag
    }


def regime_analysis(pmi_data, return_data, threshold=50.0, label=""):
    """Compare mean returns: PMI > threshold vs PMI <= threshold (Welch t-test)."""
    above, below = [], []
    for ym in sorted(return_data.keys()):
        if ym in pmi_data:
            if pmi_data[ym] > threshold:
                above.append(return_data[ym])
            else:
                below.append(return_data[ym])
    if len(below) < 3 or len(above) < 3:
        return None
    mean_a = sum(above) / len(above)
    mean_b = sum(below) / len(below)
    var_a = sum((x - mean_a) ** 2 for x in above) / (len(above) - 1)
    var_b = sum((x - mean_b) ** 2 for x in below) / (len(below) - 1)
    se_diff = math.sqrt(var_a / len(above) + var_b / len(below))
    t_stat = (mean_a - mean_b) / se_diff if se_diff > 0 else 0
    num = (var_a / len(above) + var_b / len(below)) ** 2
    denom = ((var_a / len(above)) ** 2 / (len(above) - 1) +
             (var_b / len(below)) ** 2 / (len(below) - 1))
    df = num / denom if denom > 0 else len(above) + len(below) - 2
    return {
        'label': label, 'threshold': threshold,
        'n_above': len(above), 'mean_above': mean_a,
        'n_below': len(below), 'mean_below': mean_b,
        'diff': mean_a - mean_b, 't_stat': t_stat,
        'p_val': t_to_p(t_stat, df)
    }


# =============================================================================
# SECTION 6: RUN ALL ANALYSES
# =============================================================================

print("\n" + "=" * 80)
print("A: PREDICTIVE REGRESSIONS — NOMINAL, REAL, AND USD RETURNS")
print("   (Newey-West HAC with finite-sample correction n/(n-k))")
print("=" * 80)

for ret_label, fwd_data in [('Nominal', asi_fwd), ('Real', asi_real_fwd),
                              ('USD (spliced)', asi_usd_fwd)]:
    print(f"\n  --- ASI {ret_label} ---")
    for h in [3, 6, 9, 12]:
        r = run_regression(pmi, fwd_data[h], h)
        if r['valid']:
            print(f"    {h:2d}m: B={r['beta']:+.3f}  SE={r['se_beta']:.3f}  "
                  f"t={r['t_stat']:+.2f}{stars(r['p_val']):3s}  R2={r['R2']:.3f}  n={r['n']}")


print("\n" + "=" * 80)
print("B: SECTOR-LEVEL REGRESSIONS")
print("=" * 80)

for name, nom_fwd, real_fwd in [('Banking', banking_fwd, banking_real_fwd),
                                  ('ConsGoods', consgoods_fwd, consgoods_real_fwd)]:
    print(f"\n  --- {name} ---")
    for ret_type, fwd in [('Nominal', nom_fwd), ('Real', real_fwd)]:
        for h in [6, 12]:
            r = run_regression(pmi, fwd[h], h)
            if r['valid']:
                print(f"    {name} {ret_type} {h:2d}m: B={r['beta']:+.3f}  "
                      f"t={r['t_stat']:+.2f}{stars(r['p_val']):3s}  R2={r['R2']:.3f}")


print("\n" + "=" * 80)
print("C: REGIME ANALYSIS (Expansion PMI>50 vs Contraction PMI<=50)")
print("=" * 80)

for ret_label, fwd_data in [('ASI Nom', asi_fwd), ('ASI Real', asi_real_fwd),
                              ('ASI USD', asi_usd_fwd),
                              ('Banking Nom', banking_fwd), ('ConsGoods Nom', consgoods_fwd)]:
    for h in [6, 12]:
        r = regime_analysis(pmi, fwd_data[h], 50.0, f"{ret_label} {h}m")
        if r:
            print(f"  {r['label']:25s}: Exp={r['mean_above']:+.1f}%  "
                  f"Con={r['mean_below']:+.1f}%  Diff={r['diff']:+.1f}%  "
                  f"t={r['t_stat']:+.2f}  p={r['p_val']:.4f}{stars(r['p_val'])}")


print("\n" + "=" * 80)
print("D: CRISIS EXCLUSION ROBUSTNESS")
print("=" * 80)

crisis_2016 = set((2016, m) for m in range(1, 13))
crisis_covid = {(2020, 3), (2020, 4), (2020, 5), (2020, 6)}
crisis_all = crisis_2016 | crisis_covid

for h in [6, 12]:
    for ret_label, fwd_data in [('Nominal', asi_fwd), ('Real', asi_real_fwd)]:
        print(f"\n  ASI {ret_label} {h}m:")
        for desc, excl in [("Full sample", None), ("Excl 2016", crisis_2016),
                            ("Excl COVID", crisis_covid), ("Excl both", crisis_all)]:
            r = run_regression(pmi, fwd_data[h], h, desc, exclude_months=excl)
            if r['valid']:
                print(f"    {desc:20s}: B={r['beta']:+.3f}  t={r['t_stat']:+.2f}"
                      f"{stars(r['p_val']):3s}  R2={r['R2']:.3f}  n={r['n']}")


print("\n" + "=" * 80)
print("E: DELTA PMI vs PMI LEVEL (MOMENTUM vs LEVEL)")
print("=" * 80)

for h in [6, 12]:
    r_level = run_regression(pmi, asi_fwd[h], h, "Level")
    r_delta = run_regression(delta_pmi, asi_fwd[h], h, "Delta")
    print(f"\n  ASI Nominal {h}m:")
    if r_level['valid']:
        print(f"    Level: B={r_level['beta']:+.3f}  t={r_level['t_stat']:+.2f}"
              f"{stars(r_level['p_val'])}  R2={r_level['R2']:.3f}")
    if r_delta['valid']:
        print(f"    Delta: B={r_delta['beta']:+.3f}  t={r_delta['t_stat']:+.2f}"
              f"{stars(r_delta['p_val'])}  R2={r_delta['R2']:.3f}")
    # Bivariate regression (both level + delta)
    y_bv, X_bv = [], []
    for ym in sorted(asi_fwd[h].keys()):
        if ym in pmi and ym in delta_pmi:
            y_bv.append(asi_fwd[h][ym])
            X_bv.append([1.0, pmi[ym], delta_pmi[ym]])
    n_bv = len(y_bv)
    r_bv = ols_regression(y_bv, X_bv)
    if r_bv[0]:
        co, res, R2, _, k2 = r_bv
        se_bv = newey_west_se(X_bv, res, n_bv, k2, max_lag=max(int(h * 1.5), 1))
        t_lev = co[1] / se_bv[1] if se_bv[1] > 0 else 0
        t_del = co[2] / se_bv[2] if se_bv[2] > 0 else 0
        print(f"    Both: B_lev={co[1]:+.3f} t={t_lev:+.2f}{stars(t_to_p(t_lev, n_bv-k2))} "
              f" B_del={co[2]:+.3f} t={t_del:+.2f}{stars(t_to_p(t_del, n_bv-k2))} "
              f" R2={R2:.3f}")


# =============================================================================
# SECTION 7: OUT-OF-SAMPLE WITH BRENT BENCHMARK
# =============================================================================
# FIX 3: Uses Brent price LEVEL (not 1-month log return) for apples-to-apples
# comparison with PMI level. Also tests Brent YoY change.

print("\n" + "=" * 80)
print("F: OUT-OF-SAMPLE — PMI LEVEL vs BRENT LEVEL vs BRENT YoY vs NAIVE")
print("   (FIX 3: Brent uses price level, not 1-month return)")
print("=" * 80)

train_end = (2019, 12)
test_start = (2020, 1)

for h in [6, 12]:
    for ret_label, fwd_data in [('Nominal', asi_fwd), ('Real', asi_real_fwd)]:
        # Train PMI model
        r_train_pmi = run_regression(pmi, fwd_data[h], h, "Train PMI",
                                      restrict_range=((2014, 1), train_end))
        if not r_train_pmi['valid']: continue
        a_pmi, b_pmi = r_train_pmi['alpha'], r_train_pmi['beta']

        # Train Brent LEVEL model (FIX 3)
        y_bl, x_bl = [], []
        for ym in sorted(fwd_data[h].keys()):
            if ym <= train_end and ym in brent:
                y_bl.append(fwd_data[h][ym]); x_bl.append(brent[ym])
        X_bl = [[1.0, x] for x in x_bl]
        r_bl = ols_regression(y_bl, X_bl)
        if r_bl[0] is None: continue
        a_bl, b_bl = r_bl[0][0], r_bl[0][1]

        # Train Brent YoY model
        y_by, x_by = [], []
        for ym in sorted(fwd_data[h].keys()):
            if ym <= train_end and ym in brent_yoy:
                y_by.append(fwd_data[h][ym]); x_by.append(brent_yoy[ym])
        X_by = [[1.0, x] for x in x_by]
        r_by = ols_regression(y_by, X_by)
        a_by = r_by[0][0] if r_by[0] else None
        b_by = r_by[0][1] if r_by[0] else None

        # Naive mean
        train_rets = [fwd_data[h][ym] for ym in sorted(fwd_data[h].keys())
                      if ym <= train_end and ym in pmi]
        naive_mean = sum(train_rets) / len(train_rets)

        # Test
        test_months = [ym for ym in sorted(fwd_data[h].keys())
                       if ym >= test_start and ym in pmi and ym in brent]
        if len(test_months) < 5: continue

        ep, ebl, eby, en = 0, 0, 0, 0
        dp, dbl, dby = 0, 0, 0
        nt = len(test_months)
        for ym in test_months:
            act = fwd_data[h][ym]
            pp = a_pmi + b_pmi * pmi[ym]
            pbl = a_bl + b_bl * brent[ym]
            ep += (pp - act) ** 2
            ebl += (pbl - act) ** 2
            en += (naive_mean - act) ** 2
            if (pp > naive_mean and act > naive_mean) or (pp <= naive_mean and act <= naive_mean):
                dp += 1
            if (pbl > naive_mean and act > naive_mean) or (pbl <= naive_mean and act <= naive_mean):
                dbl += 1
            if a_by is not None and ym in brent_yoy:
                pby = a_by + b_by * brent_yoy[ym]
                eby += (pby - act) ** 2
                if (pby > naive_mean and act > naive_mean) or (pby <= naive_mean and act <= naive_mean):
                    dby += 1

        rp = math.sqrt(ep / nt)
        rbl = math.sqrt(ebl / nt)
        rn = math.sqrt(en / nt)
        rby = math.sqrt(eby / nt) if eby > 0 else 0

        print(f"\n  ASI {ret_label} {h}m (n_test={nt}):")
        print(f"    RMSE: PMI={rp:.1f}  BrentLvl={rbl:.1f}  BrentYoY={rby:.1f}  Naive={rn:.1f}")
        print(f"    Ratio vs Naive: PMI={rp/rn:.3f}  BrentLvl={rbl/rn:.3f}  BrentYoY={rby/rn:.3f}")
        print(f"    DirAcc: PMI={dp/nt*100:.1f}%  BrentLvl={dbl/nt*100:.1f}%")


# =============================================================================
# SECTION 8: BACKTEST WITH ACTUAL MPR + GEOMETRIC YIELDS
# =============================================================================
# FIX 2: Uses actual CBN MPC decisions (21 rate changes) and geometric
# monthly compounding instead of simple division.

print("\n" + "=" * 80)
print("G: BACKTEST — ACTUAL MPR + GEOMETRIC YIELDS + 1-MONTH LAG")
print("   (FIX 2: geometric (1+r)^(1/12)-1 and actual CBN MPR)")
print("=" * 80)

asi_monthly_ret = {}
for i in range(1, len(months)):
    prev, curr = months[i - 1], months[i]
    if prev in asi and curr in asi:
        asi_monthly_ret[curr] = asi[curr] / asi[prev] - 1


def get_tbill_old(ym):
    """OLD: Approximate T-bill return (simple division, rough proxy rates)."""
    y, m = ym
    if y < 2022: return 12.0 / 12 / 100
    elif y == 2022: return 14.0 / 12 / 100
    elif y == 2023 and m <= 5: return 18.0 / 12 / 100
    elif y == 2023: return 19.0 / 12 / 100
    elif y == 2024 and m <= 2: return 22.5 / 12 / 100
    elif y == 2024 and m <= 5: return 24.75 / 12 / 100
    elif y == 2024 and m <= 9: return 26.25 / 12 / 100
    elif y == 2024: return 27.25 / 12 / 100
    else: return 27.50 / 12 / 100


def get_tbill_new(ym):
    """NEW: Actual MPR + geometric monthly yield: (1 + annual_rate)^(1/12) - 1."""
    if ym in mpr_data:
        annual = mpr_data[ym] / 100.0
        return (1 + annual) ** (1.0 / 12) - 1
    return (1 + 0.12) ** (1.0 / 12) - 1  # fallback


def deflate_series(cum_vals, date_list, cpi_data):
    """Deflate a cumulative equity curve by monthly CPI."""
    real = [cum_vals[0]]
    for i in range(1, len(cum_vals)):
        monthly_infl = cpi_data.get(date_list[i], 15.0) / 12 / 100
        nom_ret = cum_vals[i] / cum_vals[i - 1] - 1 if cum_vals[i - 1] > 0 else 0
        real.append(real[-1] * (1 + nom_ret - monthly_infl))
    return real


def max_drawdown(series):
    """Maximum drawdown as percentage."""
    peak = series[0]
    md = 0
    for v in series:
        if v > peak: peak = v
        dd = (peak - v) / peak
        if dd > md: md = dd
    return md * 100


bt_months = [m for m in months if m >= (2014, 2)]

configs = [
    ("Old proxy + simple div + no lag",     get_tbill_old, 0),
    ("Old proxy + simple div + 1m lag",     get_tbill_old, 1),
    ("Actual MPR + geometric + no lag",     get_tbill_new, 0),
    ("Actual MPR + geometric + 1m lag",     get_tbill_new, 1),
]

for desc, tbill_fn, lag in configs:
    strat = 1.0; bnh = 1.0; cash = 1.0
    stv = [1.0]; bhv = [1.0]; cav = [1.0]
    dates = [bt_months[0]]
    eq_months = 0; total = 0

    for i in range(1, len(bt_months)):
        sig_idx = i - 1 - lag
        curr = bt_months[i]
        if curr not in asi_monthly_ret: continue
        er = asi_monthly_ret[curr]
        cr = tbill_fn(curr)
        bnh *= (1 + er); cash *= (1 + cr)
        bhv.append(bnh); cav.append(cash)
        if sig_idx >= 0 and bt_months[sig_idx] in pmi and pmi[bt_months[sig_idx]] <= 50:
            strat *= (1 + er); eq_months += 1
        else:
            strat *= (1 + cr)
        stv.append(strat)
        dates.append(curr)
        total += 1

    str_r = deflate_series(stv, dates, cpi)
    bh_r = deflate_series(bhv, dates, cpi)
    ca_r = deflate_series(cav, dates, cpi)

    print(f"\n  {desc}:")
    print(f"    Nominal: Strategy={stv[-1]:.2f}x  BnH={bhv[-1]:.2f}x  Cash={cav[-1]:.2f}x")
    print(f"    Real:    Strategy={str_r[-1]:.2f}x  BnH={bh_r[-1]:.2f}x  Cash={ca_r[-1]:.2f}x")
    print(f"    MaxDD:   Strategy={max_drawdown(stv):.1f}%  BnH={max_drawdown(bhv):.1f}%")
    print(f"    Equity months: {eq_months}/{total} ({eq_months/total*100:.1f}%)")

# Save final backtest CSV (Actual MPR + geometric + 1m lag)
with open(f"{DATA_DIR2}/backtest_final.csv", "w") as f:
    f.write("Year,Month,Strategy,BuyHold,Cash,StratReal,BnHReal,CashReal\n")
    # Recompute for the final config
    strat = 1.0; bnh2 = 1.0; cash2 = 1.0
    stv2 = [1.0]; bhv2 = [1.0]; cav2 = [1.0]
    dates2 = [bt_months[0]]
    for i in range(1, len(bt_months)):
        sig_idx = i - 2  # 1-month lag
        curr = bt_months[i]
        if curr not in asi_monthly_ret: continue
        er = asi_monthly_ret[curr]
        cr = get_tbill_new(curr)
        bnh2 *= (1 + er); cash2 *= (1 + cr)
        bhv2.append(bnh2); cav2.append(cash2)
        if sig_idx >= 0 and bt_months[sig_idx] in pmi and pmi[bt_months[sig_idx]] <= 50:
            strat *= (1 + er)
        else:
            strat *= (1 + cr)
        stv2.append(strat)
        dates2.append(curr)
    sr2 = deflate_series(stv2, dates2, cpi)
    br2 = deflate_series(bhv2, dates2, cpi)
    cr2 = deflate_series(cav2, dates2, cpi)
    for i, ym in enumerate(dates2):
        f.write(f"{ym[0]},{ym[1]},{stv2[i]:.6f},{bhv2[i]:.6f},{cav2[i]:.6f},"
                f"{sr2[i]:.6f},{br2[i]:.6f},{cr2[i]:.6f}\n")
    print(f"\n  Backtest CSV saved: {DATA_DIR2}/backtest_final.csv")


# =============================================================================
# SECTION 9: THRESHOLD OPTIMISATION
# =============================================================================

print("\n" + "=" * 80)
print("H: PMI THRESHOLD OPTIMISATION (12m real returns)")
print("=" * 80)

print(f"  {'Threshold':>10s}  {'Above':>8s}  {'Below':>8s}  "
      f"{'Spread':>8s}  {'n_below':>7s}  {'t':>7s}  {'p':>7s}")
for threshold in [46, 47, 48, 49, 50, 51, 52]:
    r = regime_analysis(pmi, asi_real_fwd[12], threshold)
    if r and r['n_below'] >= 3:
        spread = r['mean_below'] - r['mean_above']
        print(f"  PMI <= {threshold:2d}     {r['mean_above']:+7.1f}%  {r['mean_below']:+7.1f}%  "
              f"{spread:+7.1f}%  {r['n_below']:6d}  "
              f"{-r['t_stat']:+6.2f}  {r['p_val']:.4f}{stars(r['p_val'])}")


# =============================================================================
# SECTION 10: STRUCTURAL BREAK
# =============================================================================

print("\n" + "=" * 80)
print("I: STRUCTURAL BREAK (Pre-reform vs Reform era)")
print("=" * 80)

for h in [6, 12]:
    for ret_label, fwd_data in [('Nominal', asi_fwd), ('Real', asi_real_fwd)]:
        r_pre = run_regression(pmi, fwd_data[h], h, "Pre",
                                restrict_range=((2014, 1), (2022, 12)))
        r_post = run_regression(pmi, fwd_data[h], h, "Reform",
                                 restrict_range=((2023, 1), (2026, 1)))
        print(f"  ASI {ret_label} {h}m: Pre B={r_pre['beta']:+.3f} "
              f"t={r_pre['t_stat']:+.2f}{stars(r_pre['p_val'])} | "
              f"Reform B={r_post['beta']:+.3f} "
              f"t={r_post['t_stat']:+.2f}{stars(r_post['p_val'])}")


print("\n" + "=" * 80)
print("V3 ANALYSIS COMPLETE — ALL THREE FIXES APPLIED")
print("=" * 80)
print("FIX 1: Newey-West n/(n-k) finite-sample correction")
print("FIX 2: Actual CBN MPR + geometric monthly yields")
print("FIX 3: Brent LEVEL (not 1m log return) in OOS benchmark")
print("=" * 80)
