"""
ElasticNet Rolling Window — Main ML Script
==========================================
For each rolling 10-year training window:
  1. Train ElasticNet with cross-validated hyperparameters
  2. Predict returns for the following year (out-of-sample)
  3. Build long-short portfolio (top/bottom decile)
  4. Save coefficients (feature importance over time)

Run from: Master_Thesis/scripts/
Output:   Master_Thesis/output/elasticnet/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
MERGED_PATH   = "../data_clean/merged_data.parquet"
FF_RF_PATH    = "../data_raw/ff_factors_monthly.csv"
OUT_DIR       = Path("../output/elasticnet")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED   = 42
TRAIN_YEARS   = 10
RET_COL       = "RET_ADJ"
DATE_COL      = "date"
PERMNO_COL    = "PERMNO"

FEATURE_COLS = [
    "SIZE", "mvel1", "mom1m", "MOM", "chmom",
    "turn", "dolvol", "spread", "dy", "indmom",
    "retvol", "maxret", "minret", "skew_daily", "kurt_daily",
    "bm",
    "roa", "roe", "profit_margin", "gross_profitability", "operating_profitability",
    "ebit_to_assets", "ebitda_to_assets", "cashflow_to_assets", "asset_turnover",
    "asset_growth", "sales_growth", "capex_to_assets",
    "ppent_growth", "intan_growth", "gdwl_growth", "invt_growth", "emp_growth",
    "rd_intensity", "ad_intensity", "sga_intensity",
    "leverage", "debt_to_assets", "debt_to_equity", "net_debt_to_assets",
    "current_ratio", "quick_ratio", "cash_ratio", "working_capital_to_assets",
    "accruals", "delta_roa", "delta_margin", "delta_leverage", "delta_turnover",
]

PARAM_GRID = {
    "model__alpha"    : [0.0001, 0.001, 0.01, 0.1, 1.0],
    "model__l1_ratio" : [0.1, 0.5, 0.7, 0.9, 1.0],
}

# ── HELPERS ───────────────────────────────────────────────────────────────────
def hline(): print("─" * 60)

def winsorize_df(train_df, pred_df, cols, low=0.01, high=0.99):
    """
    Winsorize using ONLY training data distribution.
    Apply same bounds to prediction data — no look-ahead bias.
    Done per month cross-sectionally within training window.
    """
    # Compute bounds from training data per month, then take median bound
    # This gives stable bounds to apply to prediction data
    lo_bounds = {}
    hi_bounds = {}
    for c in cols:
        if c not in train_df.columns:
            continue
        monthly_lo = train_df.groupby(DATE_COL)[c].quantile(low)
        monthly_hi = train_df.groupby(DATE_COL)[c].quantile(high)
        lo_bounds[c] = monthly_lo.median()
        hi_bounds[c] = monthly_hi.median()

    train_out = train_df.copy()
    pred_out  = pred_df.copy()
    for c in cols:
        if c not in train_df.columns:
            continue
        train_out[c] = train_out[c].clip(lo_bounds[c], hi_bounds[c])
        pred_out[c]  = pred_out[c].clip(lo_bounds[c], hi_bounds[c])

    return train_out, pred_out

def rank_normalize(train_df, pred_df, cols):
    """
    Cross-sectional rank normalization to [-0.5, 0.5].
    Fit ranking on training data distribution only,
    then apply to prediction data using training quantiles.
    """
    train_out = train_df.copy()
    pred_out  = pred_df.copy()

    for c in cols:
        if c not in train_df.columns:
            continue

        # Rank normalize training data per month
        def _rank(s):
            r = s.rank(method="average", na_option="keep")
            n = r.notna().sum()
            return (r - 1) / max(n - 1, 1) - 0.5

        train_out[c] = train_out.groupby(DATE_COL)[c].transform(_rank)

        # For prediction data: use training distribution to normalize
        # Map each value to its quantile position in training data
        train_vals = np.sort(train_df[c].dropna().values)
        n_train = len(train_vals)

        def _map_to_train_rank(s):
            # Find where each pred value falls in training distribution
            positions = np.searchsorted(train_vals, s.fillna(np.nan), side="left")
            positions = np.clip(positions, 0, n_train - 1)
            ranks = positions / max(n_train - 1, 1) - 0.5
            result = pd.Series(ranks, index=s.index)
            result[s.isna()] = np.nan
            return result

        pred_out[c] = pred_out.groupby(DATE_COL)[c].transform(_map_to_train_rank)

    return train_out, pred_out

def winsorize_returns(train_df, pred_df, col, low=0.01, high=0.99):
    """
    Winsorize returns using training data bounds only.
    """
    monthly_lo = train_df.groupby(DATE_COL)[col].quantile(low).median()
    monthly_hi = train_df.groupby(DATE_COL)[col].quantile(high).median()

    train_out = train_df.copy()
    pred_out  = pred_df.copy()
    train_out[col] = train_out[col].clip(monthly_lo, monthly_hi)
    pred_out[col]  = pred_out[col].clip(monthly_lo, monthly_hi)
    return train_out, pred_out

def load_rf():
    try:
        ff = pd.read_csv(FF_RF_PATH, skiprows=3)
        ff.columns = ff.columns.str.strip()
        date_col = ff.columns[0]
        ff = ff.rename(columns={date_col: "ff_date"})
        ff = ff[ff["ff_date"].astype(str).str.strip().str.match(r"^\d{6}$")].copy()
        ff["ff_date"] = pd.to_datetime(ff["ff_date"].astype(int).astype(str), format="%Y%m")
        ff["RF"] = pd.to_numeric(ff["RF"], errors="coerce") / 100
        ff["ym"] = ff["ff_date"].dt.to_period("M")
        return ff[["ym", "RF"]].set_index("ym")["RF"]
    except FileNotFoundError:
        print("  ⚠  FF risk-free rate file not found — using 0 as RF")
        return None

def compute_portfolio_metrics(returns, rf_series=None, label="Portfolio"):
    r = returns.dropna()
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    if rf_series is not None:
        idx = r.index.to_period("M")
        rf  = idx.map(rf_series).fillna(0)
        exc = r.values - rf.values
    else:
        exc = r.values
    sharpe = (exc.mean() / exc.std()) * np.sqrt(12) if exc.std() > 0 else np.nan
    cum    = (1 + r).cumprod()
    mdd    = ((cum - cum.cummax()) / cum.cummax()).min()
    return {
        "Label"       : label,
        "Ann_Return"  : round(ann_ret, 4),
        "Ann_Vol"     : round(ann_vol, 4),
        "Sharpe"      : round(sharpe, 4),
        "Max_Drawdown": round(mdd, 4),
        "N_months"    : len(r),
    }

# ── 1. LOAD RAW DATA (no preprocessing yet) ───────────────────────────────────
hline()
print("STEP 1: Loading raw data")
hline()

df = pd.read_parquet(MERGED_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values([PERMNO_COL, DATE_COL]).reset_index(drop=True)

keep = [PERMNO_COL, DATE_COL, RET_COL] + [c for c in FEATURE_COLS if c in df.columns]
df   = df[keep].copy()

present_features = [c for c in FEATURE_COLS if c in df.columns]
df["year"] = df[DATE_COL].dt.year

# ── NYSE 20th percentile size screen — Gu, Kelly & Xiu (2020) ────
print("  Applying NYSE size screen (GKX 2020)...")
crsp_screen = pd.read_parquet(
    "../data_clean/crsp_monthly_clean.parquet",
    columns=["PERMNO", "date", "ME", "EXCHCD", "PRC"]
)
crsp_screen["date"] = pd.to_datetime(crsp_screen["date"])
nyse_p20 = (
    crsp_screen[crsp_screen["EXCHCD"] == 1]
    .groupby("date")["ME"]
    .quantile(0.20)
    .reset_index()
    .rename(columns={"ME": "nyse_p20"})
)
df = df.merge(crsp_screen[["PERMNO", "date", "ME", "PRC"]], on=["PERMNO", "date"], how="left")
df = df.merge(nyse_p20, on="date", how="left")
before = len(df)
df = df[df["ME"] >= df["nyse_p20"]].copy()
df = df.drop(columns=["ME", "nyse_p20"])
print(f"  Before screen : {before:,} observations")
print(f"  After screen  : {len(df):,} observations")
print(f"  Removed       : {before - len(df):,} ({(before - len(df))/before*100:.1f}%% of sample)")
before_price = len(df)
df = df[df["PRC"] >= 1].copy()
df = df.drop(columns=["PRC"])
print(f"  After price screen: {len(df):,} (removed {before_price - len(df):,} below $1)")
print(f"  Observations  : {len(df):,}")
print(f"  Firms         : {df[PERMNO_COL].nunique():,}")
print(f"  Date range    : {df[DATE_COL].min().date()} → {df[DATE_COL].max().date()}")
print(f"  Features      : {len(present_features)}")
print("  ✓ Loaded from merged_data.parquet (cleaned data)")
print("  ✓ ML preprocessing (winsorize/rank normalize) will be done per window")

# ── DIAGNOSTIC: check for look-ahead bias ─────────────────────
hline()
print("DIAGNOSTIC: Checking for data leakage")
hline()

# Correlation between mom1m and RET_ADJ should be LOW (~0.02–0.05)
# If it's high (>0.5) then mom1m = current return (look-ahead bias)
corr_mom1m = df["mom1m"].corr(df[RET_COL])
print(f"  Corr(mom1m, RET_ADJ)  : {corr_mom1m:.4f}  (expect ~0.02–0.05)")

# Correlation between MOM and RET_ADJ should also be LOW (~0.02–0.04)
corr_mom = df["MOM"].corr(df[RET_COL])
print(f"  Corr(MOM, RET_ADJ)    : {corr_mom:.4f}  (expect ~0.02–0.04)")

# Check if mom1m literally equals RET_ADJ (perfect look-ahead)
exact_match = (df["mom1m"] == df[RET_COL]).mean()
print(f"  mom1m == RET_ADJ      : {exact_match*100:.2f}%%  (expect ~0%%)")

# Check correlation of ALL features with RET_ADJ
print("\n  Feature correlations with RET_ADJ (any > 0.3 is suspicious):")
for c in present_features:
    corr = df[c].corr(df[RET_COL])
    if abs(corr) > 0.1:
        print(f"  ⚠  {c:<35} corr={corr:.4f}")
hline()

# ── 2. DEFINE ROLLING WINDOWS ─────────────────────────────────────────────────
hline()
print("STEP 2: Defining rolling windows")
hline()

all_years = sorted(df["year"].unique())
windows   = []
for i in range(len(all_years) - TRAIN_YEARS):
    windows.append({
        "train_start": all_years[i],
        "train_end"  : all_years[i + TRAIN_YEARS - 1],
        "pred_year"  : all_years[i + TRAIN_YEARS],
    })

print(f"  Training window : {TRAIN_YEARS} years")
print(f"  Total windows   : {len(windows)}")
print(f"  First window    : train {windows[0]['train_start']}–{windows[0]['train_end']}, predict {windows[0]['pred_year']}")
print(f"  Last window     : train {windows[-1]['train_start']}–{windows[-1]['train_end']}, predict {windows[-1]['pred_year']}")

# ── 3. ROLLING WINDOW TRAINING & PREDICTION ───────────────────────────────────
hline()
print("STEP 3: Rolling window ElasticNet (preprocessing inside each window)")
hline()

all_predictions  = []
all_coefficients = []
all_metrics      = []

for w_idx, w in enumerate(windows):
    print(f"  Window {w_idx+1:02d}/{len(windows)}: "
          f"train {w['train_start']}–{w['train_end']} → predict {w['pred_year']}", end=" ")

    # ── Split raw data ────────────────────────────────────────────
    train_raw = df[df["year"].between(w["train_start"], w["train_end"])].copy()
    pred_raw  = df[df["year"] == w["pred_year"]].copy()

    if len(train_raw) < 1000 or len(pred_raw) < 100:
        print("  ⚠ skipped (insufficient data)")
        continue

    # ── Step A: Winsorize returns (training bounds only) ──────────
    train_p, pred_p = winsorize_returns(train_raw, pred_raw, RET_COL)

    # ── Step B: Winsorize features (training bounds only) ─────────
    train_p, pred_p = winsorize_df(train_p, pred_p, present_features)

    # ── Step C: Rank normalize features (training dist only) ──────
    train_p, pred_p = rank_normalize(train_p, pred_p, present_features)

    # ── Step D: Fill remaining NaN with 0 ─────────────────────────
    train_p[present_features] = train_p[present_features].fillna(0)
    pred_p[present_features]  = pred_p[present_features].fillna(0)

    # ── Fit ElasticNet ────────────────────────────────────────────
    X_train = train_p[present_features].values
    y_train = train_p[RET_COL].values
    X_pred  = pred_p[present_features].values
    y_actual = pred_p[RET_COL].values

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  ElasticNet(max_iter=5000, random_state=RANDOM_SEED)),
    ])

    tscv = TimeSeriesSplit(n_splits=5)
    gs   = GridSearchCV(
        pipe, PARAM_GRID,
        cv=tscv, scoring="r2",
        n_jobs=-1, refit=True,
    )
    gs.fit(X_train, y_train)

    best_alpha    = gs.best_params_["model__alpha"]
    best_l1_ratio = gs.best_params_["model__l1_ratio"]
    y_hat         = gs.predict(X_pred)
    oos_r2        = r2_score(y_actual, y_hat)

    print(f"| α={best_alpha:.4f} l1={best_l1_ratio:.1f} | OOS R²={oos_r2:.5f}")

    # ── Store predictions ─────────────────────────────────────────
    pred_out = pred_raw[[PERMNO_COL, DATE_COL, RET_COL]].copy()
    pred_out["predicted_return"] = y_hat
    pred_out["pred_year"]        = w["pred_year"]
    all_predictions.append(pred_out)

    # ── Store coefficients ────────────────────────────────────────
    fitted_model = gs.best_estimator_.named_steps["model"]
    coef_dict = {
        "pred_year"     : w["pred_year"],
        "train_start"   : w["train_start"],
        "train_end"     : w["train_end"],
        "best_alpha"    : best_alpha,
        "best_l1_ratio" : best_l1_ratio,
        "oos_r2"        : oos_r2,
        "n_nonzero_coef": int(np.sum(fitted_model.coef_ != 0)),
    }
    for feat, coef in zip(present_features, fitted_model.coef_):
        coef_dict[feat] = coef
    all_coefficients.append(coef_dict)

    all_metrics.append({
        "pred_year"    : w["pred_year"],
        "train_start"  : w["train_start"],
        "train_end"    : w["train_end"],
        "oos_r2"       : oos_r2,
        "best_alpha"   : best_alpha,
        "best_l1_ratio": best_l1_ratio,
        "n_train"      : len(train_raw),
        "n_pred"       : len(pred_raw),
    })

# Combine & save
predictions_df  = pd.concat(all_predictions,  ignore_index=True)
coefficients_df = pd.DataFrame(all_coefficients)
metrics_df      = pd.DataFrame(all_metrics)

predictions_df.to_parquet(OUT_DIR / "predictions.parquet", index=False)
coefficients_df.to_csv(OUT_DIR / "coefficients.csv", index=False)
metrics_df.to_csv(OUT_DIR / "window_metrics.csv", index=False)
print(f"\n  ✓ Results saved to {OUT_DIR}")

# ── 4. LONG-SHORT PORTFOLIO ───────────────────────────────────────────────────
hline()
print("STEP 4: Long-short portfolio construction")
hline()

portfolio_returns = []
for date, grp in predictions_df.groupby(DATE_COL):
    grp = grp.dropna(subset=["predicted_return", RET_COL])
    grp[RET_COL] = grp[RET_COL].clip(grp[RET_COL].quantile(0.01), grp[RET_COL].quantile(0.99))
    n   = len(grp)
    dec_n = max(1, n // 10)
    if n < dec_n * 2:
        continue
    grp_sorted = grp.sort_values("predicted_return", ascending=False)
    portfolio_returns.append({
        DATE_COL    : date,
        "long_dec"  : grp_sorted.head(dec_n)[RET_COL].mean(),
        "short_dec" : grp_sorted.tail(dec_n)[RET_COL].mean(),
        "ls_dec"    : grp_sorted.head(dec_n)[RET_COL].mean() - grp_sorted.tail(dec_n)[RET_COL].mean(),
        "n_stocks"  : n,
    })

port_df = pd.DataFrame(portfolio_returns)
port_df[DATE_COL] = pd.to_datetime(port_df[DATE_COL])
port_df = port_df.sort_values(DATE_COL)

port_df.to_csv(OUT_DIR / "portfolio_returns.csv", index=False)

print(f"  Portfolio months      : {len(port_df)}")
print(f"  Date range            : {port_df[DATE_COL].min().date()} → {port_df[DATE_COL].max().date()}")
print(f"  Mean monthly L-S (dec): {port_df['ls_dec'].mean():.4f}")

# ── 5. PERFORMANCE METRICS ────────────────────────────────────────────────────
hline()
print("STEP 5: Performance metrics")
hline()

rf_series = load_rf()
port_df   = port_df.set_index(DATE_COL)

metrics_list = [
    compute_portfolio_metrics(port_df["long_dec"],  rf_series, "Long leg (D10)"),
    compute_portfolio_metrics(port_df["short_dec"], rf_series, "Short leg (D1)"),
    compute_portfolio_metrics(port_df["ls_dec"],    rf_series, "Long-Short (D10-D1)"),
]

perf_df = pd.DataFrame(metrics_list)
perf_df.to_csv(OUT_DIR / "performance_metrics.csv", index=False)
print(perf_df.to_string(index=False))

# ── 5b. FULL DECILE PORTFOLIO RETURNS TABLE ──────────────────────────────────
hline()
print("STEP 5b: Full decile portfolio returns (D1–D10)")
hline()

decile_returns = []
for date, grp in predictions_df.groupby(DATE_COL):
    grp = grp.dropna(subset=["predicted_return", RET_COL])
    grp[RET_COL] = grp[RET_COL].clip(
        grp[RET_COL].quantile(0.01),
        grp[RET_COL].quantile(0.99)
    )
    n = len(grp)
    if n < 10:
        continue
    grp_sorted = grp.sort_values("predicted_return", ascending=True)
    # Assign decile 1 (lowest predicted) to decile 10 (highest predicted)
    grp_sorted["decile"] = pd.qcut(
        grp_sorted["predicted_return"],
        q=10,
        labels=False,
        duplicates="drop"
    ) + 1
    for d in range(1, 11):
        dec_grp = grp_sorted[grp_sorted["decile"] == d]
        if len(dec_grp) > 0:
            decile_returns.append({
                DATE_COL : date,
                "decile" : d,
                "ret"    : dec_grp[RET_COL].mean(),
                "n"      : len(dec_grp),
            })

decile_df = pd.DataFrame(decile_returns)
decile_df[DATE_COL] = pd.to_datetime(decile_df[DATE_COL])

# Compute mean monthly return and Sharpe per decile
decile_summary = []
for d in range(1, 11):
    d_rets = decile_df[decile_df["decile"] == d].set_index(DATE_COL)["ret"]
    if rf_series is not None:
        idx = d_rets.index.to_period("M")
        rf  = idx.map(rf_series).fillna(0)
        exc = d_rets.values - rf.values
    else:
        exc = d_rets.values
    sharpe_d = (exc.mean() / exc.std()) * np.sqrt(12) if exc.std() > 0 else np.nan
    ann_ret_d = (1 + d_rets).prod() ** (12 / len(d_rets)) - 1
    decile_summary.append({
        "Decile"      : d,
        "Mean_Monthly": round(d_rets.mean(), 4),
        "Ann_Return"  : round(ann_ret_d, 4),
        "Sharpe"      : round(sharpe_d, 4),
        "N_months"    : len(d_rets),
    })

decile_summary_df = pd.DataFrame(decile_summary)
decile_summary_df.to_csv(OUT_DIR / "decile_summary.csv", index=False)
print(decile_summary_df.to_string(index=False))
print(f"\n  Saved → {OUT_DIR / 'decile_summary.csv'}")

# ── 5c. DECADE-LEVEL PORTFOLIO PERFORMANCE ────────────────────────────────────
hline()
print("STEP 5c: Portfolio performance by decade")
hline()

port_df_reset = port_df.reset_index()
port_df_reset["decade"] = (port_df_reset[DATE_COL].dt.year // 10) * 10

decade_perf = []
for decade, grp in port_df_reset.groupby("decade"):
    grp = grp.set_index(DATE_COL)
    long_m  = compute_portfolio_metrics(grp["long_dec"],  rf_series, f"{decade}s Long (D10)")
    short_m = compute_portfolio_metrics(grp["short_dec"], rf_series, f"{decade}s Short (D1)")
    ls_m    = compute_portfolio_metrics(grp["ls_dec"],    rf_series, f"{decade}s L-S (D10-D1)")
    decade_perf.append({
        "Decade"        : f"{decade}s",
        "LS_Mean_Monthly": round(grp["ls_dec"].mean(), 4),
        "LS_Ann_Return" : ls_m["Ann_Return"],
        "LS_Sharpe"     : ls_m["Sharpe"],
        "LS_MaxDrawdown": ls_m["Max_Drawdown"],
        "Long_Sharpe"   : long_m["Sharpe"],
        "Short_Sharpe"  : short_m["Sharpe"],
        "N_months"      : ls_m["N_months"],
    })

decade_perf_df = pd.DataFrame(decade_perf)
decade_perf_df.to_csv(OUT_DIR / "decade_performance.csv", index=False)
print(decade_perf_df.to_string(index=False))
print(f"\n  Saved → {OUT_DIR / 'decade_performance.csv'}")

# ── 6. OOS R² SUMMARY ────────────────────────────────────────────────────────
hline()
print("STEP 6: OOS R² over time")
hline()

print(f"  Mean OOS R²   : {metrics_df['oos_r2'].mean():.5f}")
print(f"  Median OOS R² : {metrics_df['oos_r2'].median():.5f}")
print(f"  Min OOS R²    : {metrics_df['oos_r2'].min():.5f}  ({metrics_df.loc[metrics_df['oos_r2'].idxmin(), 'pred_year']})")
print(f"  Max OOS R²    : {metrics_df['oos_r2'].max():.5f}  ({metrics_df.loc[metrics_df['oos_r2'].idxmax(), 'pred_year']})")
print(f"  %% positive R² : {(metrics_df['oos_r2'] > 0).mean()*100:.1f}%%")

# ── 7. FEATURE IMPORTANCE ─────────────────────────────────────────────────────
hline()
print("STEP 7: Feature importance analysis")
hline()

coef_values  = coefficients_df[present_features].abs()
mean_abs_coef = coef_values.mean().sort_values(ascending=False)
top_features  = mean_abs_coef.head(15).index.tolist()

print("  Top 15 features by mean absolute coefficient:")
for i, (feat, val) in enumerate(mean_abs_coef.head(15).items(), 1):
    n_sel = (coefficients_df[feat] != 0).sum()
    print(f"    {i:2d}. {feat:<35} coef={val:.5f}  selected {n_sel}/{len(coefficients_df)} windows")

# ── 8. PLOTS ──────────────────────────────────────────────────────────────────
hline()
print("STEP 8: Generating plots")
hline()

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("ElasticNet Rolling Window Results", fontsize=14, fontweight="bold")

# Plot 1 — Cumulative L-S return
ax = axes[0, 0]
cum_lsd = (1 + port_df["ls_dec"]).cumprod()
ax.plot(cum_lsd.index, cum_lsd.values, lw=1.5, label="L-S D10-D1", color="steelblue")
ax.axhline(1, color="black", lw=0.5, linestyle=":")
ax.set_title("Cumulative Long-Short Return")
ax.set_ylabel("Growth of $1 (log scale)")
ax.set_yscale("log")
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(5))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.grid(alpha=0.3)

# Plot 2 — OOS R² over time
ax = axes[0, 1]
colors = np.where(metrics_df["oos_r2"] >= 0, "steelblue", "salmon")
ax.bar(metrics_df["pred_year"], metrics_df["oos_r2"], color=colors, width=0.8)
ax.axhline(0, color="black", lw=0.5)
ax.axhline(metrics_df["oos_r2"].mean(), color="darkorange", lw=1.5,
           linestyle="--", label=f"Mean={metrics_df['oos_r2'].mean():.4f}")
ax.set_title("Out-of-Sample R² per Window")
ax.set_ylabel("OOS R²")
ax.set_xlabel("Prediction Year")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 3 — Feature importance heatmap
ax = axes[1, 0]
heatmap_data = coefficients_df.set_index("pred_year")[top_features[:10]].abs()
im = ax.imshow(heatmap_data.T, aspect="auto", cmap="YlOrRd",
               extent=[heatmap_data.index.min(), heatmap_data.index.max(),
                       -0.5, len(top_features[:10]) - 0.5])
ax.set_yticks(range(len(top_features[:10])))
ax.set_yticklabels(top_features[:10], fontsize=7)
ax.set_title("Feature Importance Heatmap (|coef|)")
ax.set_xlabel("Prediction Year")
plt.colorbar(im, ax=ax, shrink=0.8)

# Plot 4 — Non-zero coefficients over time
ax = axes[1, 1]
ax.plot(coefficients_df["pred_year"], coefficients_df["n_nonzero_coef"],
        lw=1.5, color="steelblue", marker="o", markersize=3)
ax.axhline(len(present_features), color="gray", lw=0.8, linestyle="--",
           label=f"Total features ({len(present_features)})")
ax.set_title("Non-Zero Coefficients per Window")
ax.set_ylabel("Count")
ax.set_xlabel("Prediction Year")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "elasticnet_results.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved → {OUT_DIR / 'elasticnet_results.png'}")

# ── 9. DECADE SUMMARY ─────────────────────────────────────────────────────────
hline()
print("STEP 9: Feature importance by decade")
hline()

coefficients_df["decade"] = (coefficients_df["pred_year"] // 10) * 10
decade_summary = []
for decade, grp in coefficients_df.groupby("decade"):
    mean_abs = grp[present_features].abs().mean().sort_values(ascending=False)
    top5     = mean_abs.head(5)
    print(f"\n  {decade}s — top 5 predictors:")
    for feat, val in top5.items():
        pct = (grp[feat] != 0).mean() * 100
        print(f"    {feat:<35} |coef|={val:.5f}  selected {pct:.0f}%% of windows")
    for rank, (feat, val) in enumerate(top5.items(), 1):
        decade_summary.append({"decade": decade, "rank": rank,
                                "feature": feat, "mean_abs_coef": val})

pd.DataFrame(decade_summary).to_csv(OUT_DIR / "decade_importance.csv", index=False)

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
hline()
print("FINAL SUMMARY")
hline()
print(f"  Windows completed  : {len(metrics_df)}")
print(f"  Mean OOS R²        : {metrics_df['oos_r2'].mean():.5f}")
print(f"  L-S monthly return : {port_df['ls_dec'].mean():.4f}")
ls_sharpe = perf_df.loc[perf_df["Label"] == "Long-Short (D10-D1)", "Sharpe"].values[0]
print(f"  L-S Sharpe         : {ls_sharpe:.4f}")
print(f"  Output saved to    : {OUT_DIR}")
hline()