"""
Random Forest Rolling Window — ML Script
=========================================
Mirrors elasticnet_rolling.py exactly in structure:
  - Same data loading, screens, and preprocessing
  - Same rolling 10-year windows
  - Same decile portfolio construction
  - Same output format for direct comparison with ElasticNet

Hyperparameter tuning: per-decade single train/validation split
following promotor feedback. For each decade (1980s–2020s), the
first window of that decade is split into:
  - Training: first 8 years
  - Validation: last 2 years
This conservative scheme better mimics the actual forecasting task
than k-fold CV (which can overfit when adjacent years share regimes).
The selected hyperparameters are then applied to all windows in that decade.

Feature importance: permutation-based importance per window
following Breiman (2001) and Gu, Kelly & Xiu (2020).

Run from: Master_Thesis/scripts/
Output:   Master_Thesis/output/random_forest/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
MERGED_PATH   = "../data_clean/merged_data.parquet"
FF_RF_PATH    = "../data_raw/ff_factors_monthly.csv"
OUT_DIR       = Path("../output/random_forest")
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

# Random Forest hyperparameter grid for per-decade tuning
# n_estimators kept fixed at 300 (more trees rarely hurts, just slows runtime)
# Tuning the three most impactful hyperparameters:
#   max_depth: tree complexity
#   min_samples_leaf: overfitting control
#   max_features: predictor diversity at each split
RF_BASE_PARAMS = {
    "n_estimators" : 300,
    "n_jobs"       : -1,
    "random_state" : RANDOM_SEED,
}

RF_PARAM_GRID = {
    "max_depth"        : [4, 6, 8, 10],
    "min_samples_leaf" : [500, 1000, 2000],
    "max_features"     : ["sqrt", 0.5],
}

# ── HELPERS ───────────────────────────────────────────────────────────────────
def hline(): print("─" * 60)

def winsorize_df(train_df, pred_df, cols, low=0.01, high=0.99):
    """Winsorize using ONLY training data distribution — no look-ahead bias."""
    lo_bounds, hi_bounds = {}, {}
    for c in cols:
        if c not in train_df.columns:
            continue
        lo_bounds[c] = train_df.groupby(DATE_COL)[c].quantile(low).median()
        hi_bounds[c] = train_df.groupby(DATE_COL)[c].quantile(high).median()
    train_out, pred_out = train_df.copy(), pred_df.copy()
    for c in cols:
        if c not in train_df.columns:
            continue
        train_out[c] = train_out[c].clip(lo_bounds[c], hi_bounds[c])
        pred_out[c]  = pred_out[c].clip(lo_bounds[c], hi_bounds[c])
    return train_out, pred_out

def rank_normalize(train_df, pred_df, cols):
    """Cross-sectional rank normalization — training distribution only."""
    train_out, pred_out = train_df.copy(), pred_df.copy()
    for c in cols:
        if c not in train_df.columns:
            continue
        def _rank(s):
            r = s.rank(method="average", na_option="keep")
            n = r.notna().sum()
            return (r - 1) / max(n - 1, 1) - 0.5
        train_out[c] = train_out.groupby(DATE_COL)[c].transform(_rank)
        train_vals = np.sort(train_df[c].dropna().values)
        n_train = len(train_vals)
        def _map_to_train_rank(s):
            positions = np.searchsorted(train_vals, s.fillna(np.nan), side="left")
            positions = np.clip(positions, 0, n_train - 1)
            ranks = positions / max(n_train - 1, 1) - 0.5
            result = pd.Series(ranks, index=s.index)
            result[s.isna()] = np.nan
            return result
        pred_out[c] = pred_out.groupby(DATE_COL)[c].transform(_map_to_train_rank)
    return train_out, pred_out

def winsorize_returns(train_df, pred_df, col, low=0.01, high=0.99):
    """Winsorize returns using training data bounds only."""
    lo = train_df.groupby(DATE_COL)[col].quantile(low).median()
    hi = train_df.groupby(DATE_COL)[col].quantile(high).median()
    train_out, pred_out = train_df.copy(), pred_df.copy()
    train_out[col] = train_out[col].clip(lo, hi)
    pred_out[col]  = pred_out[col].clip(lo, hi)
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
    cum = (1 + r).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return {
        "Label"        : label,
        "Ann_Return"   : round(ann_ret, 4),
        "Ann_Vol"      : round(ann_vol, 4),
        "Sharpe"       : round(sharpe, 4),
        "Max_Drawdown" : round(mdd, 4),
        "N_months"     : len(r),
    }

def preprocess_window(train_raw, pred_raw, features):
    """Apply full preprocessing pipeline to a train/pred split."""
    train_p, pred_p = winsorize_returns(train_raw, pred_raw, RET_COL)
    train_p, pred_p = winsorize_df(train_p, pred_p, features)
    train_p, pred_p = rank_normalize(train_p, pred_p, features)
    train_p[features] = train_p[features].fillna(0)
    pred_p[features]  = pred_p[features].fillna(0)
    return train_p, pred_p

def tune_rf_for_decade(train_p, train_end_year, features, decade_label):
    """
    Tune RF hyperparameters for a decade using a single train/validation split
    that mimics the actual forecasting task (per Chunjie's guidance).

    Splits the 10-year training window into:
      - Training: first 8 years
      - Validation: last 2 years (no fold overlap, no shared regimes)

    This is more conservative than TimeSeriesSplit k-fold CV and better
    captures true generalization to unseen years.

    Returns the best hyperparameters for use in all windows of that decade.
    """
    from itertools import product

    print(f"\n  ▸ Tuning RF hyperparameters for {decade_label}...")

    # Single split: last 2 years of training window for validation
    val_start_year = train_end_year - 1   # last 2 years
    val_mask       = train_p["year"].values >= val_start_year
    train_mask     = ~val_mask

    X_full = train_p[features].values
    y_full = train_p[RET_COL].values

    X_tr  = X_full[train_mask]
    y_tr  = y_full[train_mask]
    X_val = X_full[val_mask]
    y_val = y_full[val_mask]

    print(f"    Training:   {train_mask.sum():,} obs (years before {val_start_year})")
    print(f"    Validation: {val_mask.sum():,} obs (years {val_start_year}-{train_end_year})")

    # Manual grid search using single train/val split
    best_score  = -np.inf
    best_params = None
    grid_keys   = list(RF_PARAM_GRID.keys())
    grid_vals   = [RF_PARAM_GRID[k] for k in grid_keys]

    for combo in product(*grid_vals):
        params = {**RF_BASE_PARAMS, **dict(zip(grid_keys, combo))}
        rf = RandomForestRegressor(**params)
        rf.fit(X_tr, y_tr)
        y_hat = rf.predict(X_val)
        score = r2_score(y_val, y_hat)

        param_str = ", ".join(f"{k}={v}" for k, v in zip(grid_keys, combo))
        print(f"    {param_str}: val R²={score:.5f}")

        if score > best_score:
            best_score  = score
            best_params = dict(zip(grid_keys, combo))

    print(f"    Best params: {best_params}")
    print(f"    Best val R²: {best_score:.5f}")
    return best_params

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
hline()
print("STEP 1: Loading data")
hline()

df = pd.read_parquet(MERGED_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values([PERMNO_COL, DATE_COL]).reset_index(drop=True)

keep = [PERMNO_COL, DATE_COL, RET_COL] + [c for c in FEATURE_COLS if c in df.columns]
df   = df[keep].copy()

present_features = [c for c in FEATURE_COLS if c in df.columns]
df["year"] = df[DATE_COL].dt.year

# NYSE 20th percentile size screen — Gu, Kelly & Xiu (2020)
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
print(f"  Before screen : {before:,} → After: {len(df):,}")
before_price = len(df)
df = df[df["PRC"] >= 1].copy()
df = df.drop(columns=["PRC"])
print(f"  After price screen: {len(df):,} (removed {before_price - len(df):,} below $1)")
print(f"  Observations  : {len(df):,}")
print(f"  Firms         : {df[PERMNO_COL].nunique():,}")
print(f"  Features      : {len(present_features)}")

# ── 2. DEFINE ROLLING WINDOWS ─────────────────────────────────────────────────
hline()
print("STEP 2: Defining rolling windows")
hline()

all_years = sorted(df["year"].unique())
windows   = []
for i in range(len(all_years) - TRAIN_YEARS):
    pred_year = all_years[i + TRAIN_YEARS]
    windows.append({
        "train_start": all_years[i],
        "train_end"  : all_years[i + TRAIN_YEARS - 1],
        "pred_year"  : pred_year,
        "decade"     : (pred_year // 10) * 10,
    })

print(f"  Training window : {TRAIN_YEARS} years")
print(f"  Total windows   : {len(windows)}")
print(f"  First window    : train {windows[0]['train_start']}–{windows[0]['train_end']}, predict {windows[0]['pred_year']}")
print(f"  Last window     : train {windows[-1]['train_start']}–{windows[-1]['train_end']}, predict {windows[-1]['pred_year']}")

# ── 3. PER-DECADE HYPERPARAMETER TUNING ───────────────────────────────────────
hline()
print("STEP 3: Per-decade hyperparameter tuning")
hline()
print("  For each decade, tune on the first window of that decade")
print("  and apply selected hyperparameters to all windows in that decade.")

# Group windows by decade and pick the first window per decade for tuning
decades = sorted(set(w["decade"] for w in windows))
decade_params = {}

for decade in decades:
    decade_windows  = [w for w in windows if w["decade"] == decade]
    tuning_window   = decade_windows[0]
    decade_label    = f"{decade}s"

    print(f"\n  Decade {decade_label}: tuning on window train "
          f"{tuning_window['train_start']}–{tuning_window['train_end']} "
          f"→ pred {tuning_window['pred_year']}")

    train_raw = df[df["year"].between(tuning_window["train_start"],
                                       tuning_window["train_end"])].copy()
    pred_raw  = df[df["year"] == tuning_window["pred_year"]].copy()

    # Preprocess for tuning
    train_p, _ = preprocess_window(train_raw, pred_raw, present_features)

    # Run grid search using single train/val split
    best_params = tune_rf_for_decade(
        train_p, tuning_window["train_end"],
        present_features, decade_label
    )
    decade_params[decade] = best_params

# Save decade params for transparency
decade_params_records = []
for decade, params in decade_params.items():
    rec = {"decade": f"{decade}s", **params}
    decade_params_records.append(rec)
pd.DataFrame(decade_params_records).to_csv(
    OUT_DIR / "decade_hyperparameters_rf.csv", index=False
)
print(f"\n  ✓ Decade hyperparameters saved → {OUT_DIR / 'decade_hyperparameters_rf.csv'}")

# ── 4. ROLLING WINDOW TRAINING & PREDICTION ───────────────────────────────────
hline()
print("STEP 4: Rolling window Random Forest with per-decade hyperparameters")
hline()

all_predictions  = []
all_importances  = []
all_metrics      = []

for w_idx, w in enumerate(windows):
    decade_p = decade_params[w["decade"]]
    print(f"  Window {w_idx+1:02d}/{len(windows)}: "
          f"train {w['train_start']}–{w['train_end']} → predict {w['pred_year']} "
          f"[{w['decade']}s params]", end=" ")

    train_raw = df[df["year"].between(w["train_start"], w["train_end"])].copy()
    pred_raw  = df[df["year"] == w["pred_year"]].copy()

    if len(train_raw) < 1000 or len(pred_raw) < 100:
        print("  ⚠ skipped (insufficient data)")
        continue

    # Preprocessing
    train_p, pred_p = preprocess_window(train_raw, pred_raw, present_features)

    X_train  = train_p[present_features].values
    y_train  = train_p[RET_COL].values
    X_pred   = pred_p[present_features].values
    y_actual = pred_p[RET_COL].values

    # Fit RF with this decade's tuned params
    rf_params = {**RF_BASE_PARAMS, **decade_p}
    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_train, y_train)

    y_hat  = rf.predict(X_pred)
    oos_r2 = r2_score(y_actual, y_hat)

    # Permutation importance on prediction data
    perm_imp = permutation_importance(
        rf, X_pred, y_actual,
        n_repeats=10,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    print(f"| OOS R²={oos_r2:.5f}")

    # Store predictions
    pred_out = pred_raw[[PERMNO_COL, DATE_COL, RET_COL]].copy()
    pred_out["predicted_return"] = y_hat
    pred_out["pred_year"]        = w["pred_year"]
    all_predictions.append(pred_out)

    # Store permutation importances
    imp_dict = {
        "pred_year"  : w["pred_year"],
        "train_start": w["train_start"],
        "train_end"  : w["train_end"],
        "oos_r2"     : oos_r2,
    }
    for feat, imp_mean in zip(present_features, perm_imp.importances_mean):
        imp_dict[feat] = imp_mean
    all_importances.append(imp_dict)

    all_metrics.append({
        "pred_year"       : w["pred_year"],
        "train_start"     : w["train_start"],
        "train_end"       : w["train_end"],
        "oos_r2"          : oos_r2,
        "max_depth"       : decade_p["max_depth"],
        "min_samples_leaf": decade_p["min_samples_leaf"],
        "max_features"    : decade_p["max_features"],
        "n_train"         : len(train_raw),
        "n_pred"          : len(pred_raw),
    })

# Combine & save
predictions_df  = pd.concat(all_predictions, ignore_index=True)
importances_df  = pd.DataFrame(all_importances)
metrics_df      = pd.DataFrame(all_metrics)

predictions_df.to_parquet(OUT_DIR / "predictions_rf.parquet", index=False)
importances_df.to_csv(OUT_DIR / "importances_rf.csv", index=False)
metrics_df.to_csv(OUT_DIR / "window_metrics_rf.csv", index=False)
print(f"\n  ✓ Results saved to {OUT_DIR}")

# ── 5. LONG-SHORT PORTFOLIO ───────────────────────────────────────────────────
hline()
print("STEP 5: Long-short portfolio construction")
hline()

portfolio_returns = []
for date, grp in predictions_df.groupby(DATE_COL):
    grp = grp.dropna(subset=["predicted_return", RET_COL])
    grp[RET_COL] = grp[RET_COL].clip(
        grp[RET_COL].quantile(0.01),
        grp[RET_COL].quantile(0.99)
    )
    n     = len(grp)
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
port_df.to_csv(OUT_DIR / "portfolio_returns_rf.csv", index=False)

print(f"  Portfolio months      : {len(port_df)}")
print(f"  Date range            : {port_df[DATE_COL].min().date()} → {port_df[DATE_COL].max().date()}")
print(f"  Mean monthly L-S (dec): {port_df['ls_dec'].mean():.4f}")

# ── 6. PERFORMANCE METRICS ────────────────────────────────────────────────────
hline()
print("STEP 6: Performance metrics")
hline()

rf_series = load_rf()
port_df   = port_df.set_index(DATE_COL)

metrics_list = [
    compute_portfolio_metrics(port_df["long_dec"],  rf_series, "Long leg (D10)"),
    compute_portfolio_metrics(port_df["short_dec"], rf_series, "Short leg (D1)"),
    compute_portfolio_metrics(port_df["ls_dec"],    rf_series, "Long-Short (D10-D1)"),
]
perf_df = pd.DataFrame(metrics_list)
perf_df.to_csv(OUT_DIR / "performance_metrics_rf.csv", index=False)
print(perf_df.to_string(index=False))

# ── 6b. FULL DECILE PORTFOLIO RETURNS TABLE ───────────────────────────────────
hline()
print("STEP 6b: Full decile portfolio returns (D1–D10)")
hline()

decile_returns = []
for date, grp in predictions_df.groupby(DATE_COL):
    grp = grp.dropna(subset=["predicted_return", RET_COL])
    grp[RET_COL] = grp[RET_COL].clip(
        grp[RET_COL].quantile(0.01),
        grp[RET_COL].quantile(0.99)
    )
    if len(grp) < 10:
        continue
    grp_sorted = grp.sort_values("predicted_return", ascending=True)
    grp_sorted["decile"] = pd.qcut(
        grp_sorted["predicted_return"],
        q=10, labels=False, duplicates="drop"
    ) + 1
    for d in range(1, 11):
        dec_grp = grp_sorted[grp_sorted["decile"] == d]
        if len(dec_grp) > 0:
            decile_returns.append({
                DATE_COL : date,
                "decile" : d,
                "ret"    : dec_grp[RET_COL].mean(),
            })

decile_df = pd.DataFrame(decile_returns)
decile_df[DATE_COL] = pd.to_datetime(decile_df[DATE_COL])

decile_summary = []
for d in range(1, 11):
    d_rets = decile_df[decile_df["decile"] == d].set_index(DATE_COL)["ret"]
    if rf_series is not None:
        idx = d_rets.index.to_period("M")
        rf  = idx.map(rf_series).fillna(0)
        exc = d_rets.values - rf.values
    else:
        exc = d_rets.values
    sharpe_d  = (exc.mean() / exc.std()) * np.sqrt(12) if exc.std() > 0 else np.nan
    ann_ret_d = (1 + d_rets).prod() ** (12 / len(d_rets)) - 1
    decile_summary.append({
        "Decile"      : d,
        "Mean_Monthly": round(d_rets.mean(), 4),
        "Ann_Return"  : round(ann_ret_d, 4),
        "Sharpe"      : round(sharpe_d, 4),
        "N_months"    : len(d_rets),
    })

decile_summary_df = pd.DataFrame(decile_summary)
decile_summary_df.to_csv(OUT_DIR / "decile_summary_rf.csv", index=False)
print(decile_summary_df.to_string(index=False))
print(f"\n  Saved → {OUT_DIR / 'decile_summary_rf.csv'}")

# ── 6c. DECADE-LEVEL PORTFOLIO PERFORMANCE ────────────────────────────────────
hline()
print("STEP 6c: Portfolio performance by decade")
hline()

port_df_reset = port_df.reset_index()
port_df_reset["decade"] = (port_df_reset[DATE_COL].dt.year // 10) * 10

decade_perf = []
for decade, grp in port_df_reset.groupby("decade"):
    grp     = grp.set_index(DATE_COL)
    long_m  = compute_portfolio_metrics(grp["long_dec"],  rf_series, f"{decade}s Long")
    short_m = compute_portfolio_metrics(grp["short_dec"], rf_series, f"{decade}s Short")
    ls_m    = compute_portfolio_metrics(grp["ls_dec"],    rf_series, f"{decade}s L-S")
    decade_perf.append({
        "Decade"         : f"{decade}s",
        "LS_Mean_Monthly": round(grp["ls_dec"].mean(), 4),
        "LS_Ann_Return"  : ls_m["Ann_Return"],
        "LS_Sharpe"      : ls_m["Sharpe"],
        "LS_MaxDrawdown" : ls_m["Max_Drawdown"],
        "Long_Sharpe"    : long_m["Sharpe"],
        "Short_Sharpe"   : short_m["Sharpe"],
        "N_months"       : ls_m["N_months"],
    })

decade_perf_df = pd.DataFrame(decade_perf)
decade_perf_df.to_csv(OUT_DIR / "decade_performance_rf.csv", index=False)
print(decade_perf_df.to_string(index=False))
print(f"\n  Saved → {OUT_DIR / 'decade_performance_rf.csv'}")

# ── 7. OOS R² SUMMARY ─────────────────────────────────────────────────────────
hline()
print("STEP 7: OOS R² over time")
hline()

print(f"  Mean OOS R²   : {metrics_df['oos_r2'].mean():.5f}")
print(f"  Median OOS R² : {metrics_df['oos_r2'].median():.5f}")
print(f"  Min OOS R²    : {metrics_df['oos_r2'].min():.5f}  ({metrics_df.loc[metrics_df['oos_r2'].idxmin(), 'pred_year']})")
print(f"  Max OOS R²    : {metrics_df['oos_r2'].max():.5f}  ({metrics_df.loc[metrics_df['oos_r2'].idxmax(), 'pred_year']})")
print(f"  %% positive R² : {(metrics_df['oos_r2'] > 0).mean()*100:.1f}%%")

# ── 8. FEATURE IMPORTANCE ─────────────────────────────────────────────────────
hline()
print("STEP 8: Feature importance analysis (permutation-based)")
hline()

imp_values    = importances_df[present_features]
mean_imp      = imp_values.mean().sort_values(ascending=False)
top_features  = mean_imp.head(15).index.tolist()

print("  Top 15 features by mean permutation importance:")
for i, (feat, val) in enumerate(mean_imp.head(15).items(), 1):
    print(f"    {i:2d}. {feat:<35} imp={val:.5f}")

# ── 9. PLOTS ──────────────────────────────────────────────────────────────────
hline()
print("STEP 9: Generating plots")
hline()

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Random Forest Rolling Window Results", fontsize=14, fontweight="bold")

# Plot 1 — Cumulative L-S return (log scale)
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

# Plot 3 — Feature importance heatmap (permutation-based)
ax = axes[1, 0]
heatmap_data = importances_df.set_index("pred_year")[top_features[:10]]
im = ax.imshow(heatmap_data.T, aspect="auto", cmap="YlOrRd",
               extent=[heatmap_data.index.min(), heatmap_data.index.max(),
                       -0.5, len(top_features[:10]) - 0.5])
ax.set_yticks(range(len(top_features[:10])))
ax.set_yticklabels(top_features[:10], fontsize=7)
ax.set_title("Feature Importance Heatmap (permutation)")
ax.set_xlabel("Prediction Year")
plt.colorbar(im, ax=ax, shrink=0.8)

# Plot 4 — Decile returns bar chart
ax = axes[1, 1]
ax.bar(decile_summary_df["Decile"], decile_summary_df["Mean_Monthly"],
       color="steelblue", alpha=0.8)
ax.set_title("Mean Monthly Return by Predicted Decile")
ax.set_xlabel("Decile (1=lowest predicted, 10=highest predicted)")
ax.set_ylabel("Mean Monthly Return")
ax.set_xticks(range(1, 11))
ax.axhline(0, color="black", lw=0.5)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "random_forest_results_rf.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved → {OUT_DIR / 'random_forest_results_rf.png'}")

# ── 10. DECADE FEATURE IMPORTANCE ─────────────────────────────────────────────
hline()
print("STEP 10: Feature importance by decade")
hline()

importances_df["decade"] = (importances_df["pred_year"] // 10) * 10
decade_summary = []
for decade, grp in importances_df.groupby("decade"):
    mean_imp_dec = grp[present_features].mean().sort_values(ascending=False)
    top5         = mean_imp_dec.head(5)
    print(f"\n  {decade}s — top 5 predictors:")
    for feat, val in top5.items():
        print(f"    {feat:<35} imp={val:.5f}")
    for rank, (feat, val) in enumerate(top5.items(), 1):
        decade_summary.append({
            "decade": decade, "rank": rank,
            "feature": feat, "mean_importance": val
        })

pd.DataFrame(decade_summary).to_csv(OUT_DIR / "decade_importance_rf.csv", index=False)

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
hline()
print("FINAL SUMMARY")
hline()
print(f"  Windows completed  : {len(metrics_df)}")
print(f"  Mean OOS R²        : {metrics_df['oos_r2'].mean():.5f}")
print(f"  L-S monthly return : {port_df['ls_dec'].mean():.4f}")
ls_sharpe = perf_df.loc[perf_df["Label"] == "Long-Short (D10-D1)", "Sharpe"].values[0]
print(f"  L-S Sharpe         : {ls_sharpe:.4f}")
print(f"  Hyperparameters    : tuned per decade (see decade_hyperparameters.csv)")
print(f"  Output saved to    : {OUT_DIR}")
hline()