"""
Data Quality & Return Validation
Reads merged_data.parquet produced by merge_ccm.py and runs all
validation checks requested by promotor.

Run from: Master_Thesis/scripts/
Output:   Master_Thesis/output/  (created automatically)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── PATHS ─────────────────────────────────────────────────────────────────────
MERGED_PATH  = "../data_clean/merged_data.parquet"
FF_RF_PATH   = "../data_raw/ff_factors_monthly.csv"   # Ken French file (see note below)
OUT_DIR      = Path("../output")
OUT_DIR.mkdir(exist_ok=True)

# ── COLUMN NAMES (exact names from your pipeline) ─────────────────────────────
RET_COL    = "RET_ADJ"
DATE_COL   = "date"
PERMNO_COL = "PERMNO"

# All predictors — names taken directly from your cleaning scripts
PREDICTOR_COLS = [
    # CRSP monthly
    "SIZE", "mvel1", "mom1m", "MOM", "chmom",
    "turn", "dolvol", "spread", "dy", "indmom",
    # CRSP daily-derived
    "retvol", "maxret", "minret", "skew_daily", "kurt_daily",
    # Compustat
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

WINSOR_LOW  = 0.01
WINSOR_HIGH = 0.99

# ── HELPERS ───────────────────────────────────────────────────────────────────
def cross_sectional_winsorize(series, low=WINSOR_LOW, high=WINSOR_HIGH):
    """Clip within each month's cross-section — no look-ahead bias."""
    lo = series.quantile(low)
    hi = series.quantile(high)
    return series.clip(lo, hi)

def desc_stats(series, name):
    s = series.dropna()
    return {
        "Variable"  : name,
        "N"         : len(s),
        "Missing_%" : round(series.isna().mean() * 100, 2),
        "Mean"      : s.mean(),
        "Median"    : s.median(),
        "Std"       : s.std(),
        "Min"       : s.min(),
        "P1"        : s.quantile(0.01),
        "P99"       : s.quantile(0.99),
        "Max"       : s.max(),
        "Skewness"  : s.skew(),
        "Kurtosis"  : s.kurtosis(),
    }

def hline():
    print("─" * 60)

# ══════════════════════════════════════════════════════════════════
# STEP 1 — LOAD
# ══════════════════════════════════════════════════════════════════
hline()
print("STEP 1: Loading merged_data.parquet")
hline()

df = pd.read_parquet(MERGED_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values([PERMNO_COL, DATE_COL]).reset_index(drop=True)

print(f"  Rows          : {len(df):,}")
print(f"  Unique firms  : {df[PERMNO_COL].nunique():,}")
print(f"  Date range    : {df[DATE_COL].min().date()} → {df[DATE_COL].max().date()}")
print(f"  Columns       : {df.shape[1]}")

# Check which predictors are actually present
present = [c for c in PREDICTOR_COLS if c in df.columns]
absent  = [c for c in PREDICTOR_COLS if c not in df.columns]
print(f"\n  Predictors found : {len(present)}/{len(PREDICTOR_COLS)}")
if absent:
    print(f"  Not found        : {absent}")

# ══════════════════════════════════════════════════════════════════
# STEP 2 — RAW RETURN SANITY CHECK
# ══════════════════════════════════════════════════════════════════
hline()
print("STEP 2: Raw return sanity check")
hline()

ret_raw = df[RET_COL].dropna()

print(f"  Count           : {len(ret_raw):,}")
print(f"  Missing %%       : {df[RET_COL].isna().mean()*100:.2f}%%")
print(f"  Mean monthly    : {ret_raw.mean():.4f}  (expect ~0.008–0.011)")
print(f"  Median          : {ret_raw.median():.4f}")
print(f"  Std             : {ret_raw.std():.4f}")
print(f"  Min             : {ret_raw.min():.4f}")
print(f"  Max             : {ret_raw.max():.4f}")
print(f"  Skewness        : {ret_raw.skew():.4f}  (expect slightly positive at stock level)")
print(f"  Kurtosis        : {ret_raw.kurtosis():.4f}  (expect > 0, fat tails)")

# Flag issues
if abs(ret_raw.mean()) > 0.05:
    print("  ⚠  Mean return too extreme — check construction")
if ret_raw.max() > 10 or ret_raw.min() < -1:
    print("  ⚠  Extreme outliers detected — winsorization needed")
if ret_raw.min() < -1:
    print("  ⚠  Returns below -100%% found — check delisting adjustment")
else:
    print("  ✓  No returns below -100%% (delisting adjustment looks correct)")

# ══════════════════════════════════════════════════════════════════
# STEP 3 — WINSORIZATION
# ══════════════════════════════════════════════════════════════════
hline()
print("STEP 3: Cross-sectional winsorization")
hline()

# Winsorize RET_ADJ cross-sectionally each month
df["RET_WIN"] = (
    df.groupby(DATE_COL)[RET_COL]
    .transform(cross_sectional_winsorize)
)

ret_win = df["RET_WIN"].dropna()
n_affected = (df[RET_COL].dropna() != df["RET_WIN"].dropna()).sum()

print(f"  Return winsorization bounds  : [{WINSOR_LOW*100:.0f}th, {WINSOR_HIGH*100:.0f}th] percentile per month")
print(f"  Observations affected        : {n_affected:,} ({n_affected/len(ret_raw)*100:.2f}%%)")
print(f"  Raw  → Win  mean : {ret_raw.mean():.5f} → {ret_win.mean():.5f}")
print(f"  Raw  → Win  std  : {ret_raw.std():.5f} → {ret_win.std():.5f}")
print(f"  Raw  → Win  max  : {ret_raw.max():.4f} → {ret_win.max():.4f}")
print(f"  Raw  → Win  min  : {ret_raw.min():.4f} → {ret_win.min():.4f}")

# Winsorize all predictors cross-sectionally
print(f"\n  Winsorizing {len(present)} predictors...")
for col in present:
    df[col + "_win"] = (
        df.groupby(DATE_COL)[col]
        .transform(cross_sectional_winsorize)
    )
print("  ✓ Done")

# ══════════════════════════════════════════════════════════════════
# STEP 4 — EQUAL-WEIGHTED PORTFOLIO TIME SERIES
# ══════════════════════════════════════════════════════════════════
hline()
print("STEP 4: Equal-weighted portfolio returns")
hline()

ew = (
    df.groupby(DATE_COL)["RET_WIN"]
    .agg(ew_ret="mean", n_stocks="count")
    .reset_index()
)
ew["cum_ret"] = (1 + ew["ew_ret"]).cumprod()

print(f"  Mean monthly EW return : {ew['ew_ret'].mean():.4f}")
print(f"  Std  monthly EW return : {ew['ew_ret'].std():.4f}")
print(f"  Avg stocks per month   : {ew['n_stocks'].mean():.0f}")
print(f"  Min stocks per month   : {ew['n_stocks'].min()}")
print(f"  Max stocks per month   : {ew['n_stocks'].max()}")

# Known crash month checks — signs must be negative
print("\n  Crash month verification (expect negative returns):")
crash_checks = {
    "Oct 1987 (Black Monday)" : "1987-10",
    "Aug 1998 (LTCM crisis)"  : "1998-08",
    "Sep 2002 (dot-com trough)": "2002-09",
    "Oct 2008 (Lehman)"       : "2008-10",
    "Mar 2020 (COVID)"        : "2020-03",
}
for label, ym in crash_checks.items():
    row = ew[ew[DATE_COL].dt.to_period("M").astype(str) == ym]
    if not row.empty:
        val = row["ew_ret"].values[0]
        flag = "✓" if val < 0 else "⚠ "
        print(f"    {flag} {label}: {val:.4f}")
    else:
        print(f"    –  {label}: not in sample")

# ══════════════════════════════════════════════════════════════════
# STEP 5 — SHARPE RATIO
# Ken French data: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
# Download "Fama/French 3 Factors" → save as data_raw/ff_factors_monthly.csv
# The file has columns: date(YYYYMM), Mkt-RF, SMB, HML, RF
# ══════════════════════════════════════════════════════════════════
hline()
print("STEP 5: Sharpe ratio")
hline()

try:
    ff = pd.read_csv(FF_RF_PATH, skiprows=3)
    ff.columns = ff.columns.str.strip()

    date_col_ff = ff.columns[0]
    ff = ff.rename(columns={date_col_ff: "ff_date"})

    # Keep only 6-digit YYYYMM rows — removes annual summary footer
    ff = ff[ff["ff_date"].astype(str).str.strip().str.match(r"^\d{6}$")].copy()
    ff["ff_date"] = pd.to_datetime(ff["ff_date"].astype(int).astype(str), format="%Y%m")
    ff["RF"] = pd.to_numeric(ff["RF"], errors="coerce") / 100

    # Merge on year-month
    ew["ym"] = ew[DATE_COL].dt.to_period("M")
    ff["ym"] = ff["ff_date"].dt.to_period("M")
    ew = ew.merge(ff[["ym", "RF"]], on="ym", how="left").drop(columns="ym")
    ew["excess_ret"] = ew["ew_ret"] - ew["RF"]

    sharpe = (ew["excess_ret"].mean() / ew["excess_ret"].std()) * np.sqrt(12)
    print(f"  Annualized Sharpe (EW portfolio) : {sharpe:.4f}")
    print(f"  Mean monthly excess return       : {ew['excess_ret'].mean():.4f}")
    print(f"  (expect Sharpe 0.3–0.6 for broad US equity)")

    if sharpe > 1.5:
        print("  ⚠  Sharpe too high — possible look-ahead or survivorship bias")
    elif sharpe < 0:
        print("  ⚠  Negative Sharpe — check return signs or RF merge")
    else:
        print("  ✓  Sharpe ratio looks economically plausible")

except FileNotFoundError:
    print(f"  ⚠  {FF_RF_PATH} not found.")
    print("     Download 'F-F_Research_Data_Factors.CSV' from Ken French's website")
    print("     and save to data_raw/ff_factors_monthly.csv, then rerun.")
    sharpe = np.nan
    ew["RF"] = np.nan
    ew["excess_ret"] = np.nan

# ══════════════════════════════════════════════════════════════════
# STEP 6 — BM UNITS CHECK
# ══════════════════════════════════════════════════════════════════
hline()
print("STEP 6: Book-to-market units check")
hline()

if "bm" in df.columns:
    bm = df["bm"].dropna()
    print(f"  BM mean   : {bm.mean():.4f}  (expect ~0.5–0.8 for US stocks)")
    print(f"  BM median : {bm.median():.4f}  (expect ~0.4–0.6)")
    print(f"  BM p1     : {bm.quantile(0.01):.4f}")
    print(f"  BM p99    : {bm.quantile(0.99):.4f}")
    print(f"  BM max    : {bm.max():.4f}")

    if bm.median() > 5:
        print("  ⚠  BM median too large — units mismatch. Check if BE needs /1000")
        print("     In merge_ccm.py change: merged['bm'] = merged['be'] / merged['ME_lag1']")
        print("     (remove the *1000 since both BE and ME_lag1 are already in $M)")
    elif bm.median() < 0.01:
        print("  ⚠  BM median too small — units mismatch in opposite direction")
    else:
        print("  ✓  BM values look plausible")
else:
    print("  ⚠  'bm' column not found in merged data")

# ══════════════════════════════════════════════════════════════════
# STEP 7 — DESCRIPTIVE STATISTICS TABLE
# ══════════════════════════════════════════════════════════════════
hline()
print("STEP 7: Descriptive statistics table")
hline()

rows = []
rows.append(desc_stats(df[RET_COL],   "RET_ADJ (raw)"))
rows.append(desc_stats(df["RET_WIN"], "RET_ADJ (winsorized)"))

for col in present:
    rows.append(desc_stats(df[col + "_win"], col))

stats_table = pd.DataFrame(rows)
float_cols  = ["Mean","Median","Std","Min","P1","P99","Max","Skewness","Kurtosis"]
stats_table[float_cols] = stats_table[float_cols].round(4)

out_csv = OUT_DIR / "descriptive_stats.csv"
stats_table.to_csv(out_csv, index=False)
print(f"  Saved → {out_csv}")
print()
print(stats_table[["Variable","Mean","Std","Skewness","Kurtosis","Missing_%"]].to_string(index=False))

# ══════════════════════════════════════════════════════════════════
# STEP 8 — VALIDATION PLOTS
# ══════════════════════════════════════════════════════════════════
hline()
print("\nSTEP 8: Generating validation plots")
hline()

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Return Validation Dashboard", fontsize=14, fontweight="bold")

# Plot 1 — Cumulative EW return
ax = axes[0, 0]
ax.plot(ew[DATE_COL], ew["cum_ret"], lw=1, color="steelblue")
ax.set_title("Equal-Weighted Cumulative Return")
ax.set_ylabel("Growth of $1")
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.grid(alpha=0.3)

# Plot 2 — Monthly EW returns bar chart
ax = axes[0, 1]
colors = np.where(ew["ew_ret"] >= 0, "steelblue", "salmon")
ax.bar(ew[DATE_COL], ew["ew_ret"], color=colors, width=20, linewidth=0)
ax.axhline(0, color="black", lw=0.5)
ax.set_title("Equal-Weighted Monthly Returns")
ax.set_ylabel("Monthly Return")
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.grid(alpha=0.3)

# Plot 3 — Return distribution raw vs winsorized
ax = axes[1, 0]
clip_lo, clip_hi = -0.5, 1.0   # display range only (not data)
ax.hist(df[RET_COL].dropna().clip(clip_lo, clip_hi),
        bins=300, alpha=0.5, color="salmon",    label="Raw",        density=True)
ax.hist(df["RET_WIN"].dropna(),
        bins=300, alpha=0.5, color="steelblue", label="Winsorized", density=True)
ax.set_title("Return Distribution: Raw vs Winsorized")
ax.set_xlabel("Monthly Return")
ax.set_ylabel("Density")
ax.set_xlim(clip_lo, clip_hi)
ax.legend()
ax.grid(alpha=0.3)

# Plot 4 — Stock count per month
ax = axes[1, 1]
ax.plot(ew[DATE_COL], ew["n_stocks"], lw=1, color="steelblue")
ax.set_title("Stocks in Universe per Month")
ax.set_ylabel("Count")
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.grid(alpha=0.3)
# Annotate expected shape
ax.annotate("Peak ~8,000\n(late 1990s)", xy=(pd.Timestamp("1997-01-01"), ew["n_stocks"].max()),
            xytext=(pd.Timestamp("1980-01-01"), ew["n_stocks"].max() * 0.9),
            arrowprops=dict(arrowstyle="->", color="gray"), fontsize=8, color="gray")

plt.tight_layout()
out_plot = OUT_DIR / "validation_plots.png"
plt.savefig(out_plot, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved → {out_plot}")

# ══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════
hline()
print("VALIDATION SUMMARY")
hline()
print(f"  Firm-month observations : {len(df):,}")
print(f"  Date range              : {df[DATE_COL].min().date()} → {df[DATE_COL].max().date()}")
print(f"  Unique firms            : {df[PERMNO_COL].nunique():,}")
print(f"  Mean monthly EW return  : {ew['ew_ret'].mean():.4f}")
print(f"  Annualized Sharpe       : {sharpe:.4f}" if not np.isnan(sharpe) else
      f"  Annualized Sharpe       : N/A (add ff_factors_monthly.csv)")
print(f"  Predictors validated    : {len(present)}/{len(PREDICTOR_COLS)}")
print(f"  Stats table             : {out_csv}")
print(f"  Plots                   : {out_plot}")
hline()