import pandas as pd
import numpy as np

# ----------------------------
# Load monthly and daily CRSP data
# ----------------------------
print("Loading CRSP monthly features...")
crsp_monthly = pd.read_parquet("../data_clean/crsp_monthly_clean.parquet")
print(f"  Monthly: {len(crsp_monthly):,} rows, {crsp_monthly['PERMNO'].nunique():,} PERMNOs")

print("\nLoading CRSP daily-derived features...")
crsp_daily = pd.read_parquet("../data_clean/crsp_daily_clean.parquet")
print(f"  Daily:   {len(crsp_daily):,} rows, {crsp_daily['PERMNO'].nunique():,} PERMNOs")

# ----------------------------
# Normalize dates to calendar month-end before merge
# Monthly has actual trading dates, daily has timestamp end-of-month
# Convert both to just year-month for matching
# ----------------------------
print("\nNormalizing dates to year-month...")
crsp_monthly["year_month"] = crsp_monthly["date"].dt.to_period("M")
crsp_daily["year_month"] = crsp_daily["date"].dt.to_period("M")

# ----------------------------
# Merge daily features into monthly
# Join on PERMNO + year_month
# ----------------------------
print("Merging daily features into monthly CRSP...")

# Daily features to merge (exclude merge keys and n_days)
daily_features = [c for c in crsp_daily.columns
                  if c not in ["PERMNO", "date", "n_days", "year_month"]]

merged = crsp_monthly.merge(
    crsp_daily[["PERMNO", "year_month"] + daily_features],
    on=["PERMNO", "year_month"],
    how="left",
    suffixes=("_monthly", "_daily")
)

# Drop the temporary year_month column
merged = merged.drop(columns=["year_month"])

# ----------------------------
# Handle overlapping columns
# Daily versions should REPLACE monthly versions for:
# - Volatility (retvol_daily > VOL_12M)
# - Turnover (turn_daily > turn)
# - Dollar volume (dolvol_daily > dolvol)
# - Spread (spread_daily > spread)
# ----------------------------

# Use daily volatility as primary volatility measure
if "retvol_daily" in merged.columns:
    merged = merged.rename(columns={"retvol_daily": "retvol"})
    merged = merged.drop(columns=["VOL_12M"], errors="ignore")
    print("  ✓ Using daily volatility (retvol) as primary measure")

# Handle duplicate liquidity measures (daily versions are more accurate)
if "turn_daily" in merged.columns and "turn_monthly" in merged.columns:
    # Daily turnover replaces monthly turnover
    merged["turn"] = merged["turn_daily"]
    merged = merged.drop(columns=["turn_monthly", "turn_daily"])
    print("  ✓ Daily turnover replaces monthly turnover")
elif "turn_daily" in merged.columns and "turn" in merged.columns:
    # Edge case: turn wasn't suffixed
    merged["turn"] = merged["turn_daily"]
    merged = merged.drop(columns=["turn_daily"])

if "dolvol_daily" in merged.columns and "dolvol_monthly" in merged.columns:
    # Daily dollar volume replaces monthly dollar volume
    merged["dolvol"] = merged["dolvol_daily"]
    merged = merged.drop(columns=["dolvol_monthly", "dolvol_daily"])
    print("  ✓ Daily dollar volume replaces monthly dollar volume")
elif "dolvol_daily" in merged.columns and "dolvol" in merged.columns:
    merged["dolvol"] = merged["dolvol_daily"]
    merged = merged.drop(columns=["dolvol_daily"])

if "spread_daily" in merged.columns and "spread_monthly" in merged.columns:
    # Daily spread replaces monthly spread
    merged["spread"] = merged["spread_daily"]
    merged = merged.drop(columns=["spread_monthly", "spread_daily"])
    print("  ✓ Daily bid-ask spread replaces monthly spread")
elif "spread_daily" in merged.columns and "spread" in merged.columns:
    merged["spread"] = merged["spread_daily"]
    merged = merged.drop(columns=["spread_daily"])

# ----------------------------
# Quality check: report match rate
# ----------------------------
n_total = len(merged)
n_with_daily = merged["maxret"].notna().sum()  # maxret only comes from daily data
match_rate = n_with_daily / n_total

print(f"\n  Match rate: {match_rate:.1%} ({n_with_daily:,} of {n_total:,} rows)")
print(f"  Rows without daily data: {n_total - n_with_daily:,}")
# ----------------------------
# Lag daily-derived features by 1 month to avoid look-ahead bias
# These features are computed from daily returns WITHIN month t,
# so they must be lagged to predict month t+1's return
# ----------------------------
daily_features_to_lag = [
    "retvol", "maxret", "minret", "skew_daily", "kurt_daily",
    "dolvol", "turn", "spread"
]
daily_features_to_lag = [c for c in daily_features_to_lag if c in merged.columns]

for c in daily_features_to_lag:
    merged[c] = merged.groupby("PERMNO")[c].shift(1)

print(f"  ✓ Lagged {len(daily_features_to_lag)} daily-derived features by 1 month")
print(f"    Features lagged: {daily_features_to_lag}")

# ----------------------------
# Save merged dataset
# ----------------------------
merged = merged.sort_values(["PERMNO", "date"])
merged.to_parquet("../data_clean/crsp_monthly_with_daily.parquet", index=False)

print("\n==============================")
print("CRSP MERGED (MONTHLY + DAILY)")
print("==============================")
print("Shape:", merged.shape)
print(f"PERMNOs: {merged['PERMNO'].nunique():,}")
print(f"Date range: {merged['date'].min()} to {merged['date'].max()}")

print("\nNew features from daily data:")
daily_only_features = [c for c in merged.columns if c in daily_features and c not in ["turn", "dolvol", "spread", "retvol"]]
for feat in daily_only_features:
    print(f"  - {feat}")

print("\nSaved to: ../data_clean/crsp_monthly_with_daily.parquet")
print("\nNext step: Merge with Compustat using CUSIP + available_date")