import pandas as pd
import numpy as np

# ----------------------------
# Load CRSP daily data
# ----------------------------
print("Loading daily CRSP data...")
crsp_daily = pd.read_parquet("../data_raw/CRSP_Daily_Stock_File.parquet")

# Standardize column names
crsp_daily.columns = [c.strip() for c in crsp_daily.columns]

# Convert key columns safely
for col in ["RET", "PRC", "VOL", "SHROUT", "BIDLO", "ASKHI", "OPENPRC", "RETX"]:
    if col in crsp_daily.columns:
        crsp_daily[col] = pd.to_numeric(crsp_daily[col], errors="coerce")

print(f"Raw data: {len(crsp_daily):,} rows")

# Keep only common shares & major exchanges
if "SHRCD" in crsp_daily.columns:
    crsp_daily = crsp_daily[crsp_daily["SHRCD"].isin([10, 11])]
if "EXCHCD" in crsp_daily.columns:
    crsp_daily = crsp_daily[crsp_daily["EXCHCD"].isin([1, 2, 3])]

print(f"After filters: {len(crsp_daily):,} rows")

# Fix negative prices (CRSP uses negative PRC as bid/ask flag)
if "PRC" in crsp_daily.columns:
    crsp_daily["PRC"] = crsp_daily["PRC"].abs()

# Date
crsp_daily["date"] = pd.to_datetime(crsp_daily["date"], errors="coerce")
crsp_daily = crsp_daily.dropna(subset=["date", "PERMNO"])

# Create year-month identifier for aggregation
crsp_daily["year_month"] = crsp_daily["date"].dt.to_period("M")

# Sort by PERMNO and date
crsp_daily = crsp_daily.sort_values(["PERMNO", "date"]).reset_index(drop=True)

# ----------------------------
# Clean returns for calculations
# Valid returns must be > -1 (can't lose more than 100%)
# ----------------------------
crsp_daily["RET_clean"] = crsp_daily["RET"]
crsp_daily.loc[crsp_daily["RET_clean"] <= -1, "RET_clean"] = np.nan

# ----------------------------
# Pre-compute daily variables needed for aggregation
# ----------------------------
crsp_daily["daily_dolvol"] = crsp_daily["PRC"] * crsp_daily["VOL"]

if "SHROUT" in crsp_daily.columns:
    crsp_daily["daily_turn"] = crsp_daily["VOL"] / (crsp_daily["SHROUT"] * 1000)

if "BIDLO" in crsp_daily.columns and "ASKHI" in crsp_daily.columns:
    midpoint = (crsp_daily["ASKHI"] + crsp_daily["BIDLO"]) / 2
    crsp_daily["daily_spread"] = (crsp_daily["ASKHI"] - crsp_daily["BIDLO"]) / midpoint
    crsp_daily["daily_spread"] = crsp_daily["daily_spread"].replace([np.inf, -np.inf], np.nan)

# ----------------------------
# Define named function for kurtosis to avoid lambda naming issues
# ----------------------------
def kurtosis_func(x):
    return x.kurtosis()

# ----------------------------
# FAST VECTORIZED AGGREGATION
# Use built-in aggregation functions instead of apply()
# ----------------------------
print("Computing monthly features (vectorized)...")

agg_dict = {
    "RET_clean": ["count", "std", "max", "min", "skew", kurtosis_func],
    "daily_dolvol": "mean",
}

if "daily_turn" in crsp_daily.columns:
    agg_dict["daily_turn"] = "mean"

if "daily_spread" in crsp_daily.columns:
    agg_dict["daily_spread"] = "mean"

monthly_features = crsp_daily.groupby(["PERMNO", "year_month"]).agg(agg_dict)

# Flatten multi-level column names
monthly_features.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                            for col in monthly_features.columns.values]

# Rename to meaningful names
rename_map = {
    "RET_clean_count": "n_days",
    "RET_clean_std": "retvol_daily_raw",
    "RET_clean_max": "maxret",
    "RET_clean_min": "minret",
    "RET_clean_skew": "skew_daily",
    "RET_clean_kurtosis_func": "kurt_daily",
    "daily_dolvol_mean": "dolvol_daily",
    "daily_turn_mean": "turn_daily",
    "daily_spread_mean": "spread_daily",
}

monthly_features = monthly_features.rename(columns=rename_map)
monthly_features = monthly_features.reset_index()

# ----------------------------
# Annualize volatility: daily_std * sqrt(252)
# ----------------------------
monthly_features["retvol_daily"] = monthly_features["retvol_daily_raw"] * np.sqrt(252)
monthly_features = monthly_features.drop(columns=["retvol_daily_raw"])

# ----------------------------
# Filter: require at least 15 trading days
# ----------------------------
monthly_features = monthly_features[monthly_features["n_days"] >= 15].copy()

# Convert year_month back to timestamp (end of month)
monthly_features["date"] = monthly_features["year_month"].dt.to_timestamp(how="end")
monthly_features = monthly_features.drop(columns=["year_month"])

# ----------------------------
# Save daily-derived monthly features
# ----------------------------
monthly_features = monthly_features.sort_values(["PERMNO", "date"])
monthly_features.to_parquet("../data_clean/crsp_daily_clean.parquet", index=False)

print("\n==============================")
print("DAILY CRSP FEATURES COMPUTED")
print("==============================")
print("Shape:", monthly_features.shape)
print(f"PERMNOs: {monthly_features['PERMNO'].nunique():,}")
print(f"Months: {monthly_features['date'].nunique():,}")
print("\nFeatures created:")
for col in monthly_features.columns:
    if col not in ["PERMNO", "date", "n_days"]:
        print(f"  - {col}")

print("\nSaved to: ../data_clean/crsp_daily_clean.parquet")
print("\nNext step: Merge these features to monthly CRSP data using PERMNO + date")