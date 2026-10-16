import pandas as pd
import numpy as np

# --------------------------------------------------
# Load cleaned CRSP (with daily features) and Compustat
# --------------------------------------------------

print("Loading CRSP monthly (with daily features)...")
crsp = pd.read_parquet("../data_clean/crsp_monthly_with_daily.parquet")
print(f"  CRSP: {len(crsp):,} rows, {crsp['PERMNO'].nunique():,} PERMNOs")

print("\nLoading Compustat...")
comp = pd.read_parquet("../data_clean/compustat_clean.parquet")
print(f"  Compustat: {len(comp):,} rows, {comp['gvkey'].nunique():,} GVKEYs")

# --------------------------------------------------
# Helper: clean CUSIP to 8-digit string
# --------------------------------------------------

def clean_cusip8(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s.endswith(".0"):   # remove .0 from float-converted strings
        s = s[:-2]
    s = s.replace(" ", "")
    return s[:8]

# --------------------------------------------------
# Ensure correct types
# --------------------------------------------------

crsp["date"] = pd.to_datetime(crsp["date"])
comp["datadate"] = pd.to_datetime(comp["datadate"])

# Handle possible casing difference in CRSP
if "cusip" in crsp.columns:
    crsp["cusip8"] = crsp["cusip"].apply(clean_cusip8)
elif "CUSIP" in crsp.columns:
    crsp["cusip8"] = crsp["CUSIP"].apply(clean_cusip8)
else:
    raise ValueError("CUSIP column not found in CRSP data")

comp["cusip8"] = comp["cusip"].apply(clean_cusip8)

# --------------------------------------------------
# Avoid look-ahead bias
# merge_date already computed in compustat_clean.py
# --------------------------------------------------

comp = comp.rename(columns={"merge_date": "available_date"})

# Drop rows with missing merge keys
crsp = crsp.dropna(subset=["cusip8", "date"]).copy()
comp = comp.dropna(subset=["cusip8", "available_date"]).copy()

print(f"\nAfter dropping missing CUSIP:")
print(f"  CRSP: {len(crsp):,} rows")
print(f"  Compustat: {len(comp):,} rows")

# Select comp variables to merge (exclude identifiers)
comp_cols = [
    c for c in comp.columns
    if c not in ["gvkey", "cusip", "cusip8", "datadate", "available_date"]
]

# --------------------------------------------------
# Merge CRSP + Compustat
# Use merge_asof with 18-month tolerance to avoid stale data
# --------------------------------------------------

print("\nMerging CRSP + Compustat...")

crsp = crsp.sort_values("date")
comp = comp.sort_values("available_date")

merged = pd.merge_asof(
    crsp,
    comp[["cusip8", "available_date"] + comp_cols],
    left_on="date",
    right_on="available_date",
    by="cusip8",
    direction="backward",
    allow_exact_matches=True,
    tolerance=pd.Timedelta(days=548)  # ~18 months
).drop(columns=["available_date"], errors="ignore")

# --------------------------------------------------
# Compute book-to-market ratio (needs both BE and ME)
# --------------------------------------------------

print("Computing book-to-market ratio...")

# Use ME_lag1 (lagged market equity) to avoid look-ahead bias
if "be" in merged.columns and "ME_lag1" in merged.columns:
    merged["bm"] = (merged["be"] * 1_000_000) / merged["ME_lag1"]
    print("  ✓ Book-to-market (bm) computed")
else:
    print("  ✗ Cannot compute bm - missing be or ME_lag1")

# --------------------------------------------------
# Quality checks
# --------------------------------------------------

n_total = len(merged)
n_matched = merged["be"].notna().sum()
match_rate = n_matched / n_total

print(f"\n{'='*50}")
print("MERGE QUALITY CHECKS")
print(f"{'='*50}")
print(f"Compustat match rate: {match_rate:.1%} ({n_matched:,} of {n_total:,} rows)")
print(f"Rows without Compustat data: {n_total - n_matched:,}")

if "bm" in merged.columns:
    n_bm = merged["bm"].notna().sum()
    print(f"Book-to-market computed: {n_bm:,} rows ({n_bm/n_total:.1%})")

# --------------------------------------------------
# Add zero-fill for sparse predictors
# --------------------------------------------------

zero_fill_cols = [
    "rd_intensity", "ad_intensity",
    "gdwl_growth", "intan_growth"
]
for c in zero_fill_cols:
    if c in merged.columns:
        merged[c] = merged[c].fillna(0)

# --------------------------------------------------
# Save merged dataset
# --------------------------------------------------

merged = merged.sort_values(["PERMNO", "date"])
merged.to_parquet("../data_clean/merged_data.parquet", index=False)

print(f"\n{'='*50}")
print("FINAL MERGED DATASET")
print(f"{'='*50}")
print(f"Shape: {merged.shape}")
print(f"PERMNOs: {merged['PERMNO'].nunique():,}")
print(f"Date range: {merged['date'].min()} to {merged['date'].max()}")
print(f"Total features: {len(merged.columns)}")

print("\nFeature breakdown:")
crsp_features = [c for c in merged.columns if c in crsp.columns and c not in ["PERMNO", "date", "CUSIP", "SICCD", "cusip8"]]
comp_features = [c for c in merged.columns if c in comp_cols]
derived_features = ["bm"] if "bm" in merged.columns else []

print(f"  CRSP features: {len(crsp_features)}")
print(f"  Compustat features: {len(comp_features)}")
print(f"  Derived features: {len(derived_features)}")
print(f"  Total predictors: {len(crsp_features) + len(comp_features) + len(derived_features)}")

print("\nSaved to: ../data_clean/merged_data.parquet")
print("\nReady for ML modeling!")