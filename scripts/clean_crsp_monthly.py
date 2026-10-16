import pandas as pd
import numpy as np

# ----------------------------
# Load CRSP monthly data
# ----------------------------
crsp = pd.read_csv("../data_raw/CRSP Monthly Stock File.csv", low_memory=False)

# Standardize column names (CRSP often uppercase)
crsp.columns = [c.strip() for c in crsp.columns]
print(f"Raw observations: {len(crsp):,}")

# Convert key columns safely
for col in ["RET", "DLRET", "DLSTCD", "PRC", "SHROUT", "VOL", "RETX", "BIDLO", "ASKHI", "SICCD"]:
    if col in crsp.columns:
        crsp[col] = pd.to_numeric(crsp[col], errors="coerce")

# Keep only common shares & major exchanges if columns exist
if "SHRCD" in crsp.columns:
    crsp = crsp[crsp["SHRCD"].isin([10, 11])]
    print(f"After share code filter (10, 11): {len(crsp):,}")
if "EXCHCD" in crsp.columns:
    crsp = crsp[crsp["EXCHCD"].isin([1, 2, 3])]
    print(f"After exchange filter (NYSE, AMEX, NASDAQ): {len(crsp):,}")

# Fix negative prices (CRSP uses negative PRC as bid/ask flag)
if "PRC" in crsp.columns:
    crsp["PRC"] = crsp["PRC"].abs()

# Date
crsp["date"] = pd.to_datetime(crsp["date"], errors="coerce")
crsp = crsp.dropna(subset=["date"])

# Market equity (SHROUT is in thousands)
# ME = |PRC| * SHROUT * 1000
crsp["ME"] = crsp["PRC"] * crsp["SHROUT"] * 1000
crsp = crsp[crsp["ME"] > 0]
print(f"After ME > 0 filter: {len(crsp):,}")

# ----------------------------
# Delisting return adjustment
# FIX: Treat missing DLRET as -30% for performance-related delistings
# (DLSTCD 500-584), and 0% otherwise (Shumway 1997)
# ----------------------------
crsp["DLRET_ADJ"] = crsp["DLRET"]
if "DLSTCD" in crsp.columns:
    perf_delist = crsp["DLSTCD"].between(500, 584)
    crsp.loc[perf_delist & crsp["DLRET"].isna(), "DLRET_ADJ"] = -0.30
crsp["DLRET_ADJ"] = crsp["DLRET_ADJ"].fillna(0.0)

crsp["RET_ADJ"] = (1 + crsp["RET"]).fillna(1.0) * (1 + crsp["DLRET_ADJ"]) - 1

# Drop if returns missing after adjustment
crsp = crsp.dropna(subset=["RET_ADJ", "ME"])
print(f"After dropping missing returns: {len(crsp):,}")

# Sort
crsp = crsp.sort_values(["PERMNO", "date"]).reset_index(drop=True)

# Compute log returns (only valid if RET_ADJ > -1)
crsp["log_ret"] = np.nan
mask = crsp["RET_ADJ"] > -1
crsp.loc[mask, "log_ret"] = np.log1p(crsp.loc[mask, "RET_ADJ"])

# ----------------------------
# SIZE: use lagged ME to avoid look-ahead bias
# ----------------------------
crsp["ME_lag1"] = crsp.groupby("PERMNO")["ME"].shift(1)
crsp["SIZE"] = np.log(crsp["ME_lag1"])

# Keep mvel1 (lagged market value) as separate feature - top GKX predictor
crsp["mvel1"] = crsp["ME_lag1"]

# ----------------------------
# NEW: mom1m - lagged 1-month return
# THE #1 most important predictor in GKX paper
# ----------------------------
crsp["mom1m"] = crsp.groupby("PERMNO")["RET_ADJ"].shift(1)

# ----------------------------
# Momentum: cumulative return from t-12 to t-2 (skip t-1)
# ----------------------------
crsp["MOM"] = (
    crsp.groupby("PERMNO")["log_ret"]
    .shift(1)
    .rolling(window=11, min_periods=11)
    .sum()
    .reset_index(level=0, drop=True)
)
crsp["MOM"] = np.expm1(crsp["MOM"])

# ----------------------------
# NEW: chmom - change in momentum (6m vs 12m)
# ----------------------------
crsp["mom6m_temp"] = (
    crsp.groupby("PERMNO")["log_ret"]
    .shift(1)
    .rolling(6, min_periods=6)
    .sum()
    .reset_index(level=0, drop=True)
)
crsp["mom6m_temp"] = np.expm1(crsp["mom6m_temp"])
crsp["chmom"] = crsp["mom6m_temp"] - crsp["MOM"]
crsp = crsp.drop(columns=["mom6m_temp"])

# ----------------------------
# Return horizons
# ----------------------------
crsp["RET_1M"] = crsp["RET_ADJ"]

crsp["RET_3M"] = (
    crsp.groupby("PERMNO")["log_ret"]
    .shift(1)
    .rolling(3, min_periods=3)
    .apply(lambda s: np.expm1(np.nansum(s)), raw=False)
    .reset_index(level=0, drop=True)
)

crsp["RET_6M"] = (
    crsp.groupby("PERMNO")["log_ret"]
    .shift(1)
    .rolling(6, min_periods=6)
    .apply(lambda s: np.expm1(np.nansum(s)), raw=False)
    .reset_index(level=0, drop=True)
)

# ----------------------------
# 12M volatility from monthly returns (will be replaced by daily volatility later)
# ----------------------------
crsp["VOL_12M"] = (
    crsp.groupby("PERMNO")["RET_ADJ"]
    .shift(1)
    .rolling(12, min_periods=12)
    .std()
    .reset_index(level=0, drop=True)
)

# ----------------------------
# NEW: Liquidity measures
# ----------------------------

# Share turnover: VOL / SHROUT (SHROUT in thousands)
if "VOL" in crsp.columns and "SHROUT" in crsp.columns:
    crsp["turn"] = crsp["VOL"] / (crsp["SHROUT"] * 1000)

# Dollar volume: |PRC| * VOL
if "VOL" in crsp.columns:
    crsp["dolvol"] = crsp["PRC"] * crsp["VOL"]

# Bid-ask spread: (ASKHI - BIDLO) / midpoint
if "BIDLO" in crsp.columns and "ASKHI" in crsp.columns:
    midpoint = (crsp["ASKHI"] + crsp["BIDLO"]) / 2
    crsp["spread"] = (crsp["ASKHI"] - crsp["BIDLO"]) / midpoint
    crsp["spread"] = crsp["spread"].replace([np.inf, -np.inf], np.nan)

# ----------------------------
# NEW: Dividend yield
# dy = (RET - RETX) approximates monthly dividend yield
# ----------------------------
if "RETX" in crsp.columns:
    crsp["dy"] = crsp["RET"] - crsp["RETX"]
    # Fill missing with 0 (no dividend)
    crsp["dy"] = crsp["dy"].fillna(0.0)

# ----------------------------
# NEW: Industry momentum
# Average return by SICCD in each month
# ----------------------------
if "SICCD" in crsp.columns:
    # Compute lagged industry average return
    ind_avg = (
        crsp.groupby(["SICCD", "date"])["RET_ADJ"]
        .mean()
        .reset_index()
        .rename(columns={"RET_ADJ": "indmom"})
    )
    crsp = crsp.merge(ind_avg, on=["SICCD", "date"], how="left")
    # Lag by 1 month to avoid look-ahead bias
    crsp["indmom"] = crsp.groupby("PERMNO")["indmom"].shift(1)

# ----------------------------
# Keep relevant columns
# ----------------------------
keep_cols = [
    "PERMNO", "date", "CUSIP", "SICCD", "EXCHCD",
    "PRC", "SHROUT", "ME", "ME_lag1",
    "RET", "DLRET", "DLRET_ADJ", "RET_ADJ",
    # Core features
    "SIZE", "mvel1", "mom1m", "MOM", "chmom",
    "RET_1M", "RET_3M", "RET_6M", "VOL_12M",
    # Liquidity
    "turn", "dolvol", "spread", "dy",
    # Industry
    "indmom"
]
keep_cols = [c for c in keep_cols if c in crsp.columns]
crsp = crsp[keep_cols].copy()

# Save
crsp.to_parquet("../data_clean/crsp_monthly_clean.parquet", index=False)
print("CRSP monthly cleaned and saved to ../data_clean/crsp_monthly_clean.parquet")
print(f"\nFeatures created: {len(keep_cols) - 9}")  # Subtract identifiers and raw columns
print("Ready to merge with daily features using PERMNO + date")