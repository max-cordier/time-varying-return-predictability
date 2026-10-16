import pandas as pd
import numpy as np

# ----------------------------
# Helpers
# ----------------------------
def safe_div(a, b):
    return np.where((b == 0) | (pd.isna(b)), np.nan, a / b)

def col(df, name):
    # return column if exists else nan series
    return df[name] if name in df.columns else np.nan

# ----------------------------
# Load Compustat annual fundamentals
# ----------------------------
comp = pd.read_csv("../data_raw/Compustat Annual Fundamentals.csv", low_memory=False)

# All vars are lowercase in your extract
comp.columns = [c.strip().lower() for c in comp.columns]

# Core filters (only apply if columns exist)
if "datafmt" in comp.columns:
    comp = comp[comp["datafmt"].astype(str).str.upper().isin(["STD"])]
if "indfmt" in comp.columns:
    comp = comp[comp["indfmt"].astype(str).str.upper().isin(["IND", "INDL"])]
if "consol" in comp.columns:
    comp = comp[comp["consol"].astype(str).str.upper().isin(["C"])]
# (Optional) US firms if you have fic
if "fic" in comp.columns:
    comp = comp[comp["fic"].astype(str).str.upper().isin(["USA", "US"])]

# Currency (you said it's CURNCD)
if "curncd" in comp.columns:
    # filter to USD if present, otherwise keep all
    if (comp["curncd"].astype(str).str.upper() == "USD").any():
        comp = comp[comp["curncd"].astype(str).str.upper() == "USD"]

# Dates
comp["datadate"] = pd.to_datetime(comp["datadate"], errors="coerce")
comp = comp.dropna(subset=["gvkey", "datadate"])

# ----------------------------
# FIX 1: Add merge_date with 6-month lag
# Accounting data is not publicly available until after filing.
# Standard Fama-French convention: use fiscal year-end data
# only from 6 months after fiscal year-end (i.e. July of year t+1
# for a December fiscal year-end). Use merge_date when joining to CRSP.
# ----------------------------
comp["merge_date"] = comp["datadate"] + pd.DateOffset(months=6)

# Sort (needed for lags)
comp = comp.sort_values(["gvkey", "datadate"]).reset_index(drop=True)

# ----------------------------
# Numeric conversion (only for columns you actually have)
# ----------------------------
num_candidates = [
    "act","ap","at","ceq","ch","che","dlc","dltt","dpacre","gdwl","intan","invt",
    "lct","lt","ppegt","ppent","rect","seq","txditc",
    "cogs","dp","ebit","ebitda","ib","ni","oiadp","revt","sale",
    "xad","xrd","xsga","capx","chech","oancf","unwcc",
    "csho","emp","prcc_f", "pstkrv", "pstkl", "pstk"
]
for c in num_candidates:
    if c in comp.columns:
        comp[c] = pd.to_numeric(comp[c], errors="coerce")

# --------------------------------------------------
# Construct Book Equity (BE)
# Standard Fama-French definition
# BE = SEQ + TXDITC − Preferred Stock
# Preferred stock hierarchy: PSTKRV > PSTKL > PSTK
# If SEQ is missing, use CEQ (common equity) as fallback
# --------------------------------------------------

# Start with SEQ, fallback to CEQ if missing
if "seq" in comp.columns and "ceq" in comp.columns:
    seq = np.where(pd.isna(comp["seq"]), comp["ceq"], comp["seq"])
elif "seq" in comp.columns:
    seq = comp["seq"]
elif "ceq" in comp.columns:
    seq = comp["ceq"]
else:
    seq = np.nan

txditc = col(comp, "txditc")

# Preferred stock hierarchy
pstk = np.nan

if "pstkrv" in comp.columns:
    pstk = comp["pstkrv"]
if "pstkl" in comp.columns:
    pstk = np.where(pd.isna(pstk), comp["pstkl"], pstk)
if "pstk" in comp.columns:
    pstk = np.where(pd.isna(pstk), comp["pstk"], pstk)

# If still missing → set to zero
pstk = np.where(pd.isna(pstk), 0.0, pstk)

# Book Equity
comp["be"] = seq + np.where(pd.isna(txditc), 0.0, txditc) - pstk

# Basic denominators
at = col(comp, "at")
sale = col(comp, "sale")
revt = col(comp, "revt")
cogs = col(comp, "cogs")
ib = col(comp, "ib")
ni = col(comp, "ni")
oiadp = col(comp, "oiadp")
ebit = col(comp, "ebit")
ebitda = col(comp, "ebitda")
oancf = col(comp, "oancf")
capx = col(comp, "capx")
lt = col(comp, "lt")
lct = col(comp, "lct")
act = col(comp, "act")
invt = col(comp, "invt")
che = col(comp, "che")
dlc = col(comp, "dlc")
dltt = col(comp, "dltt")
xsga = col(comp, "xsga")
xrd = col(comp, "xrd")
xad = col(comp, "xad")
rect = col(comp, "rect")
ap = col(comp, "ap")
unwcc = col(comp, "unwcc")
ppent = col(comp, "ppent")
intan = col(comp, "intan")
gdwl = col(comp, "gdwl")
emp = col(comp, "emp")

# ----------------------------
# Profitability (core)
# ----------------------------
comp["roa"] = safe_div(ib, at)
comp["roe"] = safe_div(ib, col(comp, "seq"))
comp["profit_margin"] = safe_div(ni, sale)
comp["gross_profitability"] = safe_div((revt - cogs), at)
comp["operating_profitability"] = safe_div(oiadp, at)
comp["ebit_to_assets"] = safe_div(ebit, at)
comp["ebitda_to_assets"] = safe_div(ebitda, at)
comp["cashflow_to_assets"] = safe_div(oancf, at)
comp["asset_turnover"] = safe_div(sale, at)

# ----------------------------
# Investment / growth
# FIX 3: Compute lagged columns explicitly before using in safe_div
# to avoid fragile index alignment with inline groupby().shift()
# ----------------------------
comp["at_lag1"]    = comp.groupby("gvkey")["at"].shift(1)
comp["sale_lag1"]  = comp.groupby("gvkey")["sale"].shift(1)
comp["ppent_lag1"] = comp.groupby("gvkey")["ppent"].shift(1)
comp["intan_lag1"] = comp.groupby("gvkey")["intan"].shift(1)
comp["gdwl_lag1"]  = comp.groupby("gvkey")["gdwl"].shift(1)
comp["invt_lag1"]  = comp.groupby("gvkey")["invt"].shift(1)
comp["emp_lag1"]   = comp.groupby("gvkey")["emp"].shift(1)

comp["asset_growth"]    = safe_div(comp["at"]    - comp["at_lag1"],    comp["at_lag1"])
comp["sales_growth"]    = safe_div(comp["sale"]  - comp["sale_lag1"],  comp["sale_lag1"])
comp["capex_to_assets"] = safe_div(capx, at)
comp["ppent_growth"]    = safe_div(comp["ppent"] - comp["ppent_lag1"], comp["ppent_lag1"])
comp["intan_growth"]    = safe_div(comp["intan"] - comp["intan_lag1"], comp["intan_lag1"])
comp["gdwl_growth"]     = safe_div(comp["gdwl"]  - comp["gdwl_lag1"],  comp["gdwl_lag1"])
comp["invt_growth"]     = safe_div(comp["invt"]  - comp["invt_lag1"],  comp["invt_lag1"])
comp["emp_growth"]      = safe_div(comp["emp"]   - comp["emp_lag1"],   comp["emp_lag1"])

comp["rd_intensity"]  = safe_div(xrd, sale)
comp["ad_intensity"]  = safe_div(xad, sale)
comp["sga_intensity"] = safe_div(xsga, sale)

# ----------------------------
# Leverage / solvency
# ----------------------------
total_debt = np.where(pd.isna(dlc), 0.0, dlc) + np.where(pd.isna(dltt), 0.0, dltt)
comp["leverage"]           = safe_div(lt, at)
comp["debt_to_assets"]     = safe_div(total_debt, at)
comp["debt_to_equity"]     = safe_div(total_debt, col(comp, "seq"))
comp["net_debt_to_assets"] = safe_div(total_debt - che, at)

# ----------------------------
# Liquidity / working capital
# ----------------------------
comp["current_ratio"]             = safe_div(act, lct)
comp["quick_ratio"]               = safe_div(act - invt, lct)
comp["cash_ratio"]                = safe_div(che, lct)
comp["working_capital_to_assets"] = safe_div((act - lct), at)

# ----------------------------
# Accruals
# FIX 2: Compute working capital as a named column first, then
# shift it safely via groupby — avoids the fragile nested apply
# that caused index misalignment in the original else-branch
# ----------------------------
dep = col(comp, "dp")
comp["_wc"] = (col(comp, "act") - col(comp, "che")) - (col(comp, "lct") - col(comp, "dlc"))
dwc = comp["_wc"] - comp.groupby("gvkey")["_wc"].shift(1)
comp["accruals"] = safe_div(dwc - dep, at)
comp.drop(columns=["_wc"], inplace=True)

# ----------------------------
# Efficiency changes (deltas)
# FIX 3 (continued): use explicit lag columns
# ----------------------------
comp["roa_lag1"]            = comp.groupby("gvkey")["roa"].shift(1)
comp["profit_margin_lag1"]  = comp.groupby("gvkey")["profit_margin"].shift(1)
comp["leverage_lag1"]       = comp.groupby("gvkey")["leverage"].shift(1)
comp["asset_turnover_lag1"] = comp.groupby("gvkey")["asset_turnover"].shift(1)

comp["delta_roa"]      = comp["roa"]            - comp["roa_lag1"]
comp["delta_margin"]   = comp["profit_margin"]  - comp["profit_margin_lag1"]
comp["delta_leverage"] = comp["leverage"]       - comp["leverage_lag1"]
comp["delta_turnover"] = comp["asset_turnover"] - comp["asset_turnover_lag1"]

# ----------------------------
# Keep identifiers needed for merge
# NOTE: merge_date (not datadate) should be used when joining to CRSP
# IMPORTANT: keep cusip (lowercase) here.
# ----------------------------
id_cols = [c for c in ["gvkey", "datadate", "merge_date", "fyear", "cusip", "tic", "costat"] if c in comp.columns]

feature_cols = [
    "be",
    "roa","roe","profit_margin","gross_profitability","operating_profitability",
    "ebit_to_assets","ebitda_to_assets","cashflow_to_assets","asset_turnover",
    "asset_growth","sales_growth","capex_to_assets","ppent_growth","intan_growth","gdwl_growth","invt_growth",
    "rd_intensity","ad_intensity","sga_intensity",
    "leverage","debt_to_assets","debt_to_equity","net_debt_to_assets",
    "current_ratio","quick_ratio","cash_ratio","working_capital_to_assets","accruals",
    "delta_roa","delta_margin","delta_leverage","delta_turnover","emp_growth"
]
feature_cols = [c for c in feature_cols if c in comp.columns]

comp = comp[id_cols + feature_cols].copy()

# Drop duplicates (keep last report per gvkey/date)
comp = comp.sort_values(["gvkey", "datadate"]).drop_duplicates(subset=["gvkey", "datadate"], keep="last")
print(f"Compustat observations after cleaning: {len(comp):,}")
print(f"Unique firms: {comp['gvkey'].nunique():,}")

# Save
comp.to_parquet("../data_clean/compustat_clean.parquet", index=False)
print("Compustat cleaned + features saved to ../data_clean/compustat_clean.parquet")
print(f"Compustat predictors created: {len(feature_cols)}")