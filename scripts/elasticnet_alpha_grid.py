"""
ElasticNet Alpha-Grid Analysis — Per Decade
============================================
Additional analysis requested by promotor (Chunjie):

"Vary alpha over a grid, track performance as the number of non-zero
coefficients changes, observe which feature enters first, second, etc.,
compare this across decades. Even if the ranking of features looks
similar across decades, the marginal contribution of features could
still differ once hyperparameters are tuned properly."

For each decade:
  1. Pool all training data within the decade (preprocessed per window
     using the same pipeline as the main ElasticNet script)
  2. Vary alpha over a wide grid (high → low)
  3. For each alpha:
     - Fit ElasticNet with l1_ratio = 1.0 (pure Lasso for cleanest
       feature entry interpretation)
     - Record which features are non-zero
     - Compute portfolio Sharpe ratio
  4. Identify entry order: at what alpha each feature first becomes
     non-zero (lower alpha means later entry)
  5. Compare entry order across decades

Run from: Master_Thesis/scripts/
Output:   Master_Thesis/output/elasticnet/alpha_grid/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
MERGED_PATH = "../data_clean/merged_data.parquet"
FF_RF_PATH = "../data_raw/ff_factors_monthly.csv"
OUT_DIR = Path("../output/elasticnet/alpha_grid")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TRAIN_YEARS = 10
RET_COL = "RET_ADJ"
DATE_COL = "date"
PERMNO_COL = "PERMNO"

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

# Alpha grid — log-spaced from high (heavily regularized) to low (lightly regularized)
# 30 values gives smooth feature entry tracking without excessive runtime
ALPHA_GRID = np.logspace(-1, -5, 30)  # 0.1 → 1e-5

# Use l1_ratio = 0.5 — standard ElasticNet midpoint, matching Chunjie's
# explicit framing ("ElasticNet, alpha and l1_ratio"). Adjust to match
# the typical value selected by your main ElasticNet pipeline if known.
L1_RATIO = 0.5


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
        pred_out[c] = pred_out[c].clip(lo_bounds[c], hi_bounds[c])
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
    pred_out[col] = pred_out[col].clip(lo, hi)
    return train_out, pred_out


def load_rf_series():
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


def compute_ls_sharpe(predictions_df, rf_series=None):
    """Compute annualized Sharpe ratio of the long-short decile portfolio."""
    portfolio_returns = []
    for date, grp in predictions_df.groupby(DATE_COL):
        grp = grp.dropna(subset=["predicted_return", RET_COL])
        if len(grp) < 20:
            continue
        grp[RET_COL] = grp[RET_COL].clip(
            grp[RET_COL].quantile(0.01), grp[RET_COL].quantile(0.99)
        )
        n = len(grp)
        dec_n = max(1, n // 10)
        grp_sorted = grp.sort_values("predicted_return", ascending=False)
        ls = grp_sorted.head(dec_n)[RET_COL].mean() - grp_sorted.tail(dec_n)[RET_COL].mean()
        portfolio_returns.append({DATE_COL: date, "ls": ls})

    if len(portfolio_returns) < 12:
        return np.nan

    pr = pd.DataFrame(portfolio_returns).set_index(DATE_COL)
    r = pr["ls"].dropna()
    if rf_series is not None:
        idx = r.index.to_period("M")
        rf = idx.map(rf_series).fillna(0)
        exc = r.values - rf.values
    else:
        exc = r.values
    return (exc.mean() / exc.std()) * np.sqrt(12) if exc.std() > 0 else np.nan


def preprocess_window(train_raw, pred_raw, features):
    """Apply full preprocessing pipeline to a train/pred split."""
    train_p, pred_p = winsorize_returns(train_raw, pred_raw, RET_COL)
    train_p, pred_p = winsorize_df(train_p, pred_p, features)
    train_p, pred_p = rank_normalize(train_p, pred_p, features)
    train_p[features] = train_p[features].fillna(0)
    pred_p[features] = pred_p[features].fillna(0)
    return train_p, pred_p


# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
hline()
print("STEP 1: Loading data")
hline()

df = pd.read_parquet(MERGED_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values([PERMNO_COL, DATE_COL]).reset_index(drop=True)

keep = [PERMNO_COL, DATE_COL, RET_COL] + [c for c in FEATURE_COLS if c in df.columns]
df = df[keep].copy()

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
df = df[df["ME"] >= df["nyse_p20"]].copy()
df = df.drop(columns=["ME", "nyse_p20"])
df = df[df["PRC"] >= 1].copy()
df = df.drop(columns=["PRC"])
print(f"  Observations  : {len(df):,}")
print(f"  Features      : {len(present_features)}")

rf_series = load_rf_series()

# ── 2. DEFINE WINDOWS BY DECADE ───────────────────────────────────────────────
hline()
print("STEP 2: Defining windows grouped by decade")
hline()

all_years = sorted(df["year"].unique())
windows = []
for i in range(len(all_years) - TRAIN_YEARS):
    pred_year = all_years[i + TRAIN_YEARS]
    windows.append({
        "train_start": all_years[i],
        "train_end": all_years[i + TRAIN_YEARS - 1],
        "pred_year": pred_year,
        "decade": (pred_year // 10) * 10,
    })

decades = sorted(set(w["decade"] for w in windows))
print(f"  Total windows : {len(windows)}")
print(f"  Decades       : {decades}")
print(f"  Alpha grid    : {len(ALPHA_GRID)} values from {ALPHA_GRID.max():.4f} to {ALPHA_GRID.min():.6f}")
print(f"  L1 ratio      : {L1_RATIO} (ElasticNet midpoint)")

# ── 3. ALPHA GRID SCAN PER DECADE ─────────────────────────────────────────────
hline()
print("STEP 3: Scanning alpha grid for each decade")
hline()
print("  For each decade, fit Lasso at each alpha value and record:")
print("    - Which features are non-zero")
print("    - Coefficient magnitudes")
print("    - Long-short Sharpe ratio of the portfolio")
print()

# Storage: rows = (decade, alpha, feature, coef_value), plus portfolio metrics
all_coef_records = []
all_portfolio_records = []
all_entry_records = []  # at what alpha each feature first enters

for decade in decades:
    decade_windows = [w for w in windows if w["decade"] == decade]
    decade_label = f"{decade}s"
    print(f"  ▸ Processing {decade_label} ({len(decade_windows)} windows)...")

    # For each alpha, we'll collect predictions across all windows in the decade,
    # then compute decade-level portfolio Sharpe
    alpha_to_predictions = {a: [] for a in ALPHA_GRID}
    alpha_to_coefs = {a: [] for a in ALPHA_GRID}  # coefs from each window

    for w_idx, w in enumerate(decade_windows):
        train_raw = df[df["year"].between(w["train_start"], w["train_end"])].copy()
        pred_raw = df[df["year"] == w["pred_year"]].copy()

        if len(train_raw) < 1000 or len(pred_raw) < 100:
            continue

        train_p, pred_p = preprocess_window(train_raw, pred_raw, present_features)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_p[present_features].values)
        y_train = train_p[RET_COL].values
        X_pred = scaler.transform(pred_p[present_features].values)

        # Fit at each alpha
        for alpha in ALPHA_GRID:
            model = ElasticNet(
                alpha=alpha, l1_ratio=L1_RATIO,
                max_iter=10000, random_state=RANDOM_SEED
            )
            model.fit(X_train, y_train)

            y_hat = model.predict(X_pred)

            # Store predictions for portfolio construction
            pred_out = pred_raw[[PERMNO_COL, DATE_COL, RET_COL]].copy()
            pred_out["predicted_return"] = y_hat
            alpha_to_predictions[alpha].append(pred_out)

            # Store coefficients
            alpha_to_coefs[alpha].append(model.coef_)

    # Compute decade-level metrics for each alpha
    print(f"    Computing portfolio Sharpe across {len(ALPHA_GRID)} alpha values...")
    for alpha in ALPHA_GRID:
        if not alpha_to_predictions[alpha]:
            continue

        # Pool predictions across all windows in decade for portfolio
        pooled = pd.concat(alpha_to_predictions[alpha], ignore_index=True)
        sharpe = compute_ls_sharpe(pooled, rf_series)

        # Average coefficients across windows in decade
        coef_matrix = np.array(alpha_to_coefs[alpha])
        mean_coefs = coef_matrix.mean(axis=0)
        n_nonzero = int((np.abs(mean_coefs) > 1e-8).sum())

        # Record portfolio metrics
        all_portfolio_records.append({
            "decade": decade,
            "alpha": alpha,
            "n_nonzero": n_nonzero,
            "ls_sharpe": sharpe,
        })

        # Record per-feature coefficients
        for feat, coef in zip(present_features, mean_coefs):
            all_coef_records.append({
                "decade": decade,
                "alpha": alpha,
                "feature": feat,
                "coef": coef,
                "abs_coef": abs(coef),
            })

    # Determine feature entry order for this decade:
    # at what alpha does each feature first become non-zero (going high to low alpha)?
    decade_coefs = pd.DataFrame([r for r in all_coef_records if r["decade"] == decade])
    for feat in present_features:
        feat_data = decade_coefs[decade_coefs["feature"] == feat].sort_values("alpha", ascending=False)
        # First alpha where coefficient is non-zero (entry alpha)
        nonzero = feat_data[feat_data["abs_coef"] > 1e-8]
        if len(nonzero) > 0:
            entry_alpha = nonzero["alpha"].iloc[0]
            entry_rank = (feat_data["alpha"] >= entry_alpha).sum()
        else:
            entry_alpha = np.nan
            entry_rank = np.nan
        all_entry_records.append({
            "decade": decade,
            "feature": feat,
            "entry_alpha": entry_alpha,
            "entry_rank": entry_rank,
        })

# Save raw data
coef_df = pd.DataFrame(all_coef_records)
portfolio_df = pd.DataFrame(all_portfolio_records)
entry_df = pd.DataFrame(all_entry_records)

coef_df.to_csv(OUT_DIR / "alpha_grid_coefficients.csv", index=False)
portfolio_df.to_csv(OUT_DIR / "alpha_grid_portfolio.csv", index=False)
entry_df.to_csv(OUT_DIR / "alpha_grid_entry_order.csv", index=False)

print(f"\n  ✓ Raw results saved to {OUT_DIR}")

# ── 4. FEATURE ENTRY ORDER TABLE ──────────────────────────────────────────────
hline()
print("STEP 4: Feature entry order across decades")
hline()
print("  For each decade, the order in which features enter the model")
print("  as alpha decreases (lower entry_rank = earlier entry = more important)")
print()

# Pivot: rows = features, columns = decades, values = entry_rank
entry_pivot = entry_df.pivot(index="feature", columns="decade", values="entry_rank")
entry_pivot.columns = [f"{int(c)}s" for c in entry_pivot.columns]

# Sort features by mean entry rank across decades (earliest-entering features first)
entry_pivot["mean_rank"] = entry_pivot.mean(axis=1, skipna=True)
entry_pivot = entry_pivot.sort_values("mean_rank")

print(entry_pivot.head(20).to_string())
print(f"\n  Saved → {OUT_DIR / 'alpha_grid_entry_order.csv'}")

# Save the pivoted version too
entry_pivot.to_csv(OUT_DIR / "alpha_grid_entry_order_pivot.csv")

# ── 5. TOP-K FEATURES PER DECADE ──────────────────────────────────────────────
hline()
print("STEP 5: Top-10 earliest-entering features per decade")
hline()
print("  These are the features the model selects first when regularization")
print("  is loosened — interpreted as the most important predictors.")

top_entry_summary = []
for decade in decades:
    decade_label = f"{decade}s"
    dec_entry = entry_df[entry_df["decade"] == decade].sort_values(
        "entry_rank", ascending=True, na_position="last"
    )
    top10 = dec_entry.head(10)
    print(f"\n  {decade_label} — top 10 earliest-entering features:")
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        if pd.isna(row["entry_alpha"]):
            print(f"    {rank:2d}. {row['feature']:<35} (never entered)")
            continue
        print(f"    {rank:2d}. {row['feature']:<35} entry α={row['entry_alpha']:.5f}")
        top_entry_summary.append({
            "decade": decade_label,
            "rank": rank,
            "feature": row["feature"],
            "entry_alpha": row["entry_alpha"],
        })

pd.DataFrame(top_entry_summary).to_csv(OUT_DIR / "top10_entry_per_decade.csv", index=False)
print(f"\n  Saved → {OUT_DIR / 'top10_entry_per_decade.csv'}")

# ── 6. PLOTS ──────────────────────────────────────────────────────────────────
hline()
print("STEP 6: Generating plots")
hline()

# Plot 1: Sharpe ratio vs number of non-zero coefficients, one line per decade
fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 0.9, len(decades)))

for color, decade in zip(colors, decades):
    sub = portfolio_df[portfolio_df["decade"] == decade].sort_values("n_nonzero")
    ax.plot(sub["n_nonzero"], sub["ls_sharpe"],
            marker="o", markersize=4, lw=1.5,
            label=f"{decade}s", color=color)

ax.axhline(0, color="black", lw=0.5, linestyle=":")
ax.set_xlabel("Number of Non-Zero Coefficients")
ax.set_ylabel("Long-Short Sharpe Ratio (Annualized)")
ax.set_title("Sharpe Ratio vs Model Sparsity by Decade")
ax.legend(title="Decade", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "sharpe_vs_nonzero.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {OUT_DIR / 'sharpe_vs_nonzero.png'}")

# Plot 2: Sharpe ratio vs alpha (log scale), one line per decade
fig, ax = plt.subplots(figsize=(10, 6))
for color, decade in zip(colors, decades):
    sub = portfolio_df[portfolio_df["decade"] == decade].sort_values("alpha", ascending=False)
    ax.plot(sub["alpha"], sub["ls_sharpe"],
            marker="o", markersize=4, lw=1.5,
            label=f"{decade}s", color=color)

ax.axhline(0, color="black", lw=0.5, linestyle=":")
ax.set_xscale("log")
ax.invert_xaxis()  # high alpha → low alpha (left to right)
ax.set_xlabel("Alpha (regularization strength, decreasing →)")
ax.set_ylabel("Long-Short Sharpe Ratio (Annualized)")
ax.set_title("Sharpe Ratio vs Regularization Strength by Decade")
ax.legend(title="Decade", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "sharpe_vs_alpha.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {OUT_DIR / 'sharpe_vs_alpha.png'}")

# Plot 3: Feature entry heatmap — entry rank for top features across decades
top_features_overall = entry_pivot.head(15).index.tolist()
heatmap_data = entry_pivot.loc[top_features_overall, [c for c in entry_pivot.columns if c != "mean_rank"]]

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(heatmap_data.values, aspect="auto", cmap="RdYlGn_r")
ax.set_xticks(range(len(heatmap_data.columns)))
ax.set_xticklabels(heatmap_data.columns)
ax.set_yticks(range(len(heatmap_data.index)))
ax.set_yticklabels(heatmap_data.index)
ax.set_title("Feature Entry Rank by Decade\n(lower = enters earlier = more important)")

# Annotate cells
for i in range(len(heatmap_data.index)):
    for j in range(len(heatmap_data.columns)):
        val = heatmap_data.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{int(val)}", ha="center", va="center",
                    color="black", fontsize=8)

plt.colorbar(im, ax=ax, label="Entry Rank")
plt.tight_layout()
plt.savefig(OUT_DIR / "entry_rank_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {OUT_DIR / 'entry_rank_heatmap.png'}")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
hline()
print("FINAL SUMMARY")
hline()
print(f"  Decades analyzed   : {len(decades)}")
print(f"  Alpha values       : {len(ALPHA_GRID)}")
print(f"  Total fits         : {len(decades) * len(ALPHA_GRID) * 9} approx")  # ~9 windows/decade
print(f"  Output saved to    : {OUT_DIR}")
print(f"\n  Key outputs:")
print(f"    - alpha_grid_coefficients.csv : raw coefficients per (decade, alpha, feature)")
print(f"    - alpha_grid_portfolio.csv    : Sharpe ratio per (decade, alpha)")
print(f"    - alpha_grid_entry_order.csv  : entry alpha and rank per (decade, feature)")
print(f"    - top10_entry_per_decade.csv  : top 10 earliest-entering features per decade")
print(f"    - sharpe_vs_nonzero.png       : Sharpe vs model sparsity")
print(f"    - sharpe_vs_alpha.png         : Sharpe vs regularization strength")
print(f"    - entry_rank_heatmap.png      : feature entry rank across decades")
hline()