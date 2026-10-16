"""
Neural Network Rolling Window — ML Script
==========================================
Mirrors elasticnet_rolling.py and random_forest_rolling.py exactly
in structure for direct comparison.

Architecture follows Gu, Kelly & Xiu (2020):
  - Feed-forward network with 3 hidden layers (32, 16, 8)
  - ReLU activations
  - Batch normalization
  - Dropout regularization (tuned per decade)
  - Adam optimizer with ReduceLROnPlateau scheduler
  - L1 regularization (tuned per decade)
  - Early stopping on validation loss

Hyperparameter tuning: per-decade single train/validation split
following promotor feedback. For each decade (1980s–2020s), the
first window of that decade is split into:
  - Training: first 7 years
  - Early stopping validation: year 8 (used during model fit)
  - Hyperparameter validation: last 2 years (years 9-10)
This conservative scheme better mimics the actual forecasting task
than k-fold CV (which can overfit when adjacent years share regimes).
The selected parameters are then applied to all windows within that decade.

Run from: Master_Thesis/scripts/
Output:   Master_Thesis/output/neural_network/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from itertools import product
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
MERGED_PATH = "../data_clean/merged_data.parquet"
FF_RF_PATH = "../data_raw/ff_factors_monthly.csv"
OUT_DIR = Path("../output/neural_network")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TRAIN_YEARS = 10
RET_COL = "RET_ADJ"
DATE_COL = "date"
PERMNO_COL = "PERMNO"

# Set seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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

# Neural Network base hyperparameters — fixed across decades following GKX (2020)
NN_BASE_PARAMS = {
    "hidden_units": [32, 16, 8],  # GKX architecture — fixed
    "learning_rate": 0.001,  # Adam default — fixed
    "batch_size": 10000,  # large batch for stability — fixed
    "max_epochs": 100,  # maximum training epochs — fixed
    "patience": 5,  # early stopping patience — fixed
}

# Tunable hyperparameters — selected per decade
# Small grid to keep runtime manageable on CPU (~5-6 hours total)
NN_PARAM_GRID = {
    "dropout_rate": [0.05, 0.20],
    "l1_lambda": [1e-5, 1e-4],
}


# ── NEURAL NETWORK ARCHITECTURE ───────────────────────────────────────────────
class GKXNet(nn.Module):
    """
    Feed-forward neural network following Gu, Kelly & Xiu (2020).
    3 hidden layers with batch normalization, ReLU activations,
    and dropout regularization.
    """

    def __init__(self, input_dim, hidden_units, dropout_rate):
        super(GKXNet, self).__init__()

        layers = []
        in_dim = input_dim

        for h_dim in hidden_units:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            in_dim = h_dim

        # Output layer — single neuron for return prediction
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(1)


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


def train_network(X_train, y_train, X_val, y_val, input_dim, params):
    """
    Train GKXNet with early stopping on validation loss.
    Uses last year of training window as validation set.
    Learning rate reduced on plateau following GKX (2020).
    """
    X_tr = torch.FloatTensor(X_train)
    y_tr = torch.FloatTensor(y_train)
    X_v = torch.FloatTensor(X_val)
    y_v = torch.FloatTensor(y_val)

    # Adaptive batch size — avoid batches larger than training set
    batch_size = min(params["batch_size"], len(X_train) // 10)
    dataset = TensorDataset(X_tr, y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GKXNet(input_dim, params["hidden_units"], params["dropout_rate"])
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    # Reduce LR when validation loss plateaus — following GKX
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_weights = None
    patience_ctr = 0

    for epoch in range(params["max_epochs"]):
        # Training
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            # L1 regularization following GKX
            l1_reg = sum(p.abs().sum() for p in model.parameters())
            loss = loss + params["l1_lambda"] * l1_reg
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_v)
            val_loss = criterion(val_preds, y_v).item()

        # Step scheduler on validation loss
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= params["patience"]:
                break

    # Restore best weights
    model.load_state_dict(best_weights)
    return model


def compute_permutation_importance(model, X_pred, y_actual, present_features, n_samples=1000):
    """
    Permutation-based feature importance for neural network.
    Measures increase in MSE when each feature is randomly permuted.
    Consistent with Random Forest approach for direct comparison.
    """
    model.eval()
    n = min(n_samples, len(X_pred))
    X_sample = torch.FloatTensor(X_pred[:n])
    y_sample = y_actual[:n]

    with torch.no_grad():
        base_preds = model(X_sample).numpy()

    base_mse = np.mean((base_preds - y_sample) ** 2)

    importances = {}
    for i, feat in enumerate(present_features):
        X_permuted = X_sample.clone()
        idx = torch.randperm(X_permuted.shape[0])
        X_permuted[:, i] = X_permuted[idx, i]

        with torch.no_grad():
            perm_preds = model(X_permuted).numpy()

        perm_mse = np.mean((perm_preds - y_sample) ** 2)
        importances[feat] = max(0, perm_mse - base_mse)

    return importances


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


def compute_portfolio_metrics(returns, rf_series=None, label="Portfolio"):
    r = returns.dropna()
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    if rf_series is not None:
        idx = r.index.to_period("M")
        rf = idx.map(rf_series).fillna(0)
        exc = r.values - rf.values
    else:
        exc = r.values
    sharpe = (exc.mean() / exc.std()) * np.sqrt(12) if exc.std() > 0 else np.nan
    cum = (1 + r).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return {
        "Label": label,
        "Ann_Return": round(ann_ret, 4),
        "Ann_Vol": round(ann_vol, 4),
        "Sharpe": round(sharpe, 4),
        "Max_Drawdown": round(mdd, 4),
        "N_months": len(r),
    }


def preprocess_window(train_raw, pred_raw, features):
    """Apply full preprocessing pipeline to a train/pred split."""
    train_p, pred_p = winsorize_returns(train_raw, pred_raw, RET_COL)
    train_p, pred_p = winsorize_df(train_p, pred_p, features)
    train_p, pred_p = rank_normalize(train_p, pred_p, features)
    train_p[features] = train_p[features].fillna(0)
    pred_p[features] = pred_p[features].fillna(0)
    return train_p, pred_p


def tune_nn_for_decade(train_p, train_end_year, present_features, decade_label):
    """
    Tune NN hyperparameters for a decade using a single train/validation split
    that mimics the actual forecasting task (per Chunjie's guidance).

    Splits the 10-year training window into:
      - Training: first 7 years
      - Early stopping validation: year 8 (used during model fit)
      - Hyperparameter validation: last 2 years (years 9-10)

    Returns the best (dropout_rate, l1_lambda) combination.
    """
    print(f"\n  ▸ Tuning NN hyperparameters for {decade_label}...")

    # Split: last 2 years for HP validation, year before that for early stopping,
    # rest for training
    hp_val_start = train_end_year - 1  # last 2 years for HP selection
    es_val_year = train_end_year - 2  # year before that for early stopping

    hp_val_mask = train_p["year"].values >= hp_val_start
    es_val_mask = train_p["year"].values == es_val_year
    train_mask = ~(hp_val_mask | es_val_mask)

    # Scale features using training portion only
    scaler = StandardScaler()
    X_full = scaler.fit_transform(train_p[present_features].values)
    y_full = train_p[RET_COL].values

    X_tr = X_full[train_mask]
    y_tr = y_full[train_mask]
    X_es = X_full[es_val_mask]
    y_es = y_full[es_val_mask]
    X_hp = X_full[hp_val_mask]
    y_hp = y_full[hp_val_mask]

    print(f"    Training:           {train_mask.sum():,} obs (years before {es_val_year})")
    print(f"    Early stopping val: {es_val_mask.sum():,} obs (year {es_val_year})")
    print(f"    HP validation:      {hp_val_mask.sum():,} obs (years {hp_val_start}-{train_end_year})")

    # Run grid search
    best_score = -np.inf
    best_params = None
    grid_keys = list(NN_PARAM_GRID.keys())
    grid_vals = [NN_PARAM_GRID[k] for k in grid_keys]

    for combo in product(*grid_vals):
        params = {**NN_BASE_PARAMS, **dict(zip(grid_keys, combo))}

        # Train with this hyperparameter combination
        model = train_network(
            X_tr, y_tr, X_es, y_es,
            input_dim=len(present_features),
            params=params
        )

        # Evaluate on hyperparameter validation set
        model.eval()
        with torch.no_grad():
            y_hat = model(torch.FloatTensor(X_hp)).numpy()
        score = r2_score(y_hp, y_hat)

        print(f"    dropout={params['dropout_rate']:.2f}, "
              f"l1={params['l1_lambda']:.0e}: val R²={score:.5f}")

        if score > best_score:
            best_score = score
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
windows = []
for i in range(len(all_years) - TRAIN_YEARS):
    pred_year = all_years[i + TRAIN_YEARS]
    windows.append({
        "train_start": all_years[i],
        "train_end": all_years[i + TRAIN_YEARS - 1],
        "pred_year": pred_year,
        "decade": (pred_year // 10) * 10,
    })

print(f"  Training window : {TRAIN_YEARS} years")
print(f"  Total windows   : {len(windows)}")
print(
    f"  First window    : train {windows[0]['train_start']}–{windows[0]['train_end']}, predict {windows[0]['pred_year']}")
print(
    f"  Last window     : train {windows[-1]['train_start']}–{windows[-1]['train_end']}, predict {windows[-1]['pred_year']}")
print(f"  Architecture    : {NN_BASE_PARAMS['hidden_units']} hidden units (fixed)")

# ── 3. PER-DECADE HYPERPARAMETER TUNING ───────────────────────────────────────
hline()
print("STEP 3: Per-decade hyperparameter tuning")
hline()
print("  For each decade, tune dropout_rate and l1_lambda on the first window")
print("  of that decade using holdout validation. Selected hyperparameters")
print("  are then applied to all windows in that decade.")

decades = sorted(set(w["decade"] for w in windows))
decade_params = {}

for decade in decades:
    decade_windows = [w for w in windows if w["decade"] == decade]
    tuning_window = decade_windows[0]
    decade_label = f"{decade}s"

    print(f"\n  Decade {decade_label}: tuning on window train "
          f"{tuning_window['train_start']}–{tuning_window['train_end']} "
          f"→ pred {tuning_window['pred_year']}")

    train_raw = df[df["year"].between(tuning_window["train_start"],
                                      tuning_window["train_end"])].copy()
    pred_raw = df[df["year"] == tuning_window["pred_year"]].copy()

    # Preprocess for tuning
    train_p, _ = preprocess_window(train_raw, pred_raw, present_features)

    # Run holdout grid search
    best_params = tune_nn_for_decade(
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
    OUT_DIR / "decade_hyperparameters.csv", index=False
)
print(f"\n  ✓ Decade hyperparameters saved → {OUT_DIR / 'decade_hyperparameters.csv'}")

# ── 4. ROLLING WINDOW TRAINING & PREDICTION ───────────────────────────────────
hline()
print("STEP 4: Rolling window Neural Network with per-decade hyperparameters")
hline()

all_predictions = []
all_importances = []
all_metrics = []

for w_idx, w in enumerate(windows):
    decade_p = decade_params[w["decade"]]
    print(f"  Window {w_idx + 1:02d}/{len(windows)}: "
          f"train {w['train_start']}–{w['train_end']} → predict {w['pred_year']} "
          f"[{w['decade']}s params]", end=" ")

    train_raw = df[df["year"].between(w["train_start"], w["train_end"])].copy()
    pred_raw = df[df["year"] == w["pred_year"]].copy()

    if len(train_raw) < 1000 or len(pred_raw) < 100:
        print("  ⚠ skipped (insufficient data)")
        continue

    # Preprocessing — identical to ElasticNet and Random Forest
    train_p, pred_p = preprocess_window(train_raw, pred_raw, present_features)

    # Scale features — important for neural network training stability
    scaler = StandardScaler()
    X_all = scaler.fit_transform(train_p[present_features].values)
    y_all = train_p[RET_COL].values
    X_pred = scaler.transform(pred_p[present_features].values)
    y_actual = pred_p[RET_COL].values

    # Use last year of training window as validation for early stopping
    # No look-ahead bias — validation always before prediction year
    val_mask = train_p["year"].values == w["train_end"]
    train_mask = ~val_mask

    X_tr = X_all[train_mask]
    y_tr = y_all[train_mask]
    X_v = X_all[val_mask]
    y_v = y_all[val_mask]

    # Train network with this decade's tuned params
    nn_params = {**NN_BASE_PARAMS, **decade_p}
    model = train_network(
        X_tr, y_tr, X_v, y_v,
        input_dim=len(present_features),
        params=nn_params
    )

    # OOS predictions
    model.eval()
    with torch.no_grad():
        y_hat = model(torch.FloatTensor(X_pred)).numpy()

    oos_r2 = r2_score(y_actual, y_hat)
    print(f"| OOS R²={oos_r2:.5f}")

    # Store predictions
    pred_out = pred_raw[[PERMNO_COL, DATE_COL, RET_COL]].copy()
    pred_out["predicted_return"] = y_hat
    pred_out["pred_year"] = w["pred_year"]
    all_predictions.append(pred_out)

    # Permutation-based feature importance
    imp_dict = compute_permutation_importance(
        model, X_pred, y_actual, present_features
    )
    imp_dict["pred_year"] = w["pred_year"]
    imp_dict["train_start"] = w["train_start"]
    imp_dict["train_end"] = w["train_end"]
    imp_dict["oos_r2"] = oos_r2
    all_importances.append(imp_dict)

    all_metrics.append({
        "pred_year": w["pred_year"],
        "train_start": w["train_start"],
        "train_end": w["train_end"],
        "oos_r2": oos_r2,
        "dropout_rate": decade_p["dropout_rate"],
        "l1_lambda": decade_p["l1_lambda"],
        "n_train": len(train_raw),
        "n_pred": len(pred_raw),
    })

# Combine & save
predictions_df = pd.concat(all_predictions, ignore_index=True)
importances_df = pd.DataFrame(all_importances)
metrics_df = pd.DataFrame(all_metrics)

predictions_df.to_parquet(OUT_DIR / "predictions.parquet", index=False)
importances_df.to_csv(OUT_DIR / "importances.csv", index=False)
metrics_df.to_csv(OUT_DIR / "window_metrics.csv", index=False)
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
    n = len(grp)
    dec_n = max(1, n // 10)
    if n < dec_n * 2:
        continue
    grp_sorted = grp.sort_values("predicted_return", ascending=False)
    portfolio_returns.append({
        DATE_COL: date,
        "long_dec": grp_sorted.head(dec_n)[RET_COL].mean(),
        "short_dec": grp_sorted.tail(dec_n)[RET_COL].mean(),
        "ls_dec": grp_sorted.head(dec_n)[RET_COL].mean() - grp_sorted.tail(dec_n)[RET_COL].mean(),
        "n_stocks": n,
    })

port_df = pd.DataFrame(portfolio_returns)
port_df[DATE_COL] = pd.to_datetime(port_df[DATE_COL])
port_df = port_df.sort_values(DATE_COL)
port_df.to_csv(OUT_DIR / "portfolio_returns.csv", index=False)

print(f"  Portfolio months      : {len(port_df)}")
print(f"  Date range            : {port_df[DATE_COL].min().date()} → {port_df[DATE_COL].max().date()}")
print(f"  Mean monthly L-S (dec): {port_df['ls_dec'].mean():.4f}")

# ── 6. PERFORMANCE METRICS ────────────────────────────────────────────────────
hline()
print("STEP 6: Performance metrics")
hline()

rf_series = load_rf_series()
port_df = port_df.set_index(DATE_COL)

metrics_list = [
    compute_portfolio_metrics(port_df["long_dec"], rf_series, "Long leg (D10)"),
    compute_portfolio_metrics(port_df["short_dec"], rf_series, "Short leg (D1)"),
    compute_portfolio_metrics(port_df["ls_dec"], rf_series, "Long-Short (D10-D1)"),
]
perf_df = pd.DataFrame(metrics_list)
perf_df.to_csv(OUT_DIR / "performance_metrics.csv", index=False)
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
                DATE_COL: date,
                "decile": d,
                "ret": dec_grp[RET_COL].mean(),
            })

decile_df = pd.DataFrame(decile_returns)
decile_df[DATE_COL] = pd.to_datetime(decile_df[DATE_COL])

decile_summary = []
for d in range(1, 11):
    d_rets = decile_df[decile_df["decile"] == d].set_index(DATE_COL)["ret"]
    if rf_series is not None:
        idx = d_rets.index.to_period("M")
        rf = idx.map(rf_series).fillna(0)
        exc = d_rets.values - rf.values
    else:
        exc = d_rets.values
    sharpe_d = (exc.mean() / exc.std()) * np.sqrt(12) if exc.std() > 0 else np.nan
    ann_ret_d = (1 + d_rets).prod() ** (12 / len(d_rets)) - 1
    decile_summary.append({
        "Decile": d,
        "Mean_Monthly": round(d_rets.mean(), 4),
        "Ann_Return": round(ann_ret_d, 4),
        "Sharpe": round(sharpe_d, 4),
        "N_months": len(d_rets),
    })

decile_summary_df = pd.DataFrame(decile_summary)
decile_summary_df.to_csv(OUT_DIR / "decile_summary.csv", index=False)
print(decile_summary_df.to_string(index=False))
print(f"\n  Saved → {OUT_DIR / 'decile_summary.csv'}")

# ── 6c. DECADE-LEVEL PORTFOLIO PERFORMANCE ────────────────────────────────────
hline()
print("STEP 6c: Portfolio performance by decade")
hline()

port_df_reset = port_df.reset_index()
port_df_reset["decade"] = (port_df_reset[DATE_COL].dt.year // 10) * 10

decade_perf = []
for decade, grp in port_df_reset.groupby("decade"):
    grp = grp.set_index(DATE_COL)
    long_m = compute_portfolio_metrics(grp["long_dec"], rf_series, f"{decade}s Long")
    short_m = compute_portfolio_metrics(grp["short_dec"], rf_series, f"{decade}s Short")
    ls_m = compute_portfolio_metrics(grp["ls_dec"], rf_series, f"{decade}s L-S")
    decade_perf.append({
        "Decade": f"{decade}s",
        "LS_Mean_Monthly": round(grp["ls_dec"].mean(), 4),
        "LS_Ann_Return": ls_m["Ann_Return"],
        "LS_Sharpe": ls_m["Sharpe"],
        "LS_MaxDrawdown": ls_m["Max_Drawdown"],
        "Long_Sharpe": long_m["Sharpe"],
        "Short_Sharpe": short_m["Sharpe"],
        "N_months": ls_m["N_months"],
    })

decade_perf_df = pd.DataFrame(decade_perf)
decade_perf_df.to_csv(OUT_DIR / "decade_performance.csv", index=False)
print(decade_perf_df.to_string(index=False))
print(f"\n  Saved → {OUT_DIR / 'decade_performance.csv'}")

# ── 7. OOS R² SUMMARY ─────────────────────────────────────────────────────────
hline()
print("STEP 7: OOS R² over time")
hline()

print(f"  Mean OOS R²   : {metrics_df['oos_r2'].mean():.5f}")
print(f"  Median OOS R² : {metrics_df['oos_r2'].median():.5f}")
print(
    f"  Min OOS R²    : {metrics_df['oos_r2'].min():.5f}  ({metrics_df.loc[metrics_df['oos_r2'].idxmin(), 'pred_year']})")
print(
    f"  Max OOS R²    : {metrics_df['oos_r2'].max():.5f}  ({metrics_df.loc[metrics_df['oos_r2'].idxmax(), 'pred_year']})")
print(f"  %% positive R² : {(metrics_df['oos_r2'] > 0).mean() * 100:.1f}%%")

# ── 8. FEATURE IMPORTANCE ─────────────────────────────────────────────────────
hline()
print("STEP 8: Feature importance analysis (permutation-based)")
hline()

imp_cols = [c for c in present_features if c in importances_df.columns]
imp_values = importances_df[imp_cols]
mean_imp = imp_values.mean().sort_values(ascending=False)
top_features = mean_imp.head(15).index.tolist()

print("  Top 15 features by mean permutation importance:")
for i, (feat, val) in enumerate(mean_imp.head(15).items(), 1):
    print(f"    {i:2d}. {feat:<35} imp={val:.5f}")

# ── 9. PLOTS ──────────────────────────────────────────────────────────────────
hline()
print("STEP 9: Generating plots")
hline()

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Neural Network Rolling Window Results", fontsize=14, fontweight="bold")

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

# Plot 3 — Feature importance heatmap
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
plt.savefig(OUT_DIR / "neural_network_results.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved → {OUT_DIR / 'neural_network_results.png'}")

# ── 10. DECADE FEATURE IMPORTANCE ─────────────────────────────────────────────
hline()
print("STEP 10: Feature importance by decade")
hline()

importances_df["decade"] = (importances_df["pred_year"] // 10) * 10
decade_summary = []
for decade, grp in importances_df.groupby("decade"):
    mean_imp_dec = grp[imp_cols].mean().sort_values(ascending=False)
    top5 = mean_imp_dec.head(5)
    print(f"\n  {decade}s — top 5 predictors:")
    for feat, val in top5.items():
        print(f"    {feat:<35} imp={val:.5f}")
    for rank, (feat, val) in enumerate(top5.items(), 1):
        decade_summary.append({
            "decade": decade, "rank": rank,
            "feature": feat, "mean_importance": val
        })

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
print(f"  Hyperparameters    : tuned per decade (see decade_hyperparameters.csv)")
print(f"  Output saved to    : {OUT_DIR}")
hline()