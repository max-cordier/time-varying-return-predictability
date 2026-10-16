"""
Regenerate all thesis figures from existing CSV outputs.

Paths are hardcoded for the Master_Thesis PyCharm project structure.
Just run this from PyCharm and all figures get saved to output/combined_figures/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============== HARDCODED PATHS ==============
PROJECT_ROOT = os.path.expanduser("~/PycharmProjects/Master_Thesis")

EN_DIR = os.path.join(PROJECT_ROOT, "output", "elasticnet")
RF_DIR = os.path.join(PROJECT_ROOT, "output", "random_forest")
NN_DIR = os.path.join(PROJECT_ROOT, "output", "neural_network")
ALPHA_DIR = os.path.join(EN_DIR, "alpha_grid")
OUT_DIR = os.path.join(PROJECT_ROOT, "output", "combined_figures")

CSV_PATHS = {
    "window_en": os.path.join(EN_DIR, "window_metrics_en.csv"),
    "window_rf": os.path.join(RF_DIR, "window_metrics_rf.csv"),
    "window_nn": os.path.join(NN_DIR, "window_metrics.csv"),
    "importance_en": os.path.join(EN_DIR, "decade_importance_en.csv"),
    "importance_rf": os.path.join(RF_DIR, "decade_importance_rf.csv"),
    "importance_nn": os.path.join(NN_DIR, "decade_importance.csv"),
    "decile_en": os.path.join(EN_DIR, "decile_summary_en.csv"),
    "decile_rf": os.path.join(RF_DIR, "decile_summary_rf.csv"),
    "decile_nn": os.path.join(NN_DIR, "decile_summary.csv"),
    "decade_perf_en": os.path.join(EN_DIR, "decade_performance_en.csv"),
    "decade_perf_rf": os.path.join(RF_DIR, "decade_performance_rf.csv"),
    "decade_perf_nn": os.path.join(NN_DIR, "decade_performance.csv"),
    "alpha_pivot": os.path.join(ALPHA_DIR, "alpha_grid_entry_order_pivot.csv"),
    "alpha_entry": os.path.join(ALPHA_DIR, "alpha_grid_entry_order.csv"),
    "alpha_portfolio": os.path.join(ALPHA_DIR, "alpha_grid_portfolio.csv"),
    "alpha_coefs": os.path.join(ALPHA_DIR, "alpha_grid_coefficients.csv"),
}

os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 16,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR_EN = "#1f77b4"
COLOR_RF = "#d62728"
COLOR_NN = "#2ca02c"
COLOR_BENCH = "#7f7f7f"

DECADES = ["1980s", "1990s", "2000s", "2010s", "2020s"]
DECADE_RANGES = {
    "1980s": (1980, 1989),
    "1990s": (1990, 1999),
    "2000s": (2000, 2009),
    "2010s": (2010, 2019),
    "2020s": (2020, 2024),
}


def save_both(fig, name):
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)


def rename_size(df, col="feature"):
    if col in df.columns:
        df = df.copy()
        df[col] = df[col].replace({"SIZE": "log_mvel1"})
    return df


# =====================================================================
# FIGURE 1: Rolling Window Methodology
# =====================================================================
def figure_1_rolling_window():
    en = pd.read_csv(CSV_PATHS["window_en"])
    rf = pd.read_csv(CSV_PATHS["window_rf"])
    nn = pd.read_csv(CSV_PATHS["window_nn"])

    def decade_avg(df):
        return {d: df.loc[(df["pred_year"] >= s) & (df["pred_year"] <= e), "oos_r2"].mean()
                for d, (s, e) in DECADE_RANGES.items()}

    en_d, rf_d, nn_d = decade_avg(en), decade_avg(rf), decade_avg(nn)

    fig, ax = plt.subplots(figsize=(13, 9))
    for i, year in enumerate(range(1980, 2025)):
        ax.barh(i + 1, 10, left=year - 10, color="#a8c5e8", height=0.7)
        ax.barh(i + 1, 1, left=year, color="#f08080", height=0.7)

    ax.barh([0], [0], color="#a8c5e8", label="Training window (10 years)")
    ax.barh([0], [0], color="#f08080", label="Prediction year (1 year)")

    for d_start in [1990, 2000, 2010, 2020]:
        ax.axvline(d_start, ls=":", color="grey", lw=1.2, alpha=0.7)

    decade_centers = {"1980s": 1985, "1990s": 1995, "2000s": 2005, "2010s": 2015, "2020s": 2022}
    for d, x in decade_centers.items():
        ax.text(x, -1.5, d, ha="center", va="top", fontsize=11, fontstyle="italic", color="grey")

    decade_y = {"1980s": 5, "1990s": 15, "2000s": 25, "2010s": 35, "2020s": 43}
    for d, y in decade_y.items():
        text = (f"{d} avg OOS R²\n"
                f"EN: {en_d[d]:+.4f}\n"
                f"RF: {rf_d[d]:+.4f}\n"
                f"NN: {nn_d[d]:+.4f}")
        ax.text(2030, y, text, fontsize=11, va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", lw=0.8))

    ax.set_xlim(1968, 2038)
    ax.set_ylim(-3, 47)
    ax.set_xlabel("Year")
    ax.set_ylabel("Window Index (Prediction Year)")
    ax.set_title("Rolling Window Methodology: 45 Training-Prediction Periods (1980–2024)")
    yticks = list(range(1, 46, 3))
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"W{i:02d}: {1979 + i}" for i in yticks])
    ax.invert_yaxis()
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(axis="x", alpha=0.2)

    save_both(fig, "fig1_rolling_window")
    print("  saved: fig1_rolling_window")


# =====================================================================
# FIGURE 2: Return Validation Dashboard (reads from parquet)
# =====================================================================
def figure_2_return_validation():
    parquet_path = os.path.join(PROJECT_ROOT, "data_clean", "crsp_monthly_clean.parquet")
    if not os.path.exists(parquet_path):
        parquet_path = os.path.join(PROJECT_ROOT, "data_clean", "crsp_monthly_with_daily.parquet")
    if not os.path.exists(parquet_path):
        print(f"  skipped: fig2 — could not find CRSP monthly parquet in data_clean/")
        return

    df = pd.read_parquet(parquet_path)

    date_col = ret_col = permno_col = None
    for c in df.columns:
        if c.lower() in ("date", "datadate", "month"):
            date_col = c
        if c.lower() in ("ret_adj", "ret", "return"):
            ret_col = c
        if c.lower() == "permno":
            permno_col = c

    if date_col is None or ret_col is None:
        print(f"  skipped: fig2 — could not find date/return columns")
        print(f"    columns found: {list(df.columns)[:20]}")
        return

    df[date_col] = pd.to_datetime(df[date_col])
    raw_returns_all = df[ret_col].dropna()

    def winsorize_group(s, lo=0.01, hi=0.99):
        if len(s) < 10:
            return s
        ql, qh = s.quantile(lo), s.quantile(hi)
        return s.clip(ql, qh)

    df["ret_winsorized"] = df.groupby(date_col)[ret_col].transform(winsorize_group)
    winsor_returns_all = df["ret_winsorized"].dropna()

    monthly = df.groupby(date_col).agg(
        ew_winsor=("ret_winsorized", "mean"),
        n_stocks=(permno_col if permno_col else ret_col, "count"),
    ).reset_index().sort_values(date_col)

    dates = monthly[date_col]
    monthly_ret = monthly["ew_winsor"]
    cum = (1 + monthly_ret).cumprod()
    n_stocks = monthly["n_stocks"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Return Validation Dashboard", fontsize=17, y=1.0)

    ax = axes[0, 0]
    ax.plot(dates, cum, color=COLOR_EN, lw=1.6)
    ax.set_title("Equal-Weighted Cumulative Return")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    positive = monthly_ret.where(monthly_ret >= 0, 0)
    negative = monthly_ret.where(monthly_ret < 0, 0)
    ax.bar(dates, positive, color=COLOR_EN, width=20)
    ax.bar(dates, negative, color=COLOR_RF, width=20)
    ax.set_title("Equal-Weighted Monthly Returns")
    ax.set_ylabel("Monthly Return")
    ax.axhline(0, color="black", lw=0.6)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    bins = np.linspace(-0.4, 1.0, 200)
    ax.hist(raw_returns_all, bins=bins, density=True, color=COLOR_RF,
            alpha=0.6, label="Raw")
    ax.hist(winsor_returns_all, bins=bins, density=True, color=COLOR_EN,
            alpha=0.6, label="Winsorized")
    ax.set_title("Return Distribution: Raw vs Winsorized")
    ax.set_xlabel("Monthly Return")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(dates, n_stocks, color=COLOR_NN, lw=1.6)
    ax.set_title("Stocks in Universe per Month")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_both(fig, "fig2_return_validation")
    print("  saved: fig2_return_validation")


# =====================================================================
# FIGURE 3: OOS R² by Prediction Year
# =====================================================================
def figure_3_oos_r2():
    en = pd.read_csv(CSV_PATHS["window_en"])
    rf = pd.read_csv(CSV_PATHS["window_rf"])
    nn = pd.read_csv(CSV_PATHS["window_nn"])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(en["pred_year"], en["oos_r2"], "o-", color=COLOR_EN, lw=2, markersize=6, label="ElasticNet")
    ax.plot(rf["pred_year"], rf["oos_r2"], "s-", color=COLOR_RF, lw=2, markersize=6, label="Random Forest")
    ax.plot(nn["pred_year"], nn["oos_r2"], "^-", color=COLOR_NN, lw=2, markersize=6, label="Neural Network")
    ax.axhline(0, color="black", lw=1)
    for d_start in [1990, 2000, 2010, 2020]:
        ax.axvline(d_start, ls=":", color="grey", lw=1, alpha=0.5)
    ax.set_xlabel("Prediction Year")
    ax.set_ylabel("Out-of-Sample R²")
    ax.set_title("Out-of-Sample R² by Prediction Year, All Models")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.3)

    save_both(fig, "fig3_oos_r2")
    print("  saved: fig3_oos_r2")


# =====================================================================
# FIGURE 4: Decile Returns
# =====================================================================
def figure_4_decile_returns():
    en = pd.read_csv(CSV_PATHS["decile_en"])
    rf = pd.read_csv(CSV_PATHS["decile_rf"])
    nn = pd.read_csv(CSV_PATHS["decile_nn"])

    fig, ax = plt.subplots(figsize=(12, 6))
    deciles = np.arange(1, 11)
    width = 0.27
    ax.bar(deciles - width, en["Mean_Monthly"] * 100, width, color=COLOR_EN, label="ElasticNet")
    ax.bar(deciles, rf["Mean_Monthly"] * 100, width, color=COLOR_RF, label="Random Forest")
    ax.bar(deciles + width, nn["Mean_Monthly"] * 100, width, color=COLOR_NN, label="Neural Network")
    ax.set_xlabel("Predicted Decile (D1 = lowest, D10 = highest)")
    ax.set_ylabel("Mean Monthly Return (%)")
    ax.set_title("Mean Monthly Returns by Predicted Decile, All Models (1980–2024)")
    ax.set_xticks(deciles)
    ax.set_xticklabels([f"D{i}" for i in deciles])
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", lw=0.8)

    save_both(fig, "fig4_decile_returns")
    print("  saved: fig4_decile_returns")


# =====================================================================
# FIGURE 5: Feature Entry Rank Heatmap
# =====================================================================
def figure_5_entry_rank_heatmap():
    df = pd.read_csv(CSV_PATHS["alpha_pivot"])
    df = rename_size(df)

    df_sorted = df.sort_values("mean_rank").head(15).reset_index(drop=True)
    matrix = df_sorted[DECADES].values
    features = df_sorted["feature"].tolist()

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=1, vmax=20)

    for i in range(len(features)):
        for j in range(len(DECADES)):
            val = matrix[i, j]
            color = "white" if val > 12 else "black"
            ax.text(j, i, f"{int(val)}", ha="center", va="center",
                    color=color, fontsize=12, fontweight="bold")

    ax.set_xticks(range(len(DECADES)))
    ax.set_xticklabels(DECADES)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel("Decade")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Entry Rank by Decade (Alpha-Grid Analysis)")
    cbar = plt.colorbar(im, ax=ax, label="Entry Rank (lower = earlier entry)")
    cbar.ax.tick_params(labelsize=11)

    save_both(fig, "fig5_entry_rank_heatmap")
    print("  saved: fig5_entry_rank_heatmap")


# =====================================================================
# FIGURE 6: Sharpe vs Non-Zero Coefficients
# =====================================================================
def figure_6_sharpe_vs_nonzero():
    path = CSV_PATHS["alpha_portfolio"]
    if not os.path.exists(path):
        print(f"  skipped: fig6 — {path} not found")
        return

    df = pd.read_csv(path)
    # Convert integer decade (1980, 1990, ...) to string ("1980s", "1990s", ...)
    if df["decade"].dtype != object:
        df["decade"] = df["decade"].astype(int).astype(str) + "s"

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, 5))
    for i, decade in enumerate(DECADES):
        sub = df[df["decade"] == decade].sort_values("n_nonzero")
        if sub.empty:
            continue
        ax.plot(sub["n_nonzero"], sub["ls_sharpe"], "o-", color=colors[i],
                lw=2, markersize=6, label=decade)
    ax.set_xlabel("Number of Non-Zero Coefficients")
    ax.set_ylabel("Annualized Long-Short Sharpe Ratio")
    ax.set_title("Long-Short Sharpe Ratio versus Number of Non-Zero Coefficients, by Decade")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.3)
    ax.axhline(0, color="black", lw=0.8)

    save_both(fig, "fig6_sharpe_vs_nonzero")
    print("  saved: fig6_sharpe_vs_nonzero")


# =====================================================================
# FIGURE 10: Sharpe vs Regularization Strength (α)
# =====================================================================
def figure_10_sharpe_vs_alpha():
    path = CSV_PATHS["alpha_portfolio"]
    if not os.path.exists(path):
        print(f"  skipped: fig10 — {path} not found")
        return

    df = pd.read_csv(path)
    # Convert integer decade (1980, 1990, ...) to string ("1980s", "1990s", ...)
    if df["decade"].dtype != object:
        df["decade"] = df["decade"].astype(int).astype(str) + "s"

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, 5))
    for i, decade in enumerate(DECADES):
        sub = df[df["decade"] == decade].sort_values("alpha", ascending=False)
        if sub.empty:
            continue
        ax.plot(sub["alpha"], sub["ls_sharpe"], "o-", color=colors[i],
                lw=2, markersize=6, label=decade)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Regularization Strength (α, log scale, reversed)")
    ax.set_ylabel("Annualized Long-Short Sharpe Ratio")
    ax.set_title("Long-Short Sharpe Ratio versus Regularization Strength (α), by Decade")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.3, which="both")
    ax.axhline(0, color="black", lw=0.8)

    save_both(fig, "fig10_sharpe_vs_alpha")
    print("  saved: fig10_sharpe_vs_alpha")


# =====================================================================
# FIGURE 10: Sharpe vs Regularization Strength (α)
# =====================================================================
def figure_10_sharpe_vs_alpha():
    path = CSV_PATHS["alpha_portfolio"]
    if not os.path.exists(path):
        print(f"  skipped: fig10 — {path} not found")
        return

    df = pd.read_csv(path)
    # Convert integer decade (1980, 1990, ...) to string ("1980s", "1990s", ...)
    if df["decade"].dtype != object:
        df["decade"] = df["decade"].astype(int).astype(str) + "s"

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, 5))
    for i, decade in enumerate(DECADES):
        sub = df[df["decade"] == decade].sort_values("alpha", ascending=False)
        if sub.empty:
            continue
        ax.plot(sub["alpha"], sub["ls_sharpe"], "o-", color=colors[i],
                lw=2, markersize=6, label=decade)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Regularization Strength (α, log scale, reversed)")
    ax.set_ylabel("Annualized Long-Short Sharpe Ratio")
    ax.set_title("Long-Short Sharpe Ratio versus Regularization Strength (α), by Decade")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.3, which="both")
    ax.axhline(0, color="black", lw=0.8)

    save_both(fig, "fig10_sharpe_vs_alpha")
    print("  saved: fig10_sharpe_vs_alpha")


# =====================================================================
# APPENDIX FIGURES: 2x2 Model Diagnostics
# =====================================================================
def appendix_diagnostic(window_key, importance_key, decile_key, decade_perf_key,
                       model_name, color, fig_name):
    window = pd.read_csv(CSV_PATHS[window_key])
    importance = pd.read_csv(CSV_PATHS[importance_key])
    importance = rename_size(importance)
    decile = pd.read_csv(CSV_PATHS[decile_key])
    decade_perf = pd.read_csv(CSV_PATHS[decade_perf_key])

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"{model_name} Rolling Window Results", fontsize=17, y=1.0)

    ax = axes[0, 0]
    ax.plot(window["pred_year"], window["oos_r2"], "o-", color=color, lw=2, markersize=5)
    ax.axhline(0, color="black", lw=0.8)
    for d_start in [1990, 2000, 2010, 2020]:
        ax.axvline(d_start, ls=":", color="grey", lw=1, alpha=0.5)
    ax.set_xlabel("Prediction Year")
    ax.set_ylabel("Out-of-Sample R²")
    ax.set_title("OOS R² by Prediction Year")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    cum, cum_x, cum_y = 1.0, [1980], [1.0]
    for _, row in decade_perf.iterrows():
        decade = row["Decade"]
        if decade not in DECADE_RANGES:
            continue
        start_year, _ = DECADE_RANGES[decade]
        monthly = row["LS_Mean_Monthly"]
        n_months = int(row["N_months"])
        for m in range(n_months):
            cum *= (1 + monthly)
            cum_x.append(start_year + (m + 1) / 12)
            cum_y.append(cum)
    ax.plot(cum_x, cum_y, color=color, lw=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative Value of $1")
    ax.set_title("Long-Short Portfolio Cumulative Return")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")

    ax = axes[1, 0]
    pivot = importance[importance["rank"] <= 5].copy()
    decade_labels = {1980: "1980s", 1990: "1990s", 2000: "2000s", 2010: "2010s", 2020: "2020s"}
    pivot["decade_label"] = pivot["decade"].map(decade_labels)
    top_features = pivot["feature"].unique()
    heat = pd.DataFrame(index=top_features, columns=DECADES, dtype=float)
    for _, row in pivot.iterrows():
        heat.loc[row["feature"], row["decade_label"]] = row["rank"]
    heat = heat.dropna(thresh=1)
    heat["mean_rank"] = heat.mean(axis=1, skipna=True)
    heat = heat.sort_values("mean_rank").drop(columns="mean_rank")

    im = ax.imshow(heat.values, cmap="RdYlGn_r", aspect="auto", vmin=1, vmax=5)
    for i in range(len(heat)):
        for j in range(len(DECADES)):
            val = heat.iloc[i, j]
            if not pd.isna(val):
                ax.text(j, i, f"{int(val)}", ha="center", va="center",
                        color="black", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(DECADES)))
    ax.set_xticklabels(DECADES)
    ax.set_yticks(range(len(heat)))
    ax.set_yticklabels(heat.index)
    ax.set_title("Top-5 Feature Rank by Decade")
    plt.colorbar(im, ax=ax, label="Rank")

    ax = axes[1, 1]
    ax.bar(decile["Decile"], decile["Mean_Monthly"] * 100, color=color, alpha=0.8)
    ax.set_xlabel("Predicted Decile")
    ax.set_ylabel("Mean Monthly Return (%)")
    ax.set_title("Decile Mean Monthly Returns")
    ax.set_xticks(range(1, 11))
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_both(fig, fig_name)
    print(f"  saved: {fig_name}")


# =====================================================================
# NEW FIGURE: Decade-by-decade cumulative performance (5 panels)
# =====================================================================
def figure_decade_cumulative():
    en = pd.read_csv(CSV_PATHS["decade_perf_en"])
    rf = pd.read_csv(CSV_PATHS["decade_perf_rf"])
    nn = pd.read_csv(CSV_PATHS["decade_perf_nn"])

    fig, axes = plt.subplots(1, 5, figsize=(22, 5.5), sharey=False)

    for idx, decade in enumerate(DECADES):
        ax = axes[idx]
        start_year, _ = DECADE_RANGES[decade]

        for name, df, color in [("ElasticNet", en, COLOR_EN),
                                ("Random Forest", rf, COLOR_RF),
                                ("Neural Network", nn, COLOR_NN)]:
            row = df[df["Decade"] == decade]
            if row.empty:
                continue
            monthly = row["LS_Mean_Monthly"].iloc[0]
            n_months = int(row["N_months"].iloc[0])
            cum, xs, ys = 100.0, [start_year], [100.0]
            for m in range(n_months):
                cum *= (1 + monthly)
                xs.append(start_year + (m + 1) / 12)
                ys.append(cum)
            ax.plot(xs, ys, color=color, lw=2.5, label=name)

        ax.set_title(decade, fontsize=15, fontweight="bold")
        ax.set_xlabel("Year")
        if idx == 0:
            ax.set_ylabel("Portfolio Value (Starting at $100)")
        ax.grid(alpha=0.3)
        ax.axhline(100, color="black", lw=0.6, alpha=0.5, ls="--")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, framealpha=0.95,
               bbox_to_anchor=(0.5, 1.02), fontsize=13)
    fig.suptitle("Long-Short Portfolio Cumulative Performance by Decade ($100 Initial Investment)",
                 fontsize=16, y=1.08)

    plt.tight_layout()
    save_both(fig, "fig_decade_cumulative")
    print("  saved: fig_decade_cumulative")

# =====================================================================
# INTRO FIGURE: Simplified decade cumulative
# =====================================================================
def figure_decade_cumulative_intro():
    """Intro figure: cumulative growth of $100 across the full 1980-2024 sample,
    showing all three ML models vs the equal-weighted market benchmark.

    The benchmark comes from the actual CRSP equal-weighted universe in
    data_clean/crsp_monthly_clean.parquet, computed month by month.
    """
    en = pd.read_csv(CSV_PATHS["decade_perf_en"])
    rf = pd.read_csv(CSV_PATHS["decade_perf_rf"])
    nn = pd.read_csv(CSV_PATHS["decade_perf_nn"])

    # Compute cumulative path for each ML model across the full sample
    def build_cumulative(df, start_year=1980):
        cum, xs, ys = 100.0, [start_year], [100.0]
        for decade in DECADES:
            row = df[df["Decade"] == decade]
            if row.empty:
                continue
            d_start, _ = DECADE_RANGES[decade]
            monthly = row["LS_Mean_Monthly"].iloc[0]
            n_months = int(row["N_months"].iloc[0])
            for m in range(n_months):
                cum *= (1 + monthly)
                xs.append(d_start + (m + 1) / 12)
                ys.append(cum)
        return xs, ys

    en_xs, en_ys = build_cumulative(en)
    rf_xs, rf_ys = build_cumulative(rf)
    nn_xs, nn_ys = build_cumulative(nn)

    # Compute equal-weighted market benchmark from CRSP parquet
    parquet_path = os.path.join(PROJECT_ROOT, "data_clean", "crsp_monthly_clean.parquet")
    if not os.path.exists(parquet_path):
        parquet_path = os.path.join(PROJECT_ROOT, "data_clean", "crsp_monthly_with_daily.parquet")

    bench_xs, bench_ys = None, None
    if os.path.exists(parquet_path):
        crsp = pd.read_parquet(parquet_path)
        # Detect columns
        date_col = ret_col = None
        for c in crsp.columns:
            if c.lower() in ("date", "datadate", "month"):
                date_col = c
            if c.lower() in ("ret_adj", "ret", "return"):
                ret_col = c
        if date_col and ret_col:
            crsp[date_col] = pd.to_datetime(crsp[date_col])
            crsp = crsp[(crsp[date_col] >= "1980-01-01") & (crsp[date_col] <= "2024-12-31")]
            monthly_market = crsp.groupby(date_col)[ret_col].mean().sort_index()
            cum_market = (1 + monthly_market).cumprod() * 100
            bench_xs = [d.year + (d.month - 1) / 12 for d in cum_market.index]
            bench_ys = cum_market.values.tolist()
            # prepend $100 at start
            bench_xs = [1980] + bench_xs
            bench_ys = [100.0] + bench_ys

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(en_xs, en_ys, color=COLOR_EN, lw=2.8, label="ElasticNet (ML)")
    ax.plot(rf_xs, rf_ys, color=COLOR_RF, lw=2.8, label="Random Forest (ML)")
    ax.plot(nn_xs, nn_ys, color=COLOR_NN, lw=2.8, label="Neural Network (ML)")
    if bench_xs is not None:
        ax.plot(bench_xs, bench_ys, color=COLOR_BENCH, lw=2.5, ls="--",
                label="Equal-Weighted Market Benchmark")

    ax.axhline(100, color="black", lw=0.8, alpha=0.5, ls=":")
    ax.set_xlabel("Year")
    ax.set_ylabel("Portfolio Value (Starting at $100, log scale)")
    ax.set_yscale("log")
    ax.set_title("Machine Learning Long-Short Portfolios vs. Market Benchmark\n($100 Initial Investment, 1980–2024)",
                 fontsize=15)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=12)
    ax.grid(alpha=0.3, which="both")
    ax.set_xlim(1980, 2025)

    plt.tight_layout()
    save_both(fig, "fig_intro_decade_cumulative")
    print("  saved: fig_intro_decade_cumulative")


# =====================================================================
# INTRO FIGURE: Simplified rolling window
# =====================================================================
def figure_1_rolling_window_intro():
    """Simplified rolling-window figure for the introduction.

    Shows the concept with 5 representative windows spread across the sample period
    (one starting near each decade boundary), without numbering them as 'Window 1-5'
    to avoid implying these are the first 5 of 45.
    """
    fig, ax = plt.subplots(figsize=(12, 5.5))

    # Show 5 representative windows: one predicting each decade boundary
    sample_pred_years = [1980, 1990, 2000, 2010, 2020]

    for i, pred_year in enumerate(sample_pred_years):
        train_start = pred_year - 10
        ax.barh(i + 1, 10, left=train_start, color="#a8c5e8",
                edgecolor="#4a7ab8", height=0.65, lw=0.8)
        ax.barh(i + 1, 1, left=pred_year, color="#f08080",
                edgecolor="#c04040", height=0.65, lw=0.8)
        # Label by training period, not by misleading window index
        ax.text(train_start - 1, i + 1, f"Train {train_start}–{pred_year - 1}, predict {pred_year}",
                ha="right", va="center", fontsize=11)

    ax.barh([0], [0], color="#a8c5e8", label="Training window (10 years of historical data)")
    ax.barh([0], [0], color="#f08080", label="Prediction year (out-of-sample forecast)")

    for d_start in [1990, 2000, 2010, 2020]:
        ax.axvline(d_start, ls=":", color="grey", lw=1.2, alpha=0.6)

    ax.annotate(
        "Each year, the model is retrained on the\nmost recent 10 years and predicts\nthe following year out-of-sample.\nThis produces 45 rolling windows in total.",
        xy=(2024, 5.5), xytext=(2026, 3.5),
        fontsize=11, ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.5", fc="#fff8dc", ec="grey", lw=0.8),
        arrowprops=dict(arrowstyle="->", color="grey", lw=1)
    )

    ax.set_xlim(1960, 2040)
    ax.set_ylim(0, 6.5)
    ax.set_xlabel("Year")
    ax.set_yticks([])
    ax.set_title("Rolling-Window Estimation: Five Representative Windows from the 45-Window Sample",
                 fontsize=14)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=11)
    ax.grid(axis="x", alpha=0.2)
    ax.invert_yaxis()

    plt.tight_layout()
    save_both(fig, "fig_intro_rolling_window")
    print("  saved: fig_intro_rolling_window")


# =====================================================================
# RUN ALL
# =====================================================================
if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output dir:   {OUT_DIR}")
    print("\nRegenerating figures...\n")

    print("Main text figures:")
    figure_1_rolling_window()
    figure_2_return_validation()
    figure_3_oos_r2()
    figure_4_decile_returns()
    figure_5_entry_rank_heatmap()
    figure_6_sharpe_vs_nonzero()

    print("\nAppendix figures:")
    appendix_diagnostic("window_en", "importance_en", "decile_en", "decade_perf_en",
                       "ElasticNet", COLOR_EN, "figA1_elasticnet")
    appendix_diagnostic("window_rf", "importance_rf", "decile_rf", "decade_perf_rf",
                       "Random Forest", COLOR_RF, "figA2_random_forest")
    appendix_diagnostic("window_nn", "importance_nn", "decile_nn", "decade_perf_nn",
                       "Neural Network", COLOR_NN, "figA3_neural_network")
    figure_10_sharpe_vs_alpha()

    print("\nNew figures (Chunjie's suggestions):")
    figure_decade_cumulative()
    figure_decade_cumulative_intro()
    figure_1_rolling_window_intro()

    print(f"\nDone. All figures saved to {OUT_DIR}/ as both .png and .pdf")