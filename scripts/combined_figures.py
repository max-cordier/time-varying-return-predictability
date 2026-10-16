"""
Combined Main-Text Figures
==========================
Generates two combined figures for the main results section,
showing all three models side-by-side:

  Figure 1: OOS R² over time, all three models on one chart
  Figure 2: Mean monthly return by predicted decile, all three models

These are the cleaner "comparison" figures requested by the promotor
to replace the model-specific 2x2 panels (which move to the appendix).

Inputs:  ../output/{elasticnet,random_forest,neural_network}/
         (window_metrics.csv and decile_summary.csv from each)
Output:  ../output/combined_figures/

Run from: Master_Thesis/scripts/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
EN_DIR  = Path("../output/elasticnet")
RF_DIR  = Path("../output/random_forest")
NN_DIR  = Path("../output/neural_network")
OUT_DIR = Path("../output/combined_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Consistent colors and styling across both figures
COLORS = {
    "ElasticNet"    : "#1f77b4",  # blue
    "Random Forest" : "#d62728",  # red
    "Neural Network": "#2ca02c",  # green
}

def hline(): print("─" * 60)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
hline()
print("STEP 1: Loading data from all three models")
hline()

en_win = pd.read_csv(EN_DIR / "window_metrics_en.csv")
rf_win = pd.read_csv(RF_DIR / "window_metrics_rf.csv")
nn_win = pd.read_csv(NN_DIR / "window_metrics.csv")

en_dec = pd.read_csv(EN_DIR / "decile_summary_en.csv")
rf_dec = pd.read_csv(RF_DIR / "decile_summary_rf.csv")
nn_dec = pd.read_csv(NN_DIR / "decile_summary.csv")

print(f"  ElasticNet     : {len(en_win)} windows, {len(en_dec)} deciles")
print(f"  Random Forest  : {len(rf_win)} windows, {len(rf_dec)} deciles")
print(f"  Neural Network : {len(nn_win)} windows, {len(nn_dec)} deciles")

# ── 2. FIGURE 1: OOS R² OVER TIME ─────────────────────────────────────────────
hline()
print("STEP 2: Building Figure 1 — OOS R² over time")
hline()

fig, ax = plt.subplots(figsize=(11, 5.5))

ax.plot(en_win["pred_year"], en_win["oos_r2"],
        marker="o", markersize=4, lw=1.5,
        color=COLORS["ElasticNet"], label="ElasticNet")
ax.plot(rf_win["pred_year"], rf_win["oos_r2"],
        marker="s", markersize=4, lw=1.5,
        color=COLORS["Random Forest"], label="Random Forest")
ax.plot(nn_win["pred_year"], nn_win["oos_r2"],
        marker="^", markersize=4, lw=1.5,
        color=COLORS["Neural Network"], label="Neural Network")

ax.axhline(0, color="black", lw=0.8, linestyle="-")

# Decade dividers and labels
decade_starts = [1980, 1990, 2000, 2010, 2020]
y_min, y_max  = ax.get_ylim()
for d in decade_starts[1:]:
    ax.axvline(d, color="gray", lw=0.5, linestyle=":", alpha=0.5)

ax.set_xlabel("Prediction Year")
ax.set_ylabel("Out-of-Sample R²")
ax.set_title("Out-of-Sample R² by Prediction Year, All Models")
ax.legend(loc="best", fontsize=10, framealpha=0.9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig1_oos_r2_combined.png", dpi=200, bbox_inches="tight")
plt.savefig(OUT_DIR / "fig1_oos_r2_combined.pdf", bbox_inches="tight")
plt.close()
print(f"  Saved → {OUT_DIR / 'fig1_oos_r2_combined.png'}")
print(f"  Saved → {OUT_DIR / 'fig1_oos_r2_combined.pdf'}")

# ── 3. FIGURE 2: DECILE RETURNS ───────────────────────────────────────────────
hline()
print("STEP 3: Building Figure 2 — Decile returns by model")
hline()

fig, ax = plt.subplots(figsize=(11, 5.5))

deciles  = en_dec["Decile"].values
n_models = 3
bar_w    = 0.27
x        = np.arange(len(deciles))

ax.bar(x - bar_w, en_dec["Mean_Monthly"] * 100, bar_w,
       color=COLORS["ElasticNet"], label="ElasticNet")
ax.bar(x,         rf_dec["Mean_Monthly"] * 100, bar_w,
       color=COLORS["Random Forest"], label="Random Forest")
ax.bar(x + bar_w, nn_dec["Mean_Monthly"] * 100, bar_w,
       color=COLORS["Neural Network"], label="Neural Network")

ax.axhline(0, color="black", lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f"D{d}" for d in deciles])
ax.set_xlabel("Predicted Decile (D1 = lowest predicted, D10 = highest predicted)")
ax.set_ylabel("Mean Monthly Return (%)")
ax.set_title("Mean Monthly Return by Predicted Decile, All Models")
ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig2_decile_returns_combined.png", dpi=200, bbox_inches="tight")
plt.savefig(OUT_DIR / "fig2_decile_returns_combined.pdf", bbox_inches="tight")
plt.close()
print(f"  Saved → {OUT_DIR / 'fig2_decile_returns_combined.png'}")
print(f"  Saved → {OUT_DIR / 'fig2_decile_returns_combined.pdf'}")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
hline()
print("FINAL SUMMARY")
hline()
print(f"  Output saved to    : {OUT_DIR}")
print(f"\n  Figure 1: OOS R² over time, all three models")
print(f"    → fig1_oos_r2_combined.png/.pdf")
print(f"    → place after Table 3 in Out-of-Sample Predictive Performance section")
print(f"\n  Figure 2: Mean monthly return by predicted decile, all three models")
print(f"    → fig2_decile_returns_combined.png/.pdf")
print(f"    → place after Table 6 in Decile Return Monotonicity section")
hline()