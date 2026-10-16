"""
Rolling Window Timeline Visualization
======================================
Generates the "killer graph" requested by the promotor: a visual
representation of the rolling window methodology showing all 45
training periods, validation periods, and prediction years, with
key findings annotated.

This figure goes in the methodology section to make the rolling
window design instantly clear to readers (especially opponents
who may not read the full paper).

Inputs:  ../output/elasticnet/window_metrics.csv (or any model's)
         — used to annotate decade-level OOS R² findings
Output:  ../output/combined_figures/methodology_timeline.png/.pdf

Run from: Master_Thesis/scripts/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
EN_DIR  = Path("../output/elasticnet")
RF_DIR  = Path("../output/random_forest")
NN_DIR  = Path("../output/neural_network")
OUT_DIR = Path("../output/combined_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_YEARS = 10
START_YEAR  = 1970   # earliest training year (1980 - 10)
END_YEAR    = 2024   # last prediction year

# Colors
TRAIN_COLOR = "#A8C5E2"  # light blue
PRED_COLOR  = "#D4756B"  # coral
GRID_COLOR  = "#E0E0E0"

def hline(): print("─" * 60)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
hline()
print("STEP 1: Loading window metrics for annotation")
hline()

en_win = pd.read_csv(EN_DIR / "window_metrics_en.csv")
rf_win = pd.read_csv(RF_DIR / "window_metrics_rf.csv")
nn_win = pd.read_csv(NN_DIR / "window_metrics.csv")

# Compute decade-level mean OOS R² for each model
def decade_mean(df):
    df = df.copy()
    df["decade"] = (df["pred_year"] // 10) * 10
    return df.groupby("decade")["oos_r2"].mean()

en_dec_r2 = decade_mean(en_win)
rf_dec_r2 = decade_mean(rf_win)
nn_dec_r2 = decade_mean(nn_win)

decades = sorted(set(en_dec_r2.index) | set(rf_dec_r2.index) | set(nn_dec_r2.index))
print(f"  Decades analyzed: {decades}")
for dec in decades:
    en_v = en_dec_r2.get(dec, np.nan)
    rf_v = rf_dec_r2.get(dec, np.nan)
    nn_v = nn_dec_r2.get(dec, np.nan)
    print(f"  {dec}s — EN: {en_v:+.4f}  RF: {rf_v:+.4f}  NN: {nn_v:+.4f}")

# Build window list
windows = []
for pred_year in range(1980, 2025):
    windows.append({
        "train_start": pred_year - TRAIN_YEARS,
        "train_end"  : pred_year - 1,
        "pred_year"  : pred_year,
    })

# ── 2. BUILD TIMELINE FIGURE ──────────────────────────────────────────────────
hline()
print("STEP 2: Building rolling window timeline figure")
hline()

fig, ax = plt.subplots(figsize=(14, 9))

# Plot each window as a horizontal bar: training + prediction
for i, w in enumerate(windows):
    y = i
    # Training window
    train_rect = Rectangle(
        (w["train_start"], y - 0.35),
        TRAIN_YEARS, 0.7,
        facecolor=TRAIN_COLOR, edgecolor="white", linewidth=0.5
    )
    ax.add_patch(train_rect)

    # Prediction year (separate rectangle, immediately after training)
    pred_rect = Rectangle(
        (w["pred_year"], y - 0.35),
        1, 0.7,
        facecolor=PRED_COLOR, edgecolor="white", linewidth=0.5
    )
    ax.add_patch(pred_rect)

# Configure axes
ax.set_xlim(START_YEAR - 1, END_YEAR + 2)
ax.set_ylim(-1, len(windows))
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Window Index (Prediction Year)", fontsize=11)

# Y-tick labels — show every 3rd window's prediction year
ytick_idx    = list(range(0, len(windows), 3))
ytick_labels = [f"W{i+1:02d}: {windows[i]['pred_year']}" for i in ytick_idx]
ax.set_yticks(ytick_idx)
ax.set_yticklabels(ytick_labels, fontsize=8)

# Decade dividers
for decade in [1980, 1990, 2000, 2010, 2020]:
    ax.axvline(decade, color="gray", lw=0.6, linestyle=":", alpha=0.6, zorder=0)

# Decade labels at top
ymax = len(windows)
for decade in [1980, 1990, 2000, 2010, 2020]:
    ax.text(decade + 5, ymax + 0.5, f"{decade}s",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            color="dimgray")

# Decade-level OOS R² annotations on the right side of the figure
annotation_x = END_YEAR + 1.5
for decade in decades:
    decade_windows_idx = [
        i for i, w in enumerate(windows)
        if (w["pred_year"] // 10) * 10 == decade
    ]
    if not decade_windows_idx:
        continue
    y_mid = np.mean(decade_windows_idx)

    en_v = en_dec_r2.get(decade, np.nan)
    rf_v = rf_dec_r2.get(decade, np.nan)
    nn_v = nn_dec_r2.get(decade, np.nan)

    annotation = (
        f"{decade}s avg OOS R²\n"
        f"EN: {en_v:+.4f}\n"
        f"RF: {rf_v:+.4f}\n"
        f"NN: {nn_v:+.4f}"
    )
    ax.text(annotation_x, y_mid, annotation,
            ha="left", va="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", edgecolor="lightgray", lw=0.5))

# Legend
train_patch = mpatches.Patch(color=TRAIN_COLOR, label="Training window (10 years)")
pred_patch  = mpatches.Patch(color=PRED_COLOR,  label="Prediction year (1 year)")
ax.legend(handles=[train_patch, pred_patch],
          loc="upper left", fontsize=10, framealpha=0.95)

ax.set_title(
    "Rolling Window Methodology: 45 Training–Prediction Periods (1980–2024)",
    fontsize=12, fontweight="bold", pad=12
)
ax.invert_yaxis()  # First window at top

# Grid for readability
ax.set_axisbelow(True)
ax.grid(axis="x", alpha=0.2)

# Extend x-axis to fit annotation boxes
ax.set_xlim(START_YEAR - 1, END_YEAR + 9)

plt.tight_layout()
plt.savefig(OUT_DIR / "methodology_timeline.png", dpi=200, bbox_inches="tight")
plt.savefig(OUT_DIR / "methodology_timeline.pdf", bbox_inches="tight")
plt.close()
print(f"  Saved → {OUT_DIR / 'methodology_timeline.png'}")
print(f"  Saved → {OUT_DIR / 'methodology_timeline.pdf'}")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
hline()
print("FINAL SUMMARY")
hline()
print(f"  Output saved to: {OUT_DIR}")
print(f"\n  Place this figure in the Methodology section, in the subsection")
print(f"  describing the rolling window design. It visually shows:")
print(f"    - All 45 training windows (10 years each, light blue)")
print(f"    - All 45 prediction years (1 year each, coral)")
print(f"    - Decade-level OOS R² for each model (annotated on right)")
print(f"    - Decade dividers showing time progression")
hline()