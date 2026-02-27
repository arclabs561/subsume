# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy"]
# ///
"""T-norm and t-conorm comparison for the fuzzy module.

Shows the three standard t-norm families as 2D contour plots (heat maps)
and 1D slices. Inspired by FuzzQE (Chen et al., AAAI 2022) Figure 2.

Two rows:
  Top: T-norms (fuzzy intersection) -- Min, Product, Lukasiewicz
  Bottom: 1D slices at b=0.7 showing how each t-norm compresses the result

Style: distill.pub-inspired. Consistent with other subsume plots.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Color palette ─────────────────────────────────────────────────────
BG = "#fafafa"
GRID = "#e5e7eb"
TEXT = "#374151"
BLUE = "#3b82f6"
GREEN = "#22c55e"
AMBER = "#f59e0b"
SLATE = "#64748b"
SLATE_LIGHT = "#94a3b8"

# ── T-norm functions ──────────────────────────────────────────────────
def tnorm_min(a, b):
    return np.minimum(a, b)

def tnorm_product(a, b):
    return a * b

def tnorm_lukasiewicz(a, b):
    return np.maximum(a + b - 1, 0)

tnorms = [
    ("Min (Godel)", tnorm_min),
    ("Product", tnorm_product),
    ("Lukasiewicz", tnorm_lukasiewicz),
]

# ── Grid ──────────────────────────────────────────────────────────────
n = 200
a = np.linspace(0, 1, n)
b = np.linspace(0, 1, n)
A, B = np.meshgrid(a, b)

# ── Figure: 2 rows x 3 cols ──────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5),
                         gridspec_kw={"height_ratios": [3, 2], "hspace": 0.35, "wspace": 0.30})
fig.patch.set_facecolor("white")

# ── Top row: contour plots ───────────────────────────────────────────
for i, (name, fn) in enumerate(tnorms):
    ax = axes[0, i]
    ax.set_facecolor(BG)
    Z = fn(A, B)

    # Filled contours
    levels = np.linspace(0, 1, 11)
    cf = ax.contourf(A, B, Z, levels=levels, cmap="Blues", alpha=0.8)
    ax.contour(A, B, Z, levels=levels, colors=SLATE_LIGHT, linewidths=0.4, alpha=0.6)

    # Contour labels for key levels
    cs = ax.contour(A, B, Z, levels=[0.25, 0.5, 0.75], colors=TEXT, linewidths=0.6)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f")

    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("$a$", fontsize=8.5, color=TEXT)
    if i == 0:
        ax.set_ylabel("$b$", fontsize=8.5, color=TEXT)
    ax.set_title(name, fontsize=9.5, fontweight="bold", color=TEXT, pad=6)
    ax.tick_params(labelsize=7, colors=SLATE_LIGHT)
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(0.8)

# Colorbar
cbar = fig.colorbar(cf, ax=axes[0, :], shrink=0.8, aspect=30, pad=0.02)
cbar.set_label("$T(a, b)$", fontsize=8, color=TEXT)
cbar.ax.tick_params(labelsize=6, colors=SLATE_LIGHT)

# ── Bottom row: 1D slices at b = 0.7 ────────────────────────────────
b_fixed = 0.7
colors = [BLUE, GREEN, AMBER]

for i, (name, fn) in enumerate(tnorms):
    ax = axes[1, i]
    ax.set_facecolor(BG)

    y = fn(a, b_fixed)
    ax.fill_between(a, 0, y, color=colors[i], alpha=0.12)
    ax.plot(a, y, color=colors[i], linewidth=1.8, label=f"$T(a, {b_fixed})$")

    # Reference: identity line T(a,1) = a
    ax.plot(a, a, color=SLATE_LIGHT, linewidth=0.8, linestyle="--", alpha=0.5,
            label="$T(a, 1) = a$")

    # Mark the fixed b value
    ax.axhline(y=b_fixed, color=SLATE_LIGHT, linewidth=0.5, linestyle=":", alpha=0.4)
    ax.text(0.02, b_fixed + 0.03, f"$b = {b_fixed}$", fontsize=6, color=SLATE, va="bottom")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("$a$", fontsize=8.5, color=TEXT)
    if i == 0:
        ax.set_ylabel("$T(a, b)$", fontsize=8.5, color=TEXT)
    ax.set_title(f"Slice at $b = {b_fixed}$", fontsize=8.5, color=SLATE, pad=4)
    ax.legend(fontsize=6.5, loc="upper left", frameon=True, fancybox=False,
              edgecolor=GRID, framealpha=0.9)
    ax.tick_params(labelsize=7, colors=SLATE_LIGHT)
    ax.grid(True, alpha=0.3, linewidth=0.4, color=GRID)
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ── Suptitle ──────────────────────────────────────────────────────────
fig.suptitle("T-norm families: fuzzy intersection operators",
             fontsize=11, fontweight="bold", color=TEXT, y=0.98)

fig.savefig("/Users/arc/Documents/dev/subsume/docs/fuzzy_tnorms.png",
            dpi=180, bbox_inches="tight", facecolor="white")
print("Saved subsume/docs/fuzzy_tnorms.png")
