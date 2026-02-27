# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "scipy"]
# ///
"""Generate Gumbel vs Gaussian vs Hard-box noise robustness plot for subsume README.

Compares containment loss under coordinate noise perturbation.
Gumbel soft boundaries absorb noise gracefully; hard/Gaussian boxes fail quickly.

Style: distill.pub-inspired. Muted palette, minimal chrome.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# ── Color palette ─────────────────────────────────────────────────────
BG = "#fafafa"
GRID = "#e5e7eb"
TEXT = "#374151"
BLUE = "#3b82f6"
RED = "#ef4444"
SLATE = "#64748b"
SLATE_LIGHT = "#94a3b8"

# ── Data ──────────────────────────────────────────────────────────────
noise = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
gumbel = np.array([0.0, 0.1, 0.2, 0.7, 1.4, 2.8, 4.8, 7.2, 11.5, 15.3, 17.6])
gaussian = np.array([51.0, 52.0, 53.5, 57.8, 62.0, 67.5, 77.0, 84.0, 94.0, 99.5, 100.0])

# Hard-box baseline (analytical)
d, margin = 10, 0.3
hard_box = []
for sigma in noise:
    if sigma == 0.0:
        hard_box.append(0.0)
    else:
        p_inside = (1.0 - 2.0 * norm.cdf(-margin / sigma)) ** d
        hard_box.append(100.0 * (1.0 - p_inside))
hard_box = np.array(hard_box)

# ── Plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 3.8))
fig.patch.set_facecolor("white")
ax.set_facecolor(BG)

for spine in ax.spines.values():
    spine.set_color(GRID)
    spine.set_linewidth(0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Fill regions to show the gap
ax.fill_between(noise, gumbel, gaussian, alpha=0.06, color=BLUE)

ax.plot(noise, gumbel, "o-", color=BLUE, linewidth=1.8, markersize=4,
        markeredgewidth=0, label="Gumbel box")
ax.plot(noise, gaussian, "s-", color=RED, linewidth=1.8, markersize=4,
        markeredgewidth=0, label="Gaussian box")
ax.plot(noise, hard_box, "^--", color=SLATE, linewidth=1.2, markersize=4,
        markeredgewidth=0, alpha=0.7, label=f"Hard box ($d$={d})")

ax.set_yscale("log")
ax.set_ylim(0.05, 150)
ax.set_xlim(0, 1.0)

ax.set_xlabel("Noise level (coordinate perturbation $\\sigma$)", fontsize=9, color=TEXT)
ax.set_ylabel("Containment loss (%, log scale)", fontsize=9, color=TEXT)
ax.set_title("Noise robustness", fontsize=10, fontweight="bold", color=TEXT, pad=8)

ax.legend(fontsize=7.5, frameon=True, fancybox=False, edgecolor=GRID,
          framealpha=0.9, loc="lower right")
ax.grid(True, alpha=0.3, linewidth=0.4, color=GRID, which="both")
ax.tick_params(labelsize=7, colors=SLATE_LIGHT)

# Annotation -- positioned to avoid overlap
ax.annotate("soft boundaries\nabsorb noise",
            xy=(0.5, 2.8), xytext=(0.62, 0.3),
            fontsize=7.5, color=BLUE, fontstyle="italic",
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))

fig.savefig("/Users/arc/Documents/dev/subsume/docs/gumbel_robustness.png",
            dpi=180, bbox_inches="tight", facecolor="white")
print("Saved subsume/docs/gumbel_robustness.png")
