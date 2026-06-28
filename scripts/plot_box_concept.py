# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Generate hero concept figure for subsume README.

Two panels, each communicating one core idea:
  (a) Containment (subsumption) -- nested axis-aligned boxes
  (b) Gumbel soft boundary -- 1D membership sigmoid vs hard box

Style: distill.pub-inspired. Sharp rectangles, muted palette, minimal chrome,
one idea per panel. No rounded corners (these are axis-aligned hyperrectangles).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# ── Color palette (muted, academic) ────────────────────────────────────
BG = "white"
GRID = "#e5e7eb"
TEXT = "#374151"
BLUE = "#3b82f6"
BLUE_DARK = "#1d4ed8"
GREEN = "#22c55e"
GREEN_DARK = "#15803d"
AMBER = "#f59e0b"
AMBER_DARK = "#b45309"
ROSE = "#f43f5e"
ROSE_DARK = "#be123c"
SLATE = "#64748b"
SLATE_LIGHT = "#94a3b8"

# ── Figure setup ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2),
                         gridspec_kw={"wspace": 0.28})
fig.patch.set_facecolor("white")

for ax in axes:
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(0.8)

# ══════════════════════════════════════════════════════════════════════
# Panel (a): Containment (subsumption)
# ══════════════════════════════════════════════════════════════════════
ax = axes[0]
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.5, 10.5)
ax.set_aspect("equal")

# animal -- large outer box
ax.add_patch(Rectangle((0.3, 0.3), 9.4, 9.4,
    linewidth=1.6, edgecolor=BLUE, facecolor=BLUE, alpha=0.06))
ax.add_patch(Rectangle((0.3, 0.3), 9.4, 9.4,
    linewidth=1.6, edgecolor=BLUE, facecolor="none"))
ax.text(5.0, 10.1, "animal", ha="center", va="bottom", fontsize=9,
        color=BLUE_DARK, fontweight="bold")

# mammal -- medium box inside animal
ax.add_patch(Rectangle((0.8, 0.8), 5.2, 6.0,
    linewidth=1.4, edgecolor=GREEN, facecolor=GREEN, alpha=0.06))
ax.add_patch(Rectangle((0.8, 0.8), 5.2, 6.0,
    linewidth=1.4, edgecolor=GREEN, facecolor="none"))
ax.text(3.4, 7.1, "mammal", ha="center", va="bottom", fontsize=8.5,
        color=GREEN_DARK, fontweight="bold")

# dog -- small box inside mammal
ax.add_patch(Rectangle((1.3, 1.3), 2.8, 3.0,
    linewidth=1.2, edgecolor=AMBER, facecolor=AMBER, alpha=0.08))
ax.add_patch(Rectangle((1.3, 1.3), 2.8, 3.0,
    linewidth=1.2, edgecolor=AMBER, facecolor="none"))
ax.text(2.7, 3.0, "dog", ha="center", va="center", fontsize=8,
        color=AMBER_DARK, fontweight="bold")

# bird -- medium box, disjoint from mammal (birds are not mammals)
ax.add_patch(Rectangle((6.3, 1.0), 3.0, 5.0,
    linewidth=1.4, edgecolor=ROSE, facecolor=ROSE, alpha=0.06))
ax.add_patch(Rectangle((6.3, 1.0), 3.0, 5.0,
    linewidth=1.4, edgecolor=ROSE, facecolor="none"))
ax.text(7.8, 3.5, "bird", ha="center", va="center", fontsize=8.5,
        color=ROSE_DARK, fontweight="bold")

# Subsumption annotation
ax.text(5.0, -0.3, "dog $\\subseteq$ mammal $\\subseteq$ animal",
        ha="center", va="top", fontsize=7.5, color=SLATE, fontstyle="italic")

# Clean axes
ax.set_xlabel("$x_1$", fontsize=8.5, color=TEXT)
ax.set_ylabel("$x_2$", fontsize=8.5, color=TEXT)
ax.set_xticks([0, 2, 4, 6, 8, 10])
ax.set_yticks([0, 2, 4, 6, 8, 10])
ax.tick_params(labelsize=7, colors=SLATE_LIGHT)
ax.grid(True, alpha=0.3, linewidth=0.4, color=GRID)
ax.set_title("(a) Containment", fontsize=10, fontweight="bold",
             color=TEXT, pad=10)

# ══════════════════════════════════════════════════════════════════════
# Panel (b): Gumbel soft boundary (1D membership)
# ══════════════════════════════════════════════════════════════════════
ax = axes[1]

x_min, x_max = 2.0, 8.0  # box boundaries
x = np.linspace(-1, 11, 500)

# Hard box: step function
hard = np.where((x >= x_min) & (x <= x_max), 1.0, 0.0)

# Gumbel membership for several temperatures
betas = [0.3, 0.8, 2.0]
beta_colors = [BLUE, GREEN, AMBER]
beta_alphas = [1.0, 0.8, 0.6]

ax.plot(x, hard, color=SLATE, linewidth=1.2, linestyle="--",
        alpha=0.5, label="hard box")

for beta, col, alp in zip(betas, beta_colors, beta_alphas):
    # Soft box = (above lower corner) AND (below upper corner). The lower corner
    # is max-Gumbel, the upper corner min-Gumbel, so the box is symmetric: both
    # boundaries cross 1/e at the edge (Dasgupta et al. 2020).
    p_above = np.exp(-np.exp(-(x - x_min) / beta))
    p_below = np.exp(-np.exp((x - x_max) / beta))
    membership = p_above * p_below
    ax.plot(x, membership, color=col, linewidth=1.8, alpha=alp,
            label=f"$\\beta = {beta}$")

# Mark box boundaries
for bnd in [x_min, x_max]:
    ax.axvline(x=bnd, color=SLATE_LIGHT, linewidth=0.6, linestyle=":")

# Boundary labels (above the curve, at the boundary)
ax.annotate("$\\ell$", xy=(x_min, 0.5), fontsize=9, color=SLATE,
            ha="right", va="center",
            xytext=(-6, 0), textcoords="offset points")
ax.annotate("$u$", xy=(x_max, 0.5), fontsize=9, color=SLATE,
            ha="left", va="center",
            xytext=(6, 0), textcoords="offset points")

ax.set_xlabel("coordinate value", fontsize=8.5, color=TEXT)
ax.set_ylabel("$P(x \\in \\mathrm{box})$", fontsize=8.5, color=TEXT)
ax.set_xlim(-1, 11)
ax.set_ylim(-0.05, 1.12)
ax.set_xticks([0, 2, 4, 6, 8, 10])
ax.tick_params(labelsize=7, colors=SLATE_LIGHT)
ax.legend(fontsize=7, loc="upper left", frameon=True, fancybox=False,
          edgecolor=GRID, framealpha=0.9)
ax.grid(True, alpha=0.3, linewidth=0.4, color=GRID)
ax.set_title("(b) Gumbel soft boundary", fontsize=10, fontweight="bold",
             color=TEXT, pad=10)

# ── Save ───────────────────────────────────────────────────────────────
fig.savefig("/Users/arc/Documents/dev/subsume/docs/box_concepts.png",
            dpi=180, bbox_inches="tight", facecolor="white")
print("Saved subsume/docs/box_concepts.png")
