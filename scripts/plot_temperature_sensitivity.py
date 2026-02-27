# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Generate temperature sensitivity plot for subsume README.

Shows how Gumbel temperature (beta) controls the crispness of
containment probability for a fixed pair of nested boxes.

Inspired by Dasgupta et al. 2020, Figure 4.

Style: distill.pub-inspired. Matches the palette of the other subsume plots.
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
ROSE = "#f43f5e"
SLATE = "#64748b"
SLATE_LIGHT = "#94a3b8"

palette = [BLUE, GREEN, AMBER, ROSE]

# ── Data ──────────────────────────────────────────────────────────────
b_min, b_max = 0.0, 10.0   # outer box
a_min, a_max = 2.0, 8.0    # inner box (margin = 2.0 on each side)
dims = [1, 5, 20, 50]
beta_values = np.logspace(np.log10(0.01), np.log10(10.0), 300)

# ── Plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 3.8))
fig.patch.set_facecolor("white")
ax.set_facecolor(BG)

for spine in ax.spines.values():
    spine.set_color(GRID)
    spine.set_linewidth(0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for i, d in enumerate(dims):
    containment = []
    for beta in beta_values:
        p_min = np.exp(-np.exp(-(a_min - b_min) / beta))
        p_max = 1.0 - np.exp(-np.exp(-(a_max - b_max) / beta))
        containment.append((p_min * p_max) ** d)
    ax.plot(beta_values, containment, linewidth=1.6, color=palette[i],
            label=f"$d = {d}$")

ax.set_xscale("log")
ax.set_xlabel("Gumbel temperature $\\beta$", fontsize=9, color=TEXT)
ax.set_ylabel("$P(A \\subseteq B)$", fontsize=9, color=TEXT)
ax.set_title("Temperature sensitivity", fontsize=10, fontweight="bold",
             color=TEXT, pad=8)
ax.set_xlim(0.01, 10.0)
ax.set_ylim(-0.02, 1.08)

ax.legend(fontsize=7, frameon=True, fancybox=False, edgecolor=GRID,
          framealpha=0.9, title="dimensions", title_fontsize=7)
ax.grid(True, alpha=0.3, linewidth=0.4, color=GRID)
ax.tick_params(labelsize=7, colors=SLATE_LIGHT)

# Regime annotations
ax.annotate("crisp ($\\beta \\to 0$)", xy=(0.015, 0.97), fontsize=7.5,
            color=SLATE, fontstyle="italic")
ax.annotate("fuzzy ($\\beta \\to \\infty$)", xy=(5.0, 0.25), fontsize=7.5,
            color=SLATE, fontstyle="italic")

# Box geometry note
ax.text(0.98, 0.02,
        f"Outer: $[{b_min:.0f},{b_max:.0f}]^d$  "
        f"Inner: $[{a_min:.0f},{a_max:.0f}]^d$  "
        f"Margin: {a_min - b_min:.0f}",
        transform=ax.transAxes, fontsize=6, color=SLATE_LIGHT,
        ha="right", va="bottom")

fig.savefig("/Users/arc/Documents/dev/subsume/docs/temperature_sensitivity.png",
            dpi=180, bbox_inches="tight", facecolor="white")
print("Saved subsume/docs/temperature_sensitivity.png")
