# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Gradient landscape comparison: hard box vs Gumbel box.

Shows why Gumbel boxes solve the local identifiability problem: hard boxes
have zero gradients (flat regions) whenever the inner point is clearly inside
or outside the box, while Gumbel boxes provide gradients everywhere.

This is the core practical advantage -- during SGD training, a hard box
embedding in a zero-gradient region receives no learning signal, while
Gumbel always provides a direction to move.

Reference: Dasgupta et al. 2020, Section 3.
Style: distill.pub-inspired. Muted palette, minimal chrome.
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

# ── Data ──────────────────────────────────────────────────────────────
x = np.linspace(-3, 5, 500)

# Hard box membership: step function at x=0
hard_membership = np.where(x >= 0, 1.0, 0.0)

# Gumbel membership: exp(-exp(-x/beta))
betas = [0.3, 0.8, 2.0]
colors = [BLUE, GREEN, AMBER]
gumbel_membership = {}
gumbel_gradient = {}
for beta in betas:
    m = np.exp(-np.exp(-x / beta))
    g = (1.0 / beta) * np.exp(-x / beta) * np.exp(-np.exp(-x / beta))
    gumbel_membership[beta] = m
    gumbel_gradient[beta] = g

# ── Figure: 2 panels ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), gridspec_kw={"wspace": 0.30})
fig.patch.set_facecolor("white")

for ax in axes:
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ── Panel (a): Membership probability ────────────────────────────────
ax = axes[0]

ax.plot(
    x,
    hard_membership,
    color=SLATE,
    linewidth=1.2,
    linestyle="--",
    alpha=0.6,
    label="hard box",
)
ax.fill_between(x, 0, hard_membership, color=SLATE, alpha=0.04, step="mid")

for beta, col in zip(betas, colors):
    ax.plot(
        x, gumbel_membership[beta], color=col, linewidth=1.6, label=rf"$\beta = {beta}$"
    )

ax.axvline(x=0, color=SLATE_LIGHT, linewidth=0.6, linestyle=":")
ax.text(
    0.15,
    0.05,
    "boundary",
    fontsize=7,
    color=SLATE,
    fontstyle="italic",
    rotation=90,
    va="bottom",
)

ax.set_xlim(-3, 5)
ax.set_ylim(-0.05, 1.10)
ax.set_xlabel("offset from boundary", fontsize=9, color=TEXT)
ax.set_ylabel(r"$P(x \in \mathrm{box})$", fontsize=9, color=TEXT)
ax.set_title(
    "(a) Membership probability", fontsize=10, fontweight="bold", color=TEXT, pad=8
)
ax.legend(
    fontsize=7,
    loc="upper left",
    frameon=True,
    fancybox=False,
    edgecolor=GRID,
    framealpha=0.9,
)
ax.grid(True, alpha=0.3, linewidth=0.4, color=GRID)
ax.tick_params(labelsize=7, colors=SLATE_LIGHT)

# ── Panel (b): Gradient magnitude ────────────────────────────────────
ax = axes[1]

ax.axhline(y=0, color=SLATE, linewidth=1.2, linestyle="--", alpha=0.4)

# Shade the zero-gradient regions for hard box
ax.axvspan(-3, -0.05, alpha=0.06, color=SLATE)
ax.axvspan(0.05, 5, alpha=0.06, color=SLATE)
ax.text(
    -1.5,
    0.55,
    "zero gradient\n(no learning signal)",
    fontsize=7,
    color=SLATE,
    ha="center",
    va="center",
    fontstyle="italic",
)
ax.text(
    2.5,
    0.55,
    "zero gradient\n(no learning signal)",
    fontsize=7,
    color=SLATE,
    ha="center",
    va="center",
    fontstyle="italic",
)

for beta, col in zip(betas, colors):
    ax.plot(
        x, gumbel_gradient[beta], color=col, linewidth=1.6, label=rf"$\beta = {beta}$"
    )

ax.axvline(x=0, color=SLATE_LIGHT, linewidth=0.6, linestyle=":")

ax.set_xlim(-3, 5)
ax.set_ylim(-0.05, max(gumbel_gradient[0.3].max() * 1.15, 0.7))
ax.set_xlabel("offset from boundary", fontsize=9, color=TEXT)
ax.set_ylabel(r"$|dP/dx|$  (gradient magnitude)", fontsize=9, color=TEXT)
ax.set_title(
    "(b) Gradient through boundary", fontsize=10, fontweight="bold", color=TEXT, pad=8
)
ax.legend(
    fontsize=7,
    loc="upper right",
    frameon=True,
    fancybox=False,
    edgecolor=GRID,
    framealpha=0.9,
)
ax.grid(True, alpha=0.3, linewidth=0.4, color=GRID)
ax.tick_params(labelsize=7, colors=SLATE_LIGHT)

# ── Save ─────────────────────────────────────────────────────────────
fig.savefig(
    "/Users/arc/Documents/dev/subsume/docs/gumbel_robustness.png",
    dpi=180,
    bbox_inches="tight",
    facecolor="white",
)
print("Saved subsume/docs/gumbel_robustness.png")
