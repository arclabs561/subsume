# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Generate conceptual box embedding diagram for subsume README.

Panels:
  (a) Containment (subsumption) -- boxes with coordinate labels
  (b) Overlap vs disjoint
  (c) Gumbel box (soft boundary) -- 2D view + 1D cross-section subplot
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np

fig = plt.figure(figsize=(14, 5.0))

# Layout: three conceptual panels on top, 1D cross-section below panel (c)
# Use gridspec for the panel (c) split
gs = fig.add_gridspec(2, 3, height_ratios=[3, 2], hspace=0.35, wspace=0.3)

# --- Panel (a): Containment (subsumption) ---
ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_xlim(-0.5, 10.5)
ax_a.set_ylim(-0.5, 10.5)
ax_a.set_aspect("equal")
ax_a.set_title("(a) Containment (subsumption)", fontsize=10, fontweight="bold", pad=8)

# Animal (large box)
animal = FancyBboxPatch((0.5, 0.5), 9, 9, boxstyle="round,pad=0.2",
                         facecolor="#dbeafe", edgecolor="#2563eb", linewidth=1.8)
ax_a.add_patch(animal)
ax_a.text(5, 9.8, "animal", ha="center", va="top", fontsize=9, color="#1e40af", fontweight="bold")

# Dog (medium box inside animal)
dog = FancyBboxPatch((1.2, 1.0), 4.0, 5.5, boxstyle="round,pad=0.15",
                      facecolor="#dcfce7", edgecolor="#16a34a", linewidth=1.6)
ax_a.add_patch(dog)
ax_a.text(3.2, 6.7, "dog", ha="center", va="top", fontsize=9, color="#166534", fontweight="bold")

# Poodle (small box inside dog)
poodle = FancyBboxPatch((1.8, 1.5), 2.5, 2.5, boxstyle="round,pad=0.1",
                         facecolor="#fef9c3", edgecolor="#ca8a04", linewidth=1.4)
ax_a.add_patch(poodle)
ax_a.text(3.05, 3.3, "poodle", ha="center", va="center", fontsize=8, color="#854d0e", fontweight="bold")

# Cat (medium box inside animal, beside dog)
cat = FancyBboxPatch((5.8, 1.0), 3.5, 5.0, boxstyle="round,pad=0.15",
                      facecolor="#fce7f3", edgecolor="#db2777", linewidth=1.6)
ax_a.add_patch(cat)
ax_a.text(7.55, 6.2, "cat", ha="center", va="top", fontsize=9, color="#9d174d", fontweight="bold")

# Axis labels with coordinate values
ax_a.set_xlabel("$x_1$", fontsize=9)
ax_a.set_ylabel("$x_2$", fontsize=9)
ax_a.set_xticks([0, 2, 4, 6, 8, 10])
ax_a.set_yticks([0, 2, 4, 6, 8, 10])
ax_a.tick_params(labelsize=7)
# Light grid to reinforce coordinate system
ax_a.grid(True, alpha=0.15, linewidth=0.5)
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)

# Annotation
ax_a.annotate("poodle $\\subseteq$ dog $\\subseteq$ animal",
              xy=(5, -0.3), fontsize=7.5, ha="center", color="#475569",
              style="italic")

# --- Panel (b): Overlap vs disjoint ---
ax_b = fig.add_subplot(gs[0, 1])
ax_b.set_xlim(-0.5, 10.5)
ax_b.set_ylim(-0.5, 10.5)
ax_b.set_aspect("equal")
ax_b.set_title("(b) Overlap vs. disjoint", fontsize=10, fontweight="bold", pad=8)

# Swimmer (overlaps with athlete)
swimmer = FancyBboxPatch((0.5, 3.0), 5.5, 4.5, boxstyle="round,pad=0.15",
                          facecolor="#dbeafe", edgecolor="#2563eb", linewidth=1.6, alpha=0.7)
ax_b.add_patch(swimmer)
ax_b.text(2.0, 7.8, "swimmer", ha="center", va="top", fontsize=9, color="#1e40af", fontweight="bold")

# Athlete (overlaps with swimmer)
athlete = FancyBboxPatch((3.5, 2.0), 6.0, 5.0, boxstyle="round,pad=0.15",
                          facecolor="#dcfce7", edgecolor="#16a34a", linewidth=1.6, alpha=0.7)
ax_b.add_patch(athlete)
ax_b.text(8.5, 7.3, "athlete", ha="center", va="top", fontsize=9, color="#166534", fontweight="bold")

# Planet (disjoint)
planet = FancyBboxPatch((6.5, 0.2), 3.2, 1.5, boxstyle="round,pad=0.1",
                         facecolor="#fef9c3", edgecolor="#ca8a04", linewidth=1.4)
ax_b.add_patch(planet)
ax_b.text(8.1, 1.1, "planet", ha="center", va="center", fontsize=8, color="#854d0e", fontweight="bold")

# Overlap region annotation
ax_b.annotate("overlap", xy=(4.8, 5.0), fontsize=8, ha="center",
              color="#7c3aed", fontweight="bold",
              path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

ax_b.annotate("disjoint", xy=(3.0, 1.0), fontsize=7.5, ha="center", color="#475569", style="italic")
ax_b.axis("off")

# --- Panel (c) top: Gumbel box 2D view ---
ax_c = fig.add_subplot(gs[0, 2])
ax_c.set_xlim(-1, 11)
ax_c.set_ylim(-1, 11)
ax_c.set_aspect("equal")
ax_c.set_title("(c) Gumbel box (soft boundary)", fontsize=10, fontweight="bold", pad=8)

# Concentric alpha-faded rectangles for soft boundary
for i, alpha in enumerate(np.linspace(0.02, 0.15, 12)):
    pad = i * 0.35
    rect = FancyBboxPatch((2.5 - pad, 2.0 - pad), 6.0 + 2*pad, 5.0 + 2*pad,
                           boxstyle="round,pad=0.2",
                           facecolor="#2563eb", edgecolor="none", alpha=alpha)
    ax_c.add_patch(rect)

# Core box
core = FancyBboxPatch((2.5, 2.0), 6.0, 5.0, boxstyle="round,pad=0.15",
                        facecolor="#dbeafe", edgecolor="#2563eb", linewidth=1.8)
ax_c.add_patch(core)
ax_c.text(5.5, 4.5, "concept", ha="center", va="center", fontsize=10,
          color="#1e40af", fontweight="bold")

# Horizontal line indicating the cross-section shown below
ax_c.axhline(y=4.5, xmin=0.0, xmax=1.0, color="#64748b", linewidth=0.8,
             linestyle="--", alpha=0.5)
ax_c.text(10.2, 4.5, "cross-section", fontsize=6.5, color="#64748b",
          va="center", ha="left", style="italic")

ax_c.axis("off")

# --- Panel (c) bottom: 1D cross-section P(x in box) ---
ax_cs = fig.add_subplot(gs[1, 2])

# Box boundaries in the x-dimension (matching the 2D box above)
x_min, x_max = 2.5, 8.5  # box min and max along x
beta = 0.5  # Gumbel temperature

x = np.linspace(-1, 11, 500)

# Hard box: step function
hard = np.where((x >= x_min) & (x <= x_max), 1.0, 0.0)

# Gumbel box membership along one dimension (Li et al. 2019):
# Let F(t) = exp(-exp(-t)) be the standard Gumbel CDF.
#   P(x >= min) = F((x - x_min) / beta)  -- rises at x_min
#   P(x <= max) = 1 - F((x - x_max) / beta)  -- falls at x_max
#   P(x in box) = P(x >= min) * P(x <= max)
p_above_min = np.exp(-np.exp(-(x - x_min) / beta))
p_below_max = 1.0 - np.exp(-np.exp(-(x - x_max) / beta))
gumbel_membership = p_above_min * p_below_max

ax_cs.plot(x, hard, color="#64748b", linewidth=1.5, linestyle="--", label="hard box")
ax_cs.plot(x, gumbel_membership, color="#2563eb", linewidth=2.0, label=f"Gumbel ($\\beta$={beta})")

ax_cs.set_xlabel("$x_1$ coordinate", fontsize=8)
ax_cs.set_ylabel("$P(x \\in \\mathrm{box})$", fontsize=8)
ax_cs.set_xlim(-1, 11)
ax_cs.set_ylim(-0.05, 1.1)
ax_cs.tick_params(labelsize=7)
ax_cs.legend(fontsize=7, loc="upper left", frameon=True)
ax_cs.grid(True, alpha=0.15, linewidth=0.5)
ax_cs.spines["top"].set_visible(False)
ax_cs.spines["right"].set_visible(False)

# Mark box boundaries
for bnd in [x_min, x_max]:
    ax_cs.axvline(x=bnd, color="#94a3b8", linewidth=0.7, linestyle=":")

# Empty out the unused bottom-left cells
ax_empty1 = fig.add_subplot(gs[1, 0])
ax_empty1.axis("off")
ax_empty2 = fig.add_subplot(gs[1, 1])
ax_empty2.axis("off")

fig.savefig("/Users/arc/Documents/dev/subsume/docs/box_concepts.png", dpi=150,
            bbox_inches="tight", facecolor="white")
print("Saved subsume/docs/box_concepts.png")
