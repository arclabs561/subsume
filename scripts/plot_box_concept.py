# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
"""Generate conceptual box embedding diagram for subsume README."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

# --- Panel 1: Containment (subsumption) ---
ax = axes[0]
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.5, 10.5)
ax.set_aspect("equal")
ax.set_title("Containment (subsumption)", fontsize=11, fontweight="bold", pad=10)

# Animal (large box)
animal = FancyBboxPatch((0.5, 0.5), 9, 9, boxstyle="round,pad=0.2",
                         facecolor="#dbeafe", edgecolor="#2563eb", linewidth=2)
ax.add_patch(animal)
ax.text(5, 9.8, "animal", ha="center", va="top", fontsize=10, color="#1e40af", fontweight="bold")

# Dog (medium box inside animal)
dog = FancyBboxPatch((1.2, 1.0), 4.0, 5.5, boxstyle="round,pad=0.15",
                      facecolor="#dcfce7", edgecolor="#16a34a", linewidth=1.8)
ax.add_patch(dog)
ax.text(3.2, 6.7, "dog", ha="center", va="top", fontsize=10, color="#166534", fontweight="bold")

# Poodle (small box inside dog)
poodle = FancyBboxPatch((1.8, 1.5), 2.5, 2.5, boxstyle="round,pad=0.1",
                         facecolor="#fef9c3", edgecolor="#ca8a04", linewidth=1.5)
ax.add_patch(poodle)
ax.text(3.05, 3.3, "poodle", ha="center", va="center", fontsize=9, color="#854d0e", fontweight="bold")

# Cat (medium box inside animal, beside dog)
cat = FancyBboxPatch((5.8, 1.0), 3.5, 5.0, boxstyle="round,pad=0.15",
                      facecolor="#fce7f3", edgecolor="#db2777", linewidth=1.8)
ax.add_patch(cat)
ax.text(7.55, 6.2, "cat", ha="center", va="top", fontsize=10, color="#9d174d", fontweight="bold")

# Annotation
ax.annotate("poodle $\\subseteq$ dog $\\subseteq$ animal",
            xy=(5, 0.0), fontsize=8.5, ha="center", color="#475569",
            style="italic")
ax.axis("off")

# --- Panel 2: Overlap vs disjoint ---
ax = axes[1]
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.5, 10.5)
ax.set_aspect("equal")
ax.set_title("Overlap vs. disjoint", fontsize=11, fontweight="bold", pad=10)

# Swimmer (overlaps with athlete)
swimmer = FancyBboxPatch((0.5, 3.0), 5.5, 4.5, boxstyle="round,pad=0.15",
                          facecolor="#dbeafe", edgecolor="#2563eb", linewidth=1.8, alpha=0.7)
ax.add_patch(swimmer)
ax.text(2.0, 7.8, "swimmer", ha="center", va="top", fontsize=10, color="#1e40af", fontweight="bold")

# Athlete (overlaps with swimmer)
athlete = FancyBboxPatch((3.5, 2.0), 6.0, 5.0, boxstyle="round,pad=0.15",
                          facecolor="#dcfce7", edgecolor="#16a34a", linewidth=1.8, alpha=0.7)
ax.add_patch(athlete)
ax.text(8.5, 7.3, "athlete", ha="center", va="top", fontsize=10, color="#166534", fontweight="bold")

# Planet (disjoint)
planet = FancyBboxPatch((6.5, 0.2), 3.2, 1.5, boxstyle="round,pad=0.1",
                         facecolor="#fef9c3", edgecolor="#ca8a04", linewidth=1.5)
ax.add_patch(planet)
ax.text(8.1, 1.1, "planet", ha="center", va="center", fontsize=9, color="#854d0e", fontweight="bold")

# Overlap region annotation
ax.annotate("overlap", xy=(4.8, 5.0), fontsize=9, ha="center",
            color="#7c3aed", fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

ax.annotate("disjoint", xy=(3.0, 1.0), fontsize=8.5, ha="center", color="#475569", style="italic")
ax.axis("off")

# --- Panel 3: Gumbel vs hard boxes ---
ax = axes[2]
ax.set_xlim(-1, 11)
ax.set_ylim(-1, 11)
ax.set_aspect("equal")
ax.set_title("Gumbel box (soft boundary)", fontsize=11, fontweight="bold", pad=10)

# Draw gradient "soft" boundary using concentric rectangles
import numpy as np
for i, alpha in enumerate(np.linspace(0.02, 0.15, 12)):
    pad = i * 0.35
    rect = FancyBboxPatch((2.5 - pad, 2.0 - pad), 6.0 + 2*pad, 5.0 + 2*pad,
                           boxstyle="round,pad=0.2",
                           facecolor="#2563eb", edgecolor="none", alpha=alpha)
    ax.add_patch(rect)

# Core box
core = FancyBboxPatch((2.5, 2.0), 6.0, 5.0, boxstyle="round,pad=0.15",
                        facecolor="#dbeafe", edgecolor="#2563eb", linewidth=2)
ax.add_patch(core)
ax.text(5.5, 4.5, "concept", ha="center", va="center", fontsize=11,
        color="#1e40af", fontweight="bold")

# Annotations
ax.annotate("P(x $\\in$ box) $\\approx$ 1.0", xy=(5.5, 3.2), fontsize=8.5,
            ha="center", color="#1e40af")
ax.annotate("", xy=(1.0, 4.5), xytext=(2.3, 4.5),
            arrowprops=dict(arrowstyle="<->", color="#64748b", lw=1.2))
ax.text(0.3, 4.5, "soft\nboundary", ha="center", va="center", fontsize=8,
        color="#64748b", style="italic")
ax.annotate("dense gradients\n(no vanishing)", xy=(5.5, 8.5), fontsize=8.5,
            ha="center", color="#475569", style="italic")
ax.axis("off")

fig.tight_layout(pad=1.5)
fig.savefig("/Users/arc/Documents/dev/subsume/docs/box_concepts.png", dpi=150,
            bbox_inches="tight", facecolor="white")
print("Saved subsume/docs/box_concepts.png")
