# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Generate temperature sensitivity plot for subsume README.

Shows how Gumbel temperature (beta) controls the crispness of
containment probability for a fixed pair of nested boxes.

Inspired by Dasgupta et al. 2020, Figure 4. Grounds the
temperature_schedule module in subsume.

Containment probability for inner box [a_min, a_max] inside outer
box [b_min, b_max] in one dimension:
    P(b_min <= a_min) * P(a_max <= b_max)
where F(t) = exp(-exp(-t)) is the standard Gumbel CDF:
    P(a_min >= b_min) = F((a_min - b_min) / beta)
    P(a_max <= b_max) = 1 - F((a_max - b_max) / beta)

For d dimensions, containment = product over all dimensions.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Fixed box pair (inner fully contained in outer with some margin).
# All dimensions identical for simplicity.
b_min, b_max = 0.0, 10.0   # outer box
a_min, a_max = 2.0, 8.0    # inner box (margin = 2.0 on each side)

dims = [1, 5, 20, 50]

beta_values = np.logspace(np.log10(0.01), np.log10(10.0), 300)

fig, ax = plt.subplots(figsize=(6, 4))

for d in dims:
    containment = []
    for beta in beta_values:
        # Per-dimension containment factors
        # P(A_min >= B_min) = F((a_min - b_min) / beta)
        # P(A_max <= B_max) = 1 - F((a_max - b_max) / beta)
        # where F(t) = exp(-exp(-t)) is the standard Gumbel CDF
        p_min = np.exp(-np.exp(-(a_min - b_min) / beta))
        p_max = 1.0 - np.exp(-np.exp(-(a_max - b_max) / beta))
        p_dim = p_min * p_max
        # d dimensions (all identical margins)
        p_total = p_dim ** d
        containment.append(p_total)
    ax.plot(beta_values, containment, linewidth=1.8, label=f"d = {d}")

ax.set_xscale("log")
ax.set_xlabel("Gumbel temperature $\\beta$", fontsize=10)
ax.set_ylabel("Containment probability $P(A \\subseteq B)$", fontsize=10)
ax.set_title("Temperature sensitivity of containment probability", fontsize=11, fontweight="bold")
ax.set_xlim(0.01, 10.0)
ax.set_ylim(-0.02, 1.05)
ax.legend(fontsize=8.5, frameon=True, title="dimensions", title_fontsize=8)
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Annotations for the two regimes
ax.annotate("crisp\n($\\beta \\to 0$)", xy=(0.02, 0.95), fontsize=8,
            color="#475569", ha="center", style="italic")
ax.annotate("fuzzy\n($\\beta \\to \\infty$)", xy=(7.0, 0.35), fontsize=8,
            color="#475569", ha="center", style="italic")

# Box geometry note
ax.text(0.98, 0.02,
        f"Outer box: [{b_min}, {b_max}]$^d$,  Inner box: [{a_min}, {a_max}]$^d$\n"
        f"Per-dimension margin: {a_min - b_min:.1f}",
        transform=ax.transAxes, fontsize=6.5, color="#94a3b8",
        ha="right", va="bottom")

fig.tight_layout()
fig.savefig("/Users/arc/Documents/dev/subsume/docs/temperature_sensitivity.png",
            dpi=150, bbox_inches="tight", facecolor="white")
print("Saved subsume/docs/temperature_sensitivity.png")
