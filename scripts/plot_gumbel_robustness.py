# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "scipy"]
# ///
"""Generate Gumbel vs Gaussian vs Hard-box noise robustness plot for subsume README.

Data source: Gumbel and Gaussian lines are from box-coref experiments
(reproduced from experimental results showing containment loss under
coordinate noise perturbation). Hard-box baseline is computed analytically.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Noise level -> containment loss (from box-coref experiments)
# These are illustrative values reproduced from experimental visualization.
noise = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
gumbel = [0.0, 0.1, 0.2, 0.7, 1.4, 2.8, 4.8, 7.2, 11.5, 15.3, 17.6]
gaussian = [51.0, 52.0, 53.5, 57.8, 62.0, 67.5, 77.0, 84.0, 94.0, 99.5, 100.0]

# Hard-box baseline (analytical):
# For a d-dimensional box with margin m from each boundary,
# P(any coordinate pushed outside) = 1 - (1 - 2*Phi(-m/sigma))^d
# Containment loss = 100 * P(any coordinate outside)
d = 10       # embedding dimension
margin = 0.3  # boundary slack

noise_arr = np.array(noise)
hard_box = []
for sigma in noise_arr:
    if sigma == 0.0:
        hard_box.append(0.0)
    else:
        p_single_inside = 1.0 - 2.0 * norm.cdf(-margin / sigma)
        p_all_inside = p_single_inside ** d
        hard_box.append(100.0 * (1.0 - p_all_inside))

# Check if log scale is warranted: Gumbel ranges ~0-18, Gaussian ~51-100, hard ~0-100
# The Gumbel values at low noise are very small compared to Gaussian.
# Span > 1 order of magnitude across series, so log scale helps.
use_log = True

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(noise, gumbel, "o-", color="#2563eb", linewidth=2, markersize=5, label="Gumbel box")
ax.plot(noise, gaussian, "s-", color="#dc2626", linewidth=2, markersize=5, label="Gaussian box")
ax.plot(noise, hard_box, "^--", color="#64748b", linewidth=1.5, markersize=5,
        label=f"Hard box (d={d}, margin={margin})")

ax.set_xlabel("Noise level (coordinate perturbation $\\sigma$)", fontsize=10)
ax.set_ylabel("Containment loss (%)", fontsize=10)
ax.set_title("Noise robustness: Gumbel vs Gaussian vs Hard box", fontsize=11, fontweight="bold")

if use_log:
    ax.set_yscale("log")
    ax.set_ylim(0.05, 150)
    ax.set_ylabel("Containment loss (%, log scale)", fontsize=10)
else:
    ax.set_ylim(0, 105)

ax.set_xlim(0, 1)
ax.legend(fontsize=8.5, frameon=True, loc="lower right")
ax.grid(True, alpha=0.3, which="both")

# Annotation
ax.annotate("Gumbel: soft boundaries\nabsorb noise gracefully",
            xy=(0.6, 4.8), xytext=(0.3, 30),
            fontsize=8, color="#1e40af",
            arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1.0))

# Data provenance note
ax.text(0.98, 0.02,
        "Gumbel/Gaussian: illustrative (box-coref experiments)\n"
        f"Hard box: analytical (d={d}, margin={margin})",
        transform=ax.transAxes, fontsize=6, color="#94a3b8",
        ha="right", va="bottom")

fig.tight_layout()
fig.savefig("/Users/arc/Documents/dev/subsume/docs/gumbel_robustness.png",
            dpi=150, bbox_inches="tight", facecolor="white")
print("Saved subsume/docs/gumbel_robustness.png")
