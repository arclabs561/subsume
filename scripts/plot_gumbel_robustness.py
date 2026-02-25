# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
"""Generate Gumbel vs Gaussian noise robustness plot for subsume README.

Data source: box-coref/experiments/visualizations/containment_robustness.png
(reproduced from experimental results showing Gumbel box containment loss
under coordinate noise perturbation).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Noise level -> containment loss (from box-coref experiments)
noise = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
gumbel = [0.0, 0.1, 0.2, 0.7, 1.4, 2.8, 4.8, 7.2, 11.5, 15.3, 17.6]
gaussian = [51.0, 52.0, 53.5, 57.8, 62.0, 67.5, 77.0, 84.0, 94.0, 99.5, 100.0]

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(noise, gumbel, "o-", color="#2563eb", linewidth=2, markersize=6, label="Gumbel box")
ax.plot(noise, gaussian, "s-", color="#dc2626", linewidth=2, markersize=6, label="Gaussian box")

ax.fill_between(noise, gumbel, alpha=0.1, color="#2563eb")
ax.fill_between(noise, gaussian, alpha=0.1, color="#dc2626")

ax.set_xlabel("Noise level (coordinate perturbation)", fontsize=11)
ax.set_ylabel("Containment loss", fontsize=11)
ax.set_title("Noise robustness: Gumbel vs Gaussian boxes", fontsize=12, fontweight="bold")
ax.legend(fontsize=10, frameon=True)
ax.set_xlim(0, 1)
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3)

# Annotation
ax.annotate("Gumbel: soft boundaries\nabsorb noise gracefully",
            xy=(0.6, 4.8), xytext=(0.3, 30),
            fontsize=9, color="#1e40af",
            arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1.2))

fig.tight_layout()
fig.savefig("/Users/arc/Documents/dev/subsume/docs/gumbel_robustness.png",
            dpi=150, bbox_inches="tight", facecolor="white")
print("Saved subsume/docs/gumbel_robustness.png")
