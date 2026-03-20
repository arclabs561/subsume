# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "scipy"]
# ///
"""Generate Gumbel vs Gaussian vs Hard-box noise robustness plot for subsume README.

Shows per-dimension containment probability under increasing coordinate noise.
Using d=1 isolates the boundary behavior that differentiates formulations,
without the exponential decay from product-over-dimensions obscuring the signal.

Style: distill.pub-inspired. Muted palette, minimal chrome.
"""

# ── Methodology ──────────────────────────────────────────────────────
#
# Geometry: 1D outer interval [0, 10], inner point at x=2 (margin=2
# from the lower boundary). Noise N(0, sigma) added to the inner point.
#
# We measure per-dimension containment probability at the BOUNDARY
# coordinate (worst case), which is the measurement that matters for
# multi-dimensional containment (total = per_dim^d).
#
# Hard box: P(x + noise in [0, 10]) = Phi(margin/sigma) for the near
#   boundary (the far boundary at distance 8 is negligible).
#
# Gumbel box (beta): membership = exp(-exp(-(x-l)/beta)) at the lower
#   boundary. Under noise, integrate membership(x+eps) * phi(eps) deps.
#
# Gaussian box (sigma_box): boundary uncertainty broadens the step
#   function into an error function. Effective std = sqrt(sigma_box^2 + sigma^2).
#   P = Phi(margin / effective_sigma).
#
# Sources: Dasgupta et al. 2020 (Gumbel membership); Li et al. 2019
# (smoothed box geometry); Chen et al. 2021 BEUrRE (Gaussian boxes).

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

# ── Color palette ─────────────────────────────────────────────────────
BG = "#fafafa"
GRID = "#e5e7eb"
TEXT = "#374151"
BLUE = "#3b82f6"
RED = "#ef4444"
SLATE = "#64748b"
SLATE_LIGHT = "#94a3b8"

# ── Parameters ────────────────────────────────────────────────────────
margin = 2.0  # distance from inner point to nearest outer boundary
beta = 0.3  # Gumbel temperature (controls boundary softness)
sigma_box = 1.5  # Gaussian box boundary uncertainty

noise_levels = np.linspace(0.0, 3.0, 61)

# ── Hard box (analytical) ────────────────────────────────────────────
# Per-dim P(contained) = Phi(margin / sigma) for the near boundary.
hard_prob = []
for sigma in noise_levels:
    if sigma == 0.0:
        hard_prob.append(1.0)
    else:
        hard_prob.append(norm.cdf(margin / sigma))
hard_prob = np.array(hard_prob)


# ── Gumbel box (numerical integration) ──────────────────────────────
# Lower-boundary membership at coordinate x:
#   m(x) = exp(-exp(-(x - boundary) / beta))
# where boundary = 0 (outer lower bound), and inner point is at x=margin.
def gumbel_lower_membership(x):
    return np.exp(-np.exp(-x / beta))


gumbel_prob = []
for sigma in noise_levels:
    if sigma == 0.0:
        gumbel_prob.append(gumbel_lower_membership(margin))
    else:

        def integrand(eps, s=sigma):
            return gumbel_lower_membership(margin + eps) * norm.pdf(eps, 0, s)

        p, _ = quad(integrand, -8 * sigma, 8 * sigma, limit=200)
        gumbel_prob.append(p)
gumbel_prob = np.array(gumbel_prob)

# ── Gaussian box (analytical) ────────────────────────────────────────
# Effective std under noise: sqrt(sigma_box^2 + sigma^2).
# Per-dim containment at the near boundary: Phi(margin / effective_sigma).
gaussian_prob = []
for sigma in noise_levels:
    eff = np.sqrt(sigma_box**2 + sigma**2)
    gaussian_prob.append(norm.cdf(margin / eff))
gaussian_prob = np.array(gaussian_prob)

# ── Plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 3.8))
fig.patch.set_facecolor("white")
ax.set_facecolor(BG)

for spine in ax.spines.values():
    spine.set_color(GRID)
    spine.set_linewidth(0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.plot(
    noise_levels,
    gumbel_prob,
    "-",
    color=BLUE,
    linewidth=1.8,
    label=rf"Gumbel ($\beta$={beta})",
)
ax.plot(
    noise_levels,
    hard_prob,
    "--",
    color=SLATE,
    linewidth=1.4,
    alpha=0.8,
    label="Hard box",
)
ax.plot(
    noise_levels,
    gaussian_prob,
    "-",
    color=RED,
    linewidth=1.8,
    label=rf"Gaussian ($\sigma_b$={sigma_box})",
)

# Fill between Gumbel and Hard to show the gap
ax.fill_between(
    noise_levels,
    gumbel_prob,
    hard_prob,
    where=(gumbel_prob > hard_prob),
    alpha=0.08,
    color=BLUE,
)

ax.set_xlim(0, 3.0)
ax.set_ylim(0.45, 1.02)

ax.set_xlabel(r"Noise level (coordinate perturbation $\sigma$)", fontsize=9, color=TEXT)
ax.set_ylabel("Per-dimension containment probability", fontsize=9, color=TEXT)
ax.set_title(
    "Noise robustness (boundary coordinate, margin=2)",
    fontsize=10,
    fontweight="bold",
    color=TEXT,
    pad=8,
)

ax.legend(
    fontsize=7.5,
    frameon=True,
    fancybox=False,
    edgecolor=GRID,
    framealpha=0.9,
    loc="lower left",
)
ax.grid(True, alpha=0.3, linewidth=0.4, color=GRID)
ax.tick_params(labelsize=7, colors=SLATE_LIGHT)

# Annotation: where Gumbel overtakes Hard box
crossover_idx = np.argmax(gumbel_prob > hard_prob)
if crossover_idx > 0:
    cx = noise_levels[crossover_idx]
    cy = gumbel_prob[crossover_idx]
    ax.annotate(
        "Gumbel overtakes\nhard box",
        xy=(cx, cy),
        xytext=(cx + 0.4, cy - 0.12),
        fontsize=7,
        color=BLUE,
        fontstyle="italic",
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8),
    )

# Annotation: Gaussian baseline
ax.annotate(
    "Gaussian: inherent\nboundary uncertainty",
    xy=(0.1, gaussian_prob[1]),
    xytext=(0.6, 0.72),
    fontsize=7,
    color=RED,
    fontstyle="italic",
    arrowprops=dict(arrowstyle="->", color=RED, lw=0.8),
)

# Note: multi-dim containment = per_dim^d
ax.text(
    0.98,
    0.02,
    r"Multi-dim containment: $P_\mathrm{total} = P_\mathrm{dim}^{\,d}$",
    transform=ax.transAxes,
    fontsize=6,
    color=SLATE_LIGHT,
    ha="right",
    va="bottom",
)

fig.savefig(
    "/Users/arc/Documents/dev/subsume/docs/gumbel_robustness.png",
    dpi=180,
    bbox_inches="tight",
    facecolor="white",
)
print("Saved subsume/docs/gumbel_robustness.png")
