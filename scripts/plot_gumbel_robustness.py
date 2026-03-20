# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "scipy"]
# ///
"""Generate Gumbel vs Gaussian vs Hard-box noise robustness plot for subsume README.

Compares containment loss under coordinate noise perturbation.
Gumbel soft boundaries absorb noise gracefully; hard/Gaussian boxes degrade faster.

Style: distill.pub-inspired. Muted palette, minimal chrome.
"""

# ── Methodology ──────────────────────────────────────────────────────
#
# Geometry: outer box [0, 10]^d, inner box [2, 8]^d, so margin = 2
# on each side. Dimension d = 10. All dimensions are symmetric.
#
# Noise model: Gaussian N(0, sigma) added independently to each
# coordinate of the inner box. We measure the probability that the
# perturbed inner point is still "contained" by the outer box, then
# report containment loss = 100 * (1 - containment_prob).
#
# Hard box: containment per dim = P(x + noise in [0, 10]) where
#   x in {2, 8}. By symmetry, p_dim = 1 - 2*Phi(-margin/sigma).
#   Total = p_dim^d.
#
# Gumbel box (beta=1.0): membership per dim uses Gumbel CDF at
#   boundaries. Lower: exp(-exp(-(x - l)/beta)), Upper: 1 - exp(-exp(-(x - u)/beta)).
#   Per-dim membership = lower * upper. Under noise, integrate
#   membership(x + eps) * phi(eps; 0, sigma) over eps, via quad.
#   Factorizes across dimensions. Total = (per_dim_integral)^d.
#
# Gaussian box (sigma_box=1.5): each boundary contributes uncertainty.
#   Under coordinate noise sigma, effective std = sqrt(sigma_box^2 + sigma^2).
#   Per-dim containment = Phi(upper_margin / eff) - Phi(-lower_margin / eff).
#   Total = per_dim^d. Non-perfect at zero noise due to infinite Gaussian tails.
#
# Sources: Dasgupta et al. "Improving Local Identifiability in
# Probabilistic Box Embeddings" (Gumbel membership); Li et al.
# "Smoothing the Geometry of Probabilistic Box Embeddings" (Gaussian).

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
d = 10  # dimensionality
outer_lo, outer_hi = 0.0, 10.0
inner_lo, inner_hi = 2.0, 8.0
margin = inner_lo - outer_lo  # = 2.0 (symmetric)
beta = 1.0  # Gumbel softness parameter
sigma_box = 1.5  # Gaussian box boundary uncertainty

noise = np.linspace(0.0, 1.0, 51)

# ── Hard box (analytical) ────────────────────────────────────────────
hard_box = []
for sigma in noise:
    if sigma == 0.0:
        hard_box.append(0.0)
    else:
        p_dim = 1.0 - 2.0 * norm.cdf(-margin / sigma)
        hard_box.append(100.0 * (1.0 - p_dim**d))
hard_box = np.array(hard_box)

# ── Gumbel box (numerical integration) ──────────────────────────────
# Membership at point x for outer box [outer_lo, outer_hi] with Gumbel CDF:
#   lower_membership = exp(-exp(-(x - outer_lo) / beta))
#   upper_membership = 1 - exp(-exp(-(outer_hi - x) / beta))
#   Note: upper boundary uses (outer_hi - x) for the "right-tail" form.
#   per_dim_membership = lower * upper


def gumbel_membership(x):
    lo_m = np.exp(-np.exp(-(x - outer_lo) / beta))
    hi_m = 1.0 - np.exp(-np.exp(-(outer_hi - x) / beta))
    return lo_m * hi_m


# Inner box coordinate is at inner_lo (or inner_hi by symmetry).
# Under noise N(0, sigma), integrate membership(inner_lo + eps) * phi(eps).
# By symmetry of the box, the lower-boundary coordinate (inner_lo=2)
# and upper-boundary coordinate (inner_hi=8) contribute identically
# to the per-dimension membership when the box is centered.
# We integrate over the lower-boundary point; by symmetry both
# boundary contributions are captured in the single per-dim membership.

gumbel_loss = []
for sigma in noise:
    if sigma == 0.0:
        p_dim = gumbel_membership(inner_lo)
        # By symmetry inner_lo and inner_hi give same membership
    else:

        def integrand(eps, s=sigma):
            return gumbel_membership(inner_lo + eps) * norm.pdf(eps, 0, s)

        p_dim, _ = quad(integrand, -10 * sigma, 10 * sigma, limit=200)
    gumbel_loss.append(100.0 * (1.0 - p_dim**d))
gumbel_loss = np.array(gumbel_loss)

# ── Gaussian box (analytical) ───────────────────────────────────────
# Effective std under coordinate noise: sqrt(sigma_box^2 + sigma^2).
# Per-dim containment: Phi(margin / eff_sigma) - Phi(-margin / eff_sigma)
#   = 1 - 2*Phi(-margin / eff_sigma), since margins are symmetric.

gaussian_loss = []
for sigma in noise:
    eff_sigma = np.sqrt(sigma_box**2 + sigma**2)
    p_dim = 1.0 - 2.0 * norm.cdf(-margin / eff_sigma)
    gaussian_loss.append(100.0 * (1.0 - p_dim**d))
gaussian_loss = np.array(gaussian_loss)

# ── Plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 3.8))
fig.patch.set_facecolor("white")
ax.set_facecolor(BG)

for spine in ax.spines.values():
    spine.set_color(GRID)
    spine.set_linewidth(0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Fill region between Gumbel and Gaussian to show the gap
ax.fill_between(noise, gumbel_loss, gaussian_loss, alpha=0.06, color=BLUE)

ax.plot(
    noise,
    gumbel_loss,
    "-",
    color=BLUE,
    linewidth=1.8,
    label=r"Gumbel box ($\beta$=1.0)",
)
ax.plot(
    noise,
    gaussian_loss,
    "-",
    color=RED,
    linewidth=1.8,
    label=r"Gaussian box ($\sigma_b$=1.5)",
)
ax.plot(
    noise,
    hard_box,
    "--",
    color=SLATE,
    linewidth=1.2,
    alpha=0.7,
    label=f"Hard box ($d$={d})",
)

ax.set_yscale("log")
ax.set_ylim(0.01, 150)
ax.set_xlim(0, 1.0)

ax.set_xlabel("Noise level (coordinate perturbation $\\sigma$)", fontsize=9, color=TEXT)
ax.set_ylabel("Containment loss (%, log scale)", fontsize=9, color=TEXT)
ax.set_title("Noise robustness", fontsize=10, fontweight="bold", color=TEXT, pad=8)

ax.legend(
    fontsize=7.5,
    frameon=True,
    fancybox=False,
    edgecolor=GRID,
    framealpha=0.9,
    loc="lower right",
)
ax.grid(True, alpha=0.3, linewidth=0.4, color=GRID, which="both")
ax.tick_params(labelsize=7, colors=SLATE_LIGHT)

# Annotation -- position depends on computed curves
gumbel_mid = gumbel_loss[len(noise) // 2]
ax.annotate(
    "soft boundaries\nabsorb noise",
    xy=(0.5, gumbel_mid),
    xytext=(0.65, gumbel_mid * 0.15),
    fontsize=7.5,
    color=BLUE,
    fontstyle="italic",
    arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8),
)

fig.savefig(
    "/Users/arc/Documents/dev/subsume/docs/gumbel_robustness.png",
    dpi=180,
    bbox_inches="tight",
    facecolor="white",
)
print("Saved subsume/docs/gumbel_robustness.png")
