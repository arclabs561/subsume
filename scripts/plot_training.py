# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Generate training convergence visualization for subsume README.

Reproduces the box_training example logic in numpy:
  - 25-entity taxonomy, 24 containment pairs
  - Direct coordinate updates (push head_min below tail_min, head_max above tail_max)
  - 200 epochs, dim=8, lr=0.05, margin=0.05

Two panels:
  (a) Training loss vs epoch (log scale)
  (b) Containment probability vs epoch (selected hierarchy pairs)

Style: distill.pub-inspired. Muted palette, minimal chrome.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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

palette = [BLUE, GREEN, AMBER, ROSE, SLATE]

# ── Taxonomy ──────────────────────────────────────────────────────────

containment_pairs = [
    ("entity", "animal"), ("entity", "plant"), ("entity", "vehicle"),
    ("animal", "mammal"), ("animal", "bird"), ("animal", "fish"),
    ("plant", "tree"), ("plant", "flower"),
    ("mammal", "dog"), ("mammal", "cat"), ("mammal", "whale"), ("mammal", "bat"),
    ("bird", "eagle"), ("bird", "sparrow"), ("bird", "penguin"),
    ("fish", "salmon"), ("fish", "tuna"),
    ("tree", "oak"), ("tree", "pine"),
    ("flower", "rose"), ("flower", "tulip"),
    ("vehicle", "car"), ("vehicle", "truck"), ("vehicle", "bicycle"),
]

entities = sorted({e for pair in containment_pairs for e in pair})
n = len(entities)
eidx = {name: i for i, name in enumerate(entities)}

# ── Initialize boxes ─────────────────────────────────────────────────

dim = 8
lr = 0.05
margin = 0.05
epochs = 200

box_min = np.zeros((n, dim), dtype=np.float32)
box_max = np.zeros((n, dim), dtype=np.float32)
for i, name in enumerate(entities):
    center = i * 0.3
    box_min[i, :] = center - 0.5
    box_max[i, :] = center + 0.5


def containment_prob(head_name: str, tail_name: str) -> float:
    h, t = eidx[head_name], eidx[tail_name]
    probs = []
    for d in range(dim):
        tail_width = box_max[t, d] - box_min[t, d]
        if tail_width <= 0:
            probs.append(1.0)
            continue
        lo = max(box_min[h, d], box_min[t, d])
        hi = min(box_max[h, d], box_max[t, d])
        overlap = max(0.0, hi - lo)
        probs.append(overlap / tail_width)
    return float(np.prod(probs))


# ── Training loop ────────────────────────────────────────────────────

violations = []
tracked_pairs = [
    ("entity", "animal"),
    ("animal", "mammal"),
    ("mammal", "dog"),
    ("plant", "tree"),
    ("vehicle", "car"),
]
prob_history = {pair: [] for pair in tracked_pairs}

for epoch in range(epochs):
    total_viol = 0.0
    for head, tail in containment_pairs:
        h, t = eidx[head], eidx[tail]
        for d in range(dim):
            if box_min[h, d] > box_min[t, d] - margin:
                v = box_min[h, d] - (box_min[t, d] - margin)
                box_min[h, d] -= lr * v
                total_viol += abs(v)
            if box_max[h, d] < box_max[t, d] + margin:
                v = (box_max[t, d] + margin) - box_max[h, d]
                box_max[h, d] += lr * v
                total_viol += abs(v)
    violations.append(total_viol)
    for pair in tracked_pairs:
        prob_history[pair].append(containment_prob(*pair))

# ── Plotting ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8),
                         gridspec_kw={"wspace": 0.30})
fig.patch.set_facecolor("white")

for ax in axes:
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ── Panel (a): Loss vs epoch ─────────────────────────────────────────
ax = axes[0]
ax.semilogy(range(epochs), violations, color=BLUE, linewidth=1.6)

# Shade the region where loss is still high
convergence_epoch = next(i for i, v in enumerate(violations) if v < 1.0)
ax.axvspan(0, convergence_epoch, alpha=0.04, color=BLUE)
ax.axhline(y=1.0, color=SLATE_LIGHT, linewidth=0.6, linestyle=":")
ax.text(convergence_epoch + 5, 1.2, "violation < 1",
        fontsize=7, color=SLATE, fontstyle="italic")

ax.set_xlabel("Epoch", fontsize=9, color=TEXT)
ax.set_ylabel("Total violation (log scale)", fontsize=9, color=TEXT)
ax.set_title("(a) Training loss", fontsize=10, fontweight="bold", color=TEXT, pad=8)
ax.grid(True, alpha=0.3, linewidth=0.4, color=GRID, which="both")
ax.tick_params(labelsize=7, colors=SLATE_LIGHT)

# ── Panel (b): Containment probability vs epoch ──────────────────────
ax = axes[1]
labels = {
    ("entity", "animal"): "entity $\\supseteq$ animal",
    ("animal", "mammal"): "animal $\\supseteq$ mammal",
    ("mammal", "dog"): "mammal $\\supseteq$ dog",
    ("plant", "tree"): "plant $\\supseteq$ tree",
    ("vehicle", "car"): "vehicle $\\supseteq$ car",
}
for i, pair in enumerate(tracked_pairs):
    ax.plot(range(epochs), prob_history[pair], color=palette[i],
            linewidth=1.4, label=labels[pair])

ax.axhline(y=1.0, color=SLATE_LIGHT, linewidth=0.6, linestyle=":")
ax.set_xlabel("Epoch", fontsize=9, color=TEXT)
ax.set_ylabel("$P$(head $\\supseteq$ tail)", fontsize=9, color=TEXT)
ax.set_title("(b) Containment probability", fontsize=10, fontweight="bold",
             color=TEXT, pad=8)
ax.set_ylim(-0.05, 1.12)
ax.legend(fontsize=6.5, loc="lower right", frameon=True, fancybox=False,
          edgecolor=GRID, framealpha=0.9, ncol=1)
ax.grid(True, alpha=0.3, linewidth=0.4, color=GRID)
ax.tick_params(labelsize=7, colors=SLATE_LIGHT)

# ── Save ─────────────────────────────────────────────────────────────
out = Path(__file__).resolve().parent.parent / "docs" / "training_convergence.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
