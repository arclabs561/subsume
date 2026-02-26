# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Generate training convergence visualization for subsume README.

Reproduces the box_training example logic in numpy:
  - 25-entity taxonomy, 24 containment pairs
  - Direct coordinate updates (push head_min below tail_min, head_max above tail_max)
  - 200 epochs, dim=8, lr=0.05, margin=0.05

Panels:
  (a) Total violation (loss) vs epoch  -- log scale
  (b) Containment probability vs epoch -- selected hierarchy checks
  (c) Before/after box positions       -- 2D projection (dims 0,1)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── Taxonomy ──────────────────────────────────────────────────────────

containment_pairs = [
    # Level 0 -> 1
    ("entity", "animal"), ("entity", "plant"), ("entity", "vehicle"),
    # Level 1 -> 2
    ("animal", "mammal"), ("animal", "bird"), ("animal", "fish"),
    ("plant", "tree"), ("plant", "flower"),
    # Level 2 -> 3
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

# ── Initialize boxes (matching Rust example) ─────────────────────────

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

# Snapshot initial boxes (dims 0,1) for panel (c)
init_min = box_min.copy()
init_max = box_max.copy()


def containment_prob(head_name: str, tail_name: str) -> float:
    """Approximate hard-box containment probability across dimensions."""
    h, t = eidx[head_name], eidx[tail_name]
    probs = []
    for d in range(dim):
        # Fraction of tail range contained in head range (per dimension)
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
# Track containment probs for selected pairs
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

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

# Colors
C_BLUE = "#2563eb"
C_GREEN = "#16a34a"
C_AMBER = "#d97706"
C_PINK = "#db2777"
C_SLATE = "#475569"

palette = [C_BLUE, C_GREEN, C_AMBER, C_PINK, C_SLATE]

# ── Panel (a): Violation (loss) vs epoch ─────────────────────────────
ax = axes[0]
ax.semilogy(range(epochs), violations, color=C_BLUE, linewidth=1.8)
ax.set_xlabel("Epoch", fontsize=9)
ax.set_ylabel("Total violation (log scale)", fontsize=9)
ax.set_title("(a) Training loss", fontsize=10, fontweight="bold")
ax.grid(True, alpha=0.15, linewidth=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=7)

# ── Panel (b): Containment probability vs epoch ─────────────────────
ax = axes[1]
for i, pair in enumerate(tracked_pairs):
    label = f"{pair[0]} > {pair[1]}"
    ax.plot(range(epochs), prob_history[pair], color=palette[i],
            linewidth=1.5, label=label)
ax.set_xlabel("Epoch", fontsize=9)
ax.set_ylabel("P(head contains tail)", fontsize=9)
ax.set_title("(b) Containment probability", fontsize=10, fontweight="bold")
ax.set_ylim(-0.05, 1.1)
ax.legend(fontsize=6.5, loc="lower right", frameon=True, ncol=1)
ax.grid(True, alpha=0.15, linewidth=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=7)

# ── Panel (c): Before/after box positions (dims 0,1) ────────────────
ax = axes[2]
ax.set_title("(c) Box positions (dims 0-1)", fontsize=10, fontweight="bold")

# Show a subset: entity, animal, mammal, dog, plant, vehicle
show = ["entity", "animal", "mammal", "dog", "plant", "vehicle"]
colors_show = {
    "entity": "#94a3b8",
    "animal": C_BLUE,
    "mammal": C_GREEN,
    "dog": "#22c55e",
    "plant": C_AMBER,
    "vehicle": C_PINK,
}

for name in show:
    i = eidx[name]
    c = colors_show[name]
    # Before (dashed, faded)
    w0 = init_max[i, 0] - init_min[i, 0]
    h0 = init_max[i, 1] - init_min[i, 1]
    ax.add_patch(plt.Rectangle(
        (init_min[i, 0], init_min[i, 1]), w0, h0,
        linewidth=1.0, edgecolor=c, facecolor="none",
        linestyle="--", alpha=0.35,
    ))
    # After (solid)
    w1 = box_max[i, 0] - box_min[i, 0]
    h1 = box_max[i, 1] - box_min[i, 1]
    ax.add_patch(plt.Rectangle(
        (box_min[i, 0], box_min[i, 1]), w1, h1,
        linewidth=1.5, edgecolor=c, facecolor=c, alpha=0.10,
    ))
    ax.add_patch(plt.Rectangle(
        (box_min[i, 0], box_min[i, 1]), w1, h1,
        linewidth=1.5, edgecolor=c, facecolor="none",
    ))
    # Label at center of final box
    cx = box_min[i, 0] + w1 / 2
    cy = box_min[i, 1] + h1 / 2
    ax.text(cx, cy, name, ha="center", va="center", fontsize=6.5,
            color=c, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.8))

ax.set_xlabel("$x_0$", fontsize=9)
ax.set_ylabel("$x_1$", fontsize=9)
ax.set_aspect("equal")
# Auto-range with padding
all_vals = np.concatenate([box_min[:, :2].ravel(), box_max[:, :2].ravel(),
                           init_min[:, :2].ravel(), init_max[:, :2].ravel()])
lo, hi = all_vals.min() - 0.5, all_vals.max() + 0.5
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.grid(True, alpha=0.15, linewidth=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=7)

# Legend for before/after
from matplotlib.lines import Line2D
legend_els = [
    Line2D([0], [0], color=C_SLATE, linewidth=1.0, linestyle="--", alpha=0.5, label="before"),
    Line2D([0], [0], color=C_SLATE, linewidth=1.5, label="after"),
]
ax.legend(handles=legend_els, fontsize=6.5, loc="upper left", frameon=True)

fig.tight_layout()
out = Path(__file__).resolve().parent.parent / "docs" / "training_convergence.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
