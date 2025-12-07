# Local Identifiability Problem

## Definition

The **local identifiability problem** arises when multiple parameter configurations produce identical loss values, creating flat regions in the loss landscape with zero gradients. This prevents learning.

## Statement

**Theorem (Local Identifiability Problem).** For hard boxes, the loss landscape has flat regions with zero gradients:

1. **Disjoint boxes**: When boxes \(A\) and \(B\) are completely disjoint, \(\text{Vol}(A \cap B) = 0\) regardless of their separation distance. Any local perturbation preserving disjointness yields zero gradient.

2. **Contained boxes**: When box \(B\) is fully contained in box \(A\), small perturbations preserving containment produce identical loss values, creating zero-gradient regions.

**Theorem (Gumbel Solution).** By modeling coordinates as Gumbel random variables, the expected volume computation involves all parameters continuously:

\[
\mathbb{E}[\text{Vol}(A \cap B)] = \int \int \text{Vol}(A(\theta_A, \epsilon_A) \cap B(\theta_B, \epsilon_B)) \, dP(\epsilon_A) \, dP(\epsilon_B)
\]

This ensemble perspective ensures that different parameter configurations produce different expected loss values, restoring local identifiability.

## Proof

Hard boxes produce zero gradients when:
- Boxes are disjoint: \(\text{Vol}(A \cap B) = 0\) with zero gradient
- Boxes are contained: Small perturbations don't change loss

Gumbel boxes solve this because:
- **Expected volumes are always positive**: Even when boxes are disjoint, \(\mathbb{E}[\text{Vol}(A \cap B)] > 0\) due to probabilistic boundaries
- **Gradients are dense**: All parameters contribute to the expected volume through the Bessel function (see [`GUMBEL_BOX_VOLUME.md`](GUMBEL_BOX_VOLUME.md))
- **Smooth loss landscape**: The probabilistic formulation eliminates flat regions

## Example

**Hard boxes (disjoint):**
- Box A: \([0.0, 0.0]\) to \([0.3, 0.3]\)
- Box B: \([0.7, 0.7]\) to \([1.0, 1.0]\)
- Intersection volume: \(0\) (zero gradient)

**Gumbel boxes (disjoint):**
- Expected intersection volume: \(> 0\) (small but positive)
- Gradient: Non-zero, allowing learning to pull boxes together or push them apart

The probabilistic formulation ensures the loss landscape has no flat regions, enabling gradient-based optimization.

