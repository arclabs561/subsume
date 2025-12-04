# Mathematical Foundations of Box Embeddings

This document provides a comprehensive mathematical treatment of box embeddings, drawing from the foundational research papers and theoretical developments in the field.

## Subsumption: The Core Logical Concept

**Subsumption** is a fundamental concept in formal logic and automated reasoning. In logic, one statement **subsumes** another when it is more general and covers all cases that the more specific statement would cover. This is exactly what box embeddings model geometrically.

### Formal Definition

In box embeddings, when box A contains box B (geometrically: B ⊆ A), we say that **A subsumes B**. This relationship:

- **Encodes entailment**: If premise box P subsumes hypothesis box H, then P entails H
- **Models hierarchies**: Parent concepts subsume child concepts (e.g., "animal" subsumes "dog")
- **Represents logical consequence**: The containment relationship directly corresponds to logical subsumption

The mathematical notation for subsumption is:
\[
\text{Box A subsumes Box B} \iff B \subseteq A \iff P(B|A) = 1
\]

## Volume Calculation Methods

### Hard Volume

The simplest volume calculation for a box \(B(\theta)\) with parameters \(\theta\) is:

\[
\text{Vol}(B(\theta)) = \prod_{i=1}^{d} \max(Z_i(\theta) - z_i(\theta), 0)
\]

where \(z_i\) is the minimum coordinate and \(Z_i\) is the maximum coordinate in dimension \(i\), and \(d\) is the embedding dimension.

**Limitation**: Hard volume produces zero gradients when boxes are disjoint, causing the "local identifiability problem" that prevents learning.

### Soft Volume (Gaussian Convolution)

To address gradient sparsity, soft volume smooths box boundaries using Gaussian convolution:

\[
\text{Vol}(x) \approx \prod_{i=1}^{d} T \cdot \text{softplus}\left(\frac{Z_i - z_i}{T}\right)
\]

where \(T\) is the volume temperature parameter. As \(T \to 0\), this approaches hard volume.

### Gumbel-Box Volume (Bessel Approximation)

The most sophisticated approach models box coordinates as Gumbel random variables. For an interval \([X, Y]\) where:
- \(X \sim \text{MinGumbel}(\mu_x, \beta)\)
- \(Y \sim \text{MaxGumbel}(\mu_y, \beta)\)

The expected volume is:

\[
\mathbb{E}[\max(Y-X, 0)] = 2\beta K_0\left(2e^{-\frac{\mu_y - \mu_x}{2\beta}}\right)
\]

where \(K_0\) is the modified Bessel function of the second kind, order zero.

For numerical stability, this is approximated as:

\[
2\beta K_0(2e^{-x/(2\beta)}) \approx \beta \log(1 + \exp(\frac{x}{\beta} - 2\gamma))
\]

where \(\gamma\) is the Euler-Mascheroni constant (\(\approx 0.5772\)).

**Advantage**: All parameters contribute to the expected volume, providing dense gradients throughout training.

## Containment Probability

The containment probability \(P(\text{other} \subseteq \text{self})\) is the core subsumption operation:

\[
P(B \subseteq A) = \frac{\text{Vol}(A \cap B)}{\text{Vol}(B)} = \frac{\text{Vol}(\text{intersection}(A, B))}{\text{Vol}(B)}
\]

For Gumbel boxes, this becomes:

\[
P(B \subseteq A) = \frac{\mathbb{E}[\text{Vol}(A \cap B)]}{\mathbb{E}[\text{Vol}(B)]}
\]

using the first-order Taylor approximation:

\[
\mathbb{E}\left[\frac{\text{Vol}(A \cap B)}{\text{Vol}(B)}\right] \approx \frac{\mathbb{E}[\text{Vol}(A \cap B)]}{\mathbb{E}[\text{Vol}(B)]}
\]

## Overlap Probability

The overlap probability measures whether two boxes have non-empty intersection:

\[
P(A \cap B \neq \emptyset) = \frac{\text{Vol}(A \cap B)}{\text{Vol}(A \cup B)}
\]

Using inclusion-exclusion principle:

\[
\text{Vol}(A \cup B) = \text{Vol}(A) + \text{Vol}(B) - \text{Vol}(A \cap B)
\]

Therefore:

\[
P(A \cap B \neq \emptyset) = \frac{\text{Vol}(A \cap B)}{\text{Vol}(A) + \text{Vol}(B) - \text{Vol}(A \cap B)}
\]

## Intersection Operations

### Hard Intersection

For boxes A and B with coordinates \((z^A, Z^A)\) and \((z^B, Z^B)\):

\[
(z_{\cap,i}, Z_{\cap,i}) = (\max(z_i^A, z_i^B), \min(Z_i^A, Z_i^B))
\]

The intersection is valid (non-empty) if \(z_{\cap,i} \leq Z_{\cap,i}\) for all dimensions \(i\).

### Gumbel Intersection

In the Gumbel-box framework, intersection coordinates are modeled as Gumbel random variables. The intersection minimum (which is the maximum of the two input minimums) follows:

\[
Z_{\cap,i} \sim \text{MaxGumbel}(\text{lse}_\beta(\mu_{z,i}^A, \mu_{z,i}^B))
\]

where \(\text{lse}_\beta(x, y) = \beta \log(e^{x/\beta} + e^{y/\beta})\) is the log-sum-exp function with temperature \(\beta\).

This maintains min-max stability: the maximum of max-stable Gumbel random variables is itself max-stable.

## Probabilistic Interpretation

Under the uniform base measure on the unit hypercube \([0,1]^d\), box volumes directly correspond to probabilities:

\[
P(c) = \text{Vol}(B(\theta))
\]

for concept \(c\) with associated box \(B(\theta)\).

Joint probabilities are computed through intersection volumes:

\[
P(h \land t) = \text{Vol}(B(\theta_h) \cap B(\theta_t))
\]

Conditional probabilities:

\[
P(h|t) = \frac{\text{Vol}(B(\theta_h) \cap B(\theta_t))}{\text{Vol}(B(\theta_t))}
\]

This probabilistic interpretation enables principled inference without ad-hoc normalization.

## Local Identifiability Problem

The local identifiability problem arises when multiple parameter configurations produce identical loss values, creating flat regions in the loss landscape with zero gradients.

### Problem Cases

1. **Disjoint boxes**: When boxes A and B are completely disjoint, \(\text{Vol}(A \cap B) = 0\) regardless of their separation distance. Any local perturbation preserving disjointness yields zero gradient.

2. **Contained boxes**: When box B is fully contained in box A, small perturbations preserving containment produce identical loss values, creating zero-gradient regions.

### Solution: Gumbel-Box Process

By modeling coordinates as Gumbel random variables, the expected volume computation involves all parameters continuously:

\[
\mathbb{E}[\text{Vol}(A \cap B)] = \int \int \text{Vol}(A(\theta_A, \epsilon_A) \cap B(\theta_B, \epsilon_B)) \, dP(\epsilon_A) \, dP(\epsilon_B)
\]

This ensemble perspective ensures that different parameter configurations produce different expected loss values, restoring local identifiability.

## Theoretical Guarantees

### Expressiveness

Box embeddings are **fully expressive** for representing arbitrary partial orders and lattice structures. Unlike translational models (TransE) or rotation-based models (RotatE), box embeddings can represent all inference patterns (symmetry, anti-symmetry, inversion, composition) with sufficient embedding dimensions.

For knowledge graph completion, full expressiveness is achieved with embedding dimension:

\[
d = |E|^{n-1} \cdot |R|
\]

where \(|E|\) is the number of entities, \(n\) is the maximum relation arity, and \(|R|\) is the number of relations. For binary relations (\(n=2\)):

\[
d = |E| \cdot |R|
\]

This is **linear** in the number of entities and relations, compared to exponential requirements for single-vector embeddings.

### Closure Properties

Boxes are **closed under intersection**: the intersection of two boxes is always a valid box (or empty set). This ensures mathematical consistency throughout the embedding space.

Boxes are **not closed under union** in the geometric sense, but the lattice join operation \(\vee\) computes the smallest enclosing box, providing an upper bound on the union.

### Idempotency

For probabilistic consistency, we require \(P(x|x) = 1\) for any event \(x\). Hard boxes satisfy this exactly:

\[
\frac{\text{Vol}(x \cap x)}{\text{Vol}(x)} = \frac{\text{Vol}(x)}{\text{Vol}(x)} = 1
\]

Gumbel boxes recover exact idempotency in the limit as variance approaches zero, providing theoretical continuity with hard boxes while enabling smooth gradients.

## Gumbel-Softmax Framework

The Gumbel-Softmax technique enables differentiable sampling from categorical distributions. For box embeddings, this is applied to min and max coordinates through the Gumbel-box process.

### Gumbel-Max Trick

Sampling from a categorical distribution with probabilities \(\pi\) can be expressed as:

\[
\text{sample} = \arg\max(\log(\pi) + g)
\]

where \(g \sim \text{Gumbel}(0, 1)\).

### Gumbel-Softmax

Replacing the non-differentiable \(\arg\max\) with a differentiable softmax:

\[
y_i = \frac{\exp\left(\frac{\log \pi_i + g_i}{\tau}\right)}{\sum_j \exp\left(\frac{\log \pi_j + g_j}{\tau}\right)}
\]

where \(\tau\) is the temperature parameter. As \(\tau \to 0\), this approaches a one-hot vector (discrete sampling). As \(\tau\) increases, the distribution becomes smoother.

### Application to Box Coordinates

In Gumbel boxes:
- Minimum coordinate \(z_i \sim \text{MinGumbel}(\mu_{z,i}, \beta)\)
- Maximum coordinate \(Z_i \sim \text{MaxGumbel}(\mu_{Z,i}, \beta)\)

The location parameters \(\mu_{z,i}\) and \(\mu_{Z,i}\) are learnable, while the scale parameter \(\beta\) controls smoothing and remains constant across dimensions to preserve min-max stability.

## Training Dynamics

### Volume Regularization

To address the "volume slackness problem" (multiple box configurations satisfying containment with perfect loss), volume regularization penalizes unnecessarily large boxes:

\[
L_{\text{reg}} = \lambda \sum_x \max(0, \text{Vol}(B_x) - V_{\text{threshold}})
\]

where \(\lambda\) weights the regularization and \(V_{\text{threshold}}\) is a target volume threshold.

### Loss Functions

For containment relationships, the loss encourages high containment probability for positive pairs and low for negative pairs:

\[
L_{\text{containment}} = \begin{cases}
-\log(P(B \subseteq A)) & \text{if positive pair} \\
\max(0, P(B \subseteq A) - \text{margin}) & \text{if negative pair}
\end{cases}
\]

For overlap relationships:

\[
L_{\text{overlap}} = \begin{cases}
-\log(P(A \cap B \neq \emptyset)) & \text{if should overlap} \\
\max(0, P(A \cap B \neq \emptyset) - \text{margin}) & \text{if should be disjoint}
\end{cases}
\]

## References

1. Vilnis et al. (2018): "Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures"
2. Dasgupta et al. (2020): "Improving Local Identifiability in Probabilistic Box Embeddings" (NeurIPS 2020)
3. Li et al. (2019): "SmoothBox: Smoothing Box Embeddings for Better Training"
4. Boratko et al. (2020): "BoxE: A Box Embedding Model for Knowledge Base Completion" (NeurIPS 2020)
5. Chen et al. (2021): "Uncertainty-Aware Knowledge Graph Embeddings"
6. Lee et al. (2022): "Box Embeddings for Event-Event Relation Extraction" (BERE)
7. Messner et al. (2022): "Temporal Knowledge Graph Completion with Box Embeddings" (BoxTE)

