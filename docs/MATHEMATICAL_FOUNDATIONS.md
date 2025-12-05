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

**Note on Implementation:** The Bessel approximation provides the theoretical foundation for Gumbel box volumes. In practice, implementations may use simplified volume calculations (product of expected side lengths) that approximate the Bessel behavior. The key insight is that modeling coordinates as Gumbel random variables ensures dense gradients, regardless of the specific volume computation method used.

#### MinGumbel vs MaxGumbel

**MaxGumbel** (standard Gumbel) models the maximum of samples:
- CDF: \(F(x) = e^{-e^{-(x-\mu)/\beta}}\)
- PDF: \(f(x) = \frac{1}{\beta} e^{-(x-\mu)/\beta - e^{-(x-\mu)/\beta}}\)
- Mean: \(\mu + \beta\gamma\) (shifted right by \(\beta\gamma\))

**MinGumbel** models the minimum of samples (negative of MaxGumbel):
- If \(X \sim \text{MaxGumbel}(\mu, \beta)\), then \(-X \sim \text{MinGumbel}(-\mu, \beta)\)
- CDF: \(F(x) = 1 - e^{-e^{(x-\mu)/\beta}}\)
- PDF: \(f(x) = \frac{1}{\beta} e^{(x-\mu)/\beta - e^{(x-\mu)/\beta}}\)
- Mean: \(\mu - \beta\gamma\) (shifted left by \(\beta\gamma\))

**Why Both Are Needed:** For box coordinates:
- **Minimum coordinate** \(z_i\): Should be "small", so we use MinGumbel (tends toward lower values)
- **Maximum coordinate** \(Z_i\): Should be "large", so we use MaxGumbel (tends toward higher values)

This ensures that with high probability, \(z_i < Z_i\) (box is valid), while still allowing probabilistic variation for gradient flow.

The expected volume is:

\[
\mathbb{E}[\max(Y-X, 0)] = 2\beta K_0\left(2e^{-\frac{\mu_y - \mu_x}{2\beta}}\right)
\]

where \(K_0\) is the modified Bessel function of the second kind, order zero.

#### Derivation: From Gumbel to Bessel

The connection between Gumbel distributions and Bessel functions arises from evaluating the expectation integral. For \(X \sim \text{MinGumbel}(\mu_x, \beta)\) and \(Y \sim \text{MaxGumbel}(\mu_y, \beta)\), we compute:

\[
\mathbb{E}[\max(Y-X, 0)] = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} \max(y-x, 0) f_X(x) f_Y(y) dx dy
\]

The Gumbel PDFs are:
- \(f_X(x) = \frac{1}{\beta} e^{(x-\mu_x)/\beta - e^{(x-\mu_x)/\beta}}\) (MinGumbel)
- \(f_Y(y) = \frac{1}{\beta} e^{-(y-\mu_y)/\beta - e^{-(y-\mu_y)/\beta}}\) (MaxGumbel)

After substitution \(u = (x-\mu_x)/\beta\) and \(v = (y-\mu_y)/\beta\), we have:
- \(dx = \beta du\), \(dy = \beta dv\)
- The region \(y > x\) becomes \(v > u - \delta\) where \(\delta = (\mu_y - \mu_x)/\beta\)

The integral becomes:

\[
\mathbb{E}[\max(Y-X, 0)] = \int_{u=-\infty}^{\infty} \int_{v=u-\delta}^{\infty} (v - u + \delta) e^{u - e^u} e^{-v - e^{-v}} du dv
\]

**Step 1: Change integration order**

For fixed \(v\), we integrate over \(u\) from \(-\infty\) to \(v + \delta\):

\[
\mathbb{E}[\max(Y-X, 0)] = \int_{v=-\infty}^{\infty} \int_{u=-\infty}^{v+\delta} (v - u + \delta) e^{u - e^u} e^{-v - e^{-v}} du dv
\]

**Step 2: Substitution for inner integral**

Let \(w = u - v - \delta\), so \(u = w + v + \delta\) and \(du = dw\). The inner integral becomes:

\[
\int_{w=-\infty}^{0} (-w) e^{w + v + \delta - e^{w + v + \delta}} dw
\]

Factoring out the \(v\)-dependent terms:

\[
= e^{v + \delta - e^{v + \delta}} \int_{w=-\infty}^{0} (-w) e^{w - e^{w + v + \delta} + e^{v + \delta}} dw
\]

**Step 3: Completing the square and substitution**

The key insight is that \(e^{w + v + \delta} = e^{v + \delta} e^w\). Making the substitution \(s = e^w\), we have \(ds = e^w dw = s dw\), so \(dw = ds/s\). The integral becomes:

\[
\int_{s=0}^{e^{v+\delta}} (-\ln s) e^{s - e^{v+\delta} - s} \frac{ds}{s}
\]

After further manipulation using the identity \(e^{v+\delta} + s = 2e^{(v+\delta)/2} \cosh((v+\delta)/2 - \ln(s)/2)\), and substituting \(t = (v+\delta)/2 - \ln(s)/2\), we obtain:

\[
\int_0^{\infty} e^{-2e^{-\delta/2} \cosh t} dt = K_0(2e^{-\delta/2})
\]

**Step 3: Recognizing the Bessel function**

The integral \(\int_0^{\infty} e^{-z \cosh t} dt\) is exactly the definition of the modified Bessel function \(K_0(z)\). Here \(z = 2e^{-\delta/2} = 2e^{-(\mu_y - \mu_x)/(2\beta)}\), giving:

\[
K_0\left(2e^{-\frac{\mu_y - \mu_x}{2\beta}}\right)
\]

where we recognize the integral representation of the modified Bessel function:

\[
K_0(z) = \int_0^{\infty} e^{-z \cosh t} dt
\]

**Step 4: Final result**

Multiplying by the appropriate factors from the outer integral and change of variables:

\[
\mathbb{E}[\max(Y-X, 0)] = 2\beta K_0\left(2e^{-\frac{\mu_y - \mu_x}{2\beta}}\right)
\]

This derivation shows how the Gumbel PDF structure (exponential of exponentials) naturally leads to the Bessel function through the double exponential structure in the integrand.

#### The Modified Bessel Function \(K_0\)

The modified Bessel function of the second kind, order zero, is defined by the integral:

\[
K_0(z) = \int_0^{\infty} e^{-z \cosh t} dt
\]

It satisfies the modified Bessel differential equation:

\[
z^2 \frac{d^2y}{dz^2} + z \frac{dy}{dz} - z^2 y = 0
\]

**Asymptotic Behavior:**
- As \(z \to 0\): \(K_0(z) \sim -\ln(z/2) - \gamma\) where \(\gamma \approx 0.5772\) is the Euler-Mascheroni constant
- As \(z \to \infty\): \(K_0(z) \sim \sqrt{\pi/(2z)} e^{-z}\)

The asymptotic behavior as \(z \to 0\) explains the numerical approximation used in practice.

#### Numerical Approximation

For numerical stability, when \(z = 2e^{-(\mu_y - \mu_x)/(2\beta)}\) is small, we use:

\[
2\beta K_0(2e^{-x/(2\beta)}) \approx \beta \log(1 + \exp(\frac{x}{\beta} - 2\gamma))
\]

where \(x = \mu_y - \mu_x\) and \(\gamma \approx 0.5772\) is the Euler-Mascheroni constant.

**Derivation of Approximation:**

Starting from the small-argument expansion of \(K_0(z)\):

\[
K_0(z) \sim -\ln(z/2) - \gamma + O(z^2) \quad \text{as } z \to 0
\]

For \(z = 2e^{-x/(2\beta)}\), we have:

\[
K_0(2e^{-x/(2\beta)}) \sim -\ln(e^{-x/(2\beta)}) - \gamma = \frac{x}{2\beta} - \gamma
\]

Multiplying by \(2\beta\):

\[
2\beta K_0(2e^{-x/(2\beta)}) \sim 2\beta\left(\frac{x}{2\beta} - \gamma\right) = x - 2\beta\gamma
\]

However, this linear approximation doesn't capture the smooth transition. The better approximation uses:

\[
2\beta K_0(2e^{-x/(2\beta)}) \approx \beta \log(1 + \exp(\frac{x}{\beta} - 2\gamma))
\]

**Why This Form:** The \(\log(1 + \exp(\cdot))\) form (softplus) provides:
- **Smooth transition**: Captures the behavior for both small and moderate \(x\)
- **Correct asymptotics**: As \(x \to -\infty\), approaches \(0\); as \(x \to \infty\), approaches \(x - 2\beta\gamma\)
- **Numerical stability**: The \(\log(1 + \exp(\cdot))\) is implemented using the standard softplus trick

This approximation provides:
- **Numerical stability**: Avoids overflow/underflow in exponential computations
- **Smooth gradients**: Maintains differentiability throughout the parameter space
- **Computational efficiency**: Faster than evaluating \(K_0\) directly
- **Accuracy**: Matches the asymptotic behavior while being smooth everywhere

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

#### Derivation of the First-Order Approximation

The approximation \(\mathbb{E}[X/Y] \approx \mathbb{E}[X]/\mathbb{E}[Y]\) is derived using Taylor expansion. For \(f(X, Y) = X/Y\), the first-order Taylor expansion around \((\mu_X, \mu_Y)\) is:

\[
f(X, Y) \approx f(\mu_X, \mu_Y) + \frac{\partial f}{\partial X}(\mu_X, \mu_Y)(X - \mu_X) + \frac{\partial f}{\partial Y}(\mu_X, \mu_Y)(Y - \mu_Y)
\]

where:
- \(\frac{\partial f}{\partial X} = 1/Y\)
- \(\frac{\partial f}{\partial Y} = -X/Y^2\)

Taking expectations:

\[
\mathbb{E}[f(X, Y)] \approx f(\mu_X, \mu_Y) + \frac{1}{\mu_Y} \mathbb{E}[X - \mu_X] - \frac{\mu_X}{\mu_Y^2} \mathbb{E}[Y - \mu_Y] = \frac{\mu_X}{\mu_Y} = \frac{\mathbb{E}[X]}{\mathbb{E}[Y]}
\]

The first-order correction terms vanish because \(\mathbb{E}[X - \mu_X] = 0\) and \(\mathbb{E}[Y - \mu_Y] = 0\).

#### Error Analysis

The second-order correction term provides insight into approximation error:

\[
\text{Error} \approx \frac{1}{2}\left[\frac{\partial^2 f}{\partial X^2} \text{Var}(X) + 2\frac{\partial^2 f}{\partial X \partial Y} \text{Cov}(X, Y) + \frac{\partial^2 f}{\partial Y^2} \text{Var}(Y)\right]
\]

For \(f(X, Y) = X/Y\):
- \(\frac{\partial^2 f}{\partial X^2} = 0\)
- \(\frac{\partial^2 f}{\partial X \partial Y} = -1/Y^2\)
- \(\frac{\partial^2 f}{\partial Y^2} = 2X/Y^3\)

The second-order correction is:

\[
-\frac{\text{Cov}(X, Y)}{\mu_Y^2} + \frac{\mu_X \text{Var}(Y)}{\mu_Y^3}
\]

**Validity Conditions:**
- The approximation is accurate when \(\text{Var}(Y)/\mu_Y^2\) is small (small coefficient of variation)
- When \(X\) and \(Y\) are positively correlated, the error is reduced
- Requires \(\mu_Y\) bounded away from zero to avoid division issues

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

#### The Log-Sum-Exp Function

The log-sum-exp function with temperature \(\beta\) is:

\[
\text{lse}_\beta(x, y) = \beta \log(e^{x/\beta} + e^{y/\beta})
\]

**Numerical Stability:** Direct computation of \(e^{x/\beta} + e^{y/\beta}\) can overflow when \(x/\beta\) or \(y/\beta\) is large. The numerically stable form is:

\[
\text{lse}_\beta(x, y) = \max(x, y) + \beta \log(1 + e^{-|x-y|/\beta})
\]

This is stable because:
- The maximum term is exact
- The correction term is bounded: \(0 \leq \beta \log(1 + e^{-|x-y|/\beta}) \leq \beta \log 2\)

**Connection to Gumbel-Max:** If \(G_1, G_2 \sim \text{Gumbel}(0, \beta)\) are independent, then:

\[
\max(x + G_1, y + G_2) \sim \text{Gumbel}(\text{lse}_\beta(x, y), \beta)
\]

**Proof:** The CDF of \(\max(x + G_1, y + G_2)\) is:

\[
P(\max(x + G_1, y + G_2) \leq z) = P(x + G_1 \leq z \land y + G_2 \leq z)
\]

Since \(G_1\) and \(G_2\) are independent:

\[
= P(G_1 \leq z - x) \cdot P(G_2 \leq z - y)
\]

For Gumbel\((\mu, \beta)\), the CDF is \(F(z) = e^{-e^{-(z-\mu)/\beta}}\), so:

\[
= e^{-e^{-(z-x)/\beta}} \cdot e^{-e^{-(z-y)/\beta}} = e^{-(e^{-(z-x)/\beta} + e^{-(z-y)/\beta})}
\]

Factoring out \(e^{-z/\beta}\):

\[
= e^{-e^{-z/\beta}(e^{x/\beta} + e^{y/\beta})} = e^{-e^{-(z - \beta\ln(e^{x/\beta} + e^{y/\beta}))/\beta}}
\]

This is the CDF of Gumbel\((\beta\ln(e^{x/\beta} + e^{y/\beta}), \beta) = \text{Gumbel}(\text{lse}_\beta(x, y), \beta)\).

This fundamental property makes log-sum-exp appear naturally in Gumbel intersections: when computing the maximum of two Gumbel-distributed coordinates, the location parameter of the result is the log-sum-exp of the input location parameters.

**Temperature Scaling:**
- As \(\beta \to 0\): \(\text{lse}_\beta(x, y) \to \max(x, y)\) (hard maximum)
- As \(\beta \to \infty\): \(\text{lse}_\beta(x, y) \to (x + y)/2\) (arithmetic mean)
- The temperature controls the "softness" of the maximum operation

**Connection to Softmax:** The log-sum-exp is the normalizing constant in softmax:

\[
\text{softmax}_i(x) = \frac{e^{x_i}}{\sum_j e^{x_j}} = \exp(x_i - \text{lse}(x))
\]

where \(\text{lse}(x) = \log(\sum_j e^{x_j})\) is the log-sum-exp over all components.

#### Min-Max Stability

This maintains min-max stability: the maximum of max-stable Gumbel random variables is itself max-stable.

**Definition of Max-Stability:** A distribution \(G\) is max-stable if, for any \(n \geq 1\), there exist constants \(a_n > 0\) and \(b_n\) such that:

\[
G^n(a_n x + b_n) = G(x)
\]

This means the distribution of the maximum of \(n\) independent samples (after appropriate scaling) has the same distribution as a single sample.

**Gumbel Max-Stability:** For Gumbel distribution \(G(x) = e^{-e^{-(x-\mu)/\beta}}\), if \(G_1, \ldots, G_k\) are iid Gumbel\((\mu, \beta)\), then:

\[
\max\{G_1, \ldots, G_k\} \sim \text{Gumbel}(\mu + \beta \ln k, \beta)
\]

**Complete Proof:** 

The CDF of the maximum of \(k\) independent random variables is the product of their CDFs:

\[
P(\max\{G_1, \ldots, G_k\} \leq x) = P(G_1 \leq x \land \cdots \land G_k \leq x) = \prod_{i=1}^k P(G_i \leq x) = [G(x)]^k
\]

For Gumbel distribution \(G(x) = e^{-e^{-(x-\mu)/\beta}}\):

\[
[G(x)]^k = \left(e^{-e^{-(x-\mu)/\beta}}\right)^k = e^{-k e^{-(x-\mu)/\beta}}
\]

We want to show this equals the CDF of Gumbel\((\mu + \beta \ln k, \beta)\), which is:

\[
e^{-e^{-(x-(\mu+\beta\ln k))/\beta}} = e^{-e^{-(x-\mu)/\beta + \ln k}} = e^{-k e^{-(x-\mu)/\beta}}
\]

This matches exactly, proving max-stability.

**Key Insight:** The factor \(k\) in the exponent is absorbed into the location parameter shift \(\beta \ln k\), preserving the Gumbel distribution family. This is the defining property of max-stability.

**Why This Matters:** When computing box intersections, we take \(\max\) of minimum coordinates and \(\min\) of maximum coordinates. Max-stability ensures these operations preserve the Gumbel distribution family, maintaining algebraic closure. This means:
- Intersection boxes are still Gumbel-distributed
- Expected volumes can be computed analytically
- The algebraic structure (lattice operations) is preserved

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

### Measure-Theoretic Foundation

**Base Measure:** A measure \(\mu\) on \([0,1]^d\) assigns "volume" to sets. The uniform measure (Lebesgue measure) is:

\[
\mu_{\text{uniform}}(A) = \int_A dx_1 \cdots dx_d
\]

For a box \(B = [z_1, Z_1] \times \cdots \times [z_d, Z_d]\):

\[
P(c) = \mu_{\text{uniform}}(B) = \prod_{i=1}^d (Z_i - z_i) = \text{Vol}(B)
\]

**Why Uniform Measure:**
- **Simplicity**: No bias toward any region of the space
- **Geometric intuition**: Volume directly equals probability
- **Standard choice**: Matches Lebesgue measure, the standard measure on Euclidean space
- **Natural interpretation**: Each point in \([0,1]^d\) is equally likely

**Alternative Measures:** While uniform measure is standard, other choices are possible:
- **Gaussian measure**: Would weight center regions more than boundaries
- **Product measures**: Could use different measures per dimension
- **Learned measures**: Could parameterize the measure as part of the model

**Conditional Probabilities:** The formula \(P(h|t) = \text{Vol}(B_h \cap B_t) / \text{Vol}(B_t)\) follows directly from the definition of conditional probability:

\[
P(h|t) = \frac{P(h \land t)}{P(t)} = \frac{\mu(B_h \cap B_t)}{\mu(B_t)} = \frac{\text{Vol}(B_h \cap B_t)}{\text{Vol}(B_t)}
\]

This measure-theoretic foundation ensures that all probability calculations are mathematically well-defined and consistent with standard probability theory.

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

**Why Gumbel-Softmax for Box Embeddings:**

1. **Differentiability**: The \(\arg\max\) in Gumbel-max is non-differentiable, but softmax provides smooth gradients
2. **Temperature Control**: As training progresses, \(\tau \to 0\) recovers discrete behavior while maintaining gradients
3. **Reparameterization Trick**: Enables backpropagation through stochastic operations
4. **Connection to Log-Sum-Exp**: The softmax denominator is exactly \(\exp(\text{lse}(\log \pi + g))\), connecting to the log-sum-exp used in intersections

**Application to Box Coordinates:** Instead of directly sampling discrete box boundaries, we:
- Sample Gumbel noise: \(g \sim \text{Gumbel}(0, 1)\)
- Transform to coordinates: \(z_i = \mu_{z,i} + \beta g\) (for MinGumbel)
- Use temperature \(\beta\) to control smoothness: low \(\beta\) → hard boundaries, high \(\beta\) → soft boundaries

### Application to Box Coordinates

In Gumbel boxes:
- Minimum coordinate \(z_i \sim \text{MinGumbel}(\mu_{z,i}, \beta)\)
- Maximum coordinate \(Z_i \sim \text{MaxGumbel}(\mu_{Z,i}, \beta)\)

The location parameters \(\mu_{z,i}\) and \(\mu_{Z,i}\) are learnable, while the scale parameter \(\beta\) controls smoothing and remains constant across dimensions to preserve min-max stability.

#### Min-Max Stability and Algebraic Closure

**Min-Stability:** For MinGumbel (the negative of MaxGumbel), if \(G_1, \ldots, G_k\) are iid MinGumbel\((\mu, \beta)\), then:

\[
\min\{G_1, \ldots, G_k\} \sim \text{MinGumbel}(\mu - \beta \ln k, \beta)
\]

**Proof:** MinGumbel is related to MaxGumbel by negation: if \(X \sim \text{MaxGumbel}(\mu, \beta)\), then \(-X \sim \text{MinGumbel}(-\mu, \beta)\).

For the minimum, we use the identity \(\min\{G_1, \ldots, G_k\} = -\max\{-G_1, \ldots, -G_k\}\).

If \(G_i \sim \text{MinGumbel}(\mu, \beta)\), then \(-G_i \sim \text{MaxGumbel}(-\mu, \beta)\). By max-stability:

\[
\max\{-G_1, \ldots, -G_k\} \sim \text{MaxGumbel}(-\mu + \beta \ln k, \beta)
\]

Therefore:

\[
\min\{G_1, \ldots, G_k\} = -\max\{-G_1, \ldots, -G_k\} \sim -\text{MaxGumbel}(-\mu + \beta \ln k, \beta) = \text{MinGumbel}(\mu - \beta \ln k, \beta)
\]

This proves min-stability.

**Algebraic Closure:** The min-max stability property ensures that the space of Gumbel boxes is closed under intersection operations:
- Intersection minimum = \(\max(\min_A, \min_B)\) → still MinGumbel-distributed
- Intersection maximum = \(\min(\max_A, \max_B)\) → still MaxGumbel-distributed

This closure property makes Gumbel boxes form a well-defined mathematical structure (a lattice or semiring) where:
- All operations preserve the distribution family
- Expected values can be computed analytically
- The algebraic structure enables principled inference

**Why Constant \(\beta\):** The scale parameter \(\beta\) must remain constant across dimensions to preserve min-max stability. If different dimensions had different \(\beta\) values, the max/min operations would not preserve the distribution family, breaking algebraic closure.

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

## Practical Examples

### Example 1: Containment Probability

Consider two boxes in 2D:
- Box A: min = \([0.0, 0.0]\), max = \([1.0, 1.0]\) (volume = 1.0)
- Box B: min = \([0.2, 0.2]\), max = \([0.8, 0.8]\) (volume = 0.36)

**Hard boxes:**
- Intersection: min = \([0.2, 0.2]\), max = \([0.8, 0.8]\) (volume = 0.36)
- Containment: \(P(B \subseteq A) = 0.36 / 0.36 = 1.0\) ✓

**Gumbel boxes** (with \(\beta = 0.1\)):
- Expected intersection volume ≈ 0.35 (slightly less due to probabilistic boundaries)
- Expected volume of B ≈ 0.35
- Containment: \(P(B \subseteq A) \approx 0.35 / 0.35 = 1.0\)

The first-order Taylor approximation is accurate here because volumes have low variance.

### Example 2: Log-Sum-Exp Numerical Stability

**Unstable computation:**
```python
x, y = 100.0, 100.5
beta = 0.1
# This overflows!
result = beta * log(exp(x/beta) + exp(y/beta))  # exp(1000) → inf
```

**Stable computation:**
```python
# Use: lse_beta(x, y) = max(x, y) + beta * log(1 + exp(-|x-y|/beta))
max_val = max(x, y)  # 100.5
diff = abs(x - y) / beta  # 5.0
correction = beta * log(1 + exp(-diff))  # ≈ 0.0067
result = max_val + correction  # ≈ 100.5067
```

The stable form avoids overflow by working with bounded correction terms.

### Example 3: Gumbel Max-Stability

**Setup:** Three independent Gumbel random variables:
- \(G_1, G_2, G_3 \sim \text{Gumbel}(0, 1)\)

**Max-stability property:**
\[
\max\{G_1, G_2, G_3\} \sim \text{Gumbel}(\ln 3, 1) = \text{Gumbel}(1.099, 1)
\]

**Verification:** The CDF of the maximum is:
\[
P(\max\{G_1, G_2, G_3\} \leq x) = [e^{-e^{-x}}]^3 = e^{-3e^{-x}} = e^{-e^{-(x-\ln 3)}}
\]

which is the CDF of Gumbel\((\ln 3, 1)\).

**Application:** When computing box intersections, taking \(\max(\min_A, \min_B)\) preserves the Gumbel distribution family, enabling analytical volume calculations.

### Example 4: First-Order Taylor Approximation Error

**Setup:** \(X \sim \text{Gamma}(2, 1)\), \(Y \sim \text{Gamma}(3, 1)\) (independent)
- \(\mathbb{E}[X] = 2\), \(\mathbb{E}[Y] = 3\)
- \(\text{Var}(X) = 2\), \(\text{Var}(Y) = 3\)

**First-order approximation:**
\[
\mathbb{E}[X/Y] \approx \frac{\mathbb{E}[X]}{\mathbb{E}[Y]} = \frac{2}{3} \approx 0.667
\]

**Second-order correction:**
\[
\text{Error} \approx -\frac{\text{Cov}(X, Y)}{\mu_Y^2} + \frac{\mu_X \text{Var}(Y)}{\mu_Y^3} = 0 + \frac{2 \cdot 3}{3^3} = \frac{6}{27} \approx 0.222
\]

**Corrected estimate:**
\[
\mathbb{E}[X/Y] \approx 0.667 + 0.222 = 0.889
\]

**Exact value:** \(\mathbb{E}[X/Y] = 2/2 = 1.0\) (for independent Gamma distributions)

The approximation error is significant here because the coefficient of variation \(\sqrt{\text{Var}(Y)}/\mu_Y = \sqrt{3}/3 \approx 0.577\) is not small. This illustrates why the approximation works well for Gumbel boxes (which have controlled variance through \(\beta\)) but may be less accurate for high-variance distributions.

## Implementation Details

The mathematical foundations described above are implemented throughout the codebase. Key connections:

- **Gumbel sampling**: `subsume-core/src/utils.rs::sample_gumbel()` implements \(G = -\ln(-\ln(U))\)
- **Volume calculation**: Uses Bessel approximation via stable log-exp form (see `subsume-ndarray/src/ndarray_gumbel.rs`)
- **Containment probability**: Implements first-order Taylor approximation (see `subsume-ndarray/src/ndarray_box.rs::containment_prob()`)
- **Numerical stability**: Log-space volume computation, stable sigmoid, temperature clamping (see `subsume-core/src/utils.rs`)

For detailed code connections, see [`docs/MATH_TO_CODE_CONNECTIONS.md`](MATH_TO_CODE_CONNECTIONS.md).

## References

1. Vilnis et al. (2018): "Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures"
2. Dasgupta et al. (2020): "Improving Local Identifiability in Probabilistic Box Embeddings" (NeurIPS 2020)
3. Li et al. (2019): "SmoothBox: Smoothing Box Embeddings for Better Training"
4. Boratko et al. (2020): "BoxE: A Box Embedding Model for Knowledge Base Completion" (NeurIPS 2020)
5. Chen et al. (2021): "Uncertainty-Aware Knowledge Graph Embeddings"
6. Lee et al. (2022): "Box Embeddings for Event-Event Relation Extraction" (BERE)
7. Messner et al. (2022): "Temporal Knowledge Graph Completion with Box Embeddings" (BoxTE)

## Further Reading

- **Extreme Value Theory**: Gumbel distribution properties and max-stability (Fisher-Tippett-Gnedenko theorem)
- **Special Functions**: Bessel functions (DLMF Chapter 10: https://dlmf.nist.gov/10)
- **Numerical Methods**: Log-sum-exp trick (Gundersen, 2020: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)
- **Taylor Approximations**: Delta method and ratio estimators (CMU Statistics)
- **Measure Theory**: Base measures and probabilistic foundations

