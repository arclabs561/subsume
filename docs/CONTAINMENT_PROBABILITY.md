# Containment Probability

## Definition

The **containment probability** \(P(B \subseteq A)\) measures whether box \(B\) is geometrically contained within box \(A\). For hard boxes:

\[
P(B \subseteq A) = \frac{\text{Vol}(A \cap B)}{\text{Vol}(B)}
\]

For Gumbel boxes, this becomes a ratio of expected volumes.

## Statement

**Theorem (First-Order Approximation).** For Gumbel boxes with random volumes \(\text{Vol}(A \cap B)\) and \(\text{Vol}(B)\):

\[
\mathbb{E}\left[\frac{\text{Vol}(A \cap B)}{\text{Vol}(B)}\right] \approx \frac{\mathbb{E}[\text{Vol}(A \cap B)]}{\mathbb{E}[\text{Vol}(B)]}
\]

The approximation is accurate when the coefficient of variation \(\text{Var}(\text{Vol}(B))/\mathbb{E}[\text{Vol}(B)]^2\) is small (i.e., when volume variance is controlled by the scale parameter \(\beta\)).

## Proof

For \(f(X, Y) = X/Y\), the first-order Taylor expansion around \((\mu_X, \mu_Y)\) is:

\[
f(X, Y) \approx f(\mu_X, \mu_Y) + \frac{\partial f}{\partial X}(\mu_X, \mu_Y)(X - \mu_X) + \frac{\partial f}{\partial Y}(\mu_X, \mu_Y)(Y - \mu_Y)
\]

with partial derivatives:
- \(\frac{\partial f}{\partial X} = 1/Y\)
- \(\frac{\partial f}{\partial Y} = -X/Y^2\)

Taking expectations:

\[
\mathbb{E}[f(X, Y)] \approx \frac{\mu_X}{\mu_Y} + \frac{1}{\mu_Y} \mathbb{E}[X - \mu_X] - \frac{\mu_X}{\mu_Y^2} \mathbb{E}[Y - \mu_Y] = \frac{\mu_X}{\mu_Y}
\]

The first-order correction terms vanish because \(\mathbb{E}[X - \mu_X] = 0\) and \(\mathbb{E}[Y - \mu_Y] = 0\).

## Error Analysis

The second-order correction term is:

\[
\text{Error} \approx -\frac{\text{Cov}(X, Y)}{\mu_Y^2} + \frac{\mu_X \text{Var}(Y)}{\mu_Y^3}
\]

The approximation is valid when:
- \(\text{Var}(Y)/\mu_Y^2\) is small (small coefficient of variation)
- \(X\) and \(Y\) are positively correlated (reduces error)
- \(\mu_Y\) is bounded away from zero

## Example

For two boxes in 2D:
- Box A: \([0.0, 0.0]\) to \([1.0, 1.0]\) (volume = 1.0)
- Box B: \([0.2, 0.2]\) to \([0.8, 0.8]\) (volume = 0.36)

**Hard boxes:**
- Intersection: \([0.2, 0.2]\) to \([0.8, 0.8]\) (volume = 0.36)
- Containment: \(P(B \subseteq A) = 0.36 / 0.36 = 1.0\)

**Gumbel boxes** (with \(\beta = 0.1\)):
- Expected intersection volume ≈ 0.35
- Expected volume of B ≈ 0.35
- Containment: \(P(B \subseteq A) \approx 0.35 / 0.35 = 1.0\)

The approximation is accurate because volumes have low variance controlled by \(\beta\).

