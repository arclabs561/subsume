# Mathematical Quick Reference

A concise reference guide for the key mathematical formulas and concepts in box embeddings.

## Core Formulas

### Volume Calculations

**Hard Volume:**
\[
\text{Vol}(B) = \prod_{i=1}^{d} \max(Z_i - z_i, 0)
\]

**Soft Volume (Gaussian Convolution):**
\[
\text{Vol}(B) \approx \prod_{i=1}^{d} T \cdot \text{softplus}\left(\frac{Z_i - z_i}{T}\right)
\]

**Gumbel-Box Volume (Bessel Approximation):**
\[
\mathbb{E}[\text{Vol}(B)] = 2\beta K_0\left(2e^{-\frac{\mu_y - \mu_x}{2\beta}}\right) \approx \beta \log(1 + \exp(\frac{\mu_y - \mu_x}{\beta} - 2\gamma))
\]

### Containment and Overlap

**Containment Probability:**
\[
P(B \subseteq A) = \frac{\text{Vol}(A \cap B)}{\text{Vol}(B)} \approx \frac{\mathbb{E}[\text{Vol}(A \cap B)]}{\mathbb{E}[\text{Vol}(B)]}
\]

**Overlap Probability:**
\[
P(A \cap B \neq \emptyset) = \frac{\text{Vol}(A \cap B)}{\text{Vol}(A) + \text{Vol}(B) - \text{Vol}(A \cap B)}
\]

### Intersection Operations

**Hard Intersection:**
\[
(z_{\cap,i}, Z_{\cap,i}) = (\max(z_i^A, z_i^B), \min(Z_i^A, Z_i^B))
\]

**Gumbel Intersection:**
\[
Z_{\cap,i} \sim \text{MaxGumbel}(\text{lse}_\beta(\mu_{z,i}^A, \mu_{z,i}^B))
\]

## Key Functions

### Log-Sum-Exp

**Basic Form:**
\[
\text{lse}_\beta(x, y) = \beta \log(e^{x/\beta} + e^{y/\beta})
\]

**Stable Form:**
\[
\text{lse}_\beta(x, y) = \max(x, y) + \beta \log(1 + e^{-|x-y|/\beta})
\]

**Limits:**
- \(\beta \to 0\): \(\text{lse}_\beta(x, y) \to \max(x, y)\)
- \(\beta \to \infty\): \(\text{lse}_\beta(x, y) \to (x + y)/2\)

### Modified Bessel Function \(K_0\)

**Definition:**
\[
K_0(z) = \int_0^{\infty} e^{-z \cosh t} dt
\]

**Asymptotic Behavior:**
- \(z \to 0\): \(K_0(z) \sim -\ln(z/2) - \gamma\)
- \(z \to \infty\): \(K_0(z) \sim \sqrt{\pi/(2z)} e^{-z}\)

## Gumbel Distribution Properties

### Max-Stability

If \(G_1, \ldots, G_k \sim \text{Gumbel}(\mu, \beta)\) are independent:

\[
\max\{G_1, \ldots, G_k\} \sim \text{Gumbel}(\mu + \beta \ln k, \beta)
\]

### Min-Stability

If \(G_1, \ldots, G_k \sim \text{MinGumbel}(\mu, \beta)\) are independent:

\[
\min\{G_1, \ldots, G_k\} \sim \text{MinGumbel}(\mu - \beta \ln k, \beta)
\]

### Gumbel-Max Property

If \(G_1, G_2 \sim \text{Gumbel}(0, \beta)\) are independent:

\[
\max(x + G_1, y + G_2) \sim \text{Gumbel}(\text{lse}_\beta(x, y), \beta)
\]

## Approximations

### First-Order Taylor Approximation

For \(f(X, Y) = X/Y\):

\[
\mathbb{E}[X/Y] \approx \frac{\mathbb{E}[X]}{\mathbb{E}[Y]}
\]

**Error (second-order):**
\[
\text{Error} \approx -\frac{\text{Cov}(X, Y)}{\mu_Y^2} + \frac{\mu_X \text{Var}(Y)}{\mu_Y^3}
\]

**Validity:** When \(\text{Var}(Y)/\mu_Y^2\) is small (small coefficient of variation).

## Constants

- **Euler-Mascheroni constant**: \(\gamma \approx 0.5772\)
- **Minimum safe temperature**: \(T_{\min} = 10^{-3}\)
- **Maximum safe temperature**: \(T_{\max} = 10.0\)

## Distribution Definitions

### Gumbel Distribution

**CDF:**
\[
F(x; \mu, \beta) = e^{-e^{-(x-\mu)/\beta}}
\]

**PDF (MaxGumbel):**
\[
f(x; \mu, \beta) = \frac{1}{\beta} e^{-(x-\mu)/\beta - e^{-(x-\mu)/\beta}}
\]

**PDF (MinGumbel):**
\[
f(x; \mu, \beta) = \frac{1}{\beta} e^{(x-\mu)/\beta - e^{(x-\mu)/\beta}}
\]

**Mean:** \(\mu + \beta\gamma\) (MaxGumbel) or \(\mu - \beta\gamma\) (MinGumbel)

**Variance:** \(\frac{\pi^2}{6}\beta^2\)

## Measure-Theoretic Foundations

**Uniform Base Measure:**
\[
P(c) = \mu_{\text{uniform}}(B) = \text{Vol}(B) = \prod_{i=1}^d (Z_i - z_i)
\]

**Conditional Probability:**
\[
P(h|t) = \frac{\text{Vol}(B_h \cap B_t)}{\text{Vol}(B_t)}
\]

## Expressiveness

**Full Expressiveness Dimension:**
\[
d = |E|^{n-1} \cdot |R|
\]

For binary relations (\(n=2\)):
\[
d = |E| \cdot |R|
\]

## See Also

- [`MATHEMATICAL_FOUNDATIONS.md`](MATHEMATICAL_FOUNDATIONS.md) — Complete derivations and explanations
- [`MATH_TO_CODE_CONNECTIONS.md`](MATH_TO_CODE_CONNECTIONS.md) — Implementation details
- [`MATH_EXPLANATION_GAPS.md`](MATH_EXPLANATION_GAPS.md) — Detailed analysis of concepts

