# Mathematical Quick Reference

Concise formulas and key results. For detailed explanations, see the one-pagers in [`MATHEMATICAL_FOUNDATIONS.md`](MATHEMATICAL_FOUNDATIONS.md).

## Volume

**Hard Volume:**
\[
\text{Vol}(B) = \prod_{i=1}^{d} \max(Z_i - z_i, 0)
\]

**Gumbel-Box Volume:**
\[
\mathbb{E}[\max(Y-X, 0)] = 2\beta K_0\left(2e^{-\frac{\mu_y - \mu_x}{2\beta}}\right) \approx \beta \log(1 + \exp(\frac{\mu_y - \mu_x}{\beta} - 2\gamma))
\]

## Containment and Overlap

**Containment Probability:**
\[
P(B \subseteq A) = \frac{\text{Vol}(A \cap B)}{\text{Vol}(B)} \approx \frac{\mathbb{E}[\text{Vol}(A \cap B)]}{\mathbb{E}[\text{Vol}(B)]}
\]

**Overlap Probability:**
\[
P(A \cap B \neq \emptyset) = \frac{\text{Vol}(A \cap B)}{\text{Vol}(A) + \text{Vol}(B) - \text{Vol}(A \cap B)}
\]

## Intersection

**Hard Intersection:**
\[
(z_{\cap,i}, Z_{\cap,i}) = (\max(z_i^A, z_i^B), \min(Z_i^A, Z_i^B))
\]

**Gumbel Intersection:**
\[
Z_{\cap,i} \sim \text{MaxGumbel}(\text{lse}_\beta(\mu_{z,i}^A, \mu_{z,i}^B), \beta)
\]

## Log-Sum-Exp

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

## Gumbel Properties

**Max-Stability:**
\[
\max\{G_1, \ldots, G_k\} \sim \text{Gumbel}(\mu + \beta \ln k, \beta) \quad \text{for } G_i \sim \text{Gumbel}(\mu, \beta)
\]

**Min-Stability:**
\[
\min\{G_1, \ldots, G_k\} \sim \text{MinGumbel}(\mu - \beta \ln k, \beta) \quad \text{for } G_i \sim \text{MinGumbel}(\mu, \beta)
\]

**Gumbel-Max Property:**
\[
\max(x + G_1, y + G_2) \sim \text{Gumbel}(\text{lse}_\beta(x, y), \beta) \quad \text{for } G_1, G_2 \sim \text{Gumbel}(0, \beta)
\]

**Note:** In Gumbel boxes, minimum coordinates use MinGumbel (tends toward lower values) and maximum coordinates use MaxGumbel (tends toward higher values), ensuring \(z_i < Z_i\) with high probability while maintaining gradient flow.

## Bessel Function

**Definition:**
\[
K_0(z) = \int_0^{\infty} e^{-z \cosh t} \, dt
\]

**Asymptotics:**
- \(z \to 0\): \(K_0(z) \sim -\ln(z/2) - \gamma\)
- \(z \to \infty\): \(K_0(z) \sim \sqrt{\pi/(2z)} e^{-z}\)

## Constants

- **Euler-Mascheroni constant**: \(\gamma \approx 0.5772\)
- **Gumbel variance**: \(\frac{\pi^2}{6}\beta^2\)
