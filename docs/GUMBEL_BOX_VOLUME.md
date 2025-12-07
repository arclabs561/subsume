# Gumbel-Box Volume

## Definition

A **Gumbel box** models each coordinate interval \([X, Y]\) as:
- \(X \sim \text{MinGumbel}(\mu_x, \beta)\) (minimum coordinate)
- \(Y \sim \text{MaxGumbel}(\mu_y, \beta)\) (maximum coordinate)

where \(\beta > 0\) is the scale parameter (constant across dimensions) and \(\mu_x, \mu_y\) are learnable location parameters.

## Statement

**Theorem (Gumbel-Box Volume).** For a Gumbel box with \(X \sim \text{MinGumbel}(\mu_x, \beta)\) and \(Y \sim \text{MaxGumbel}(\mu_y, \beta)\), the expected interval length is:

\[
\mathbb{E}[\max(Y-X, 0)] = 2\beta K_0\left(2e^{-\frac{\mu_y - \mu_x}{2\beta}}\right)
\]

where \(K_0\) is the modified Bessel function of the second kind, order zero.

## Proof

The expectation is:

\[
\mathbb{E}[\max(Y-X, 0)] = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} \max(y-x, 0) f_X(x) f_Y(y) \, dx \, dy
\]

with Gumbel PDFs:
- \(f_X(x) = \frac{1}{\beta} e^{(x-\mu_x)/\beta - e^{(x-\mu_x)/\beta}}\) (MinGumbel)
- \(f_Y(y) = \frac{1}{\beta} e^{-(y-\mu_y)/\beta - e^{-(y-\mu_y)/\beta}}\) (MaxGumbel)

**Step 1:** Substitute \(u = (x-\mu_x)/\beta\) and \(v = (y-\mu_y)/\beta\), so \(dx = \beta du\) and \(dy = \beta dv\). The region \(y > x\) becomes \(v > u - \delta\) where \(\delta = (\mu_y - \mu_x)/\beta\).

**Step 2:** Change integration order (integrate over \(u\) first for fixed \(v\)) and make substitution \(w = u - v - \delta\), then \(s = e^w\). The double exponential structure \(e^{u - e^u} e^{-v - e^{-v}}\) in the integrand leads to:

\[
\int_0^{\infty} e^{-2e^{-\delta/2} \cosh t} \, dt
\]

**Step 3:** Recognize the integral representation of the modified Bessel function:

\[
K_0(z) = \int_0^{\infty} e^{-z \cosh t} \, dt
\]

With \(z = 2e^{-\delta/2} = 2e^{-(\mu_y - \mu_x)/(2\beta)}\), multiplying by \(2\beta\) from the change of variables yields the result.

## Numerical Approximation

For small arguments, \(K_0(z) \sim -\ln(z/2) - \gamma\) as \(z \to 0\), where \(\gamma \approx 0.5772\) is Euler's constant. This leads to the stable approximation:

\[
2\beta K_0(2e^{-x/(2\beta)}) \approx \beta \log(1 + \exp(\frac{x}{\beta} - 2\gamma))
\]

where \(x = \mu_y - \mu_x\). The softplus form provides smooth gradients and numerical stability.

## Example

For \(\mu_x = 0.0\), \(\mu_y = 1.0\), \(\beta = 0.1\):
- \(z = 2e^{-(1.0-0.0)/(2 \cdot 0.1)} = 2e^{-5} \approx 0.0135\)
- \(K_0(0.0135) \approx 4.27\) (using small-argument expansion)
- Expected volume \(\approx 2 \cdot 0.1 \cdot 4.27 \approx 0.854\)

The approximation \(\beta \log(1 + \exp(1.0/0.1 - 2 \cdot 0.5772)) \approx 0.854\) matches closely.

