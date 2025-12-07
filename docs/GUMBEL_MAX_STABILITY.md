# Gumbel Max-Stability

## Definition

A distribution \(G\) is **max-stable** if, for any \(n \geq 1\), there exist constants \(a_n > 0\) and \(b_n\) such that:

\[
G^n(a_n x + b_n) = G(x)
\]

This means the maximum of \(n\) independent samples (after appropriate scaling) has the same distribution as a single sample.

## Statement

**Theorem (Gumbel Max-Stability).** If \(G_1, \ldots, G_k \sim \text{Gumbel}(\mu, \beta)\) are independent, then:

\[
\max\{G_1, \ldots, G_k\} \sim \text{Gumbel}(\mu + \beta \ln k, \beta)
\]

The location parameter shifts by \(\beta \ln k\), preserving the Gumbel family.

## Proof

The CDF of the maximum of \(k\) independent random variables is the product of their CDFs:

\[
P(\max\{G_1, \ldots, G_k\} \leq x) = \prod_{i=1}^k P(G_i \leq x) = [G(x)]^k
\]

For Gumbel distribution \(G(x) = e^{-e^{-(x-\mu)/\beta}}\):

\[
[G(x)]^k = \left(e^{-e^{-(x-\mu)/\beta}}\right)^k = e^{-k e^{-(x-\mu)/\beta}}
\]

The CDF of \(\text{Gumbel}(\mu + \beta \ln k, \beta)\) is:

\[
e^{-e^{-(x-(\mu+\beta\ln k))/\beta}} = e^{-e^{-(x-\mu)/\beta + \ln k}} = e^{-k e^{-(x-\mu)/\beta}}
\]

This matches exactly, proving max-stability.

## Min-Stability

**Corollary (Min-Stability).** For MinGumbel, if \(G_1, \ldots, G_k \sim \text{MinGumbel}(\mu, \beta)\) are independent:

\[
\min\{G_1, \ldots, G_k\} \sim \text{MinGumbel}(\mu - \beta \ln k, \beta)
\]

**Proof:** Using \(\min\{G_1, \ldots, G_k\} = -\max\{-G_1, \ldots, -G_k\}\) and the relationship between MinGumbel and MaxGumbel: if \(G_i \sim \text{MinGumbel}(\mu, \beta)\), then \(-G_i \sim \text{MaxGumbel}(-\mu, \beta)\). By max-stability, \(\max\{-G_1, \ldots, -G_k\} \sim \text{MaxGumbel}(-\mu + \beta \ln k, \beta)\), so \(\min\{G_1, \ldots, G_k\} \sim \text{MinGumbel}(\mu - \beta \ln k, \beta)\).

## Why It Matters

Max-stability ensures that box intersection operations preserve the Gumbel distribution family:
- Intersection minimum = \(\max(\min_A, \min_B)\) → still MinGumbel-distributed
- Intersection maximum = \(\min(\max_A, \max_B)\) → still MaxGumbel-distributed

This **algebraic closure** enables analytical volume calculations and maintains the mathematical structure throughout operations.

## Example

Three independent Gumbel random variables:
- \(G_1, G_2, G_3 \sim \text{Gumbel}(0, 1)\)

**Max-stability property:**
\[
\max\{G_1, G_2, G_3\} \sim \text{Gumbel}(\ln 3, 1) = \text{Gumbel}(1.099, 1)
\]

**Verification:** The CDF of the maximum is:
\[
P(\max\{G_1, G_2, G_3\} \leq x) = [e^{-e^{-x}}]^3 = e^{-3e^{-x}} = e^{-e^{-(x-\ln 3)}}
\]

which is the CDF of \(\text{Gumbel}(\ln 3, 1)\).

