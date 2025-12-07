# Log-Sum-Exp and Gumbel Intersection

## Definition

The **log-sum-exp function** with temperature \(\beta > 0\) is:

\[
\text{lse}_\beta(x, y) = \beta \log(e^{x/\beta} + e^{y/\beta})
\]

For Gumbel boxes, intersection coordinates are computed using log-sum-exp.

## Statement

**Theorem (Gumbel-Max Property).** If \(G_1, G_2 \sim \text{Gumbel}(0, \beta)\) are independent, then:

\[
\max(x + G_1, y + G_2) \sim \text{Gumbel}(\text{lse}_\beta(x, y), \beta)
\]

The location parameter of the maximum is the log-sum-exp of the input locations.

## Proof

The CDF of \(\max(x + G_1, y + G_2)\) is:

\[
P(\max(x + G_1, y + G_2) \leq z) = P(x + G_1 \leq z \land y + G_2 \leq z)
\]

Since \(G_1\) and \(G_2\) are independent:

\[
= P(G_1 \leq z - x) \cdot P(G_2 \leq z - y)
\]

For \(\text{Gumbel}(0, \beta)\), the CDF is \(F(z) = e^{-e^{-z/\beta}}\), so:

\[
= e^{-e^{-(z-x)/\beta}} \cdot e^{-e^{-(z-y)/\beta}} = e^{-(e^{-(z-x)/\beta} + e^{-(z-y)/\beta})}
\]

Factoring out \(e^{-z/\beta}\):

\[
= e^{-e^{-z/\beta}(e^{x/\beta} + e^{y/\beta})} = e^{-e^{-(z - \beta\ln(e^{x/\beta} + e^{y/\beta}))/\beta}}
\]

This is the CDF of \(\text{Gumbel}(\beta\ln(e^{x/\beta} + e^{y/\beta}), \beta) = \text{Gumbel}(\text{lse}_\beta(x, y), \beta)\).

## Numerical Stability

Direct computation of \(e^{x/\beta} + e^{y/\beta}\) can overflow. The stable form is:

\[
\text{lse}_\beta(x, y) = \max(x, y) + \beta \log(1 + e^{-|x-y|/\beta})
\]

The correction term is bounded: \(0 \leq \beta \log(1 + e^{-|x-y|/\beta}) \leq \beta \log 2\).

## Limits

- As \(\beta \to 0\): \(\text{lse}_\beta(x, y) \to \max(x, y)\) (hard maximum)
- As \(\beta \to \infty\): \(\text{lse}_\beta(x, y) \to (x + y)/2\) (arithmetic mean)

The temperature controls the "softness" of the maximum operation.

## Application to Box Intersection

For Gumbel boxes, intersection coordinates use log-sum-exp:
- Intersection minimum: \(z_{\cap,i} \sim \text{MaxGumbel}(\text{lse}_\beta(\mu_{z,i}^A, \mu_{z,i}^B), \beta)\) (maximum of two minimums)
- Intersection maximum: \(Z_{\cap,i} \sim \text{MinGumbel}(\text{lse}_\beta(\mu_{Z,i}^A, \mu_{Z,i}^B), \beta)\) (minimum of two maximums)

This preserves the Gumbel distribution family through intersection operations, maintaining algebraic closure (see [`GUMBEL_MAX_STABILITY.md`](GUMBEL_MAX_STABILITY.md)).

## Example

For \(x = 100.0\), \(y = 100.5\), \(\beta = 0.1\):

**Unstable computation:**
\[
\text{lse}_\beta(x, y) = 0.1 \cdot \log(e^{1000} + e^{1005}) \quad \text{(overflows!)}
\]

**Stable computation:**
\[
\text{lse}_\beta(x, y) = 100.5 + 0.1 \cdot \log(1 + e^{-5}) \approx 100.5 + 0.0067 = 100.5067
\]

The stable form avoids overflow by working with bounded correction terms.

