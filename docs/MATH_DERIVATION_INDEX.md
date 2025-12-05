# Mathematical Derivations: Index and Cross-References

This document provides an index of all mathematical derivations in the documentation, with cross-references for easy navigation.

## Complete Derivations

### 1. Gumbel Max-Stability Proof
**Location:** `MATHEMATICAL_FOUNDATIONS.md`, "Min-Max Stability" section

**Shows:** If \(G_1, \ldots, G_k \sim \text{Gumbel}(\mu, \beta)\), then \(\max\{G_1, \ldots, G_k\} \sim \text{Gumbel}(\mu + \beta \ln k, \beta)\)

**Key Steps:**
1. CDF of maximum = product of individual CDFs
2. Algebraic manipulation of exponential terms
3. Recognition of shifted Gumbel CDF

**See Also:** Min-stability proof in same section

---

### 2. Gumbel-Max Property (Log-Sum-Exp Connection)
**Location:** `MATHEMATICAL_FOUNDATIONS.md`, "The Log-Sum-Exp Function" section

**Shows:** If \(G_1, G_2 \sim \text{Gumbel}(0, \beta)\), then \(\max(x + G_1, y + G_2) \sim \text{Gumbel}(\text{lse}_\beta(x, y), \beta)\)

**Key Steps:**
1. CDF of maximum using independence
2. Factorization of exponential terms
3. Recognition of log-sum-exp in the location parameter

**See Also:** Log-sum-exp numerical stability, Temperature scaling

---

### 3. Bessel Function Derivation (Gumbel to \(K_0\))
**Location:** `MATHEMATICAL_FOUNDATIONS.md`, "Derivation: From Gumbel to Bessel" section

**Shows:** How \(\mathbb{E}[\max(Y-X, 0)]\) for Gumbel random variables reduces to \(2\beta K_0(2e^{-(\mu_y - \mu_x)/(2\beta)})\)

**Key Steps:**
1. Write expectation as double integral over Gumbel PDFs
2. Change of variables \(u = (x-\mu_x)/\beta\), \(v = (y-\mu_y)/\beta\)
3. Change integration order
4. Substitution \(w = u - v - \delta\)
5. Further substitution recognizing \(\cosh\) structure
6. Identification with Bessel function integral representation

**See Also:** Modified Bessel function definition, Asymptotic behavior, Numerical approximation

---

### 4. First-Order Taylor Approximation
**Location:** `MATHEMATICAL_FOUNDATIONS.md`, "Derivation of the First-Order Approximation" section

**Shows:** \(\mathbb{E}[X/Y] \approx \mathbb{E}[X]/\mathbb{E}[Y]\) for random variables \(X\) and \(Y\)

**Key Steps:**
1. Taylor expansion of \(f(X, Y) = X/Y\) around \((\mu_X, \mu_Y)\)
2. Compute partial derivatives
3. Take expectations (first-order terms vanish)
4. Second-order error analysis

**See Also:** Error analysis, Validity conditions

---

### 5. Log-Sum-Exp Numerical Stability
**Location:** `MATHEMATICAL_FOUNDATIONS.md`, "The Log-Sum-Exp Function" section

**Shows:** How to compute \(\text{lse}_\beta(x, y) = \beta \log(e^{x/\beta} + e^{y/\beta})\) stably

**Key Steps:**
1. Factor out maximum: \(e^{x/\beta} + e^{y/\beta} = e^{\max(x,y)/\beta}(1 + e^{-|x-y|/\beta})\)
2. Take logarithm and simplify
3. Show bounded correction term

**See Also:** Gumbel-max property, Temperature scaling

---

### 6. Bessel Function Numerical Approximation
**Location:** `MATHEMATICAL_FOUNDATIONS.md`, "Numerical Approximation" section

**Shows:** \(2\beta K_0(2e^{-x/(2\beta)}) \approx \beta \log(1 + \exp(x/\beta - 2\gamma))\)

**Key Steps:**
1. Small-argument expansion of \(K_0(z)\)
2. Substitute \(z = 2e^{-x/(2\beta)}\)
3. Recognize softplus form for smoothness

**See Also:** Asymptotic behavior of \(K_0\), Euler-Mascheroni constant

---

### 7. Min-Stability Proof
**Location:** `MATHEMATICAL_FOUNDATIONS.md`, "Min-Max Stability and Algebraic Closure" section

**Shows:** If \(G_1, \ldots, G_k \sim \text{MinGumbel}(\mu, \beta)\), then \(\min\{G_1, \ldots, G_k\} \sim \text{MinGumbel}(\mu - \beta \ln k, \beta)\)

**Key Steps:**
1. Relationship between MinGumbel and MaxGumbel (negation)
2. Apply max-stability to negated variables
3. Negate result to get min-stability

**See Also:** Max-stability proof, Algebraic closure

---

## Partial Derivations / Proof Sketches

### 8. Expressiveness Dimension
**Location:** `MATHEMATICAL_FOUNDATIONS.md`, "Expressiveness" section

**States:** \(d = |E|^{n-1} \cdot |R|\) for full expressiveness

**Note:** Full proof requires lattice theory and is beyond scope. Reference to papers provided.

---

### 9. Idempotency Recovery
**Location:** `MATHEMATICAL_FOUNDATIONS.md`, "Idempotency" section

**States:** Gumbel boxes recover exact idempotency as variance → 0

**Note:** This follows from continuity of the Gumbel distribution family.

---

## Formula Derivations (Without Full Proofs)

### 10. Containment Probability Formula
**Location:** `MATHEMATICAL_FOUNDATIONS.md`, "Containment Probability" section

**Formula:** \(P(B \subseteq A) = \text{Vol}(A \cap B) / \text{Vol}(B)\)

**Rationale:** Direct from definition of conditional probability under uniform measure

---

### 11. Overlap Probability Formula
**Location:** `MATHEMATICAL_FOUNDATIONS.md`, "Overlap Probability" section

**Formula:** \(P(A \cap B \neq \emptyset) = \text{Vol}(A \cap B) / \text{Vol}(A \cup B)\)

**Rationale:** Uses inclusion-exclusion principle for union volume

---

### 12. Hard Intersection Formula
**Location:** `MATHEMATICAL_FOUNDATIONS.md`, "Hard Intersection" section

**Formula:** \((z_{\cap,i}, Z_{\cap,i}) = (\max(z_i^A, z_i^B), \min(Z_i^A, Z_i^B))\)

**Rationale:** Standard geometric intersection of axis-aligned boxes

---

## Quick Reference

For formulas without derivations, see [`MATH_QUICK_REFERENCE.md`](MATH_QUICK_REFERENCE.md).

For code connections, see [`MATH_TO_CODE_CONNECTIONS.md`](MATH_TO_CODE_CONNECTIONS.md).

For gaps and missing explanations, see [`MATH_EXPLANATION_GAPS.md`](MATH_EXPLANATION_GAPS.md).

