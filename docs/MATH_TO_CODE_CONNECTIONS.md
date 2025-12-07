# Mathematical Foundations: Code Connections

This document connects the mathematical one-pagers to their implementations in the codebase, showing how theory translates to practice. For the mathematical foundations, see the one-pagers in [`MATHEMATICAL_FOUNDATIONS.md`](MATHEMATICAL_FOUNDATIONS.md).

## Gumbel Distribution and Max-Stability

### Theory
Gumbel distributions are max-stable: if \(G_1, \ldots, G_k \sim \text{Gumbel}(\mu, \beta)\), then:
\[
\max\{G_1, \ldots, G_k\} \sim \text{Gumbel}(\mu + \beta \ln k, \beta)
\]

### Implementation
**Location:** `subsume-core/src/utils.rs::sample_gumbel()`

```rust
pub fn sample_gumbel(u: f32, epsilon: f32) -> f32 {
    let u_clamped = u.clamp(epsilon, 1.0 - epsilon);
    -(-u_clamped.ln()).ln()
}
```

**Connection:** The function implements \(G = -\ln(-\ln(U))\) where \(U \sim \text{Uniform}(0,1)\). The clamping to \([\epsilon, 1-\epsilon]\) ensures numerical stability, avoiding:
- `ln(0)` when \(U \approx 0\) (would give \(-\infty\))
- `ln(1)` when \(U \approx 1\) (would give \(0\), then \(-\ln(0) = \infty\))

**Why Max-Stability Matters:** When computing box intersections, we take \(\max(\min_A, \min_B)\). If both minimums are Gumbel-distributed, max-stability ensures the result is also Gumbel-distributed, preserving the algebraic structure.

---

## Bessel Function and Volume Calculation

### Theory
For Gumbel boxes with \(X \sim \text{MinGumbel}(\mu_x, \beta)\) and \(Y \sim \text{MaxGumbel}(\mu_y, \beta)\):
\[
\mathbb{E}[\max(Y-X, 0)] = 2\beta K_0\left(2e^{-\frac{\mu_y - \mu_x}{2\beta}}\right)
\]

For numerical stability, this is approximated as:
\[
2\beta K_0(2e^{-x/(2\beta)}) \approx \beta \log(1 + \exp(\frac{x}{\beta} - 2\gamma))
\]

### Implementation
**Location:** `subsume-ndarray/src/ndarray_gumbel.rs` and `subsume-candle/src/candle_gumbel.rs`

**Implementation Note:** The current implementation uses the softplus approximation for computational efficiency. The Bessel function \(K_0\) provides the theoretical foundation, but the softplus form \(\beta \log(1 + \exp(x/\beta - 2\gamma))\) is used in practice because it:
- Matches the Bessel function asymptotically (see [`docs/typst-output/pdf/gumbel-box-volume.pdf`](typst-output/pdf/gumbel-box-volume.pdf) for the derivation)
- Provides numerical stability (avoids overflow/underflow)
- Is computationally efficient (no special function evaluation needed)
- Maintains smooth gradients throughout parameter space

**Theoretical vs Practical:** 
- **Theory**: Gumbel boxes use expected volumes with Bessel function \(K_0\) (see PDF for complete derivation)
- **Practice**: Implementation uses the softplus approximation, which is mathematically equivalent in the regimes where it's used

**Key Insight:** The approximation uses the small-argument expansion of \(K_0(z)\):
- As \(z \to 0\): \(K_0(z) \sim -\ln(z/2) - \gamma\)
- This leads to the \(\beta \log(1 + \exp(\cdot))\) form (softplus)

**Why This Works:** The softplus approximation \(\beta \log(1 + \exp(x/\beta - 2\gamma))\) provides:
- Smooth gradients throughout parameter space
- Numerical stability (avoids overflow/underflow)
- Computational efficiency (faster than evaluating \(K_0\) directly)
- Correct asymptotic limits

---

## First-Order Taylor Approximation

### Theory
For containment probability with Gumbel boxes:
\[
\mathbb{E}\left[\frac{\text{Vol}(A \cap B)}{\text{Vol}(B)}\right] \approx \frac{\mathbb{E}[\text{Vol}(A \cap B)]}{\mathbb{E}[\text{Vol}(B)]}
\]

### Implementation
**Location:** `subsume-ndarray/src/ndarray_box.rs::containment_prob()` and `subsume-candle/src/candle_box.rs::containment_prob()`

```rust
fn containment_prob(&self, other: &Self, temperature: Self::Scalar) -> Result<Self::Scalar, BoxError> {
    let intersection = self.intersection(other)?;
    let intersection_vol = intersection.volume(temperature)?;
    let other_vol = other.volume(temperature)?;
    
    if other_vol <= 0.0 {
        return Err(BoxError::ZeroVolume);
    }
    
    Ok((intersection_vol / other_vol).clamp(0.0, 1.0))
}
```

**Connection:** The code computes \(\mathbb{E}[\text{Vol}(A \cap B)] / \mathbb{E}[\text{Vol}(B)]\) directly, using the first-order approximation. The approximation is valid when:
- The coefficient of variation \(\text{Var}(Y)/\mu_Y^2\) is small
- The volumes are bounded away from zero (checked by the `ZeroVolume` error)

**Error Handling:** The check for `other_vol <= 0.0` ensures we don't divide by zero, which would violate the validity conditions of the Taylor approximation.

---

## Log-Sum-Exp and Gumbel Intersections

### Theory
For Gumbel boxes, intersection coordinates use log-sum-exp:
\[
Z_{\cap,i} \sim \text{MaxGumbel}(\text{lse}_\beta(\mu_{z,i}^A, \mu_{z,i}^B))
\]

where:
\[
\text{lse}_\beta(x, y) = \max(x, y) + \beta \log(1 + e^{-|x-y|/\beta})
\]

### Implementation
**Note:** The current implementation uses standard box intersection (max/min operations) on the location parameters \(\mu\). The Gumbel properties come into play in:
1. **Volume calculation**: Uses standard volume formula (product of side lengths), but the Gumbel framework ensures dense gradients
2. **Membership probability**: Uses Gumbel-Softmax probabilities via `gumbel_membership_prob()`
3. **Sampling**: Uses Gumbel sampling for generating points via `sample_gumbel()` and `map_gumbel_to_bounds()`

**Theoretical vs Practical:** The Bessel approximation is the theoretical foundation explaining why Gumbel boxes work (dense gradients, local identifiability), but the implementation uses the simpler standard volume calculation. The key benefit is that Gumbel coordinates ensure all parameters contribute to gradients, even when using the standard volume formula.

**Location:** `subsume-core/src/utils.rs::gumbel_membership_prob()`

```rust
pub fn gumbel_membership_prob(x: f32, min: f32, max: f32, temp: f32) -> f32 {
    let temp_safe = clamp_temperature_default(temp);
    
    if temp_safe < MIN_TEMPERATURE {
        // Hard bounds: return 1.0 if in bounds, 0.0 otherwise
        if x >= min && x <= max {
            return 1.0;
        } else {
            return 0.0;
        }
    }
    
    // P(x > min) using stable sigmoid
    let min_prob = stable_sigmoid((x - min) / temp_safe);
    // P(x < max) using stable sigmoid
    let max_prob = stable_sigmoid((max - x) / temp_safe);
    
    min_prob * max_prob
}
```

**Connection:** This implements \(P(\min \leq x \leq \max)\) for Gumbel-distributed bounds. The stable sigmoid ensures numerical stability, similar to the log-sum-exp trick.

**Temperature Scaling:**
- **Low temperature** (\(\beta \to 0\)): Approaches hard bounds (0 or 1)
- **High temperature** (\(\beta \to \infty\)): Smooth probability distribution

---

## Numerical Stability Patterns

### Log-Space Volume Computation

### Theory
For high-dimensional boxes, direct volume computation can underflow:
\[
\text{Vol} = \prod_{i=1}^{d} (Z_i - z_i)
\]

Solution: Compute in log-space:
\[
\log(\text{Vol}) = \sum_{i=1}^{d} \log(Z_i - z_i)
\]
\[
\text{Vol} = \exp\left(\sum_{i=1}^{d} \log(Z_i - z_i)\right)
\]

### Implementation
**Location:** `subsume-core/src/utils.rs::log_space_volume()`

```rust
pub fn log_space_volume<I>(side_lengths: I) -> (f32, f32)
where
    I: Iterator<Item = f32>,
{
    let mut log_sum = 0.0;
    let mut has_zero = false;
    
    for side_len in side_lengths {
        if side_len <= EPSILON {
            has_zero = true;
            break;
        }
        log_sum += side_len.ln();
    }
    
    if has_zero {
        (f32::NEG_INFINITY, 0.0)
    } else {
        let volume = log_sum.exp();
        (log_sum, volume)
    }
}
```

**Connection:** This implements the log-space computation to avoid underflow in high dimensions. For example:
- 20 dimensions with side length 0.5: \(0.5^{20} \approx 9.5 \times 10^{-7}\)
- This can underflow to 0.0 in f32, but log-space computation is stable

**When to Use:**
- Dimension > 10
- Side lengths < 1.0
- Computing many volumes in a loop (accumulated numerical error)

---

## Stable Sigmoid

### Theory
The sigmoid function \(\sigma(x) = 1/(1 + e^{-x})\) can overflow when \(x\) is large and negative.

Stable form:
- If \(x < 0\): use \(e^x / (1 + e^x)\)
- If \(x \geq 0\): use \(1 / (1 + e^{-x})\)

### Implementation
**Location:** `subsume-core/src/utils.rs::stable_sigmoid()`

```rust
pub fn stable_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}
```

**Connection:** This pattern appears throughout the codebase wherever sigmoid-like functions are used (Gumbel membership probability, softplus approximations, etc.).

---

## Temperature Clamping

### Theory
Temperature parameters control the "softness" of operations:
- **Low temperature** (\(\beta \to 0\)): Hard, discrete-like behavior
- **High temperature** (\(\beta \to \infty\)): Smooth, continuous behavior

Extreme values cause numerical issues:
- **Too low**: Vanishing gradients, exponential underflow
- **Too high**: Loss of correspondence to discrete distributions

### Implementation
**Location:** `subsume-core/src/utils.rs::clamp_temperature()`

```rust
pub const MIN_TEMPERATURE: f32 = 1e-3;
pub const MAX_TEMPERATURE: f32 = 10.0;

pub fn clamp_temperature_default(temp: f32) -> f32 {
    clamp_temperature(temp, MIN_TEMPERATURE, MAX_TEMPERATURE)
}
```

**Connection:** All temperature-dependent operations use clamped temperatures to ensure numerical stability. This is especially important for:
- Gumbel membership probability
- Volume calculations
- Intersection operations

---

## Measure-Theoretic Foundations

### Theory
Under uniform base measure on \([0,1]^d\):
\[
P(c) = \text{Vol}(B(\theta)) = \mu_{\text{uniform}}(B)
\]

### Implementation
**Location:** Volume calculations throughout the codebase

The uniform measure assumption is implicit in all volume calculations. Boxes are constrained to \([0,1]^d\) (or can be normalized to this range), and volumes are computed as:
\[
\text{Vol}(B) = \prod_{i=1}^{d} (Z_i - z_i)
\]

**Connection:** The code doesn't explicitly implement different base measures, but the structure allows for this extension. The uniform measure is the natural choice because:
- It's the standard Lebesgue measure
- Volume directly equals probability
- No bias toward any region

---

## Summary: Mathematical Patterns in Code

1. **Max-Stability**: Preserved through Gumbel sampling and intersection operations
2. **Bessel Approximation**: Used in volume calculations via stable log-exp form
3. **Taylor Approximation**: Implicit in containment probability computation
4. **Log-Sum-Exp**: Pattern appears in stable sigmoid and log-space computations
5. **Numerical Stability**: Systematic use of log-space, clamping, and stable forms
6. **Temperature Scaling**: Controlled through clamping and parameter scheduling

These patterns ensure that the mathematical theory translates correctly to numerical implementation while maintaining stability and efficiency.

