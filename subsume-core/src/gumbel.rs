//! Gumbel box embeddings for probabilistic containment.
//!
//! # Mathematical Foundations
//!
//! Gumbel boxes model box coordinates as Gumbel random variables to solve the
//! **local identifiability problem** that prevents learning with hard boxes.
//!
//! ## Gumbel Distribution Properties
//!
//! ### Max-Stability: The Key Property
//!
//! **Max-Stability** is the property that makes Gumbel distributions special for box embeddings.
//! It says: if you take the maximum of several Gumbel random variables, the result is still
//! a Gumbel random variable (just with different parameters).
//!
//! **Formal statement**: If \(G_1, \ldots, G_k \sim \text{Gumbel}(\mu, \beta)\) are independent, then:
//!
//! \[
//! \max\{G_1, \ldots, G_k\} \sim \text{Gumbel}(\mu + \beta \ln k, \beta)
//! \]
//!
//! **Why this matters**: When computing box intersections, we take the maximum of coordinates
//! from different boxes. Max-stability ensures the result is still a Gumbel distribution,
//! maintaining **algebraic closure**—the intersection of Gumbel boxes is still a Gumbel box.
//!
//! **Paradigm problem**: Consider intersecting two boxes A and B. Their max coordinates are
//! Gumbel random variables. The intersection's max coordinate is max(max_A, max_B). Max-stability
//! tells us this is still Gumbel, so we can continue intersecting boxes without leaving the
//! Gumbel family. This is crucial for training, as it means all operations stay differentiable
//! and we maintain algebraic closure.
//!
//! **Min-Stability:** Similarly, for MinGumbel:
//!
//! \[
//! \min\{G_1, \ldots, G_k\} \sim \text{MinGumbel}(\mu - \beta \ln k, \beta)
//! \]
//!
//! ## Volume Calculation: From Gumbel to Bessel
//!
//! ### The Problem
//!
//! For a Gumbel box, the min coordinate \(X \sim \text{MinGumbel}(\mu_x, \beta)\) and max
//! coordinate \(Y \sim \text{MaxGumbel}(\mu_y, \beta)\) are random variables. The box volume
//! is \(\max(Y-X, 0)\), which is also a random variable. We need its expected value.
//!
//! ### The Derivation (Step-by-Step)
//!
//! **Step 1**: The volume is \(\max(Y-X, 0)\). We want \(\mathbb{E}[\max(Y-X, 0)]\).
//!
//! **Step 2**: This requires integrating over the joint distribution of \(X\) and \(Y\).
//! The Gumbel PDFs are known, but the integration is complex.
//!
//! **Step 3**: **Dasgupta et al. (2020)** showed that this integral simplifies to a
//! Bessel function:
//!
//! \[
//! \mathbb{E}[\max(Y-X, 0)] = 2\beta K_0\left(2e^{-\frac{\mu_y - \mu_x}{2\beta}}\right)
//! \]
//!
//! where \(K_0\) is the modified Bessel function of the second kind, order zero.
//!
//! **Step 4**: For numerical stability (Bessel functions can be expensive to compute),
//! we use a log-space approximation:
//!
//! \[
//! 2\beta K_0(2e^{-x/(2\beta)}) \approx \beta \log(1 + \exp(\frac{x}{\beta} - 2\gamma))
//! \]
//!
//! where \(\gamma \approx 0.5772\) is the Euler-Mascheroni constant and \(x = \mu_y - \mu_x\).
//!
//! **Why this works**: The Bessel function \(K_0\) naturally appears when integrating
//! Gumbel distributions. The approximation uses the fact that \(K_0(z) \approx -\ln(z/2) - \gamma\)
//! for small \(z\), which we can compute stably in log-space.
//!
//! **Reference**: The complete derivation is in **Dasgupta et al. (2020)**, Section 3,
//! "Deriving the Expected Volume Formula". The key insight is that the Gumbel CDF leads
//! to an integral that evaluates to a Bessel function. See the [mathematical foundations](../docs/MATHEMATICAL_FOUNDATIONS.md)
//! for the full derivation from Gumbel PDFs to Bessel functions.
//!
//! ## Temperature Parameter
//!
//! The `temperature` parameter (denoted `β` or `τ`) controls the "softness":
//!
//! - **β → 0**: Hard bounds (standard boxes, discrete-like behavior)
//! - **β → ∞**: Soft bounds (approaches uniform distribution)
//!
//! The temperature must remain **constant across dimensions** to preserve min-max stability.
//!
//! See [`docs/MATHEMATICAL_FOUNDATIONS.md`](../../../docs/MATHEMATICAL_FOUNDATIONS.md) for:
//! - Complete derivation from Gumbel PDFs to Bessel functions
//! - Proofs of max-stability and min-stability
//! - Log-sum-exp function and numerical stability
//! - Gumbel-Softmax framework details
//!
//! See [`docs/MATH_TO_CODE_CONNECTIONS.md`](../../../docs/MATH_TO_CODE_CONNECTIONS.md) for
//! how these mathematical concepts map to code implementations.

use crate::{Box, BoxError};

/// A Gumbel box embedding with temperature-controlled softness.
///
/// Gumbel boxes use the Gumbel-Softmax reparameterization trick to make
/// box bounds differentiable during training. The `temperature` parameter
/// controls the "hardness" of the bounds:
///
/// - `temperature → 0`: Hard bounds (standard boxes)
/// - `temperature → ∞`: Soft bounds (approaches uniform distribution)
///
/// # Example
///
/// ```rust,ignore
/// // This example requires a backend implementation (e.g., subsume-ndarray)
/// use subsume_core::GumbelBox;
/// use subsume_ndarray::NdarrayGumbelBox;
/// use ndarray::array;
///
/// let gumbel_box = NdarrayGumbelBox::new(
///     array![0.0, 0.0],
///     array![1.0, 1.0],
///     0.5, // temperature
/// ).unwrap();
///
/// // Sample a point from the box distribution
/// let sample = gumbel_box.sample();
///
/// // Compute membership probability for a point
/// let point = array![0.5, 0.5];
/// let prob = gumbel_box.membership_probability(&point).unwrap();
/// ```
///
/// # Research Background
///
/// Gumbel boxes solve the **local identifiability problem** identified in **Dasgupta et al. (2020)**,
/// "Improving Local Identifiability in Probabilistic Box Embeddings" (NeurIPS 2020).
///
/// **Key Papers**:
/// - **Dasgupta et al. (2020)**: Introduces Gumbel-box process to solve identifiability
///   - Paper: [arXiv:2004.13131](https://arxiv.org/abs/2004.13131) | [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/2cfa8a9da6cdae0c7ac0b94a2c3f4c0d-Abstract.html)
///   - Key contribution: Expected volume formula using Bessel function \(K_0\)
///   - Shows ~6 F1 improvement over SmoothBox on entailment tasks
///   - Section 3: Derives Bessel function from Gumbel PDFs
///
/// - **Jang et al. (2016)**: "Categorical Reparameterization with Gumbel-Softmax"
///   - Introduces Gumbel-Softmax trick for differentiable categorical sampling
///   - Foundation for probabilistic box embeddings
///
/// - **Vilnis et al. (2018)**: "Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures"
///   - Foundational work on probabilistic box embeddings
///   - Establishes volume as probability measure
///
/// **The Problem**: Hard boxes create "flat regions" where gradients vanish when boxes are disjoint.
/// Gumbel boxes ensure all parameters contribute to expected volume, providing dense gradients.
pub trait GumbelBox: Box {
    /// Get the temperature parameter (controls softness of bounds).
    fn temperature(&self) -> Self::Scalar;

    /// Compute membership probability for a point.
    ///
    /// This computes P(point ∈ self) using the Gumbel-Softmax framework.
    ///
    /// ## Mathematical Formulation
    ///
    /// For a point \(x\) and box with bounds \([min, max]\) at temperature \(\tau\):
    ///
    /// \[
    /// P(\text{point} \in \text{self}) = \sigma\left(\frac{x - \text{min}}{\tau}\right) \cdot \sigma\left(\frac{\text{max} - x}{\tau}\right)
    /// \]
    ///
    /// where \(\sigma\) is the sigmoid function. This is the product of:
    /// - \(P(x > \text{min})\): Probability that point is above minimum bound
    /// - \(P(x < \text{max})\): Probability that point is below maximum bound
    ///
    /// **Why this works:** The Gumbel-Softmax framework models box boundaries as
    /// probabilistic rather than hard. The sigmoid functions provide smooth gradients
    /// even when the point is near or outside the boundaries.
    ///
    /// ## Interpretation
    ///
    /// - **1.0**: Point is definitely inside the box (high confidence)
    /// - **0.0**: Point is definitely outside the box (high confidence)
    /// - **0.5**: Point is on the boundary (uncertain)
    ///
    /// As temperature decreases (\(\tau \to 0\)), the probability becomes sharper
    /// (more like hard membership). As temperature increases (\(\tau \to \infty\)), the
    /// probability becomes smoother (approaches uniform).
    ///
    /// **Temperature scheduling**: During training, it's common to start with high temperature
    /// (smooth gradients) and gradually decrease to low temperature (sharp boundaries) for
    /// better final performance.
    ///
    /// ## Numerical Stability
    ///
    /// This function uses [`stable_sigmoid`](crate::utils::stable_sigmoid) to avoid
    /// overflow when computing \(\exp(-x/\tau)\) for large \(|x|/\tau\).
    ///
    /// The stable sigmoid implementation:
    /// - If \(x < 0\): uses \(e^x / (1 + e^x)\) to avoid computing \(e^{-x}\) when \(x\) is large and negative
    /// - If \(x \geq 0\): uses \(1 / (1 + e^{-x})\) which is stable for positive \(x\)
    ///
    /// See [`docs/MATHEMATICAL_FOUNDATIONS.md`](../../../docs/MATHEMATICAL_FOUNDATIONS.md)
    /// section "Gumbel-Softmax Framework" for more details.
    ///
    /// # Errors
    ///
    /// Returns `BoxError::DimensionMismatch` if point dimension doesn't match box dimension.
    fn membership_probability(&self, point: &Self::Vector) -> Result<Self::Scalar, BoxError>;

    /// Sample a point from the box distribution.
    ///
    /// Uses Gumbel-Softmax to sample a point that lies within the box bounds.
    /// The sampling is uniform within the box (all points equally likely), but the
    /// Gumbel distribution ensures the sampling is differentiable with respect to
    /// the box parameters.
    ///
    /// ## Sampling Process
    ///
    /// 1. Sample Gumbel noise for each dimension
    /// 2. Transform Gumbel samples to [0, 1] using tanh and temperature scaling
    /// 3. Map to box bounds [min, max] in each dimension
    ///
    /// This ensures samples are always within bounds while maintaining differentiability.
    ///
    /// ## Use Cases
    ///
    /// - **Training**: Sampling enables stochastic gradient estimation
    /// - **Inference**: Can sample multiple points to estimate box properties
    /// - **Visualization**: Sample points to visualize box shape and location
    fn sample(&self) -> Self::Vector;
}
