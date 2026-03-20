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
//! **Formal statement**: If $G_1, \ldots, G_k \sim \text{Gumbel}(\mu, \beta)$ are independent, then:
//!
//! $$
//! \max\{G_1, \ldots, G_k\} \sim \text{Gumbel}(\mu + \beta \ln k, \beta)
//! $$
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
//! $$
//! \min\{G_1, \ldots, G_k\} \sim \text{MinGumbel}(\mu - \beta \ln k, \beta)
//! $$
//!
//! ## Volume Calculation: From Gumbel to Bessel
//!
//! ### The Problem
//!
//! For a Gumbel box, the min coordinate $X \sim \text{MinGumbel}(\mu_x, \beta)$ and max
//! coordinate $Y \sim \text{MaxGumbel}(\mu_y, \beta)$ are random variables. The box volume
//! is $\max(Y-X, 0)$, which is also a random variable. We need its expected value.
//!
//! ### The Derivation (Step-by-Step)
//!
//! **Step 1**: The volume is $\max(Y-X, 0)$. We want $\mathbb{E}[\max(Y-X, 0)]$.
//!
//! **Step 2**: This requires integrating over the joint distribution of $X$ and $Y$.
//! The Gumbel PDFs are known, but the integration is complex.
//!
//! **Step 3**: **Dasgupta et al. (2020)** showed that this integral simplifies to a
//! Bessel function:
//!
//! $$
//! \mathbb{E}[\max(Y-X, 0)] = 2\beta K_0\left(2e^{-\frac{\mu_y - \mu_x}{2\beta}}\right)
//! $$
//!
//! where $K_0$ is the modified Bessel function of the second kind, order zero.
//!
//! **Step 4**: For numerical stability (Bessel functions can be expensive to compute),
//! we use a log-space approximation:
//!
//! $$
//! 2\beta K_0(2e^{-x/(2\beta)}) \approx \beta \log(1 + \exp(\frac{x}{\beta} - 2\gamma))
//! $$
//!
//! where $\gamma \approx 0.5772$ is the Euler-Mascheroni constant and $x = \mu_y - \mu_x$.
//!
//! **Why this works**: The Bessel function $K_0$ naturally appears when integrating
//! Gumbel distributions. The approximation uses the fact that $K_0(z) \approx -\ln(z/2) - \gamma$
//! for small `z`, which we can compute stably in log-space.
//!
//! **Reference**: The complete derivation is in **Dasgupta et al. (2020)**, Section 3,
//! "Deriving the Expected Volume Formula". The key insight is that the Gumbel CDF leads
//! to an integral that evaluates to a Bessel function. See the [mathematical foundations](https://github.com/arclabs561/subsume/blob/main/docs/MATHEMATICAL_FOUNDATIONS.md)
//! for the full derivation from Gumbel PDFs to Bessel functions.
//!
//! **For detailed study:** The PDF version [`docs/typst-output/pdf/gumbel-box-volume.pdf`](https://github.com/arclabs561/subsume/blob/main/docs/typst-output/pdf/gumbel-box-volume.pdf)
//! provides a complete step-by-step derivation with professional typesetting, including the connection
//! to extreme value theory and numerical approximation methods.
//!
//! ## Temperature Parameter
//!
//! The `temperature` parameter (denoted `β` or `τ`) controls the "softness":
//!
//! - **β → 0**: Hard bounds (standard boxes, discrete-like behavior)
//! - **β → ∞**: Soft bounds (spread increases; volume shrinks toward zero)
//!
//! The temperature must remain **constant across dimensions** to preserve min-max stability.
//!
//! See [`docs/MATHEMATICAL_FOUNDATIONS.md`](https://github.com/arclabs561/subsume/blob/main/docs/MATHEMATICAL_FOUNDATIONS.md) for:
//! - Complete derivation from Gumbel PDFs to Bessel functions
//! - Proofs of max-stability and min-stability
//! - Log-sum-exp function and numerical stability
//! - Gumbel-Softmax framework details
//!
//! See [`docs/MATH_TO_CODE_CONNECTIONS.md`](https://github.com/arclabs561/subsume/blob/main/docs/MATH_TO_CODE_CONNECTIONS.md) for
//! how these mathematical concepts map to code implementations.
