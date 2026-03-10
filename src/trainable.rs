//! Trainable geometric representations with learnable parameters.
//!
//! Contains both [`TrainableBox`] (axis-aligned hyperrectangles) and
//! [`TrainableCone`] (Cartesian products of 2D angular sectors, ConE model).

use crate::optimizer::AMSGradState;
#[cfg(feature = "ndarray-backend")]
use crate::BoxError;
use serde::{Deserialize, Serialize};

/// A simple hard box used by the trainer implementation.
///
/// This is intentionally local to `subsume` (no tensor deps).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct DenseBox {
    pub min: Vec<f32>,
    pub max: Vec<f32>,
}

impl DenseBox {
    pub fn new(min: Vec<f32>, max: Vec<f32>) -> Self {
        Self { min, max }
    }

    #[inline]
    pub fn volume(&self) -> f32 {
        self.min
            .iter()
            .zip(self.max.iter())
            .map(|(&a, &b)| (b - a).max(0.0))
            .product::<f32>()
    }

    #[inline]
    pub fn intersection_volume(&self, other: &Self) -> f32 {
        let mut v = 1.0f32;
        for i in 0..self.min.len().min(other.min.len()) {
            let lo = self.min[i].max(other.min[i]);
            let hi = self.max[i].min(other.max[i]);
            let side = (hi - lo).max(0.0);
            v *= side;
            if v == 0.0 {
                break;
            }
        }
        v
    }

    /// Containment-style probability: \(P(\text{other} \subseteq \text{self})\).
    ///
    /// For hard boxes: Vol(self ∩ other) / Vol(other), with 0 when other has 0 volume.
    #[inline]
    pub fn conditional_probability(&self, other: &Self) -> f32 {
        let denom = other.volume();
        if denom <= 0.0 {
            return 0.0;
        }
        (self.intersection_volume(other) / denom).clamp(0.0, 1.0)
    }
}

/// A trainable box embedding with learnable parameters.
///
/// Uses reparameterization to ensure min <= max:
/// - min = mu - exp(delta)/2
/// - max = mu + exp(delta)/2
///
/// This ensures boxes are always valid (min <= max).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainableBox {
    /// Mean position in each dimension (d-dimensional vector).
    pub mu: Vec<f32>,
    /// Log-width in each dimension (width = exp(delta)).
    pub delta: Vec<f32>,
    /// Dimension
    pub dim: usize,
}

impl TrainableBox {
    /// Create a new trainable box.
    ///
    /// # Arguments
    ///
    /// * `mu` - Mean position (center of box)
    /// * `delta` - Log-width (width = exp(delta))
    ///
    /// The box will have:
    /// - min = mu - exp(delta) / 2
    /// - max = mu + exp(delta) / 2
    #[must_use]
    pub fn new(mu: Vec<f32>, delta: Vec<f32>) -> Self {
        assert_eq!(
            mu.len(),
            delta.len(),
            "mu and delta must have same dimension"
        );
        let dim = mu.len();
        Self { mu, delta, dim }
    }

    /// Initialize from a vector embedding.
    ///
    /// Creates a small box around the vector with initial width `init_width`.
    #[must_use]
    pub fn from_vector(vector: &[f32], init_width: f32) -> Self {
        let mu = vector.to_vec();
        let delta: Vec<f32> = vec![init_width.ln(); mu.len()];
        Self::new(mu, delta)
    }

    /// Convert to a BoxEmbedding (for inference).
    #[must_use]
    pub(crate) fn to_box(&self) -> DenseBox {
        let min: Vec<f32> = self
            .mu
            .iter()
            .zip(self.delta.iter())
            .map(|(&m, &d)| m - (d.exp() / 2.0))
            .collect();
        let max: Vec<f32> = self
            .mu
            .iter()
            .zip(self.delta.iter())
            .map(|(&m, &d)| m + (d.exp() / 2.0))
            .collect();
        DenseBox::new(min, max)
    }

    /// Convert to an [`NdarrayBox`] for querying through the [`Box`](crate::Box) trait.
    ///
    /// This bridges the training representation (mutable, gradient-compatible)
    /// to the inference representation (immutable, trait-based). The resulting
    /// box has temperature 1.0 (hard box).
    ///
    /// [`NdarrayBox`]: crate::ndarray_backend::NdarrayBox
    #[cfg(feature = "ndarray-backend")]
    #[cfg_attr(docsrs, doc(cfg(feature = "ndarray-backend")))]
    pub fn to_ndarray_box(&self) -> Result<crate::ndarray_backend::NdarrayBox, BoxError> {
        let dense = self.to_box();
        crate::ndarray_backend::NdarrayBox::new(
            ndarray::Array1::from(dense.min),
            ndarray::Array1::from(dense.max),
            1.0,
        )
    }

    /// Update box parameters using AMSGrad optimizer.
    pub fn update_amsgrad(
        &mut self,
        grad_mu: &[f32],
        grad_delta: &[f32],
        state: &mut AMSGradState,
    ) {
        state.t += 1;
        let t = state.t as f32;

        // Update first moment (m)
        for (i, &grad) in grad_mu.iter().enumerate().take(self.dim) {
            state.m[i] = state.beta1 * state.m[i] + (1.0 - state.beta1) * grad;
        }

        // Update second moment (v) and max (v_hat)
        for (i, &grad) in grad_mu.iter().enumerate().take(self.dim) {
            let v_new = state.beta2 * state.v[i] + (1.0 - state.beta2) * grad * grad;
            state.v[i] = v_new;
            state.v_hat[i] = state.v_hat[i].max(v_new);
        }

        // Bias correction for first moment
        let m_hat: Vec<f32> = state
            .m
            .iter()
            .map(|&m| m / (1.0 - state.beta1.powf(t)))
            .collect();

        // Update mu
        for (i, &m_hat_val) in m_hat.iter().enumerate().take(self.dim) {
            let update = state.lr * m_hat_val / (state.v_hat[i].sqrt() + state.epsilon);
            self.mu[i] -= update;

            // Ensure finite
            if !self.mu[i].is_finite() {
                self.mu[i] = 0.0;
            }
        }

        // Similar for delta
        let mut m_delta = vec![0.0_f32; self.dim];
        let mut v_delta = vec![0.0_f32; self.dim];
        let mut v_hat_delta = vec![0.0_f32; self.dim];

        for i in 0..self.dim {
            m_delta[i] = state.beta1 * m_delta[i] + (1.0 - state.beta1) * grad_delta[i];
            let v_new: f32 =
                state.beta2 * v_delta[i] + (1.0 - state.beta2) * grad_delta[i] * grad_delta[i];
            v_delta[i] = v_new;
            v_hat_delta[i] = v_hat_delta[i].max(v_new);
        }

        let m_hat_delta: Vec<f32> = m_delta
            .iter()
            .map(|&m| m / (1.0 - state.beta1.powf(t)))
            .collect();

        for i in 0..self.dim {
            let update = state.lr * m_hat_delta[i] / (v_hat_delta[i].sqrt() + state.epsilon);
            self.delta[i] -= update;

            // Clamp delta to reasonable range (width between 0.01 and 10.0)
            self.delta[i] = self.delta[i].clamp(0.01_f32.ln(), 10.0_f32.ln());

            // Ensure finite
            if !self.delta[i].is_finite() {
                self.delta[i] = 0.5_f32.ln();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DenseCone: lightweight cone for the trainer (no tensor deps)
// ---------------------------------------------------------------------------

/// A simple cone used by the trainer implementation.
///
/// Represents a Cartesian product of `d` independent 2D angular sectors.
/// Each dimension has an axis angle in \[-pi, pi\] and an aperture (half-width)
/// in \[0, pi\]. Follows the ConE model (Zhang & Wang, NeurIPS 2021).
///
/// Intentionally local to `subsume` (no tensor deps), analogous to [`DenseBox`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct DenseCone {
    /// Per-dimension axis angles, each in \[-pi, pi\].
    pub axes: Vec<f32>,
    /// Per-dimension apertures (half-widths), each in \[0, pi\].
    pub apertures: Vec<f32>,
}

impl DenseCone {
    pub fn new(axes: Vec<f32>, apertures: Vec<f32>) -> Self {
        Self { axes, apertures }
    }

    /// Dimension (number of angular sectors).
    #[inline]
    pub fn dim(&self) -> usize {
        self.axes.len()
    }

    /// Compute the ConE distance score: lower = better containment.
    ///
    /// `self` is the query cone, `entity` is the entity being scored.
    /// Uses per-dimension `|sin((e - q_axis) / 2)|` with inside/outside decomposition.
    #[inline]
    pub fn cone_distance(&self, entity: &Self, cen: f32) -> f32 {
        let mut dist_out = 0.0_f32;
        let mut dist_in = 0.0_f32;

        for i in 0..self.dim() {
            let e = entity.axes[i];
            let q_axis = self.axes[i];
            let q_aper = self.apertures[i];

            let distance_to_axis = ((e - q_axis) / 2.0).sin().abs();
            let distance_base = (q_aper / 2.0).sin().abs();

            if distance_to_axis < distance_base {
                dist_in += distance_to_axis.min(distance_base);
            } else {
                let delta1 = e - (q_axis - q_aper);
                let delta2 = e - (q_axis + q_aper);
                let d1 = (delta1 / 2.0).sin().abs();
                let d2 = (delta2 / 2.0).sin().abs();
                dist_out += d1.min(d2);
            }
        }

        dist_out + cen * dist_in
    }

    /// Convert distance to a containment-like score in \[0, 1\].
    ///
    /// Uses `gamma - distance * modulus` mapped through sigmoid. Higher = better.
    #[inline]
    pub fn containment_score(&self, entity: &Self, cen: f32, gamma: f32, modulus: f32) -> f32 {
        let dist = self.cone_distance(entity, cen);
        sigmoid_f32(gamma - dist * modulus)
    }
}

// ---------------------------------------------------------------------------
// TrainableCone
// ---------------------------------------------------------------------------

/// A trainable cone embedding with per-dimension learnable parameters.
///
/// Follows the ConE model (Zhang & Wang, NeurIPS 2021): each dimension has an
/// independent axis angle and aperture (half-width). The parameterization uses
/// unconstrained scalars that are mapped to valid ranges during the forward pass:
///
/// - **raw_axes\[i\]**: unconstrained; `axis[i] = tanh(raw_axes[i]) * pi` maps to \[-pi, pi\]
/// - **raw_apertures\[i\]**: unconstrained; `aperture[i] = tanh(2 * raw_apertures[i]) * pi/2 + pi/2`
///   maps to \(0, pi\)
///
/// This ensures all parameters receive gradients regardless of the current geometry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainableCone {
    /// Raw (unconstrained) per-dimension axis angles.
    /// Actual axes = tanh(raw_axes) * pi.
    pub raw_axes: Vec<f32>,
    /// Raw (unconstrained) per-dimension apertures.
    /// Actual apertures = tanh(2 * raw_apertures) * pi/2 + pi/2.
    pub raw_apertures: Vec<f32>,
    /// Dimension.
    pub dim: usize,
}

impl TrainableCone {
    /// Create a new trainable cone from raw (unconstrained) parameters.
    #[must_use]
    pub fn new(raw_axes: Vec<f32>, raw_apertures: Vec<f32>) -> Self {
        assert_eq!(
            raw_axes.len(),
            raw_apertures.len(),
            "raw_axes and raw_apertures must have same dimension"
        );
        let dim = raw_axes.len();
        Self {
            raw_axes,
            raw_apertures,
            dim,
        }
    }

    /// Initialize from a vector embedding with a given initial aperture.
    ///
    /// Each dimension's axis is initialized from the vector components (mapped
    /// through atanh to get the raw parameter), and all apertures are set to
    /// `init_aperture`.
    #[must_use]
    pub fn from_vector(vector: &[f32], init_aperture: f32) -> Self {
        let pi = std::f32::consts::PI;
        // Invert: axis = tanh(raw) * pi  =>  raw = atanh(axis / pi)
        let raw_axes: Vec<f32> = vector
            .iter()
            .map(|&v| {
                let clamped = (v / pi).clamp(-0.999, 0.999);
                clamped.atanh()
            })
            .collect();
        // Invert: aperture = tanh(2 * raw) * pi/2 + pi/2
        //   =>  (aperture - pi/2) / (pi/2) = tanh(2 * raw)
        //   =>  raw = atanh((aperture - pi/2) / (pi/2)) / 2
        let ratio = ((init_aperture - pi / 2.0) / (pi / 2.0)).clamp(-0.999, 0.999);
        let raw_aper = ratio.atanh() / 2.0;
        let raw_apertures = vec![raw_aper; vector.len()];
        Self::new(raw_axes, raw_apertures)
    }

    /// Compute the actual per-dimension axis angles, each in \(-pi, pi\).
    #[must_use]
    pub fn axes(&self) -> Vec<f32> {
        self.raw_axes
            .iter()
            .map(|&r| r.tanh() * std::f32::consts::PI)
            .collect()
    }

    /// Compute the actual per-dimension apertures, each in \(0, pi\).
    #[must_use]
    pub fn apertures(&self) -> Vec<f32> {
        let pi = std::f32::consts::PI;
        self.raw_apertures
            .iter()
            .map(|&r| (2.0 * r).tanh() * (pi / 2.0) + (pi / 2.0))
            .collect()
    }

    /// Compute the mean aperture across dimensions (convenience).
    #[must_use]
    pub fn mean_aperture(&self) -> f32 {
        let aps = self.apertures();
        aps.iter().sum::<f32>() / aps.len() as f32
    }

    /// Convert to a [`DenseCone`] (for loss computation / inference).
    #[must_use]
    pub(crate) fn to_cone(&self) -> DenseCone {
        DenseCone::new(self.axes(), self.apertures())
    }

    /// Number of learnable scalar parameters: dim (axes) + dim (apertures) = 2 * dim.
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        2 * self.dim
    }

    /// Compute the ConE distance score between this (as query) and another (as entity).
    ///
    /// Lower distance = better containment. Convenience wrapper around `DenseCone`.
    pub fn cone_distance(&self, entity: &Self, cen: f32) -> f32 {
        self.to_cone().cone_distance(&entity.to_cone(), cen)
    }

    /// Compute a containment-like score in \[0, 1\].
    ///
    /// Uses `gamma - distance * modulus` through sigmoid.
    pub fn containment_score(&self, entity: &Self, cen: f32, gamma: f32, modulus: f32) -> f32 {
        self.to_cone()
            .containment_score(&entity.to_cone(), cen, gamma, modulus)
    }

    /// Update cone parameters using AMSGrad optimizer.
    ///
    /// Gradients are passed as two slices:
    /// - `grad_axes` (dim elements): gradient w.r.t. raw_axes
    /// - `grad_apertures` (dim elements): gradient w.r.t. raw_apertures
    pub fn update_amsgrad(
        &mut self,
        grad_axes: &[f32],
        grad_apertures: &[f32],
        state: &mut AMSGradState,
    ) {
        let n = self.num_parameters();
        let mut grads = Vec::with_capacity(n);
        grads.extend_from_slice(&grad_axes[..self.dim]);
        grads.extend_from_slice(&grad_apertures[..self.dim]);

        state.t += 1;
        let t = state.t as f32;

        // Update moments.
        for (i, &g) in grads.iter().enumerate().take(n) {
            state.m[i] = state.beta1 * state.m[i] + (1.0 - state.beta1) * g;
            let v_new = state.beta2 * state.v[i] + (1.0 - state.beta2) * g * g;
            state.v[i] = v_new;
            state.v_hat[i] = state.v_hat[i].max(v_new);
        }

        let bias_correction = 1.0 - state.beta1.powf(t);

        // Update raw_axes.
        for i in 0..self.dim {
            let m_hat = state.m[i] / bias_correction;
            let update = state.lr * m_hat / (state.v_hat[i].sqrt() + state.epsilon);
            self.raw_axes[i] -= update;
            self.raw_axes[i] = self.raw_axes[i].clamp(-6.0, 6.0);
            if !self.raw_axes[i].is_finite() {
                self.raw_axes[i] = 0.0;
            }
        }

        // Update raw_apertures.
        for i in 0..self.dim {
            let idx = self.dim + i;
            let m_hat = state.m[idx] / bias_correction;
            let update = state.lr * m_hat / (state.v_hat[idx].sqrt() + state.epsilon);
            self.raw_apertures[i] -= update;
            self.raw_apertures[i] = self.raw_apertures[i].clamp(-6.0, 6.0);
            if !self.raw_apertures[i].is_finite() {
                self.raw_apertures[i] = 0.0;
            }
        }
    }
}

/// Stable sigmoid for f32.
#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- TrainableCone tests --

    #[test]
    fn trainable_cone_apertures_in_valid_range() {
        // Within [-3, 3] (well within trainable range), apertures are strictly in (0, pi).
        // Beyond ~|4|, tanh(2*x) saturates to +/-1.0 at f32 precision, hitting exact 0 or pi.
        for raw_a in [-3.0, -1.0, 0.0, 1.0, 3.0] {
            let cone = TrainableCone::new(vec![0.0, 0.0], vec![raw_a, raw_a]);
            let aps = cone.apertures();
            for (i, &a) in aps.iter().enumerate() {
                assert!(
                    a > 0.0 && a < std::f32::consts::PI,
                    "aperture[{i}] must be in (0, pi), got {a} for raw_aperture={raw_a}",
                );
            }
        }
        // At f32 saturation, apertures land on exact boundaries [0, pi].
        // This is fine -- training clamps raw_apertures to [-6, 6].
        for raw_a in [-10.0, 10.0] {
            let cone = TrainableCone::new(vec![0.0], vec![raw_a]);
            let a = cone.apertures()[0];
            assert!(a >= 0.0 && a <= std::f32::consts::PI);
        }
    }

    #[test]
    fn trainable_cone_axes_in_valid_range() {
        for raw_a in [-10.0, -1.0, 0.0, 1.0, 10.0] {
            let cone = TrainableCone::new(vec![raw_a, raw_a], vec![0.0, 0.0]);
            let axes = cone.axes();
            for (i, &a) in axes.iter().enumerate() {
                assert!(
                    a >= -std::f32::consts::PI && a <= std::f32::consts::PI,
                    "axes[{i}] must be in [-pi, pi], got {a} for raw_axis={raw_a}",
                );
            }
        }
    }

    #[test]
    fn trainable_cone_from_vector_roundtrip() {
        let init_aperture = 1.0_f32;
        let cone = TrainableCone::from_vector(&[1.0, 0.0, -0.5], init_aperture);
        let aps = cone.apertures();
        for &a in &aps {
            assert!(
                (a - init_aperture).abs() < 0.05,
                "aperture should roundtrip, expected {init_aperture} got {a}",
            );
        }
    }

    #[test]
    fn trainable_cone_to_dense_cone() {
        // raw_axes = [0, 0] -> axes = [tanh(0)*pi, tanh(0)*pi] = [0, 0]
        // raw_apertures = [0, 0] -> apertures = [tanh(0)*pi/2 + pi/2, ...] = [pi/2, pi/2]
        let cone = TrainableCone::new(vec![0.0, 0.0], vec![0.0, 0.0]);
        let dense = cone.to_cone();
        for &a in &dense.axes {
            assert!(a.abs() < 1e-6, "axis should be 0, got {a}");
        }
        for &a in &dense.apertures {
            assert!(
                (a - std::f32::consts::FRAC_PI_2).abs() < 1e-6,
                "aperture should be pi/2, got {a}"
            );
        }
    }

    #[test]
    fn dense_cone_distance_wide_contains_narrow() {
        // Wide cone (large apertures) should have low distance to narrow entity with same axes.
        let wide = DenseCone::new(vec![0.5, 0.5], vec![2.5, 2.5]);
        let narrow = DenseCone::new(vec![0.5, 0.5], vec![0.3, 0.3]);
        let d = wide.cone_distance(&narrow, 0.02);
        assert!(
            d < 0.1,
            "wide cone should have low distance to narrow entity, got {d}"
        );
    }

    #[test]
    fn dense_cone_distance_far_entity_has_high_distance() {
        let query = DenseCone::new(vec![0.0, 0.0], vec![0.3, 0.3]);
        let near = DenseCone::new(vec![0.1, 0.1], vec![0.1, 0.1]);
        let far = DenseCone::new(vec![3.0, 3.0], vec![0.1, 0.1]);

        let d_near = query.cone_distance(&near, 0.02);
        let d_far = query.cone_distance(&far, 0.02);

        assert!(
            d_far > d_near,
            "far entity should have higher distance: near={d_near}, far={d_far}"
        );
    }

    #[test]
    fn trainable_cone_update_amsgrad_does_not_panic() {
        let mut cone = TrainableCone::new(vec![0.0, 0.0], vec![0.0, 0.0]);
        let mut state = AMSGradState::new(cone.num_parameters(), 0.01);
        let grad_axes = vec![0.1, -0.1];
        let grad_apertures = vec![0.05, 0.05];
        cone.update_amsgrad(&grad_axes, &grad_apertures, &mut state);
        assert!(cone.raw_axes.iter().all(|x| x.is_finite()));
        assert!(cone.raw_apertures.iter().all(|x| x.is_finite()));
    }

    // -- TrainableBox bridge tests --

    #[cfg(feature = "ndarray-backend")]
    #[test]
    fn trainable_box_to_ndarray_box_roundtrip() {
        use crate::Box as BoxTrait;

        let tb = TrainableBox::new(vec![1.0, 2.0, 3.0], vec![0.0, 0.5, -0.5]);
        let nb = tb.to_ndarray_box().unwrap();

        // Verify dimensions match
        assert_eq!(nb.dim(), 3);

        // Verify coordinates: min = mu - exp(delta)/2, max = mu + exp(delta)/2
        let dense = tb.to_box();
        let volume = nb.volume(1.0).unwrap();
        let expected_vol: f32 = dense
            .min
            .iter()
            .zip(dense.max.iter())
            .map(|(&a, &b)| b - a)
            .product();
        assert!(
            (volume - expected_vol).abs() < 1e-5,
            "volume mismatch: got {volume}, expected {expected_vol}"
        );
    }
}
