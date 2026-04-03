//! Density matrix EL++ training losses and training loop.
//!
//! Implements EL++ ontology embedding using density matrices (pure-state quantum
//! embeddings) instead of boxes. Each concept is a rank-1 density matrix
//! `rho = |v><v| / ||v||^2`; subsumption is modeled via fidelity (trace overlap).
//!
//! # Normal form losses
//!
//! | NF | Axiom | Loss |
//! |----|-------|------|
//! | NF1 | C1 ⊓ C2 ⊑ D | Fidelity of (normalized product state) with D |
//! | NF2 | C ⊑ D | `max(0, margin - F(C, D))` |
//! | NF3 | C ⊑ ∃r.D | Fidelity of C with role-transformed D |
//! | NF4 | ∃r.C ⊑ D | Fidelity of role-transformed C with D |
//! | DISJ | C ⊓ D ⊑ ⊥ | `F(C, D)^2` (push toward orthogonality) |
//!
//! Roles are represented as real-valued linear maps on the interleaved
//! complex parameter space (a `2d x 2d` matrix stored as flat `Vec<f32>`).

use crate::density::{fidelity, DensityRegion};
use crate::el_training::{Axiom, Ontology};
use crate::optimizer::{get_learning_rate, AMSGradState};
use rand::Rng;
use rand::SeedableRng;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for density matrix EL++ training.
#[derive(Debug, Clone)]
pub struct DensityElConfig {
    /// Hilbert space dimension (number of complex components per concept).
    pub dim: usize,
    /// Subsumption margin: fidelity must exceed this for zero loss.
    pub margin: f32,
    /// Learning rate.
    pub lr: f32,
    /// Number of training epochs.
    pub epochs: usize,
    /// Number of negative samples per positive axiom.
    pub negative_samples: usize,
    /// Weight for negative sampling loss.
    pub neg_weight: f32,
    /// Weight for disjointness loss.
    pub disj_weight: f32,
    /// Weight for existential (NF3/NF4) loss.
    pub existential_weight: f32,
    /// Warmup epochs for learning rate.
    pub warmup_epochs: usize,
    /// Random seed.
    pub seed: u64,
    /// Log interval (0 = no logging).
    pub log_interval: usize,
}

impl Default for DensityElConfig {
    fn default() -> Self {
        Self {
            dim: 16,
            margin: 0.5,
            lr: 0.01,
            epochs: 200,
            negative_samples: 2,
            neg_weight: 1.0,
            disj_weight: 1.0,
            existential_weight: 1.0,
            warmup_epochs: 10,
            seed: 42,
            log_interval: 20,
        }
    }
}

// ---------------------------------------------------------------------------
// Loss functions
// ---------------------------------------------------------------------------

/// NF2 subsumption loss: penalizes when `F(child, parent) < margin`.
///
/// ```text
/// L = max(0, margin - F(child, parent))
/// ```
///
/// Returns 0 when fidelity meets or exceeds the margin.
pub fn nf1_loss_density(child: &DensityRegion, parent: &DensityRegion, margin: f32) -> f32 {
    let f = fidelity(child, parent).unwrap_or(0.0);
    (margin - f).max(0.0)
}

/// Disjointness loss: penalizes non-zero fidelity between concepts that
/// should be orthogonal.
///
/// ```text
/// L = F(a, b)^2
/// ```
pub fn disjointness_loss_density(a: &DensityRegion, b: &DensityRegion) -> f32 {
    let f = fidelity(a, b).unwrap_or(0.0);
    f * f
}

// ---------------------------------------------------------------------------
// Gradient computation
// ---------------------------------------------------------------------------

/// Compute the gradient of fidelity w.r.t. the raw parameters of `a` and `b`.
///
/// Fidelity = |<a|b>|^2 / (||a||^2 * ||b||^2)
///
/// Returns (grad_a, grad_b, fidelity_value).
fn fidelity_grads(a_params: &[f32], b_params: &[f32]) -> (Vec<f32>, Vec<f32>, f32) {
    let n = a_params.len();
    debug_assert_eq!(n, b_params.len());

    // Compute <a|b> = sum conj(a_i)*b_i
    let mut ip_re = 0.0f32;
    let mut ip_im = 0.0f32;
    let mut norm_a_sq = 0.0f32;
    let mut norm_b_sq = 0.0f32;

    for i in (0..n).step_by(2) {
        let a_re = a_params[i];
        let a_im = a_params[i + 1];
        let b_re = b_params[i];
        let b_im = b_params[i + 1];

        ip_re += a_re * b_re + a_im * b_im;
        ip_im += a_re * b_im - a_im * b_re;
        norm_a_sq += a_re * a_re + a_im * a_im;
        norm_b_sq += b_re * b_re + b_im * b_im;
    }

    let ip_sq = ip_re * ip_re + ip_im * ip_im;
    let denom = norm_a_sq * norm_b_sq;
    if denom < 1e-12 {
        return (vec![0.0; n], vec![0.0; n], 0.0);
    }
    let f = ip_sq / denom;

    // dF/d(a_params[k]):
    // F = |<a|b>|^2 / (||a||^2 * ||b||^2)
    // Let S = |<a|b>|^2, Na = ||a||^2, Nb = ||b||^2
    // dF/da_k = (dS/da_k * Na - S * dNa/da_k) / (Na^2 * Nb)
    //
    // dS/da_k = 2 * (ip_re * d(ip_re)/da_k + ip_im * d(ip_im)/da_k)
    // d(ip_re)/d(a_re_j) = b_re_j,  d(ip_re)/d(a_im_j) = b_im_j
    // d(ip_im)/d(a_re_j) = b_im_j,  d(ip_im)/d(a_im_j) = -b_re_j
    // dNa/d(a_re_j) = 2*a_re_j,     dNa/d(a_im_j) = 2*a_im_j

    let mut grad_a = vec![0.0f32; n];
    let mut grad_b = vec![0.0f32; n];
    let inv_denom = 1.0 / denom;

    for i in (0..n).step_by(2) {
        let a_re = a_params[i];
        let a_im = a_params[i + 1];
        let b_re = b_params[i];
        let b_im = b_params[i + 1];

        // dS/d(a_re)
        let ds_da_re = 2.0 * (ip_re * b_re + ip_im * b_im);
        // dS/d(a_im)
        let ds_da_im = 2.0 * (ip_re * b_im - ip_im * b_re);

        grad_a[i] = ds_da_re * inv_denom - f * 2.0 * a_re / norm_a_sq;
        grad_a[i + 1] = ds_da_im * inv_denom - f * 2.0 * a_im / norm_a_sq;

        // Symmetric for b: conj goes on a, so inner product w.r.t. b:
        // d(ip_re)/d(b_re_j) = a_re_j,  d(ip_re)/d(b_im_j) = -a_im_j  (wait: <a|b>=conj(a)*b)
        // Actually: ip_re = sum(a_re*b_re + a_im*b_im)
        // d(ip_re)/d(b_re) = a_re, d(ip_re)/d(b_im) = a_im
        // ip_im = sum(a_re*b_im - a_im*b_re)
        // d(ip_im)/d(b_re) = -a_im, d(ip_im)/d(b_im) = a_re
        let ds_db_re = 2.0 * (ip_re * a_re - ip_im * a_im);
        let ds_db_im = 2.0 * (ip_re * a_im + ip_im * a_re);

        grad_b[i] = ds_db_re * inv_denom - f * 2.0 * b_re / norm_b_sq;
        grad_b[i + 1] = ds_db_im * inv_denom - f * 2.0 * b_im / norm_b_sq;
    }

    (grad_a, grad_b, f)
}

// ---------------------------------------------------------------------------
// Embedding store for density matrix parameters
// ---------------------------------------------------------------------------

struct DensityStore {
    /// Each entry: interleaved [re_0, im_0, re_1, im_1, ...] with length 2*dim.
    params: Vec<Vec<f32>>,
    opts: Vec<AMSGradState>,
}

impl DensityStore {
    fn new(count: usize, dim: usize, lr: f32, rng: &mut impl Rng) -> Self {
        let params: Vec<Vec<f32>> = (0..count)
            .map(|_| {
                (0..2 * dim)
                    .map(|_| rng.random_range(-1.0f32..1.0))
                    .collect()
            })
            .collect();
        let opts = (0..count).map(|_| AMSGradState::new(2 * dim, lr)).collect();
        Self { params, opts }
    }

    fn set_lr(&mut self, lr: f32) {
        for o in &mut self.opts {
            o.set_lr(lr);
        }
    }

    /// Apply gradient to entity `idx`. `grad` has length 2*dim.
    fn apply_grad(&mut self, idx: usize, grad: &[f32], scale: f32) {
        let n = grad.len();
        let opt = &mut self.opts[idx];
        opt.t += 1;
        let t = opt.t as f32;

        for i in 0..n {
            let g = grad[i] * scale;
            opt.m[i] = opt.beta1 * opt.m[i] + (1.0 - opt.beta1) * g;
            let v_new = opt.beta2 * opt.v[i] + (1.0 - opt.beta2) * g * g;
            opt.v[i] = v_new;
            opt.v_hat[i] = opt.v_hat[i].max(v_new);
        }

        let bias_correction = 1.0 - opt.beta1.powf(t);
        let params = &mut self.params[idx];
        for i in 0..n {
            let m_hat = opt.m[i] / bias_correction;
            let update = opt.lr * m_hat / (opt.v_hat[i].sqrt() + opt.epsilon);
            params[i] -= update;
            if !params[i].is_finite() {
                params[i] = 0.01;
            }
        }
    }
}

/// Role maps for density matrix embeddings.
///
/// Each role is a `2d x 2d` real matrix that acts on the interleaved
/// complex parameter vector. Stored as flat row-major `Vec<f32>`.
struct RoleStore {
    /// Each entry: flat `[2d * 2d]` matrix.
    params: Vec<Vec<f32>>,
    opts: Vec<AMSGradState>,
}

impl RoleStore {
    fn new(count: usize, dim: usize, lr: f32, rng: &mut impl Rng) -> Self {
        let n = 2 * dim;
        let params: Vec<Vec<f32>> = (0..count)
            .map(|_| {
                // Initialize near identity: I + small noise
                let mut m = vec![0.0f32; n * n];
                for i in 0..n {
                    m[i * n + i] = 1.0;
                }
                for v in m.iter_mut() {
                    *v += rng.random_range(-0.1f32..0.1);
                }
                m
            })
            .collect();
        let opts = (0..count).map(|_| AMSGradState::new(n * n, lr)).collect();
        Self { params, opts }
    }

    fn set_lr(&mut self, lr: f32) {
        for o in &mut self.opts {
            o.set_lr(lr);
        }
    }

    /// Apply role transform to a concept vector: out = M * v
    fn transform(&self, role: usize, v: &[f32]) -> Vec<f32> {
        let n = v.len();
        let m = &self.params[role];
        let mut out = vec![0.0f32; n];
        for i in 0..n {
            let mut sum = 0.0f32;
            for j in 0..n {
                sum += m[i * n + j] * v[j];
            }
            out[i] = sum;
        }
        out
    }

    /// Apply gradient to role `idx`.
    fn apply_grad(&mut self, idx: usize, grad: &[f32], scale: f32) {
        let n = grad.len();
        let opt = &mut self.opts[idx];
        opt.t += 1;
        let t = opt.t as f32;

        for i in 0..n {
            let g = grad[i] * scale;
            opt.m[i] = opt.beta1 * opt.m[i] + (1.0 - opt.beta1) * g;
            let v_new = opt.beta2 * opt.v[i] + (1.0 - opt.beta2) * g * g;
            opt.v[i] = v_new;
            opt.v_hat[i] = opt.v_hat[i].max(v_new);
        }

        let bias_correction = 1.0 - opt.beta1.powf(t);
        let params = &mut self.params[idx];
        for i in 0..n {
            let m_hat = opt.m[i] / bias_correction;
            let update = opt.lr * m_hat / (opt.v_hat[i].sqrt() + opt.epsilon);
            params[i] -= update;
            if !params[i].is_finite() {
                params[i] = 0.0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Training result
// ---------------------------------------------------------------------------

/// Result of density matrix EL++ training.
#[derive(Debug, Clone)]
pub struct DensityElResult {
    /// Concept embeddings: interleaved complex params. Shape: `[num_concepts][2*dim]`.
    pub concept_params: Vec<Vec<f32>>,
    /// Role matrices: flat row-major `[2d x 2d]`. Shape: `[num_roles][4*dim*dim]`.
    pub role_params: Vec<Vec<f32>>,
    /// Hilbert space dimension.
    pub dim: usize,
    /// Per-epoch average loss.
    pub epoch_losses: Vec<f32>,
}

impl DensityElResult {
    /// Get a `DensityRegion` for concept `idx`.
    pub fn concept(&self, idx: usize) -> DensityRegion {
        DensityRegion::new(self.concept_params[idx].clone(), self.dim).unwrap_or_else(|_| {
            let mut p = vec![0.0f32; 2 * self.dim];
            p[0] = 1.0;
            DensityRegion::new(p, self.dim).unwrap()
        })
    }

    /// Compute fidelity between two concepts.
    pub fn concept_fidelity(&self, a: usize, b: usize) -> f32 {
        let da = self.concept(a);
        let db = self.concept(b);
        fidelity(&da, &db).unwrap_or(0.0)
    }
}

// ---------------------------------------------------------------------------
// Training loop
// ---------------------------------------------------------------------------

/// Train density matrix EL++ embeddings on an ontology.
pub fn train_density_el(ontology: &Ontology, config: &DensityElConfig) -> DensityElResult {
    let dim = config.dim;
    let nc = ontology.num_concepts().max(1);
    let nr = ontology.num_roles().max(1);
    let n = 2 * dim; // parameter vector length

    let mut rng = rand::rngs::SmallRng::seed_from_u64(config.seed);
    let mut concepts = DensityStore::new(nc, dim, config.lr, &mut rng);
    let mut roles = RoleStore::new(nr, dim, config.lr, &mut rng);

    let mut epoch_losses = Vec::with_capacity(config.epochs);
    let mut axiom_indices: Vec<usize> = (0..ontology.axioms.len()).collect();

    for epoch in 0..config.epochs {
        let lr = get_learning_rate(epoch, config.epochs, config.lr, config.warmup_epochs);
        concepts.set_lr(lr);
        roles.set_lr(lr);

        // Shuffle axioms
        for i in (1..axiom_indices.len()).rev() {
            let j = rng.random_range(0..=i);
            axiom_indices.swap(i, j);
        }

        let mut total_loss = 0.0f32;
        let mut count = 0usize;

        for &ax_idx in &axiom_indices {
            let axiom = &ontology.axioms[ax_idx];
            match *axiom {
                Axiom::SubClassOf { sub, sup } => {
                    // NF2: C ⊑ D -- maximize fidelity(sub, sup)
                    let (ga, gb, f) = fidelity_grads(&concepts.params[sub], &concepts.params[sup]);
                    let deficit = (config.margin - f).max(0.0);
                    total_loss += deficit;

                    if deficit > 0.0 {
                        // Gradient ascent on fidelity: negate the gradient
                        concepts.apply_grad(sub, &ga, -1.0);
                        if sup != sub {
                            concepts.apply_grad(sup, &gb, -1.0);
                        }
                    }

                    // Negative samples: push random concepts apart from sup
                    for _ in 0..config.negative_samples {
                        let neg = rng.random_range(0..nc);
                        if neg == sub || neg == sup {
                            continue;
                        }
                        let (gn, gs, f_neg) =
                            fidelity_grads(&concepts.params[neg], &concepts.params[sup]);
                        // Disjointness loss: f_neg^2
                        let neg_loss = f_neg * f_neg;
                        total_loss += config.neg_weight * neg_loss;

                        // Gradient descent on f_neg^2: d/dp(f^2) = 2f * df/dp
                        let scale = 2.0 * f_neg * config.neg_weight;
                        concepts.apply_grad(neg, &gn, scale);
                        if sup != neg {
                            concepts.apply_grad(sup, &gs, scale);
                        }
                    }
                }
                Axiom::Disjoint { a, b } => {
                    // C ⊓ D ⊑ ⊥: push fidelity to 0
                    let (ga, gb, f) = fidelity_grads(&concepts.params[a], &concepts.params[b]);
                    let loss = f * f;
                    total_loss += config.disj_weight * loss;

                    let scale = 2.0 * f * config.disj_weight;
                    concepts.apply_grad(a, &ga, scale);
                    if b != a {
                        concepts.apply_grad(b, &gb, scale);
                    }
                }
                Axiom::Existential {
                    role,
                    filler,
                    target,
                } => {
                    // NF4: ∃r.C ⊑ D
                    // Transform filler by role, then check fidelity with target
                    let transformed = roles.transform(role, &concepts.params[filler]);
                    let (g_tf, g_tgt, f) = fidelity_grads(&transformed, &concepts.params[target]);
                    let deficit = (config.margin - f).max(0.0);
                    total_loss += config.existential_weight * deficit;

                    if deficit > 0.0 {
                        let scale = -config.existential_weight;

                        // Gradient for target (direct)
                        concepts.apply_grad(target, &g_tgt, scale);

                        // Chain rule through role transform for filler
                        // transformed = M * v_filler
                        // d(loss)/d(v_filler) = M^T * d(loss)/d(transformed)
                        let m = &roles.params[role];
                        let mut g_filler = vec![0.0f32; n];
                        for i in 0..n {
                            let mut sum = 0.0f32;
                            for j in 0..n {
                                sum += m[j * n + i] * g_tf[j]; // M^T
                            }
                            g_filler[i] = sum;
                        }
                        concepts.apply_grad(filler, &g_filler, scale);

                        // d(loss)/d(M[i][j]) = d(loss)/d(transformed[i]) * v_filler[j]
                        let mut g_role = vec![0.0f32; n * n];
                        for i in 0..n {
                            for j in 0..n {
                                g_role[i * n + j] = g_tf[i] * concepts.params[filler][j];
                            }
                        }
                        roles.apply_grad(role, &g_role, scale);
                    }
                }
                Axiom::ExistentialRight { sub, role, filler } => {
                    // NF3: C ⊑ ∃r.D
                    // sub should have high fidelity with role-transformed filler
                    let transformed = roles.transform(role, &concepts.params[filler]);
                    let (g_sub_v, g_tf, f) = fidelity_grads(&concepts.params[sub], &transformed);
                    let deficit = (config.margin - f).max(0.0);
                    total_loss += config.existential_weight * deficit;

                    if deficit > 0.0 {
                        let scale = -config.existential_weight;

                        concepts.apply_grad(sub, &g_sub_v, scale);

                        // Chain rule through role transform
                        let m = &roles.params[role];
                        let mut g_filler = vec![0.0f32; n];
                        for i in 0..n {
                            let mut sum = 0.0f32;
                            for j in 0..n {
                                sum += m[j * n + i] * g_tf[j];
                            }
                            g_filler[i] = sum;
                        }
                        concepts.apply_grad(filler, &g_filler, scale);

                        let mut g_role = vec![0.0f32; n * n];
                        for i in 0..n {
                            for j in 0..n {
                                g_role[i * n + j] = g_tf[i] * concepts.params[filler][j];
                            }
                        }
                        roles.apply_grad(role, &g_role, scale);
                    }
                }
                Axiom::Intersection { c1, c2, target } => {
                    // NF1: C1 ⊓ C2 ⊑ D
                    // Intersection as element-wise product of complex vectors (then normalize).
                    // Product state: psi_inter[i] = a[i] * b[i] (complex multiply)
                    let a = concepts.params[c1].clone();
                    let b = concepts.params[c2].clone();
                    let mut inter = vec![0.0f32; n];
                    for i in (0..n).step_by(2) {
                        inter[i] = a[i] * b[i] - a[i + 1] * b[i + 1];
                        inter[i + 1] = a[i] * b[i + 1] + a[i + 1] * b[i];
                    }

                    // Check if product is non-zero
                    let inter_norm_sq: f32 = inter.iter().map(|x| x * x).sum();
                    if inter_norm_sq < 1e-12 {
                        count += 1;
                        continue;
                    }

                    let target_params = concepts.params[target].clone();
                    let (g_inter, g_tgt, f) = fidelity_grads(&inter, &target_params);
                    let deficit = (config.margin - f).max(0.0);
                    total_loss += deficit;

                    if deficit > 0.0 {
                        concepts.apply_grad(target, &g_tgt, -1.0);

                        // Chain rule through element-wise complex product
                        // inter[2k] = a[2k]*b[2k] - a[2k+1]*b[2k+1]
                        // inter[2k+1] = a[2k]*b[2k+1] + a[2k+1]*b[2k]
                        let mut g_c1 = vec![0.0f32; n];
                        let mut g_c2 = vec![0.0f32; n];
                        for i in (0..n).step_by(2) {
                            let gi_re = g_inter[i];
                            let gi_im = g_inter[i + 1];
                            g_c1[i] = gi_re * b[i] + gi_im * b[i + 1];
                            g_c1[i + 1] = -gi_re * b[i + 1] + gi_im * b[i];
                            g_c2[i] = gi_re * a[i] + gi_im * a[i + 1];
                            g_c2[i + 1] = -gi_re * a[i + 1] + gi_im * a[i];
                        }
                        concepts.apply_grad(c1, &g_c1, -1.0);
                        if c2 != c1 {
                            concepts.apply_grad(c2, &g_c2, -1.0);
                        }
                    }
                }
                Axiom::RoleInclusion { sub, sup } => {
                    // r ⊑ s: for every concept, applying r should give similar result to s.
                    // Approximate: Frobenius norm of (M_sub - M_sup)
                    let m_sub = &roles.params[sub];
                    let m_sup = &roles.params[sup];
                    let mut loss = 0.0f32;
                    let mut grad = vec![0.0f32; n * n];
                    for i in 0..m_sub.len() {
                        let diff = m_sub[i] - m_sup[i];
                        loss += diff * diff;
                        grad[i] = 2.0 * diff;
                    }
                    loss = loss.sqrt();
                    total_loss += loss;

                    if loss > 1e-8 {
                        let inv = 1.0 / (2.0 * loss);
                        for g in grad.iter_mut() {
                            *g *= inv;
                        }
                        roles.apply_grad(sub, &grad, 1.0);
                        // Negate for sup
                        for g in grad.iter_mut() {
                            *g = -*g;
                        }
                        if sup != sub {
                            roles.apply_grad(sup, &grad, 1.0);
                        }
                    }
                }
                Axiom::RoleComposition { r, s, t } => {
                    // r ∘ s ⊑ t: M_r * M_s should approximate M_t
                    // Frobenius norm of (M_r * M_s - M_t)
                    let m_r = &roles.params[r];
                    let m_s = &roles.params[s];
                    let m_t = &roles.params[t];

                    // Compute M_r * M_s
                    let mut composed = vec![0.0f32; n * n];
                    for i in 0..n {
                        for j in 0..n {
                            let mut sum = 0.0f32;
                            for k in 0..n {
                                sum += m_r[i * n + k] * m_s[k * n + j];
                            }
                            composed[i * n + j] = sum;
                        }
                    }

                    let mut loss = 0.0f32;
                    let mut diff_mat = vec![0.0f32; n * n];
                    for i in 0..composed.len() {
                        let d = composed[i] - m_t[i];
                        diff_mat[i] = d;
                        loss += d * d;
                    }
                    loss = loss.sqrt();
                    total_loss += loss;

                    if loss > 1e-8 {
                        let inv = 1.0 / loss;
                        // d(||C - T||)/d(C[i][j]) = (C-T)[i][j] / ||C-T||
                        // d(C)/d(M_r[i][k]) = M_s[k][j] (summed over j in chain)
                        let mut g_r = vec![0.0f32; n * n];
                        let mut g_s = vec![0.0f32; n * n];
                        let mut g_t = vec![0.0f32; n * n];

                        for i in 0..n {
                            for j in 0..n {
                                let dc = diff_mat[i * n + j] * inv;
                                g_t[i * n + j] = -dc;
                                for k in 0..n {
                                    g_r[i * n + k] += dc * m_s[k * n + j];
                                    g_s[k * n + j] += dc * m_r[i * n + k];
                                }
                            }
                        }

                        roles.apply_grad(r, &g_r, 1.0);
                        if s != r {
                            roles.apply_grad(s, &g_s, 1.0);
                        }
                        if t != r && t != s {
                            roles.apply_grad(t, &g_t, 1.0);
                        }
                    }
                }
            }
            count += 1;
        }

        let avg_loss = if count > 0 {
            total_loss / count as f32
        } else {
            0.0
        };
        epoch_losses.push(avg_loss);

        if config.log_interval > 0 && (epoch + 1) % config.log_interval == 0 {
            eprintln!(
                "epoch {}/{}: avg_loss = {avg_loss:.6}, lr = {lr:.6}",
                epoch + 1,
                config.epochs
            );
        }
    }

    DensityElResult {
        concept_params: concepts.params,
        role_params: roles.params,
        dim,
        epoch_losses,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::density::DensityRegion;
    use crate::el_training::Ontology;

    fn make_real(v: &[f32]) -> DensityRegion {
        let mut params = Vec::with_capacity(v.len() * 2);
        for &x in v {
            params.push(x);
            params.push(0.0);
        }
        DensityRegion::new(params, v.len()).unwrap()
    }

    #[test]
    fn nf1_loss_zero_when_fidelity_exceeds_margin() {
        let a = make_real(&[1.0, 0.0, 0.0]);
        // Same state: fidelity = 1.0, margin = 0.5 -> loss = 0
        let loss = nf1_loss_density(&a, &a, 0.5);
        assert!(loss.abs() < 1e-6, "expected 0, got {loss}");
    }

    #[test]
    fn nf1_loss_positive_for_orthogonal() {
        let a = make_real(&[1.0, 0.0]);
        let b = make_real(&[0.0, 1.0]);
        let loss = nf1_loss_density(&a, &b, 0.5);
        assert!((loss - 0.5).abs() < 1e-6, "expected 0.5, got {loss}");
    }

    #[test]
    fn disjointness_loss_zero_for_orthogonal() {
        let a = make_real(&[1.0, 0.0, 0.0]);
        let b = make_real(&[0.0, 1.0, 0.0]);
        let loss = disjointness_loss_density(&a, &b);
        assert!(loss.abs() < 1e-6, "expected 0, got {loss}");
    }

    #[test]
    fn disjointness_loss_one_for_identical() {
        let a = make_real(&[1.0, 0.0]);
        let loss = disjointness_loss_density(&a, &a);
        assert!((loss - 1.0).abs() < 1e-6, "expected 1, got {loss}");
    }

    #[test]
    fn training_reduces_loss() {
        let mut ont = Ontology::new();
        let animal = ont.concept("Animal");
        let dog = ont.concept("Dog");
        let cat = ont.concept("Cat");
        let mammal = ont.concept("Mammal");
        let fish = ont.concept("Fish");

        ont.axioms.push(Axiom::SubClassOf {
            sub: dog,
            sup: animal,
        });
        ont.axioms.push(Axiom::SubClassOf {
            sub: cat,
            sup: animal,
        });
        ont.axioms.push(Axiom::SubClassOf {
            sub: dog,
            sup: mammal,
        });
        ont.axioms.push(Axiom::SubClassOf {
            sub: cat,
            sup: mammal,
        });
        ont.axioms.push(Axiom::Disjoint { a: dog, b: cat });
        ont.axioms.push(Axiom::Disjoint { a: dog, b: fish });
        ont.axioms.push(Axiom::Disjoint { a: cat, b: fish });

        let config = DensityElConfig {
            dim: 8,
            margin: 0.3,
            lr: 0.05,
            epochs: 100,
            negative_samples: 2,
            neg_weight: 0.5,
            disj_weight: 1.0,
            existential_weight: 1.0,
            warmup_epochs: 5,
            seed: 42,
            log_interval: 0,
        };

        let result = train_density_el(&ont, &config);

        // Loss should decrease
        let first_loss = result.epoch_losses[0];
        let last_loss = *result.epoch_losses.last().unwrap();
        assert!(
            last_loss < first_loss,
            "loss should decrease: first={first_loss}, last={last_loss}"
        );

        // Fidelity for parent-child should be higher than disjoint pairs
        let f_dog_animal = result.concept_fidelity(dog, animal);
        let f_dog_cat = result.concept_fidelity(dog, cat);
        assert!(
            f_dog_animal > f_dog_cat,
            "F(Dog,Animal)={f_dog_animal} should > F(Dog,Cat)={f_dog_cat}"
        );
    }

    #[test]
    fn fidelity_grads_finite() {
        let a = vec![1.0f32, 0.0, 0.5, 0.5];
        let b = vec![0.5, -0.5, 1.0, 0.0];
        let (ga, gb, f) = fidelity_grads(&a, &b);
        assert!(f.is_finite());
        assert!(ga.iter().all(|x| x.is_finite()));
        assert!(gb.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn fidelity_grads_self_are_zero() {
        // For identical vectors, fidelity = 1.0
        // Gradient of fidelity at maximum should be ~0
        let a = vec![1.0f32, 0.0, 0.0, 0.0];
        let (ga, gb, f) = fidelity_grads(&a, &a);
        assert!((f - 1.0).abs() < 1e-5, "f={f}");
        // At maximum, gradient should vanish
        let ga_norm: f32 = ga.iter().map(|x| x * x).sum();
        assert!(ga_norm < 1e-4, "grad_a should be ~0, norm={ga_norm}");
        let gb_norm: f32 = gb.iter().map(|x| x * x).sum();
        assert!(gb_norm < 1e-4, "grad_b should be ~0, norm={gb_norm}");
    }

    #[test]
    fn training_with_existentials() {
        let mut ont = Ontology::new();
        let human = ont.concept("Human");
        let animal = ont.concept("Animal");
        let role = ont.role("hasParent");

        // NF4: ∃hasParent.Human ⊑ Animal
        ont.axioms.push(Axiom::Existential {
            role,
            filler: human,
            target: animal,
        });
        // NF2: Human ⊑ Animal
        ont.axioms.push(Axiom::SubClassOf {
            sub: human,
            sup: animal,
        });

        let config = DensityElConfig {
            dim: 8,
            epochs: 50,
            log_interval: 0,
            ..DensityElConfig::default()
        };

        let result = train_density_el(&ont, &config);
        let first = result.epoch_losses[0];
        let last = *result.epoch_losses.last().unwrap();
        assert!(
            last <= first + 0.01,
            "loss should not increase significantly: first={first}, last={last}"
        );
    }
}
