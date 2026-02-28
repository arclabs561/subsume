//! End-to-end EL++ ontology embedding training.
//!
//! Loads an ontology (concept/role axioms), initializes box embeddings in
//! center/offset representation, and trains them using the geometric losses
//! from [`crate::el`].
//!
//! # Axiom format
//!
//! One axiom per line, whitespace-separated:
//!
//! | Axiom | Syntax | EL++ normal form |
//! |-------|--------|------------------|
//! | Subsumption | `SubClassOf C D` | NF2: C ⊑ D |
//! | Disjointness | `Disjoint C D` | C ⊓ D ⊑ ⊥ |
//! | Existential restriction | `Existential R C D` | NF4: ∃R.C ⊑ D |
//! | Role inclusion | `RoleInclusion R S` | RI6: R ⊑ S |
//! | Role composition | `RoleComposition R S T` | RI7: R ∘ S ⊑ T |
//!
//! Lines starting with `#` or empty lines are ignored.
//!
//! # Example
//!
//! ```text
//! SubClassOf Dog Animal
//! SubClassOf Cat Animal
//! Disjoint Dog Cat
//! Existential hasParent Animal Animal
//! ```

use crate::el;
use crate::optimizer::{get_learning_rate, AMSGradState};
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use std::collections::HashMap;
use std::io::BufRead;

// ---------------------------------------------------------------------------
// Ontology representation
// ---------------------------------------------------------------------------

/// A parsed EL++ axiom.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Axiom {
    /// NF2: C ⊑ D
    SubClassOf {
        /// Subclass concept index.
        sub: usize,
        /// Superclass concept index.
        sup: usize,
    },
    /// C ⊓ D ⊑ ⊥ (disjointness)
    Disjoint {
        /// First concept index.
        a: usize,
        /// Second concept index.
        b: usize,
    },
    /// NF4: ∃R.C ⊑ D
    Existential {
        /// Role index.
        role: usize,
        /// Filler concept index.
        filler: usize,
        /// Target concept index.
        target: usize,
    },
    /// RI6: R ⊑ S
    RoleInclusion {
        /// Sub-role index.
        sub: usize,
        /// Super-role index.
        sup: usize,
    },
    /// RI7: R ∘ S ⊑ T
    RoleComposition {
        /// First role index.
        r: usize,
        /// Second role index.
        s: usize,
        /// Target role index.
        t: usize,
    },
}

/// An EL++ ontology: named concepts, roles, and axioms.
#[derive(Debug, Clone)]
pub struct Ontology {
    /// Concept name -> index.
    pub concept_index: HashMap<String, usize>,
    /// Index -> concept name.
    pub concept_names: Vec<String>,
    /// Role name -> index.
    pub role_index: HashMap<String, usize>,
    /// Index -> role name.
    pub role_names: Vec<String>,
    /// Axioms.
    pub axioms: Vec<Axiom>,
}

impl Ontology {
    /// Create an empty ontology.
    pub fn new() -> Self {
        Self {
            concept_index: HashMap::new(),
            concept_names: Vec::new(),
            role_index: HashMap::new(),
            role_names: Vec::new(),
            axioms: Vec::new(),
        }
    }

    /// Get or create a concept index for the given name.
    pub fn concept(&mut self, name: &str) -> usize {
        if let Some(&idx) = self.concept_index.get(name) {
            idx
        } else {
            let idx = self.concept_names.len();
            self.concept_names.push(name.to_string());
            self.concept_index.insert(name.to_string(), idx);
            idx
        }
    }

    /// Get or create a role index for the given name.
    pub fn role(&mut self, name: &str) -> usize {
        if let Some(&idx) = self.role_index.get(name) {
            idx
        } else {
            let idx = self.role_names.len();
            self.role_names.push(name.to_string());
            self.role_index.insert(name.to_string(), idx);
            idx
        }
    }

    /// Number of concepts.
    pub fn num_concepts(&self) -> usize {
        self.concept_names.len()
    }

    /// Number of roles.
    pub fn num_roles(&self) -> usize {
        self.role_names.len()
    }

    /// Parse axioms from a reader (one axiom per line).
    ///
    /// See module-level docs for the format.
    pub fn parse<R: BufRead>(reader: R) -> Result<Self, String> {
        let mut ont = Self::new();
        for (line_num, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| format!("line {}: {e}", line_num + 1))?;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }
            let axiom = match parts[0] {
                "SubClassOf" => {
                    if parts.len() != 3 {
                        return Err(format!(
                            "line {}: SubClassOf expects 2 arguments, got {}",
                            line_num + 1,
                            parts.len() - 1
                        ));
                    }
                    let sub = ont.concept(parts[1]);
                    let sup = ont.concept(parts[2]);
                    Axiom::SubClassOf { sub, sup }
                }
                "Disjoint" => {
                    if parts.len() != 3 {
                        return Err(format!(
                            "line {}: Disjoint expects 2 arguments, got {}",
                            line_num + 1,
                            parts.len() - 1
                        ));
                    }
                    let a = ont.concept(parts[1]);
                    let b = ont.concept(parts[2]);
                    Axiom::Disjoint { a, b }
                }
                "Existential" => {
                    if parts.len() != 4 {
                        return Err(format!(
                            "line {}: Existential expects 3 arguments, got {}",
                            line_num + 1,
                            parts.len() - 1
                        ));
                    }
                    let role = ont.role(parts[1]);
                    let filler = ont.concept(parts[2]);
                    let target = ont.concept(parts[3]);
                    Axiom::Existential {
                        role,
                        filler,
                        target,
                    }
                }
                "RoleInclusion" => {
                    if parts.len() != 3 {
                        return Err(format!(
                            "line {}: RoleInclusion expects 2 arguments, got {}",
                            line_num + 1,
                            parts.len() - 1
                        ));
                    }
                    let sub = ont.role(parts[1]);
                    let sup = ont.role(parts[2]);
                    Axiom::RoleInclusion { sub, sup }
                }
                "RoleComposition" => {
                    if parts.len() != 4 {
                        return Err(format!(
                            "line {}: RoleComposition expects 3 arguments, got {}",
                            line_num + 1,
                            parts.len() - 1
                        ));
                    }
                    let r = ont.role(parts[1]);
                    let s = ont.role(parts[2]);
                    let t = ont.role(parts[3]);
                    Axiom::RoleComposition { r, s, t }
                }
                other => {
                    return Err(format!(
                        "line {}: unknown axiom type: {other}",
                        line_num + 1
                    ));
                }
            };
            ont.axioms.push(axiom);
        }
        Ok(ont)
    }
}

impl Default for Ontology {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Training configuration
// ---------------------------------------------------------------------------

/// Configuration for EL++ ontology embedding training.
#[derive(Debug, Clone)]
pub struct ElTrainingConfig {
    /// Embedding dimension.
    pub dim: usize,
    /// Learning rate.
    pub learning_rate: f32,
    /// Number of training epochs.
    pub epochs: usize,
    /// Margin for inclusion/disjointness losses.
    pub margin: f32,
    /// Number of negative subsumption samples per positive axiom.
    pub negative_samples: usize,
    /// Weight for negative (non-subsumption) loss.
    pub negative_weight: f32,
    /// Weight for disjointness loss.
    pub disjointness_weight: f32,
    /// Weight for existential loss.
    pub existential_weight: f32,
    /// Weight for role inclusion loss.
    pub role_inclusion_weight: f32,
    /// Weight for role composition loss.
    pub role_composition_weight: f32,
    /// Warmup epochs (linear LR warmup).
    pub warmup_epochs: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Log interval (print every N epochs). 0 = no logging.
    pub log_interval: usize,
}

impl Default for ElTrainingConfig {
    fn default() -> Self {
        Self {
            dim: 30,
            learning_rate: 5e-3,
            epochs: 200,
            margin: 0.1,
            negative_samples: 2,
            negative_weight: 1.0,
            disjointness_weight: 1.0,
            existential_weight: 1.0,
            role_inclusion_weight: 1.0,
            role_composition_weight: 1.0,
            warmup_epochs: 10,
            seed: 42,
            log_interval: 20,
        }
    }
}

// ---------------------------------------------------------------------------
// Training result
// ---------------------------------------------------------------------------

/// Result of training EL++ embeddings.
#[derive(Debug, Clone)]
pub struct ElTrainingResult {
    /// Concept embeddings: centers. Shape: `[num_concepts][dim]`.
    pub concept_centers: Vec<Vec<f32>>,
    /// Concept embeddings: offsets. Shape: `[num_concepts][dim]`.
    pub concept_offsets: Vec<Vec<f32>>,
    /// Role embeddings: centers. Shape: `[num_roles][dim]`.
    pub role_centers: Vec<Vec<f32>>,
    /// Role embeddings: offsets. Shape: `[num_roles][dim]`.
    pub role_offsets: Vec<Vec<f32>>,
    /// Per-epoch total loss.
    pub epoch_losses: Vec<f32>,
}

impl ElTrainingResult {
    /// Compute the inclusion loss between two concepts (C ⊑ D).
    ///
    /// Low loss means concept `sub` is geometrically contained in concept `sup`.
    pub fn subsumption_score(&self, sub: usize, sup: usize) -> f32 {
        el::el_inclusion_loss(
            &self.concept_centers[sub],
            &self.concept_offsets[sub],
            &self.concept_centers[sup],
            &self.concept_offsets[sup],
            0.0,
        )
        .unwrap_or(f32::MAX)
    }
}

// ---------------------------------------------------------------------------
// Gradient accumulator (avoids borrow-checker issues with &mut vec[i])
// ---------------------------------------------------------------------------

/// Per-dimension gradient for a center/offset box.
struct BoxGrad {
    center: Vec<f32>,
    offset: Vec<f32>,
}

impl BoxGrad {
    fn zeros(dim: usize) -> Self {
        Self {
            center: vec![0.0; dim],
            offset: vec![0.0; dim],
        }
    }
}

/// Flat storage for all embeddings. Avoids multiple mutable borrows
/// by using index-based access and separate gradient accumulators.
struct EmbeddingStore {
    /// Interleaved: centers[i*dim..(i+1)*dim], offsets[i*dim..(i+1)*dim]
    centers: Vec<Vec<f32>>,
    offsets: Vec<Vec<f32>>,
    opts: Vec<AMSGradState>,
}

impl EmbeddingStore {
    fn new(count: usize, dim: usize, lr: f32, rng: &mut impl Rng) -> Self {
        let centers: Vec<Vec<f32>> = (0..count)
            .map(|_| (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect())
            .collect();
        let offsets: Vec<Vec<f32>> = (0..count)
            .map(|_| (0..dim).map(|_| rng.random_range(0.1..1.0)).collect())
            .collect();
        let opts = (0..count).map(|_| AMSGradState::new(2 * dim, lr)).collect();
        Self {
            centers,
            offsets,
            opts,
        }
    }

    fn set_lr(&mut self, lr: f32) {
        for o in &mut self.opts {
            o.set_lr(lr);
        }
    }

    /// Apply a gradient update to entity `idx`.
    fn apply_grad(&mut self, idx: usize, grad: &BoxGrad) {
        let dim = grad.center.len();
        let opt = &mut self.opts[idx];

        // Pack gradients: [center_grads..., offset_grads...]
        opt.t += 1;
        let t = opt.t as f32;

        // Update moments for center
        for i in 0..dim {
            let g = grad.center[i];
            opt.m[i] = opt.beta1 * opt.m[i] + (1.0 - opt.beta1) * g;
            let v_new = opt.beta2 * opt.v[i] + (1.0 - opt.beta2) * g * g;
            opt.v[i] = v_new;
            opt.v_hat[i] = opt.v_hat[i].max(v_new);
        }

        // Update moments for offset
        for i in 0..dim {
            let idx_o = dim + i;
            let g = grad.offset[i];
            opt.m[idx_o] = opt.beta1 * opt.m[idx_o] + (1.0 - opt.beta1) * g;
            let v_new = opt.beta2 * opt.v[idx_o] + (1.0 - opt.beta2) * g * g;
            opt.v[idx_o] = v_new;
            opt.v_hat[idx_o] = opt.v_hat[idx_o].max(v_new);
        }

        let bias_correction = 1.0 - opt.beta1.powf(t);

        // Update center
        let center = &mut self.centers[idx];
        for i in 0..dim {
            let m_hat = opt.m[i] / bias_correction;
            let update = opt.lr * m_hat / (opt.v_hat[i].sqrt() + opt.epsilon);
            center[i] -= update;
            if !center[i].is_finite() {
                center[i] = 0.0;
            }
        }

        // Update offset (keep positive)
        let offset = &mut self.offsets[idx];
        for i in 0..dim {
            let idx_o = dim + i;
            let m_hat = opt.m[idx_o] / bias_correction;
            let update = opt.lr * m_hat / (opt.v_hat[idx_o].sqrt() + opt.epsilon);
            offset[i] -= update;
            offset[i] = offset[i].max(0.01);
            if !offset[i].is_finite() {
                offset[i] = 0.5;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Gradient computation (read-only access to embeddings, returns gradients)
// ---------------------------------------------------------------------------

/// Compute inclusion loss gradients for A ⊑ B.
/// Returns (grad_a, grad_b, loss).
fn inclusion_grads(
    ca: &[f32],
    oa: &[f32],
    cb: &[f32],
    ob: &[f32],
    margin: f32,
) -> (BoxGrad, BoxGrad, f32) {
    let dim = ca.len();
    let mut ga = BoxGrad::zeros(dim);
    let mut gb = BoxGrad::zeros(dim);

    let mut sum_sq = 0.0f32;
    let mut relu_terms = vec![0.0f32; dim];
    for i in 0..dim {
        let diff = ca[i] - cb[i];
        let v = diff.abs() + oa[i] - ob[i] - margin;
        let rv = v.max(0.0);
        relu_terms[i] = rv;
        sum_sq += rv * rv;
    }
    let norm = sum_sq.sqrt();
    if norm < 1e-8 {
        return (ga, gb, norm);
    }

    for i in 0..dim {
        if relu_terms[i] <= 0.0 {
            continue;
        }
        let diff = ca[i] - cb[i];
        let sign = if diff >= 0.0 { 1.0 } else { -1.0 };
        let scale = relu_terms[i] / norm;

        ga.center[i] = sign * scale;
        gb.center[i] = -sign * scale;
        ga.offset[i] = scale;
        gb.offset[i] = -scale;
    }

    (ga, gb, norm)
}

/// Compute disjointness loss gradients.
/// Returns (grad_a, grad_b, loss).
fn disjointness_grads(
    ca: &[f32],
    oa: &[f32],
    cb: &[f32],
    ob: &[f32],
    margin: f32,
) -> (BoxGrad, BoxGrad, f32) {
    let dim = ca.len();
    let mut ga = BoxGrad::zeros(dim);
    let mut gb = BoxGrad::zeros(dim);

    let mut sum_sq = 0.0f32;
    let mut relu_terms = vec![0.0f32; dim];
    for i in 0..dim {
        let diff = ca[i] - cb[i];
        let v = -diff.abs() + oa[i] + ob[i] - margin;
        let rv = v.max(0.0);
        relu_terms[i] = rv;
        sum_sq += rv * rv;
    }
    let norm = sum_sq.sqrt();
    if norm < 1e-8 {
        return (ga, gb, norm);
    }

    for i in 0..dim {
        if relu_terms[i] <= 0.0 {
            continue;
        }
        let diff = ca[i] - cb[i];
        let sign = if diff >= 0.0 { 1.0 } else { -1.0 };
        let scale = relu_terms[i] / norm;

        // Push centers apart, shrink offsets
        ga.center[i] = -sign * scale;
        gb.center[i] = sign * scale;
        ga.offset[i] = scale;
        gb.offset[i] = scale;
    }

    (ga, gb, norm)
}

/// Compute negative inclusion gradients (push A OUT of B).
fn negative_inclusion_grads(
    ca: &[f32],
    _oa: &[f32],
    cb: &[f32],
    _ob: &[f32],
) -> (BoxGrad, BoxGrad) {
    let dim = ca.len();
    let mut ga = BoxGrad::zeros(dim);
    let mut gb = BoxGrad::zeros(dim);
    let scale = 0.1;

    for i in 0..dim {
        let diff = ca[i] - cb[i];
        let sign = if diff >= 0.0 { 1.0 } else { -1.0 };
        ga.center[i] = -sign * scale;
        gb.center[i] = sign * scale;
        gb.offset[i] = scale;
    }

    (ga, gb)
}

// ---------------------------------------------------------------------------
// Training loop
// ---------------------------------------------------------------------------

/// Train EL++ box embeddings on an ontology.
///
/// Returns trained embeddings (concept and role centers/offsets) and per-epoch losses.
pub fn train_el_embeddings(ontology: &Ontology, config: &ElTrainingConfig) -> ElTrainingResult {
    let dim = config.dim;
    let nc = ontology.num_concepts().max(1);
    let nr = ontology.num_roles().max(1);

    let mut rng = rand::rngs::SmallRng::seed_from_u64(config.seed);
    let mut concepts = EmbeddingStore::new(nc, dim, config.learning_rate, &mut rng);
    let mut roles = EmbeddingStore::new(nr, dim, config.learning_rate, &mut rng);

    let mut epoch_losses = Vec::with_capacity(config.epochs);
    let mut axiom_indices: Vec<usize> = (0..ontology.axioms.len()).collect();

    for epoch in 0..config.epochs {
        let lr = get_learning_rate(
            epoch,
            config.epochs,
            config.learning_rate,
            config.warmup_epochs,
        );
        concepts.set_lr(lr);
        roles.set_lr(lr);

        axiom_indices.shuffle(&mut rng);
        let mut total_loss = 0.0f32;
        let mut count = 0usize;

        for &ax_idx in &axiom_indices {
            let axiom = &ontology.axioms[ax_idx];
            match *axiom {
                Axiom::SubClassOf { sub, sup } => {
                    // Positive: C ⊑ D
                    let (ga, gb, loss) = inclusion_grads(
                        &concepts.centers[sub],
                        &concepts.offsets[sub],
                        &concepts.centers[sup],
                        &concepts.offsets[sup],
                        config.margin,
                    );
                    total_loss += loss;
                    concepts.apply_grad(sub, &ga);
                    if sup != sub {
                        concepts.apply_grad(sup, &gb);
                    }

                    // Negative samples
                    for _ in 0..config.negative_samples {
                        let neg = rng.random_range(0..nc);
                        if neg == sub || neg == sup {
                            continue;
                        }
                        let neg_loss = el::el_inclusion_loss(
                            &concepts.centers[neg],
                            &concepts.offsets[neg],
                            &concepts.centers[sup],
                            &concepts.offsets[sup],
                            0.0,
                        )
                        .unwrap();
                        let neg_penalty = (config.margin - neg_loss).max(0.0);
                        total_loss += config.negative_weight * neg_penalty;

                        if neg_penalty > 0.0 {
                            let (gn, gs) = negative_inclusion_grads(
                                &concepts.centers[neg],
                                &concepts.offsets[neg],
                                &concepts.centers[sup],
                                &concepts.offsets[sup],
                            );
                            concepts.apply_grad(neg, &gn);
                            if sup != neg {
                                concepts.apply_grad(sup, &gs);
                            }
                        }
                    }
                }
                Axiom::Disjoint { a, b } => {
                    let (ga, gb, loss) = disjointness_grads(
                        &concepts.centers[a],
                        &concepts.offsets[a],
                        &concepts.centers[b],
                        &concepts.offsets[b],
                        config.margin,
                    );
                    total_loss += config.disjointness_weight * loss;
                    concepts.apply_grad(a, &ga);
                    if b != a {
                        concepts.apply_grad(b, &gb);
                    }
                }
                Axiom::Existential {
                    role,
                    filler,
                    target,
                } => {
                    // ∃R.C ⊑ D: existential_box(R, C) should be contained in D
                    let mut ex_center = vec![0.0f32; dim];
                    let mut ex_offset = vec![0.0f32; dim];
                    el::existential_box(
                        &roles.centers[role],
                        &roles.offsets[role],
                        &concepts.centers[filler],
                        &concepts.offsets[filler],
                        &mut ex_center,
                        &mut ex_offset,
                    )
                    .unwrap();

                    // Inclusion loss: ex ⊑ target
                    let (g_ex, g_target, loss) = inclusion_grads(
                        &ex_center,
                        &ex_offset,
                        &concepts.centers[target],
                        &concepts.offsets[target],
                        config.margin,
                    );
                    total_loss += config.existential_weight * loss;

                    // Chain rule: ex_center = role_center + filler_center
                    // ex_offset = max(0, filler_offset - role_offset)
                    let mut g_role = BoxGrad::zeros(dim);
                    let mut g_filler = BoxGrad::zeros(dim);
                    for i in 0..dim {
                        g_role.center[i] = g_ex.center[i];
                        g_filler.center[i] = g_ex.center[i];

                        if concepts.offsets[filler][i] > roles.offsets[role][i] {
                            g_filler.offset[i] = g_ex.offset[i];
                            g_role.offset[i] = -g_ex.offset[i];
                        }
                    }

                    roles.apply_grad(role, &g_role);
                    concepts.apply_grad(filler, &g_filler);
                    concepts.apply_grad(target, &g_target);
                }
                Axiom::RoleInclusion { sub, sup } => {
                    let (ga, gb, loss) = inclusion_grads(
                        &roles.centers[sub],
                        &roles.offsets[sub],
                        &roles.centers[sup],
                        &roles.offsets[sup],
                        config.margin,
                    );
                    total_loss += config.role_inclusion_weight * loss;
                    roles.apply_grad(sub, &ga);
                    if sup != sub {
                        roles.apply_grad(sup, &gb);
                    }
                }
                Axiom::RoleComposition { r, s, t } => {
                    // R ∘ S ⊑ T
                    let mut comp_center = vec![0.0f32; dim];
                    let mut comp_offset = vec![0.0f32; dim];
                    el::compose_roles(
                        &roles.centers[r],
                        &roles.offsets[r],
                        &roles.centers[s],
                        &roles.offsets[s],
                        &mut comp_center,
                        &mut comp_offset,
                    )
                    .unwrap();

                    let (g_comp, g_t, loss) = inclusion_grads(
                        &comp_center,
                        &comp_offset,
                        &roles.centers[t],
                        &roles.offsets[t],
                        config.margin,
                    );
                    total_loss += config.role_composition_weight * loss;

                    // Chain rule: comp_center = r_c + s_c, comp_offset = r_o + s_o
                    let mut g_r = BoxGrad::zeros(dim);
                    let mut g_s = BoxGrad::zeros(dim);
                    for i in 0..dim {
                        g_r.center[i] = g_comp.center[i];
                        g_s.center[i] = g_comp.center[i];
                        g_r.offset[i] = g_comp.offset[i];
                        g_s.offset[i] = g_comp.offset[i];
                    }

                    roles.apply_grad(r, &g_r);
                    if s != r {
                        roles.apply_grad(s, &g_s);
                    }
                    if t != r && t != s {
                        roles.apply_grad(t, &g_t);
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

    ElTrainingResult {
        concept_centers: concepts.centers,
        concept_offsets: concepts.offsets,
        role_centers: roles.centers,
        role_offsets: roles.offsets,
        epoch_losses,
    }
}

/// Evaluate subsumption prediction accuracy on a set of axioms.
///
/// For each `SubClassOf(C, D)` axiom, checks whether the model ranks
/// the true superclass D among the top-k predictions (lowest inclusion loss).
/// Returns (hits_at_1, hits_at_10, mrr) over `SubClassOf` axioms only.
pub fn evaluate_subsumption(result: &ElTrainingResult, axioms: &[Axiom]) -> (f32, f32, f32) {
    let nc = result.concept_centers.len();
    if nc == 0 {
        return (0.0, 0.0, 0.0);
    }

    let mut hits1 = 0usize;
    let mut hits10 = 0usize;
    let mut rr_sum = 0.0f32;
    let mut total = 0usize;

    for axiom in axioms {
        if let Axiom::SubClassOf { sub, sup } = axiom {
            let mut scores: Vec<(usize, f32)> = (0..nc)
                .map(|c| {
                    let loss = result.subsumption_score(*sub, c);
                    (c, loss)
                })
                .collect();
            scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let rank = scores
                .iter()
                .position(|(c, _)| c == sup)
                .map(|p| p + 1)
                .unwrap_or(nc);

            if rank == 1 {
                hits1 += 1;
            }
            if rank <= 10 {
                hits10 += 1;
            }
            rr_sum += 1.0 / rank as f32;
            total += 1;
        }
    }

    if total == 0 {
        return (0.0, 0.0, 0.0);
    }
    (
        hits1 as f32 / total as f32,
        hits10 as f32 / total as f32,
        rr_sum / total as f32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_ontology() -> Ontology {
        let input = "\
# A small animal ontology
SubClassOf Dog Animal
SubClassOf Cat Animal
SubClassOf Poodle Dog
SubClassOf Animal LivingThing
Disjoint Dog Cat
";
        Ontology::parse(input.as_bytes()).unwrap()
    }

    #[test]
    fn parse_basic_ontology() {
        let ont = small_ontology();
        assert_eq!(ont.num_concepts(), 5); // Dog, Animal, Cat, Poodle, LivingThing
        assert_eq!(ont.axioms.len(), 5);
        assert_eq!(ont.axioms[0], Axiom::SubClassOf { sub: 0, sup: 1 });
    }

    #[test]
    fn parse_all_axiom_types() {
        let input = "\
SubClassOf A B
Disjoint C D
Existential hasParent Human Human
RoleInclusion hasChild hasDescendant
RoleComposition hasParent hasSibling hasUncle
";
        let ont = Ontology::parse(input.as_bytes()).unwrap();
        assert_eq!(ont.axioms.len(), 5);
        assert_eq!(ont.num_concepts(), 5); // A, B, C, D, Human
        assert_eq!(ont.num_roles(), 5); // hasParent, hasChild, hasDescendant, hasSibling, hasUncle
    }

    #[test]
    fn parse_errors() {
        assert!(Ontology::parse("Unknown A B".as_bytes()).is_err());
        assert!(Ontology::parse("SubClassOf A".as_bytes()).is_err());
        assert!(Ontology::parse("SubClassOf A B C".as_bytes()).is_err());
    }

    #[test]
    fn train_small_ontology_loss_decreases() {
        let ont = small_ontology();
        let config = ElTrainingConfig {
            dim: 16,
            epochs: 100,
            learning_rate: 0.01,
            log_interval: 0,
            seed: 42,
            ..Default::default()
        };
        let result = train_el_embeddings(&ont, &config);

        let first_10_avg: f32 = result.epoch_losses[..10].iter().sum::<f32>() / 10.0;
        let last_10_avg: f32 = result.epoch_losses[90..].iter().sum::<f32>() / 10.0;
        assert!(
            last_10_avg < first_10_avg,
            "loss should decrease: first_10={first_10_avg:.4}, last_10={last_10_avg:.4}"
        );
    }

    #[test]
    fn train_subsumption_prediction() {
        let ont = small_ontology();
        let config = ElTrainingConfig {
            dim: 30,
            epochs: 300,
            learning_rate: 0.01,
            negative_samples: 3,
            log_interval: 0,
            seed: 42,
            ..Default::default()
        };
        let result = train_el_embeddings(&ont, &config);

        // Dog ⊑ Animal should have lower loss than Cat ⊑ Dog
        let dog = ont.concept_index["Dog"];
        let cat = ont.concept_index["Cat"];
        let animal = ont.concept_index["Animal"];

        let dog_animal = result.subsumption_score(dog, animal);
        let cat_dog = result.subsumption_score(cat, dog);
        assert!(
            dog_animal < cat_dog,
            "Dog ⊑ Animal ({dog_animal:.4}) should score lower than Cat ⊑ Dog ({cat_dog:.4})"
        );
    }

    #[test]
    fn evaluate_subsumption_basic() {
        let ont = small_ontology();
        let config = ElTrainingConfig {
            dim: 30,
            epochs: 300,
            learning_rate: 0.01,
            negative_samples: 3,
            log_interval: 0,
            seed: 42,
            ..Default::default()
        };
        let result = train_el_embeddings(&ont, &config);

        let (hits1, hits10, mrr) = evaluate_subsumption(&result, &ont.axioms);
        assert!(mrr > 0.0, "MRR should be positive, got {mrr}");
        assert!(hits10 > 0.0, "Hits@10 should be positive, got {hits10}");
        eprintln!("Evaluation: Hits@1={hits1:.2}, Hits@10={hits10:.2}, MRR={mrr:.4}");
    }
}
