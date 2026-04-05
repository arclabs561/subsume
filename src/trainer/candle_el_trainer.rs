//! GPU-accelerated EL++ ontology embedding trainer via candle autograd.
//!
//! Trains box embeddings for EL++ ontology completion using:
//! - Mini-batch training with per-NF-type balanced sampling (Box2EL-style)
//! - AdamW optimizer with cosine LR schedule
//! - Inclusion loss: squared for NF2/NF4, unsquared for NF1/NF3 (see ablation notes)
//! - Disjointness-target negative sampling
//! - Center L2 distance evaluation (Box2EL protocol)
//!
//! # Architecture
//!
//! Concept embeddings: center `[nc, dim]` + offset `[nc, dim]` (abs of raw params).
//! Role embeddings: center `[nr, dim]` + offset `[nr, dim]`.
//! Existential box: `center = role_center + filler_center`,
//!   `offset = relu(filler_offset - role_offset)`.

use candle_core::{Device, Result, Tensor, Var};

use crate::el_training::{Axiom, Ontology};

/// GPU-accelerated EL++ trainer.
///
/// Uses Box2EL-style architecture with per-concept bump translations
/// for NF3/NF4 existential axioms. Bumps create concept-pair-specific
/// transformations that make centers discriminative for subsumption ranking.
pub struct CandleElTrainer {
    /// Concept centers: `[num_concepts, dim]`.
    pub(crate) concept_centers: Var,
    /// Concept raw offsets: `[num_concepts, dim]` (abs applied at use).
    pub(crate) concept_offsets: Var,
    /// Per-concept bump translations: `[num_concepts, dim]`.
    ///
    /// Used in NF3/NF4: concept C is bumped by D's bump vector before
    /// checking inclusion in the role box. This creates directional
    /// subsumption encoding (Box2EL's key architectural contribution).
    pub(crate) bumps: Var,
    /// Role head boxes: `[num_roles, dim*2]` (center + offset).
    pub(crate) role_heads: Var,
    /// Role tail boxes: `[num_roles, dim*2]` (center + offset).
    pub(crate) role_tails: Var,
    /// Embedding dimension.
    pub(crate) dim: usize,
    /// Number of concepts.
    pub(crate) num_concepts: usize,
    /// Number of roles.
    pub(crate) num_roles: usize,
    /// Margin for inclusion loss.
    pub(crate) margin: f32,
    /// Target separation distance for negatives.
    pub(crate) neg_dist: f32,
    /// Weight for NF4 negative sampling loss (0.0 disables, default 1.0).
    pub(crate) nf4_neg_weight: f32,
    /// Device.
    pub(crate) device: Device,
}

impl CandleElTrainer {
    /// Embedding dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of concepts.
    #[must_use]
    pub fn num_concepts(&self) -> usize {
        self.num_concepts
    }

    /// Number of roles.
    #[must_use]
    pub fn num_roles(&self) -> usize {
        self.num_roles
    }

    /// Weight for NF4 negative sampling loss (0.0 disables).
    #[must_use]
    pub fn nf4_neg_weight(&self) -> f32 {
        self.nf4_neg_weight
    }

    /// Set the NF4 negative sampling weight.
    pub fn set_nf4_neg_weight(&mut self, w: f32) {
        self.nf4_neg_weight = w;
    }

    /// Reference to the device.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Concept centers tensor: `[num_concepts, dim]`.
    #[must_use]
    pub fn concept_centers(&self) -> &Var {
        &self.concept_centers
    }

    /// Concept raw offsets tensor: `[num_concepts, dim]` (abs applied at use).
    #[must_use]
    pub fn concept_offsets(&self) -> &Var {
        &self.concept_offsets
    }

    /// Per-concept bump translations tensor: `[num_concepts, dim]`.
    #[must_use]
    pub fn bumps(&self) -> &Var {
        &self.bumps
    }

    /// Role head boxes tensor: `[num_roles, dim*2]` (center + offset).
    #[must_use]
    pub fn role_heads(&self) -> &Var {
        &self.role_heads
    }

    /// Role tail boxes tensor: `[num_roles, dim*2]` (center + offset).
    #[must_use]
    pub fn role_tails(&self) -> &Var {
        &self.role_tails
    }
}

impl CandleElTrainer {
    /// Create a new trainer from an ontology.
    pub fn new(
        num_concepts: usize,
        num_roles: usize,
        dim: usize,
        margin: f32,
        neg_dist: f32,
        device: &Device,
    ) -> Result<Self> {
        // Uniform [-1, 1] then L2-normalize (matching Box2EL exactly)
        let cc_raw = Tensor::rand(-1.0_f32, 1.0, (num_concepts, dim), device)?;
        let cc_norm =
            cc_raw.broadcast_div(&cc_raw.sqr()?.sum(1)?.sqrt()?.reshape((num_concepts, 1))?)?;
        let concept_centers = Var::from_tensor(&cc_norm)?;
        // All embeddings: uniform [-1,1] then L2-normalize (Box2EL init_embeddings)
        let co_raw = Tensor::rand(-1.0_f32, 1.0, (num_concepts, dim), device)?;
        let co_norm =
            co_raw.broadcast_div(&co_raw.sqr()?.sum(1)?.sqrt()?.reshape((num_concepts, 1))?)?;
        let concept_offsets = Var::from_tensor(&co_norm)?;

        let bump_raw = Tensor::rand(-1.0_f32, 1.0, (num_concepts, dim), device)?;
        let bump_norm =
            bump_raw.broadcast_div(&bump_raw.sqr()?.sum(1)?.sqrt()?.reshape((num_concepts, 1))?)?;
        let bumps = Var::from_tensor(&bump_norm)?;

        let nr = num_roles.max(1);
        let rh_raw = Tensor::rand(-1.0_f32, 1.0, (nr, dim * 2), device)?;
        let rh_norm = rh_raw.broadcast_div(&rh_raw.sqr()?.sum(1)?.sqrt()?.reshape((nr, 1))?)?;
        let role_heads = Var::from_tensor(&rh_norm)?;

        let rt_raw = Tensor::rand(-1.0_f32, 1.0, (nr, dim * 2), device)?;
        let rt_norm = rt_raw.broadcast_div(&rt_raw.sqr()?.sum(1)?.sqrt()?.reshape((nr, 1))?)?;
        let role_tails = Var::from_tensor(&rt_norm)?;

        Ok(Self {
            concept_centers,
            concept_offsets,
            bumps,
            role_heads,
            role_tails,
            dim,
            num_concepts,
            num_roles,
            margin,
            neg_dist,
            // Default 0.0 matches Box2EL (no NF4 negatives). Box2EL only
            // applies negative sampling to NF3. Setting > 0.0 enables NF4
            // negatives but competes for gradient bandwidth with NF1/NF3.
            nf4_neg_weight: 0.0,
            device: device.clone(),
        })
    }

    /// Inclusion loss: `||relu(|c_a - c_b| + o_a - o_b - margin)||` per sample (L2 norm).
    ///
    /// Returns the L2 norm (not squared). The caller squares it if needed
    /// (Box2EL uses `.square().mean()` which provides stronger gradient).
    fn inclusion_loss(
        centers_a: &Tensor,
        offsets_a: &Tensor,
        centers_b: &Tensor,
        offsets_b: &Tensor,
        margin: f32,
    ) -> Result<Tensor> {
        let diffs = centers_a.sub(centers_b)?.abs()?;
        let violation = diffs
            .add(offsets_a)?
            .sub(offsets_b)?
            .affine(1.0, -(margin as f64))?
            .relu()?;
        let norm_sq = violation.sqr()?.sum(1)?;
        norm_sq.affine(1.0, 1e-8)?.sqrt()
    }

    /// Negative loss (Box2EL `neg_loss`): `||relu(|c_a - c_b| - o_a - o_b + margin)||` per sample.
    /// Used for NF3 negatives. Returns L2 norm (not squared).
    fn neg_loss_fn(
        centers_a: &Tensor,
        offsets_a: &Tensor,
        centers_b: &Tensor,
        offsets_b: &Tensor,
        margin: f32,
    ) -> Result<Tensor> {
        let diffs = centers_a.sub(centers_b)?.abs()?;
        let gap = diffs
            .sub(offsets_a)?
            .sub(offsets_b)?
            .affine(1.0, margin as f64)?
            .relu()?;
        let norm_sq = gap.sqr()?.sum(1)?;
        norm_sq.affine(1.0, 1e-8)?.sqrt()
    }

    /// Disjointness score: same computation as [`neg_loss_fn`].
    ///
    /// Kept as a named alias for readability at call sites where the
    /// semantic intent is measuring box separation rather than negative loss.
    #[inline]
    fn disjointness_score(
        centers_a: &Tensor,
        offsets_a: &Tensor,
        centers_b: &Tensor,
        offsets_b: &Tensor,
        margin: f32,
    ) -> Result<Tensor> {
        Self::neg_loss_fn(centers_a, offsets_a, centers_b, offsets_b, margin)
    }

    /// Get concept boxes (center, abs(offset)) for given IDs.
    fn concept_boxes(&self, ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let centers = self.concept_centers.as_tensor().index_select(ids, 0)?;
        let offsets = self
            .concept_offsets
            .as_tensor()
            .index_select(ids, 0)?
            .abs()?;
        Ok((centers, offsets))
    }

    /// Get bump vectors for given concept IDs.
    fn concept_bumps(&self, ids: &Tensor) -> Result<Tensor> {
        self.bumps.as_tensor().index_select(ids, 0)
    }

    /// Get role head/tail boxes from the combined embedding.
    /// Returns (center, abs(offset)) for the requested role box type.
    fn role_box(&self, ids: &Tensor, head: bool) -> Result<(Tensor, Tensor)> {
        let embed = if head {
            self.role_heads.as_tensor().index_select(ids, 0)?
        } else {
            self.role_tails.as_tensor().index_select(ids, 0)?
        };
        let centers = embed.narrow(1, 0, self.dim)?;
        let offsets = embed.narrow(1, self.dim, self.dim)?.abs()?;
        Ok((centers, offsets))
    }

    /// Train on an ontology with AdamW and mini-batch sampling.
    #[allow(clippy::too_many_arguments)]
    pub fn fit(
        &self,
        ontology: &Ontology,
        epochs: usize,
        lr: f64,
        batch_size: usize,
        negative_samples: usize,
        reg_factor: f32,
    ) -> Result<Vec<f32>> {
        use candle_nn::{AdamW, Optimizer, ParamsAdamW};

        let mut vars = vec![
            self.concept_centers.clone(),
            self.concept_offsets.clone(),
            self.bumps.clone(),
        ];
        if self.num_roles > 0 {
            vars.push(self.role_heads.clone());
            vars.push(self.role_tails.clone());
        }

        let params = ParamsAdamW {
            lr,
            weight_decay: 0.0,
            ..Default::default()
        };
        let mut opt = AdamW::new(vars, params)?;

        // Group axioms by type for balanced sampling
        let mut nf2_axioms: Vec<(usize, usize)> = Vec::new(); // (sub, sup)
        let mut nf1_axioms: Vec<(usize, usize, usize)> = Vec::new(); // (c1, c2, target)
        let mut nf3_axioms: Vec<(usize, usize, usize)> = Vec::new(); // (sub, role, filler)
        let mut nf4_axioms: Vec<(usize, usize, usize)> = Vec::new(); // (role, filler, target)

        let mut skipped = 0usize;
        for ax in &ontology.axioms {
            match *ax {
                Axiom::SubClassOf { sub, sup } => nf2_axioms.push((sub, sup)),
                Axiom::Intersection { c1, c2, target } => nf1_axioms.push((c1, c2, target)),
                Axiom::ExistentialRight { sub, role, filler } => {
                    nf3_axioms.push((sub, role, filler))
                }
                Axiom::Existential {
                    role,
                    filler,
                    target,
                } => nf4_axioms.push((role, filler, target)),
                _ => skipped += 1,
            }
        }

        if skipped > 0 {
            eprintln!(
                "CandleElTrainer: skipped {skipped} axioms (RI6, RI7, Disjoint not yet supported)"
            );
        }

        let nc = self.num_concepts;
        let mut epoch_losses = Vec::with_capacity(epochs);
        let mut master_rng: u64 = 42;
        let lcg = |s: &mut u64| -> usize {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*s >> 33) as usize
        };

        for epoch in 0..epochs {
            // Branch RNG per NF type so adding/removing random calls in one
            // NF section doesn't shift the trajectory for other sections.
            let mut rng_nf2 = master_rng.wrapping_add(1);
            let mut rng_nf1 = master_rng.wrapping_add(2);
            let mut rng_nf3 = master_rng.wrapping_add(3);
            let mut rng_nf4 = master_rng.wrapping_add(4);
            lcg(&mut master_rng); // advance master for next epoch
                                  // Cosine LR decay (helps convergence even though Box2EL uses constant)
            let current_lr = crate::optimizer::cosine_lr(epoch, epochs, lr, 0.01);
            opt.set_learning_rate(current_lr);

            // Accumulate all NF losses into one tensor (Box2EL does one backward per epoch)
            let mut epoch_loss = Tensor::zeros((), candle_core::DType::F32, &self.device)?;

            // NF2: C ⊑ D -- sample a batch of subsumption axioms
            if !nf2_axioms.is_empty() {
                let bs = batch_size.min(nf2_axioms.len());
                let mut sub_ids = Vec::with_capacity(bs);
                let mut sup_ids = Vec::with_capacity(bs);
                for _ in 0..bs {
                    let idx = lcg(&mut rng_nf2) % nf2_axioms.len();
                    let (s, d) = nf2_axioms[idx];
                    sub_ids.push(s as u32);
                    sup_ids.push(d as u32);
                }

                let sub_t = Tensor::from_vec(sub_ids, (bs,), &self.device)?;
                let sup_t = Tensor::from_vec(sup_ids, (bs,), &self.device)?;
                let (c_sub, o_sub) = self.concept_boxes(&sub_t)?;
                let (c_sup, o_sup) = self.concept_boxes(&sup_t)?;

                // Box2EL-style: inclusion_loss.square().mean() (no center-contrastive)
                // Squaring provides quadratic gradient that moves centers strongly
                let pos_loss = Self::inclusion_loss(&c_sub, &o_sub, &c_sup, &o_sup, self.margin)?
                    .sqr()?
                    .mean(0)?;

                // Negative: Box2EL disjointness target
                let mut neg_loss_sum = Tensor::zeros((), candle_core::DType::F32, &self.device)?;
                for _ in 0..negative_samples {
                    let neg_ids: Vec<u32> =
                        (0..bs).map(|_| (lcg(&mut rng_nf2) % nc) as u32).collect();
                    let neg_t = Tensor::from_vec(neg_ids, (bs,), &self.device)?;
                    let (c_neg, o_neg) = self.concept_boxes(&neg_t)?;
                    let disj =
                        Self::disjointness_score(&c_sub, &o_sub, &c_neg, &o_neg, self.margin)?;
                    let target = Tensor::full(self.neg_dist, disj.shape(), &self.device)?;
                    let gap = target.sub(&disj)?;
                    let neg_loss = gap.sqr()?.mean(0)?;
                    neg_loss_sum = neg_loss_sum.add(&neg_loss)?;
                }

                let batch_loss = pos_loss.add(&neg_loss_sum)?;
                epoch_loss = epoch_loss.add(&batch_loss)?;
            }

            // NF1: C1 ⊓ C2 ⊑ D -- extended intersection then masked inclusion
            //
            // TransBox (Yang et al., WWW 2025) Section 4.4: standard box intersection
            // becomes empty exponentially fast in high dimensions. Extended intersection
            // uses a per-dimension overlap mask to restrict the inclusion loss to
            // dimensions where both boxes are present.
            //
            if !nf1_axioms.is_empty() {
                let bs = batch_size.min(nf1_axioms.len());
                let mut c1_ids = Vec::with_capacity(bs);
                let mut c2_ids = Vec::with_capacity(bs);
                let mut d_ids = Vec::with_capacity(bs);
                for _ in 0..bs {
                    let idx = lcg(&mut rng_nf1) % nf1_axioms.len();
                    let (c1, c2, d) = nf1_axioms[idx];
                    c1_ids.push(c1 as u32);
                    c2_ids.push(c2 as u32);
                    d_ids.push(d as u32);
                }

                let c1_t = Tensor::from_vec(c1_ids, (bs,), &self.device)?;
                let c2_t = Tensor::from_vec(c2_ids, (bs,), &self.device)?;
                let d_t = Tensor::from_vec(d_ids, (bs,), &self.device)?;

                let (cc1, oc1) = self.concept_boxes(&c1_t)?;
                let (cc2, oc2) = self.concept_boxes(&c2_t)?;
                let (cd, od) = self.concept_boxes(&d_t)?;

                // Extended intersection (TransBox Eq. 8): per-dimension overlap mask.
                let min1 = cc1.sub(&oc1)?;
                let max1 = cc1.add(&oc1)?;
                let min2 = cc2.sub(&oc2)?;
                let max2 = cc2.add(&oc2)?;

                // overlap_width > 0 where the two boxes overlap in that dimension.
                let overlap_width = max1.minimum(&max2)?.sub(&min1.maximum(&min2)?)?;

                // Soft mask: sigmoid(k * overlap_width) -- 1 where overlapping, 0 where disjoint.
                // Using k=20 for a sharp but differentiable mask.
                let mask = (overlap_width.affine(20.0, 0.0))?.tanh()?.relu()?;

                // Intersection center/offset (clamped to non-negative width).
                let inter_min = min1.maximum(&min2)?;
                let inter_max = max1.minimum(&max2)?.maximum(&inter_min)?;
                let inter_center = inter_min.add(&inter_max)?.affine(0.5, 0.0)?;
                let inter_offset = inter_max.sub(&inter_min)?.affine(0.5, 0.0)?;

                // Masked inclusion loss (TransBox Eq. 10):
                // Only measure containment in dimensions where both boxes overlap.
                let diffs = inter_center.sub(&cd)?.abs()?;
                let violation = diffs
                    .add(&inter_offset)?
                    .sub(&od)?
                    .affine(1.0, -(self.margin as f64))?
                    .relu()?;
                // Apply mask: violation only counts in overlapping dimensions.
                let masked_violation = violation.mul(&mask)?;
                let nf1_incl = masked_violation
                    .sqr()?
                    .sum(1)?
                    .affine(1.0, 1e-8)?
                    .sqrt()?
                    .mean(0)?;

                // Shrinkage term: encourage intersection offset to be small in
                // non-overlapping dimensions (push boxes toward overlap).
                let inv_mask = mask.affine(-1.0, 1.0)?; // 1 where disjoint
                let shrink = inter_offset.mul(&inv_mask)?.sqr()?.sum(1)?.mean(0)?;

                let nf1_loss = nf1_incl.add(&shrink.affine(0.1, 0.0)?)?;
                epoch_loss = epoch_loss.add(&nf1_loss)?;
            }

            // NF3: C ⊑ ∃r.D -- Box2EL bump-based existential
            // dist1 = inclusion(C + bump_D, head_r)
            // dist2 = inclusion(D + bump_C, tail_r)
            // loss = (dist1 + dist2) / 2
            if !nf3_axioms.is_empty() {
                let bs = batch_size.min(nf3_axioms.len());
                let mut sub_ids = Vec::with_capacity(bs);
                let mut role_ids = Vec::with_capacity(bs);
                let mut filler_ids = Vec::with_capacity(bs);
                for _ in 0..bs {
                    let idx = lcg(&mut rng_nf3) % nf3_axioms.len();
                    let (s, r, f) = nf3_axioms[idx];
                    sub_ids.push(s as u32);
                    role_ids.push(r as u32);
                    filler_ids.push(f as u32);
                }

                let sub_t = Tensor::from_vec(sub_ids, (bs,), &self.device)?;
                let role_t = Tensor::from_vec(role_ids, (bs,), &self.device)?;
                let filler_t = Tensor::from_vec(filler_ids, (bs,), &self.device)?;

                let (c_sub, o_sub) = self.concept_boxes(&sub_t)?;
                let (c_filler, o_filler) = self.concept_boxes(&filler_t)?;
                let bump_sub = self.concept_bumps(&sub_t)?;
                let bump_filler = self.concept_bumps(&filler_t)?;
                let (c_head, o_head) = self.role_box(&role_t, true)?;
                let (c_tail, o_tail) = self.role_box(&role_t, false)?;

                // C bumped by D's bump -> should be in head box of role r.
                // Gradient flows through BOTH centers and bumps (matching Box2EL).
                let c_sub_bumped = c_sub.add(&bump_filler)?;
                let dist1 =
                    Self::inclusion_loss(&c_sub_bumped, &o_sub, &c_head, &o_head, self.margin)?;

                // D bumped by C's bump -> should be in tail box of role r
                let c_filler_bumped = c_filler.add(&bump_sub)?;
                let dist2 = Self::inclusion_loss(
                    &c_filler_bumped,
                    &o_filler,
                    &c_tail,
                    &o_tail,
                    self.margin,
                )?;

                // NF3 uses unsquared L2 norm (same rationale as NF1 above).
                // Squaring collapses NF3 on GO from H@1=0.260 to 0.003.
                let nf3_loss = dist1.add(&dist2)?.affine(0.5, 0.0)?.mean(0)?;

                // NF3 negatives: corrupt BOTH head and tail (matching Box2EL exactly).
                // For each neg sample: replace D with random -> check (C+bump_rand, head_r)
                //                      replace C with random -> check (rand+bump_C, tail_r)
                let mut nf3_neg_sum = Tensor::zeros((), candle_core::DType::F32, &self.device)?;
                for _ in 0..negative_samples {
                    // Corrupt tail (D): keep C, replace D with random
                    let neg_tail_ids: Vec<u32> =
                        (0..bs).map(|_| (lcg(&mut rng_nf3) % nc) as u32).collect();
                    let neg_tail_t = Tensor::from_vec(neg_tail_ids, (bs,), &self.device)?;
                    let bump_neg_tail = self.concept_bumps(&neg_tail_t)?;

                    let c_sub_bumped_neg = c_sub.add(&bump_neg_tail)?;
                    let neg_loss1 = Self::neg_loss_fn(
                        &c_sub_bumped_neg,
                        &o_sub,
                        &c_head,
                        &o_head,
                        self.margin,
                    )?;

                    // Corrupt head (C): keep D, replace C with random
                    let neg_head_ids: Vec<u32> =
                        (0..bs).map(|_| (lcg(&mut rng_nf3) % nc) as u32).collect();
                    let neg_head_t = Tensor::from_vec(neg_head_ids, (bs,), &self.device)?;
                    let (c_neg_head, o_neg_head) = self.concept_boxes(&neg_head_t)?;

                    let c_neg_bumped = c_neg_head.add(&bump_sub)?;
                    let neg_loss2 = Self::neg_loss_fn(
                        &c_neg_bumped,
                        &o_neg_head,
                        &c_tail,
                        &o_tail,
                        self.margin,
                    )?;

                    // Box2EL: (neg_dist - neg_loss).square().mean() for BOTH
                    let target1 = Tensor::full(self.neg_dist, neg_loss1.shape(), &self.device)?;
                    let target2 = Tensor::full(self.neg_dist, neg_loss2.shape(), &self.device)?;
                    let nl1 = target1.sub(&neg_loss1)?.sqr()?.mean(0)?;
                    let nl2 = target2.sub(&neg_loss2)?.sqr()?.mean(0)?;
                    nf3_neg_sum = nf3_neg_sum.add(&nl1)?.add(&nl2)?;
                }

                let nf3_total = nf3_loss.add(&nf3_neg_sum)?;
                epoch_loss = epoch_loss.add(&nf3_total)?;
            }

            // NF4: ∃r.C ⊑ D -- same bump approach, reversed direction
            if !nf4_axioms.is_empty() {
                let bs = batch_size.min(nf4_axioms.len());
                let mut role_ids = Vec::with_capacity(bs);
                let mut filler_ids = Vec::with_capacity(bs);
                let mut target_ids = Vec::with_capacity(bs);
                for _ in 0..bs {
                    let idx = lcg(&mut rng_nf4) % nf4_axioms.len();
                    let (r, f, t) = nf4_axioms[idx];
                    role_ids.push(r as u32);
                    filler_ids.push(f as u32);
                    target_ids.push(t as u32);
                }

                let role_t = Tensor::from_vec(role_ids, (bs,), &self.device)?;
                let filler_t = Tensor::from_vec(filler_ids, (bs,), &self.device)?;
                let target_t = Tensor::from_vec(target_ids, (bs,), &self.device)?;

                // Box2EL NF4: inclusion_loss(head_r.translate(-bump_C), D)
                // Different from NF3! Only uses head box, one bump, one target.
                let (c_target, o_target) = self.concept_boxes(&target_t)?;
                let bump_filler = self.concept_bumps(&filler_t)?;
                let (c_head, o_head) = self.role_box(&role_t, true)?;

                // Translate head box by -bump_C (negate filler bump)
                let c_head_shifted = c_head.sub(&bump_filler)?;
                let nf4_loss = Self::inclusion_loss(
                    &c_head_shifted,
                    &o_head,
                    &c_target,
                    &o_target,
                    self.margin,
                )?
                .sqr()?
                .mean(0)?;

                // NF4 negatives: corrupt filler (C) and target (D).
                // Skipped entirely when nf4_neg_weight == 0.0 to preserve
                // RNG sequence for other NF types.
                let nf4_total = if self.nf4_neg_weight > 0.0 {
                    let mut nf4_neg_sum = Tensor::zeros((), candle_core::DType::F32, &self.device)?;
                    for _ in 0..negative_samples {
                        // Corrupt target (D): keep role+filler, replace D with random
                        let neg_target_ids: Vec<u32> =
                            (0..bs).map(|_| (lcg(&mut rng_nf4) % nc) as u32).collect();
                        let neg_target_t = Tensor::from_vec(neg_target_ids, (bs,), &self.device)?;
                        let (c_neg_target, o_neg_target) = self.concept_boxes(&neg_target_t)?;

                        let neg_loss1 = Self::neg_loss_fn(
                            &c_head_shifted,
                            &o_head,
                            &c_neg_target,
                            &o_neg_target,
                            self.margin,
                        )?;

                        // Corrupt filler (C): keep role+target, replace C with random
                        let neg_filler_ids: Vec<u32> =
                            (0..bs).map(|_| (lcg(&mut rng_nf4) % nc) as u32).collect();
                        let neg_filler_t = Tensor::from_vec(neg_filler_ids, (bs,), &self.device)?;
                        let bump_neg_filler = self.concept_bumps(&neg_filler_t)?;
                        let c_head_neg_shifted = c_head.sub(&bump_neg_filler)?;

                        let neg_loss2 = Self::neg_loss_fn(
                            &c_head_neg_shifted,
                            &o_head,
                            &c_target,
                            &o_target,
                            self.margin,
                        )?;

                        let target1 = Tensor::full(self.neg_dist, neg_loss1.shape(), &self.device)?;
                        let target2 = Tensor::full(self.neg_dist, neg_loss2.shape(), &self.device)?;
                        let nl1 = target1.sub(&neg_loss1)?.sqr()?.mean(0)?;
                        let nl2 = target2.sub(&neg_loss2)?.sqr()?.mean(0)?;
                        nf4_neg_sum = nf4_neg_sum.add(&nl1)?.add(&nl2)?;
                    }
                    nf4_loss.add(&nf4_neg_sum.affine(self.nf4_neg_weight as f64, 0.0)?)?
                } else {
                    nf4_loss
                };
                epoch_loss = epoch_loss.add(&nf4_total)?;
            }

            // Regularization: bump norms (Box2EL: reg_factor * mean(||bump||_2))
            if reg_factor > 0.0 {
                let bump_reg = self
                    .bumps
                    .as_tensor()
                    .sqr()?
                    .sum(1)?
                    .sqrt()?
                    .mean(0)?
                    .affine(reg_factor as f64, 0.0)?;
                epoch_loss = epoch_loss.add(&bump_reg)?;
            }

            // Single backward pass for entire epoch (matching Box2EL)
            let loss_val = epoch_loss.to_scalar::<f32>()?;
            opt.backward_step(&epoch_loss)?;

            epoch_losses.push(loss_val);

            // NaN detection: stop early to preserve debuggable state
            if loss_val.is_nan() || loss_val.is_infinite() {
                eprintln!(
                    "  WARNING: loss diverged at epoch {} (loss={loss_val}). Stopping.",
                    epoch + 1
                );
                break;
            }

            if (epoch + 1) % 100 == 0 || epoch == 0 {
                let c_mean = self
                    .concept_centers
                    .as_tensor()
                    .abs()?
                    .mean_all()?
                    .to_scalar::<f32>()?;
                let o_mean = self
                    .concept_offsets
                    .as_tensor()
                    .abs()?
                    .mean_all()?
                    .to_scalar::<f32>()?;
                eprintln!(
                    "  epoch {:>5}/{epochs}: loss={loss_val:.4} lr={current_lr:.6} |c|={c_mean:.3} |o|={o_mean:.3}",
                    epoch + 1
                );
            }
        }

        Ok(epoch_losses)
    }
}

/// Compute the rank of `target` among all concepts by L2 distance from `query`
/// to rows of `centers` (flat `[nc * dim]`). Concepts in `exclude` are skipped.
///
/// Returns `(rank, dist_to_target)` where rank is 1-based.
fn l2_rank(
    query: &[f32],
    centers: &[f32],
    nc: usize,
    dim: usize,
    target: usize,
    exclude: &[usize],
) -> (usize, f32) {
    let target_dist_sq = {
        let off = target * dim;
        let mut d = 0.0f32;
        for i in 0..dim {
            let diff = query[i] - centers[off + i];
            d += diff * diff;
        }
        d
    };

    let mut rank = 1usize;
    for c in 0..nc {
        if c == target || exclude.contains(&c) {
            continue;
        }
        let off = c * dim;
        let mut d = 0.0f32;
        for i in 0..dim {
            let diff = query[i] - centers[off + i];
            d += diff * diff;
        }
        // Compare squared distances (sqrt is monotone, skip it).
        if d < target_dist_sq {
            rank += 1;
        }
    }
    (rank, target_dist_sq.sqrt())
}

/// Accumulate H@1, H@10, MRR from a sequence of ranks.
fn metrics_from_ranks(ranks: &[usize]) -> (f32, f32, f32) {
    if ranks.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let n = ranks.len() as f32;
    let h1 = ranks.iter().filter(|&&r| r == 1).count() as f32 / n;
    let h10 = ranks.iter().filter(|&&r| r <= 10).count() as f32 / n;
    let mrr: f32 = ranks.iter().map(|&r| 1.0 / r as f32).sum::<f32>() / n;
    (h1, h10, mrr)
}

/// Shared ranking evaluation: for each test item, score all candidates,
/// find the rank of the target, and accumulate H@1, H@10, MRR.
///
/// `nc` = number of concepts, `dim` = embedding dimension.
/// `scorer` returns `(target_idx, Vec<(candidate_idx, score)>)` for each test item.
/// Lower score = better rank.
fn rank_evaluate<const N: usize>(
    nc: usize,
    items: &[[usize; N]],
    mut scorer: impl FnMut(&[usize; N]) -> Option<(usize, Vec<(usize, f32)>)>,
) -> (f32, f32, f32) {
    let mut hits1 = 0usize;
    let mut hits10 = 0usize;
    let mut rr_sum = 0.0f32;
    let mut total = 0usize;

    for item in items {
        let Some((target, mut scores)) = scorer(item) else {
            continue;
        };
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let rank = scores
            .iter()
            .position(|(c, _)| *c == target)
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

    if total == 0 {
        return (0.0, 0.0, 0.0);
    }
    (
        hits1 as f32 / total as f32,
        hits10 as f32 / total as f32,
        rr_sum / total as f32,
    )
}

impl CandleElTrainer {
    /// Evaluate subsumption (NF2: C ⊑ D) by center L2 distance ranking.
    ///
    /// For each test pair (C, D), ranks all concepts by L2 distance to C's center
    /// and reports the rank of D. This matches the Box2EL evaluation protocol
    /// where centers encode subsumption hierarchy position.
    pub fn evaluate_subsumption(&self, test_axioms: &[(usize, usize)]) -> Result<(f32, f32, f32)> {
        let centers: Vec<f32> = self
            .concept_centers
            .as_tensor()
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let nc = self.num_concepts;
        let dim = self.dim;

        let mut ranks = Vec::with_capacity(test_axioms.len());
        for &(sub, sup) in test_axioms {
            if sub >= nc || sup >= nc {
                continue;
            }
            let query = &centers[sub * dim..(sub + 1) * dim];
            let (rank, _) = l2_rank(query, &centers, nc, dim, sup, &[sub]);
            ranks.push(rank);
        }
        Ok(metrics_from_ranks(&ranks))
    }

    /// Evaluate NF1 (C1 ⊓ C2 ⊑ D) by intersection-center L2 distance ranking.
    ///
    /// Computes the box intersection of C1 and C2, then ranks all concepts
    /// by L2 distance from the intersection center to find D.
    pub fn evaluate_nf1(&self, test_axioms: &[(usize, usize, usize)]) -> Result<(f32, f32, f32)> {
        let centers: Vec<f32> = self
            .concept_centers
            .as_tensor()
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let offsets: Vec<f32> = self
            .concept_offsets
            .as_tensor()
            .abs()?
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let nc = self.num_concepts;
        let dim = self.dim;

        let mut ranks = Vec::with_capacity(test_axioms.len());
        let mut inter_center = vec![0.0f32; dim];
        for &(c1, c2, d) in test_axioms {
            if c1 >= nc || c2 >= nc || d >= nc {
                continue;
            }
            let c1_off = c1 * dim;
            let c2_off = c2 * dim;
            for i in 0..dim {
                let min1 = centers[c1_off + i] - offsets[c1_off + i];
                let max1 = centers[c1_off + i] + offsets[c1_off + i];
                let min2 = centers[c2_off + i] - offsets[c2_off + i];
                let max2 = centers[c2_off + i] + offsets[c2_off + i];
                let inter_min = min1.max(min2);
                let inter_max = max1.min(max2).max(inter_min);
                inter_center[i] = (inter_min + inter_max) / 2.0;
            }
            let (rank, _) = l2_rank(&inter_center, &centers, nc, dim, d, &[c1, c2]);
            ranks.push(rank);
        }
        Ok(metrics_from_ranks(&ranks))
    }

    /// Evaluate NF3 (C ⊑ ∃r.D) by bumped-center L2 distance ranking.
    ///
    /// For each test triple (C, r, D): compute C + bump_D for each candidate D,
    /// measure L2 distance to role head center, rank candidates. The candidate D
    /// whose bump brings C closest to the role head wins.
    pub fn evaluate_nf3(&self, test_axioms: &[(usize, usize, usize)]) -> Result<(f32, f32, f32)> {
        let centers: Vec<f32> = self
            .concept_centers
            .as_tensor()
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let bump_vecs: Vec<f32> = self
            .bumps
            .as_tensor()
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let role_heads_data: Vec<f32> = self
            .role_heads
            .as_tensor()
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let nc = self.num_concepts;
        let nr = self.num_roles;
        let dim = self.dim;

        // NF3 scoring: dist(sub_center + bump[d], role_head[r]) for each candidate d.
        // The query depends on the candidate, so we compute the target's score first,
        // then count how many candidates score better (avoiding Vec allocation + sort).
        let mut ranks = Vec::with_capacity(test_axioms.len());
        for &(sub, role, filler) in test_axioms {
            if sub >= nc || filler >= nc || role >= nr {
                continue;
            }
            let sub_off = sub * dim;
            let rh_off = role * dim * 2;

            // Score for the correct filler (squared distance -- sqrt is monotone).
            let target_dist_sq = {
                let bump_off = filler * dim;
                let mut d = 0.0f32;
                for i in 0..dim {
                    let bumped = centers[sub_off + i] + bump_vecs[bump_off + i];
                    let diff = bumped - role_heads_data[rh_off + i];
                    d += diff * diff;
                }
                d
            };

            let mut rank = 1usize;
            for c in 0..nc {
                let bump_off = c * dim;
                let mut d = 0.0f32;
                for i in 0..dim {
                    let bumped = centers[sub_off + i] + bump_vecs[bump_off + i];
                    let diff = bumped - role_heads_data[rh_off + i];
                    d += diff * diff;
                }
                if d < target_dist_sq {
                    rank += 1;
                }
            }
            ranks.push(rank);
        }
        Ok(metrics_from_ranks(&ranks))
    }

    /// Evaluate NF4 (∃r.C ⊑ D) by shifted-head L2 distance ranking.
    ///
    /// For each test triple (r, C, D): compute head_r - bump_C, then rank all
    /// concepts D by L2 distance from that shifted center to D's center.
    pub fn evaluate_nf4(&self, test_axioms: &[(usize, usize, usize)]) -> Result<(f32, f32, f32)> {
        let centers: Vec<f32> = self
            .concept_centers
            .as_tensor()
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let bump_vecs: Vec<f32> = self
            .bumps
            .as_tensor()
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let role_heads_data: Vec<f32> = self
            .role_heads
            .as_tensor()
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let nc = self.num_concepts;
        let nr = self.num_roles;
        let dim = self.dim;

        let mut ranks = Vec::with_capacity(test_axioms.len());
        let mut query = vec![0.0f32; dim];
        for &(role, filler, target) in test_axioms {
            if filler >= nc || target >= nc || role >= nr {
                continue;
            }
            let rh_off = role * dim * 2;
            let bump_off = filler * dim;
            for i in 0..dim {
                query[i] = role_heads_data[rh_off + i] - bump_vecs[bump_off + i];
            }
            let (rank, _) = l2_rank(&query, &centers, nc, dim, target, &[]);
            ranks.push(rank);
        }
        Ok(metrics_from_ranks(&ranks))
    }

    /// Evaluate subsumption (NF2: C ⊑ D) by inclusion loss ranking.
    ///
    /// For each test pair (C, D), ranks all concepts by inclusion loss
    /// `inclusion_loss(C, candidate)` (lower = better containment) instead
    /// of center L2 distance. This uses learned box widths in addition to centers.
    pub fn evaluate_subsumption_by_inclusion(
        &self,
        test_axioms: &[(usize, usize)],
    ) -> Result<(f32, f32, f32)> {
        let centers: Vec<f32> = self
            .concept_centers
            .as_tensor()
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let offsets: Vec<f32> = self
            .concept_offsets
            .as_tensor()
            .abs()?
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let nc = self.num_concepts;
        let dim = self.dim;
        let margin = self.margin;

        let items: Vec<[usize; 2]> = test_axioms.iter().map(|(a, b)| [*a, *b]).collect();
        Ok(rank_evaluate(nc, &items, |&[sub, sup]| {
            if sub >= nc || sup >= nc {
                return None;
            }
            let sub_off = sub * dim;
            let scores: Vec<(usize, f32)> = (0..nc)
                .filter(|&c| c != sub)
                .map(|c| {
                    let c_off = c * dim;
                    let loss_sq: f32 = (0..dim)
                        .map(|d| {
                            let diff = (centers[sub_off + d] - centers[c_off + d]).abs();
                            let v = (diff + offsets[sub_off + d] - offsets[c_off + d] - margin)
                                .max(0.0);
                            v * v
                        })
                        .sum();
                    (c, (loss_sq + 1e-8).sqrt())
                })
                .collect();
            Some((sup, scores))
        }))
    }

    /// Evaluate NF1 (C1 ⊓ C2 ⊑ D) by inclusion loss ranking.
    ///
    /// Computes the box intersection of C1 and C2, then ranks all concepts
    /// by `inclusion_loss(intersection, candidate)` instead of center L2 distance.
    pub fn evaluate_nf1_by_inclusion(
        &self,
        test_axioms: &[(usize, usize, usize)],
    ) -> Result<(f32, f32, f32)> {
        let centers: Vec<f32> = self
            .concept_centers
            .as_tensor()
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let offsets: Vec<f32> = self
            .concept_offsets
            .as_tensor()
            .abs()?
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let nc = self.num_concepts;
        let dim = self.dim;
        let margin = self.margin;

        let items: Vec<[usize; 3]> = test_axioms.iter().map(|(a, b, c)| [*a, *b, *c]).collect();
        Ok(rank_evaluate(nc, &items, |&[c1, c2, d]| {
            if c1 >= nc || c2 >= nc || d >= nc {
                return None;
            }
            let c1_off = c1 * dim;
            let c2_off = c2 * dim;
            let mut inter_center = vec![0.0f32; dim];
            let mut inter_offset = vec![0.0f32; dim];
            for i in 0..dim {
                let min1 = centers[c1_off + i] - offsets[c1_off + i];
                let max1 = centers[c1_off + i] + offsets[c1_off + i];
                let min2 = centers[c2_off + i] - offsets[c2_off + i];
                let max2 = centers[c2_off + i] + offsets[c2_off + i];
                let inter_min = min1.max(min2);
                let inter_max = max1.min(max2).max(inter_min);
                inter_center[i] = (inter_min + inter_max) / 2.0;
                inter_offset[i] = (inter_max - inter_min) / 2.0;
            }

            let scores: Vec<(usize, f32)> = (0..nc)
                .filter(|&c| c != c1 && c != c2)
                .map(|c| {
                    let c_off = c * dim;
                    let loss_sq: f32 = (0..dim)
                        .map(|i| {
                            let diff = (inter_center[i] - centers[c_off + i]).abs();
                            let v = (diff + inter_offset[i] - offsets[c_off + i] - margin).max(0.0);
                            v * v
                        })
                        .sum();
                    (c, (loss_sq + 1e-8).sqrt())
                })
                .collect();
            Some((d, scores))
        }))
    }

    /// Evaluate NF3 (C ⊑ ∃r.D) by inclusion loss ranking.
    ///
    /// For each test triple (C, r, D): for each candidate D, compute
    /// `inclusion_loss(C + bump_D, o_C, head_center, head_offset)` and rank.
    /// Lower inclusion loss = candidate D's bump brings C more inside the role head box.
    pub fn evaluate_nf3_by_inclusion(
        &self,
        test_axioms: &[(usize, usize, usize)],
    ) -> Result<(f32, f32, f32)> {
        let centers: Vec<f32> = self
            .concept_centers
            .as_tensor()
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let offsets: Vec<f32> = self
            .concept_offsets
            .as_tensor()
            .abs()?
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let bump_vecs: Vec<f32> = self
            .bumps
            .as_tensor()
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let role_heads_data: Vec<f32> = self
            .role_heads
            .as_tensor()
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let nc = self.num_concepts;
        let nr = self.num_roles;
        let dim = self.dim;
        let margin = self.margin;

        let items: Vec<[usize; 3]> = test_axioms.iter().map(|(a, b, c)| [*a, *b, *c]).collect();
        Ok(rank_evaluate(nc, &items, |&[sub, role, filler]| {
            if sub >= nc || filler >= nc || role >= nr {
                return None;
            }
            let sub_off = sub * dim;
            let rh_off = role * dim * 2;

            let scores: Vec<(usize, f32)> = (0..nc)
                .map(|d| {
                    let bump_off = d * dim;
                    let loss_sq: f32 = (0..dim)
                        .map(|i| {
                            let bumped_center = centers[sub_off + i] + bump_vecs[bump_off + i];
                            let head_center = role_heads_data[rh_off + i];
                            let head_offset = role_heads_data[rh_off + dim + i].abs();
                            let diff = (bumped_center - head_center).abs();
                            let v = (diff + offsets[sub_off + i] - head_offset - margin).max(0.0);
                            v * v
                        })
                        .sum();
                    (d, (loss_sq + 1e-8).sqrt())
                })
                .collect();
            Some((filler, scores))
        }))
    }

    /// Evaluate NF4 (∃r.C ⊑ D) by inclusion loss ranking.
    ///
    /// For each test triple (r, C, D): compute shifted head = head_r - bump_C,
    /// then rank all concepts D by `inclusion_loss(shifted_head, o_head, D_center, D_offset)`.
    pub fn evaluate_nf4_by_inclusion(
        &self,
        test_axioms: &[(usize, usize, usize)],
    ) -> Result<(f32, f32, f32)> {
        let centers: Vec<f32> = self
            .concept_centers
            .as_tensor()
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let offsets: Vec<f32> = self
            .concept_offsets
            .as_tensor()
            .abs()?
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let bump_vecs: Vec<f32> = self
            .bumps
            .as_tensor()
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let role_heads_data: Vec<f32> = self
            .role_heads
            .as_tensor()
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let nc = self.num_concepts;
        let nr = self.num_roles;
        let dim = self.dim;
        let margin = self.margin;

        let items: Vec<[usize; 3]> = test_axioms.iter().map(|(a, b, c)| [*a, *b, *c]).collect();
        Ok(rank_evaluate(nc, &items, |&[role, filler, target]| {
            if filler >= nc || target >= nc || role >= nr {
                return None;
            }
            let rh_off = role * dim * 2;
            let bump_off = filler * dim;

            let scores: Vec<(usize, f32)> = (0..nc)
                .map(|c| {
                    let c_off = c * dim;
                    let loss_sq: f32 = (0..dim)
                        .map(|i| {
                            let shifted_center =
                                role_heads_data[rh_off + i] - bump_vecs[bump_off + i];
                            let head_offset = role_heads_data[rh_off + dim + i].abs();
                            let diff = (shifted_center - centers[c_off + i]).abs();
                            let v = (diff + head_offset - offsets[c_off + i] - margin).max(0.0);
                            v * v
                        })
                        .sum();
                    (c, (loss_sq + 1e-8).sqrt())
                })
                .collect();
            Some((target, scores))
        }))
    }

    /// Run all four inclusion-based evaluations and print a comparison table
    /// against center-distance evaluation.
    ///
    /// Returns `(nf2, nf1, nf3, nf4)` inclusion results as `(h@1, h@10, mrr)` tuples.
    /// Prints a side-by-side table to stderr for quick comparison.
    #[allow(clippy::type_complexity)]
    pub fn evaluate_all_by_inclusion(
        &self,
        nf2_test: &[(usize, usize)],
        nf1_test: &[(usize, usize, usize)],
        nf3_test: &[(usize, usize, usize)],
        nf4_test: &[(usize, usize, usize)],
    ) -> Result<(
        (f32, f32, f32),
        (f32, f32, f32),
        (f32, f32, f32),
        (f32, f32, f32),
    )> {
        // Center-distance baselines
        let cd_nf2 = if nf2_test.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            self.evaluate_subsumption(nf2_test)?
        };
        let cd_nf1 = if nf1_test.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            self.evaluate_nf1(nf1_test)?
        };
        let cd_nf3 = if nf3_test.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            self.evaluate_nf3(nf3_test)?
        };
        let cd_nf4 = if nf4_test.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            self.evaluate_nf4(nf4_test)?
        };

        // Inclusion-based
        let inc_nf2 = if nf2_test.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            self.evaluate_subsumption_by_inclusion(nf2_test)?
        };
        let inc_nf1 = if nf1_test.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            self.evaluate_nf1_by_inclusion(nf1_test)?
        };
        let inc_nf3 = if nf3_test.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            self.evaluate_nf3_by_inclusion(nf3_test)?
        };
        let inc_nf4 = if nf4_test.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            self.evaluate_nf4_by_inclusion(nf4_test)?
        };

        eprintln!("  +---------+-----------------------------+-----------------------------+");
        eprintln!("  |  NF     |     Center-Distance         |       Inclusion-Loss        |");
        eprintln!("  |         |  H@1     H@10     MRR       |  H@1     H@10     MRR       |");
        eprintln!("  +---------+-----------------------------+-----------------------------+");
        for (label, cd, inc) in [
            ("NF2", cd_nf2, inc_nf2),
            ("NF1", cd_nf1, inc_nf1),
            ("NF3", cd_nf3, inc_nf3),
            ("NF4", cd_nf4, inc_nf4),
        ] {
            eprintln!(
                "  | {label:<7} | {:.3}    {:.3}    {:.4}    | {:.3}    {:.3}    {:.4}    |",
                cd.0, cd.1, cd.2, inc.0, inc.1, inc.2,
            );
        }
        eprintln!("  +---------+-----------------------------+-----------------------------+");

        Ok((inc_nf2, inc_nf1, inc_nf3, inc_nf4))
    }

    /// Save model weights to a safetensors file.
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let tensors: std::collections::HashMap<String, Tensor> = [
            (
                "concept_centers".to_string(),
                self.concept_centers.as_tensor().clone(),
            ),
            (
                "concept_offsets".to_string(),
                self.concept_offsets.as_tensor().clone(),
            ),
            ("bumps".to_string(), self.bumps.as_tensor().clone()),
            (
                "role_heads".to_string(),
                self.role_heads.as_tensor().clone(),
            ),
            (
                "role_tails".to_string(),
                self.role_tails.as_tensor().clone(),
            ),
        ]
        .into_iter()
        .collect();
        candle_core::safetensors::save(&tensors, path)?;
        Ok(())
    }

    /// Load model weights from a safetensors file.
    ///
    /// Validates that loaded tensor shapes match the trainer's
    /// `(num_concepts, dim)` and `(num_roles, dim * 2)` dimensions.
    pub fn load<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<()> {
        let tensors = candle_core::safetensors::load(path, &self.device)?;
        let get = |name: &str| -> Result<Tensor> {
            tensors
                .get(name)
                .cloned()
                .ok_or_else(|| candle_core::Error::Msg(format!("missing tensor: {name}")))
        };

        let check = |t: &Tensor, name: &str, expected: &[usize]| -> Result<()> {
            if t.dims() != expected {
                return Err(candle_core::Error::Msg(format!(
                    "{name}: expected shape {expected:?}, got {:?}",
                    t.dims()
                )));
            }
            Ok(())
        };

        let concept_shape = &[self.num_concepts, self.dim];
        let bump_shape = &[self.num_concepts, self.dim];
        let role_shape = &[self.num_roles, self.dim * 2];

        let cc = get("concept_centers")?;
        check(&cc, "concept_centers", concept_shape)?;
        let co = get("concept_offsets")?;
        check(&co, "concept_offsets", concept_shape)?;
        let b = get("bumps")?;
        check(&b, "bumps", bump_shape)?;
        let rh = get("role_heads")?;
        check(&rh, "role_heads", role_shape)?;
        let rt = get("role_tails")?;
        check(&rt, "role_tails", role_shape)?;

        self.concept_centers = Var::from_tensor(&cc)?;
        self.concept_offsets = Var::from_tensor(&co)?;
        self.bumps = Var::from_tensor(&b)?;
        self.role_heads = Var::from_tensor(&rh)?;
        self.role_tails = Var::from_tensor(&rt)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_el_trainer_creates() {
        let device = Device::Cpu;
        let trainer = CandleElTrainer::new(100, 5, 32, 0.1, 2.0, &device).unwrap();
        assert_eq!(trainer.num_concepts, 100);
        assert_eq!(trainer.num_roles, 5);
    }

    #[test]
    fn test_candle_el_trainer_fits() {
        let device = Device::Cpu;
        let trainer = CandleElTrainer::new(20, 3, 16, 0.1, 2.0, &device).unwrap();

        let mut ont = Ontology::new();
        for i in 0..20 {
            ont.concept(&format!("C{i}"));
        }
        for i in 0..3 {
            ont.role(&format!("R{i}"));
        }
        ont.axioms.push(Axiom::SubClassOf { sub: 0, sup: 1 });
        ont.axioms.push(Axiom::SubClassOf { sub: 2, sup: 3 });
        ont.axioms.push(Axiom::Intersection {
            c1: 0,
            c2: 2,
            target: 4,
        });
        ont.axioms.push(Axiom::ExistentialRight {
            sub: 5,
            role: 0,
            filler: 6,
        });

        let losses = trainer.fit(&ont, 50, 0.01, 4, 1, 0.0).unwrap();
        assert_eq!(losses.len(), 50);
        assert!(losses[0].is_finite());
        assert!(losses.last().unwrap() < &losses[0], "loss should decrease");
    }

    #[test]
    fn test_candle_el_eval_works() {
        let device = Device::Cpu;
        let trainer = CandleElTrainer::new(20, 2, 16, 0.1, 2.0, &device).unwrap();

        let mut ont = Ontology::new();
        for i in 0..20 {
            ont.concept(&format!("C{i}"));
        }
        ont.role("R0");
        ont.role("R1");
        for i in 0..15 {
            ont.axioms.push(Axiom::SubClassOf {
                sub: i,
                sup: (i + 1) % 20,
            });
        }

        let _losses = trainer.fit(&ont, 200, 0.01, 8, 2, 0.0).unwrap();

        let test_pairs: Vec<(usize, usize)> = ont
            .axioms
            .iter()
            .filter_map(|a| match a {
                Axiom::SubClassOf { sub, sup } => Some((*sub, *sup)),
                _ => None,
            })
            .collect();

        let (h1, h10, mrr) = trainer.evaluate_subsumption(&test_pairs).unwrap();
        assert!(
            mrr > 0.0,
            "MRR should be positive on training data, got {mrr}"
        );
        eprintln!("CandleElTrainer eval: H@1={h1:.3} H@10={h10:.3} MRR={mrr:.4}");
    }

    #[test]
    fn test_load_rejects_wrong_shape() {
        let device = Device::Cpu;
        let trainer = CandleElTrainer::new(10, 2, 8, 0.1, 2.0, &device).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");

        // Save with current (10, 8) shape
        trainer.save(&path).unwrap();

        // Create a new trainer with different dimensions
        let mut wrong = CandleElTrainer::new(20, 2, 8, 0.1, 2.0, &device).unwrap();
        let err = wrong.load(&path);
        assert!(err.is_err(), "load should reject shape mismatch");
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("expected shape"),
            "error should name the shape: {msg}"
        );
    }
}
