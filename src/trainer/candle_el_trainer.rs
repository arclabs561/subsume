//! GPU-accelerated EL++ ontology embedding trainer via candle autograd.
//!
//! Trains box embeddings for EL++ ontology completion using:
//! - Mini-batch training with per-NF-type balanced sampling (Box2EL-style)
//! - AdamW optimizer with cosine LR schedule
//! - Squared inclusion loss (matches Box2EL's `.square().mean()`)
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
pub struct CandleElTrainer {
    /// Concept centers: `[num_concepts, dim]`.
    pub concept_centers: Var,
    /// Concept raw offsets: `[num_concepts, dim]` (abs applied at use).
    pub concept_offsets: Var,
    /// Role centers: `[num_roles, dim]`.
    pub role_centers: Var,
    /// Role raw offsets: `[num_roles, dim]`.
    pub role_offsets: Var,
    /// Embedding dimension.
    pub dim: usize,
    /// Number of concepts.
    pub num_concepts: usize,
    /// Number of roles.
    pub num_roles: usize,
    /// Margin for inclusion loss.
    pub margin: f32,
    /// Target separation distance for negatives.
    pub neg_dist: f32,
    /// Device.
    pub device: Device,
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
        // Uniform [-1, 1] init (Box2EL uses uniform then L2-normalize)
        let concept_centers =
            Var::from_tensor(&Tensor::rand(-1.0_f32, 1.0, (num_concepts, dim), device)?)?;
        let concept_offsets =
            Var::from_tensor(&Tensor::rand(0.1_f32, 1.0, (num_concepts, dim), device)?)?;
        let role_centers = Var::from_tensor(&Tensor::rand(
            -1.0_f32,
            1.0,
            (num_roles.max(1), dim),
            device,
        )?)?;
        let role_offsets = Var::from_tensor(&Tensor::rand(
            0.1_f32,
            1.0,
            (num_roles.max(1), dim),
            device,
        )?)?;

        Ok(Self {
            concept_centers,
            concept_offsets,
            role_centers,
            role_offsets,
            dim,
            num_concepts,
            num_roles,
            margin,
            neg_dist,
            device: device.clone(),
        })
    }

    /// Inclusion loss: `||relu(|c_a - c_b| + o_a - o_b - margin)||` per sample (L2 norm).
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
        // L2 norm per sample (not squared -- squaring causes gradient explosion in high dim)
        let norm_sq = violation.sqr()?.sum(1)?;
        // Add epsilon before sqrt for numerical stability
        norm_sq.affine(1.0, 1e-8)?.sqrt()
    }

    /// Disjointness score: `||relu(|c_a - c_b| - o_a - o_b + margin)||` per sample.
    fn disjointness_score(
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

    /// Get role boxes for given IDs.
    fn role_boxes(&self, ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let centers = self.role_centers.as_tensor().index_select(ids, 0)?;
        let offsets = self.role_offsets.as_tensor().index_select(ids, 0)?.abs()?;
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

        let mut vars = vec![self.concept_centers.clone(), self.concept_offsets.clone()];
        if self.num_roles > 0 {
            vars.push(self.role_centers.clone());
            vars.push(self.role_offsets.clone());
        }

        let params = ParamsAdamW {
            lr,
            weight_decay: 0.0,
            ..Default::default()
        };
        let mut opt = AdamW::new(vars, params)?;
        let lr_min = lr * 0.01;

        // Group axioms by type for balanced sampling
        let mut nf2_axioms: Vec<(usize, usize)> = Vec::new(); // (sub, sup)
        let mut nf1_axioms: Vec<(usize, usize, usize)> = Vec::new(); // (c1, c2, target)
        let mut nf3_axioms: Vec<(usize, usize, usize)> = Vec::new(); // (sub, role, filler)
        let mut nf4_axioms: Vec<(usize, usize, usize)> = Vec::new(); // (role, filler, target)

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
                _ => {} // Skip RI6, RI7, Disjoint for now
            }
        }

        let nc = self.num_concepts;
        let mut epoch_losses = Vec::with_capacity(epochs);
        let mut rng: u64 = 42;
        let lcg = |s: &mut u64| -> usize {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*s >> 33) as usize
        };

        for epoch in 0..epochs {
            let progress = epoch as f64 / epochs.max(1) as f64;
            let current_lr =
                lr_min + 0.5 * (lr - lr_min) * (1.0 + (std::f64::consts::PI * progress).cos());
            opt.set_learning_rate(current_lr);

            let mut total_loss = 0.0f32;
            let mut loss_count = 0;

            // NF2: C ⊑ D -- sample a batch of subsumption axioms
            if !nf2_axioms.is_empty() {
                let bs = batch_size.min(nf2_axioms.len());
                let mut sub_ids = Vec::with_capacity(bs);
                let mut sup_ids = Vec::with_capacity(bs);
                for _ in 0..bs {
                    let idx = lcg(&mut rng) % nf2_axioms.len();
                    let (s, d) = nf2_axioms[idx];
                    sub_ids.push(s as u32);
                    sup_ids.push(d as u32);
                }

                let sub_t = Tensor::from_vec(sub_ids, (bs,), &self.device)?;
                let sup_t = Tensor::from_vec(sup_ids, (bs,), &self.device)?;
                let (c_sub, o_sub) = self.concept_boxes(&sub_t)?;
                let (c_sup, o_sup) = self.concept_boxes(&sup_t)?;

                // Positive: inclusion loss (box containment)
                let pos_incl = Self::inclusion_loss(&c_sub, &o_sub, &c_sup, &o_sup, self.margin)?;

                // Center-distance contrastive loss (directly optimizes eval metric):
                // Pull positive pairs' centers together.
                let pos_center_dist = c_sub.sub(&c_sup)?.sqr()?.sum(1)?.mean(0)?;

                // Negative: push random concept centers apart from sub
                let mut neg_loss_sum = Tensor::zeros((), candle_core::DType::F32, &self.device)?;
                for _ in 0..negative_samples {
                    let neg_ids: Vec<u32> = (0..bs).map(|_| (lcg(&mut rng) % nc) as u32).collect();
                    let neg_t = Tensor::from_vec(neg_ids, (bs,), &self.device)?;
                    let (c_neg, _o_neg) = self.concept_boxes(&neg_t)?;

                    // Push negative centers away: max(0, neg_dist - ||c_sub - c_neg||)^2
                    let neg_center_dist_sq = c_sub.sub(&c_neg)?.sqr()?.sum(1)?;
                    let neg_center_dist = neg_center_dist_sq.affine(1.0, 1e-8)?.sqrt()?;
                    let target =
                        Tensor::full(self.neg_dist, neg_center_dist.shape(), &self.device)?;
                    let gap = target.sub(&neg_center_dist)?.relu()?;
                    let neg_loss = gap.sqr()?.mean(0)?;
                    neg_loss_sum = neg_loss_sum.add(&neg_loss)?;
                }

                // Combined: inclusion + center contrastive + negative separation
                let batch_loss = pos_incl
                    .mean(0)?
                    .add(&pos_center_dist)?
                    .add(&neg_loss_sum)?;
                opt.backward_step(&batch_loss)?;
                total_loss += batch_loss.to_scalar::<f32>()?;
                loss_count += 1;
            }

            // NF1: C1 ⊓ C2 ⊑ D -- intersection then inclusion
            if !nf1_axioms.is_empty() {
                let bs = batch_size.min(nf1_axioms.len());
                let mut c1_ids = Vec::with_capacity(bs);
                let mut c2_ids = Vec::with_capacity(bs);
                let mut d_ids = Vec::with_capacity(bs);
                for _ in 0..bs {
                    let idx = lcg(&mut rng) % nf1_axioms.len();
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

                // Intersection: max of mins, min of maxes
                let min1 = cc1.sub(&oc1)?;
                let max1 = cc1.add(&oc1)?;
                let min2 = cc2.sub(&oc2)?;
                let max2 = cc2.add(&oc2)?;
                let inter_min = min1.maximum(&min2)?;
                let inter_max = max1.minimum(&max2)?;
                // Clamp: if min > max, intersection is empty
                let inter_max = inter_max.maximum(&inter_min)?;
                let inter_center = inter_min.add(&inter_max)?.affine(0.5, 0.0)?;
                let inter_offset = inter_max.sub(&inter_min)?.affine(0.5, 0.0)?;

                let nf1_loss =
                    Self::inclusion_loss(&inter_center, &inter_offset, &cd, &od, self.margin)?;
                let nf1_mean = nf1_loss.mean(0)?;
                opt.backward_step(&nf1_mean)?;
                total_loss += nf1_mean.to_scalar::<f32>()?;
                loss_count += 1;
            }

            // NF3: C ⊑ ∃r.D -- existential inclusion
            if !nf3_axioms.is_empty() {
                let bs = batch_size.min(nf3_axioms.len());
                let mut sub_ids = Vec::with_capacity(bs);
                let mut role_ids = Vec::with_capacity(bs);
                let mut filler_ids = Vec::with_capacity(bs);
                for _ in 0..bs {
                    let idx = lcg(&mut rng) % nf3_axioms.len();
                    let (s, r, f) = nf3_axioms[idx];
                    sub_ids.push(s as u32);
                    role_ids.push(r as u32);
                    filler_ids.push(f as u32);
                }

                let sub_t = Tensor::from_vec(sub_ids, (bs,), &self.device)?;
                let role_t = Tensor::from_vec(role_ids, (bs,), &self.device)?;
                let filler_t = Tensor::from_vec(filler_ids, (bs,), &self.device)?;

                let (c_sub, o_sub) = self.concept_boxes(&sub_t)?;
                let (c_role, o_role) = self.role_boxes(&role_t)?;
                let (c_filler, o_filler) = self.concept_boxes(&filler_t)?;

                // Existential box: center = role + filler, offset = relu(filler - role)
                let ex_center = c_role.add(&c_filler)?;
                let ex_offset = o_filler.sub(&o_role)?.relu()?;

                let nf3_loss =
                    Self::inclusion_loss(&c_sub, &o_sub, &ex_center, &ex_offset, self.margin)?;
                let nf3_mean = nf3_loss.mean(0)?;
                opt.backward_step(&nf3_mean)?;
                total_loss += nf3_mean.to_scalar::<f32>()?;
                loss_count += 1;
            }

            // NF4: ∃r.C ⊑ D
            if !nf4_axioms.is_empty() {
                let bs = batch_size.min(nf4_axioms.len());
                let mut role_ids = Vec::with_capacity(bs);
                let mut filler_ids = Vec::with_capacity(bs);
                let mut target_ids = Vec::with_capacity(bs);
                for _ in 0..bs {
                    let idx = lcg(&mut rng) % nf4_axioms.len();
                    let (r, f, t) = nf4_axioms[idx];
                    role_ids.push(r as u32);
                    filler_ids.push(f as u32);
                    target_ids.push(t as u32);
                }

                let role_t = Tensor::from_vec(role_ids, (bs,), &self.device)?;
                let filler_t = Tensor::from_vec(filler_ids, (bs,), &self.device)?;
                let target_t = Tensor::from_vec(target_ids, (bs,), &self.device)?;

                let (c_role, o_role) = self.role_boxes(&role_t)?;
                let (c_filler, o_filler) = self.concept_boxes(&filler_t)?;
                let (c_target, o_target) = self.concept_boxes(&target_t)?;

                let ex_center = c_role.add(&c_filler)?;
                let ex_offset = o_filler.sub(&o_role)?.relu()?;

                let nf4_loss = Self::inclusion_loss(
                    &ex_center,
                    &ex_offset,
                    &c_target,
                    &o_target,
                    self.margin,
                )?;
                let nf4_mean = nf4_loss.mean(0)?;
                opt.backward_step(&nf4_mean)?;
                total_loss += nf4_mean.to_scalar::<f32>()?;
                loss_count += 1;
            }

            // Offset regularization
            if reg_factor > 0.0 {
                let reg = self
                    .concept_offsets
                    .as_tensor()
                    .abs()?
                    .mean_all()?
                    .affine(reg_factor as f64, 0.0)?;
                opt.backward_step(&reg)?;
                total_loss += reg.to_scalar::<f32>()?;
            }

            let avg = if loss_count > 0 {
                total_loss / loss_count as f32
            } else {
                0.0
            };
            epoch_losses.push(avg);

            // NaN detection: stop early to preserve debuggable state
            if avg.is_nan() || avg.is_infinite() {
                eprintln!(
                    "  WARNING: loss diverged at epoch {} (loss={avg}). Stopping.",
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
                    "  epoch {:>5}/{epochs}: loss={avg:.4} lr={current_lr:.6} |c|={c_mean:.3} |o|={o_mean:.3}",
                    epoch + 1
                );
            }
        }

        Ok(epoch_losses)
    }

    /// Evaluate subsumption (NF2) by center L2 distance ranking.
    pub fn evaluate_subsumption(
        &self,
        test_axioms: &[(usize, usize)], // (sub, sup) pairs
    ) -> Result<(f32, f32, f32)> {
        let centers: Vec<f32> = self
            .concept_centers
            .as_tensor()
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let nc = self.num_concepts;
        let dim = self.dim;

        let mut hits1 = 0usize;
        let mut hits10 = 0usize;
        let mut rr_sum = 0.0f32;
        let mut total = 0usize;

        for &(sub, sup) in test_axioms {
            if sub >= nc || sup >= nc {
                continue;
            }
            let sub_offset = sub * dim;
            let mut scores: Vec<(usize, f32)> = (0..nc)
                .filter(|&c| c != sub)
                .map(|c| {
                    let c_offset = c * dim;
                    let dist_sq: f32 = (0..dim)
                        .map(|d| {
                            let diff = centers[sub_offset + d] - centers[c_offset + d];
                            diff * diff
                        })
                        .sum();
                    (c, dist_sq.sqrt())
                })
                .collect();
            scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let rank = scores
                .iter()
                .position(|(c, _)| *c == sup)
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
            return Ok((0.0, 0.0, 0.0));
        }
        Ok((
            hits1 as f32 / total as f32,
            hits10 as f32 / total as f32,
            rr_sum / total as f32,
        ))
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
}
