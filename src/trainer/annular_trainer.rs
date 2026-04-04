//! Annular sector trainer with finite-difference gradients.

use crate::annular::{AnnularRelation, AnnularSector};
use crate::dataset::Triple;
use crate::trainer::CpuBoxTrainingConfig;
use crate::BoxError;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

pub struct AnnularTrainer { rng: StdRng, step: usize }

impl AnnularTrainer {
    pub fn new(seed: u64) -> Self { Self { rng: StdRng::seed_from_u64(seed), step: 0 } }

    pub fn init_embeddings(&mut self, num_entities: usize, num_relations: usize) -> (Vec<AnnularSector>, Vec<AnnularRelation>) {
        let entities: Vec<AnnularSector> = (0..num_entities).map(|_| {
            let cre = self.rng.random_range(-0.1..0.1);
            let cim = self.rng.random_range(-0.1..0.1);
            let ri = self.rng.random_range(0.1..0.5);
            let ro = ri + self.rng.random_range(0.3..1.0);
            let ts = self.rng.random_range(0.0..std::f32::consts::PI);
            let te = ts + self.rng.random_range(0.5..std::f32::consts::PI);
            AnnularSector::new(cre, cim, ri, ro, ts, te).unwrap()
        }).collect();
        let relations: Vec<AnnularRelation> = (0..num_relations).map(|_| {
            AnnularRelation::new(self.rng.random_range(-0.1..0.1), self.rng.random_range(0.9..1.1), self.rng.random_range(0.9..1.1)).unwrap()
        }).collect();
        (entities, relations)
    }

    pub fn score_triple(head: &AnnularSector, relation: &AnnularRelation, tail: &AnnularSector) -> f32 {
        crate::annular::surface_distance(&relation.apply(head), tail)
    }

    fn compute_loss(head: &AnnularSector, rel: &AnnularRelation, tail: &AnnularSector, margin: f32) -> f32 {
        crate::annular::surface_distance(&rel.apply(head), tail)
    }

    fn compute_pair_gradients(head: &AnnularSector, rel: &AnnularRelation, tail: &AnnularSector, neg_tail: &AnnularSector, margin: f32) -> (f32, AnnularGradients) {
        let mut grads = AnnularGradients::new();
        let pos_score = Self::compute_loss(head, rel, tail, margin);
        let neg_score = Self::compute_loss(head, rel, neg_tail, margin);
        let loss = (margin + pos_score - neg_score).max(0.0);
        if loss <= 1e-8 { return (0.0, grads); }
        let eps = 1e-4f32;

        // Helper: perturb a single f32 field of a sector
        let perturb_sector = |s: &AnnularSector, field: usize, v: f32| -> Option<AnnularSector> {
            let vals = [s.center_re(), s.center_im(), s.r_inner(), s.r_outer(), s.theta_start(), s.theta_end()];
            let mut nv = vals; nv[field] = v;
            AnnularSector::new(nv[0], nv[1], nv[2], nv[3], nv[4], nv[5]).ok()
        };
        let perturb_rel = |r: &AnnularRelation, field: usize, v: f32| -> Option<AnnularRelation> {
            let vals = [r.rotation(), r.radial_scale(), r.angular_scale()];
            let mut nv = vals; nv[field] = v;
            AnnularRelation::new(nv[0], nv[1], nv[2]).ok()
        };

        // Head gradients
        for field in 0..6 {
            let vals = [head.center_re(), head.center_im(), head.r_inner(), head.r_outer(), head.theta_start(), head.theta_end()];
            let mut nv = vals; nv[field] += eps;
            if let Some(p) = perturb_sector(head, field, nv[field]) {
                let pl = Self::compute_loss(&p, rel, tail, margin);
                let nl = Self::compute_loss(&p, rel, neg_tail, margin);
                let lp = (margin + pl - nl).max(0.0);
                let g = (lp - loss) / eps;
                match field {
                    0 => grads.head_center_re = g,
                    1 => grads.head_center_im = g,
                    2 => grads.head_r_inner = g,
                    3 => grads.head_r_outer = g,
                    4 => grads.head_theta_start = g,
                    5 => grads.head_theta_end = g,
                    _ => {}
                }
            }
        }
        // Relation gradients
        for field in 0..3 {
            let vals = [rel.rotation(), rel.radial_scale(), rel.angular_scale()];
            let mut nv = vals; nv[field] += eps;
            if let Some(r) = perturb_rel(rel, field, nv[field]) {
                let pl = Self::compute_loss(head, &r, tail, margin);
                let nl = Self::compute_loss(head, &r, neg_tail, margin);
                let lp = (margin + pl - nl).max(0.0);
                let g = (lp - loss) / eps;
                match field {
                    0 => grads.rel_rotation = g,
                    1 => grads.rel_radial_scale = g,
                    2 => grads.rel_angular_scale = g,
                    _ => {}
                }
            }
        }
        // Tail gradients
        for field in 0..6 {
            let vals = [tail.center_re(), tail.center_im(), tail.r_inner(), tail.r_outer(), tail.theta_start(), tail.theta_end()];
            let mut nv = vals; nv[field] += eps;
            if let Some(p) = perturb_sector(tail, field, nv[field]) {
                let pl = Self::compute_loss(head, rel, &p, margin);
                let lp = (margin + pl - neg_score).max(0.0);
                let g = (lp - loss) / eps;
                match field {
                    0 => grads.tail_center_re = g,
                    1 => grads.tail_center_im = g,
                    2 => grads.tail_r_inner = g,
                    3 => grads.tail_r_outer = g,
                    4 => grads.tail_theta_start = g,
                    5 => grads.tail_theta_end = g,
                    _ => {}
                }
            }
        }
        // Neg tail gradients
        for field in 0..6 {
            let vals = [neg_tail.center_re(), neg_tail.center_im(), neg_tail.r_inner(), neg_tail.r_outer(), neg_tail.theta_start(), neg_tail.theta_end()];
            let mut nv = vals; nv[field] += eps;
            if let Some(p) = perturb_sector(neg_tail, field, nv[field]) {
                let nl = Self::compute_loss(head, rel, &p, margin);
                let lp = (margin + pos_score - nl).max(0.0);
                let g = (lp - loss) / eps;
                match field {
                    0 => grads.neg_tail_center_re = g,
                    1 => grads.neg_tail_center_im = g,
                    2 => grads.neg_tail_r_inner = g,
                    3 => grads.neg_tail_r_outer = g,
                    4 => grads.neg_tail_theta_start = g,
                    5 => grads.neg_tail_theta_end = g,
                    _ => {}
                }
            }
        }
        (loss, grads)
    }

    pub fn train_epoch(&mut self, entities: &mut [AnnularSector], relations: &mut [AnnularRelation], triples: &[Triple], config: &CpuBoxTrainingConfig, entity_to_idx: &HashMap<String, usize>, relation_to_idx: &HashMap<String, usize>) -> f32 {
        let num_entities = entities.len();
        let mut total_loss = 0.0f32; let mut count = 0usize;
        let lr = config.learning_rate; let beta1 = 0.9f32; let beta2 = 0.999f32; let eps = 1e-8f32;
        let mut indices: Vec<usize> = (0..triples.len()).collect();
        for i in (1..indices.len()).rev() { let j = self.rng.random_range(0..=i); indices.swap(i, j); }
        let mut m: HashMap<String, f32> = HashMap::new(); let mut v: HashMap<String, f32> = HashMap::new();

        for &idx in &indices {
            let triple = &triples[idx];
            let hi = match entity_to_idx.get(&triple.head) { Some(&i) => i, None => continue };
            let ri = match relation_to_idx.get(&triple.relation) { Some(&i) => i, None => continue };
            let ti = match entity_to_idx.get(&triple.tail) { Some(&i) => i, None => continue };
            let nti = loop { let n = self.rng.random_range(0..num_entities); if n != ti { break n; } };

            let head = &entities[hi]; let rel = &relations[ri];
            let tail = &entities[ti]; let neg_tail = &entities[nti];
            let (loss, grads) = Self::compute_pair_gradients(head, rel, tail, neg_tail, config.margin);
            total_loss += loss; count += 1;
            self.step += 1;
            let t = self.step as f32;
            let bias1 = 1.0 - beta1.powf(t); let bias2 = 1.0 - beta2.powf(t);

            apply_adam(&mut m, &mut v, &format!("h{hi}_cre"), entities[hi].center_re_mut(), grads.head_center_re, lr, beta1, beta2, eps, bias1, bias2);
            apply_adam(&mut m, &mut v, &format!("h{hi}_cim"), entities[hi].center_im_mut(), grads.head_center_im, lr, beta1, beta2, eps, bias1, bias2);
            apply_adam(&mut m, &mut v, &format!("h{hi}_ri"), entities[hi].r_inner_mut(), grads.head_r_inner, lr, beta1, beta2, eps, bias1, bias2);
            apply_adam(&mut m, &mut v, &format!("h{hi}_ro"), entities[hi].r_outer_mut(), grads.head_r_outer, lr, beta1, beta2, eps, bias1, bias2);
            apply_adam(&mut m, &mut v, &format!("h{hi}_ts"), entities[hi].theta_start_mut(), grads.head_theta_start, lr, beta1, beta2, eps, bias1, bias2);
            apply_adam(&mut m, &mut v, &format!("h{hi}_te"), entities[hi].theta_end_mut(), grads.head_theta_end, lr, beta1, beta2, eps, bias1, bias2);
            entities[hi].clamp_valid();

            apply_adam(&mut m, &mut v, &format!("r{ri}_rot"), relations[ri].rotation_mut(), grads.rel_rotation, lr, beta1, beta2, eps, bias1, bias2);
            apply_adam(&mut m, &mut v, &format!("r{ri}_rs"), relations[ri].radial_scale_mut(), grads.rel_radial_scale, lr, beta1, beta2, eps, bias1, bias2);
            apply_adam(&mut m, &mut v, &format!("r{ri}_as"), relations[ri].angular_scale_mut(), grads.rel_angular_scale, lr, beta1, beta2, eps, bias1, bias2);
            relations[ri].clamp_valid();

            apply_adam(&mut m, &mut v, &format!("t{ti}_cre"), entities[ti].center_re_mut(), grads.tail_center_re, lr, beta1, beta2, eps, bias1, bias2);
            apply_adam(&mut m, &mut v, &format!("t{ti}_cim"), entities[ti].center_im_mut(), grads.tail_center_im, lr, beta1, beta2, eps, bias1, bias2);
            apply_adam(&mut m, &mut v, &format!("t{ti}_ri"), entities[ti].r_inner_mut(), grads.tail_r_inner, lr, beta1, beta2, eps, bias1, bias2);
            apply_adam(&mut m, &mut v, &format!("t{ti}_ro"), entities[ti].r_outer_mut(), grads.tail_r_outer, lr, beta1, beta2, eps, bias1, bias2);
            apply_adam(&mut m, &mut v, &format!("t{ti}_ts"), entities[ti].theta_start_mut(), grads.tail_theta_start, lr, beta1, beta2, eps, bias1, bias2);
            apply_adam(&mut m, &mut v, &format!("t{ti}_te"), entities[ti].theta_end_mut(), grads.tail_theta_end, lr, beta1, beta2, eps, bias1, bias2);
            entities[ti].clamp_valid();

            apply_adam(&mut m, &mut v, &format!("nt{nti}_cre"), entities[nti].center_re_mut(), grads.neg_tail_center_re, lr, beta1, beta2, eps, bias1, bias2);
            apply_adam(&mut m, &mut v, &format!("nt{nti}_cim"), entities[nti].center_im_mut(), grads.neg_tail_center_im, lr, beta1, beta2, eps, bias1, bias2);
            apply_adam(&mut m, &mut v, &format!("nt{nti}_ri"), entities[nti].r_inner_mut(), grads.neg_tail_r_inner, lr, beta1, beta2, eps, bias1, bias2);
            apply_adam(&mut m, &mut v, &format!("nt{nti}_ro"), entities[nti].r_outer_mut(), grads.neg_tail_r_outer, lr, beta1, beta2, eps, bias1, bias2);
            apply_adam(&mut m, &mut v, &format!("nt{nti}_ts"), entities[nti].theta_start_mut(), grads.neg_tail_theta_start, lr, beta1, beta2, eps, bias1, bias2);
            apply_adam(&mut m, &mut v, &format!("nt{nti}_te"), entities[nti].theta_end_mut(), grads.neg_tail_theta_end, lr, beta1, beta2, eps, bias1, bias2);
            entities[nti].clamp_valid();
        }
        if count == 0 { 0.0 } else { total_loss / count as f32 }
    }

    pub fn evaluate(&self, entities: &[AnnularSector], relations: &[AnnularRelation], test_triples: &[crate::dataset::TripleIds], filter: Option<&crate::trainer::evaluation::FilteredTripleIndexIds>) -> crate::trainer::EvaluationResults {
        let num_entities = entities.len();
        let score = |h: usize, r: usize, t: usize| -> f32 {
            let dist = crate::annular::surface_distance(&relations[r].apply(&entities[h]), &entities[t]);
            (-dist).exp()
        };
        crate::trainer::evaluation::evaluate_link_prediction_generic(test_triples, num_entities, filter, score, score)
            .unwrap_or_else(|_| crate::trainer::EvaluationResults { mrr: 0.0, head_mrr: 0.0, tail_mrr: 0.0, hits_at_1: 0.0, hits_at_3: 0.0, hits_at_10: 0.0, mean_rank: f32::MAX, per_relation: vec![] })
    }
}

struct AnnularGradients {
    head_center_re: f32, head_center_im: f32, head_r_inner: f32, head_r_outer: f32,
    head_theta_start: f32, head_theta_end: f32,
    rel_rotation: f32, rel_radial_scale: f32, rel_angular_scale: f32,
    tail_center_re: f32, tail_center_im: f32, tail_r_inner: f32, tail_r_outer: f32,
    tail_theta_start: f32, tail_theta_end: f32,
    neg_tail_center_re: f32, neg_tail_center_im: f32, neg_tail_r_inner: f32, neg_tail_r_outer: f32,
    neg_tail_theta_start: f32, neg_tail_theta_end: f32,
}
impl AnnularGradients {
    fn new() -> Self { Self {
        head_center_re: 0.0, head_center_im: 0.0, head_r_inner: 0.0, head_r_outer: 0.0,
        head_theta_start: 0.0, head_theta_end: 0.0, rel_rotation: 0.0, rel_radial_scale: 0.0, rel_angular_scale: 0.0,
        tail_center_re: 0.0, tail_center_im: 0.0, tail_r_inner: 0.0, tail_r_outer: 0.0,
        tail_theta_start: 0.0, tail_theta_end: 0.0, neg_tail_center_re: 0.0, neg_tail_center_im: 0.0,
        neg_tail_r_inner: 0.0, neg_tail_r_outer: 0.0, neg_tail_theta_start: 0.0, neg_tail_theta_end: 0.0,
    }}
}

fn apply_adam(m: &mut HashMap<String, f32>, v: &mut HashMap<String, f32>, key: &str, param: &mut f32, grad: f32, lr: f32, beta1: f32, beta2: f32, eps: f32, bias1: f32, bias2: f32) {
    let m_val = m.entry(key.to_string()).or_insert(0.0);
    let v_val = v.entry(key.to_string()).or_insert(0.0);
    *m_val = beta1 * *m_val + (1.0 - beta1) * grad;
    *v_val = beta2 * *v_val + (1.0 - beta2) * grad * grad;
    let m_hat = *m_val / bias1;
    let v_hat = (*v_val / bias2).max(0.0);
    *param -= lr * m_hat / (v_hat.sqrt() + eps);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{TripleIds, Vocab};

    #[test]
    fn trainer_init() {
        let mut t = AnnularTrainer::new(42);
        let (e, r) = t.init_embeddings(10, 3);
        assert_eq!(e.len(), 10); assert_eq!(r.len(), 3);
    }

    #[test]
    fn score_identical_is_low() {
        let h = AnnularSector::new(0.0, 0.0, 0.5, 1.5, 0.0, std::f32::consts::PI).unwrap();
        let r = AnnularRelation::identity();
        let t = AnnularSector::new(0.0, 0.0, 0.5, 1.5, 0.0, std::f32::consts::PI).unwrap();
        assert!(AnnularTrainer::score_triple(&h, &r, &t) < 0.1);
    }

    #[test]
    fn score_disjoint_is_high() {
        let h = AnnularSector::new(0.0, 0.0, 0.5, 1.0, 0.0, 0.5).unwrap();
        let r = AnnularRelation::identity();
        let t = AnnularSector::new(10.0, 0.0, 2.0, 3.0, std::f32::consts::PI, std::f32::consts::PI + 0.5).unwrap();
        assert!(AnnularTrainer::score_triple(&h, &r, &t) > 5.0);
    }

    #[test]
    fn gradients_are_finite() {
        let h = AnnularSector::new(0.0, 0.0, 0.5, 1.5, 0.0, std::f32::consts::PI).unwrap();
        let r = AnnularRelation::new(0.1, 1.0, 1.0).unwrap();
        let t = AnnularSector::new(0.0, 0.0, 0.5, 1.5, 0.0, std::f32::consts::PI).unwrap();
        let nt = AnnularSector::new(5.0, 0.0, 0.5, 1.0, std::f32::consts::PI, std::f32::consts::PI + 0.5).unwrap();
        let (loss, grads) = AnnularTrainer::compute_pair_gradients(&h, &r, &t, &nt, 1.0);
        assert!(loss.is_finite());
        assert!(grads.head_center_re.is_finite());
        assert!(grads.rel_rotation.is_finite());
    }

    #[test]
    fn train_epoch_runs() {
        let mut t = AnnularTrainer::new(42);
        let (mut e, mut r) = t.init_embeddings(20, 3);
        let triples = vec![Triple { head: "e0".to_string(), relation: "r0".to_string(), tail: "e1".to_string() }];
        let em: HashMap<String, usize> = [("e0".into(), 0), ("e1".into(), 1)].into_iter().collect();
        let rm: HashMap<String, usize> = [("r0".into(), 0)].into_iter().collect();
        let cfg = CpuBoxTrainingConfig { learning_rate: 0.01, margin: 1.0, ..Default::default() };
        assert!(t.train_epoch(&mut e, &mut r, &triples, &cfg, &em, &rm).is_finite());
    }

    #[test]
    fn train_and_evaluate_synthetic() {
        let mut vocab = Vocab::default();
        let e0 = vocab.intern("e0".to_string()); let e1 = vocab.intern("e1".to_string());
        let e2 = vocab.intern("e2".to_string()); let e3 = vocab.intern("e3".to_string());
        let triples = vec![
            Triple { head: "e0".to_string(), relation: "r0".to_string(), tail: "e1".to_string() },
            Triple { head: "e2".to_string(), relation: "r0".to_string(), tail: "e3".to_string() },
            Triple { head: "e0".to_string(), relation: "r1".to_string(), tail: "e2".to_string() },
        ];
        let test = vec![TripleIds { head: e0, relation: 0, tail: e1 }, TripleIds { head: e2, relation: 0, tail: e3 }, TripleIds { head: e0, relation: 1, tail: e2 }];
        let em: HashMap<String, usize> = [("e0".into(), 0), ("e1".into(), 1), ("e2".into(), 2), ("e3".into(), 3)].into_iter().collect();
        let rm: HashMap<String, usize> = [("r0".into(), 0), ("r1".into(), 1)].into_iter().collect();
        let mut t = AnnularTrainer::new(42);
        let (mut e, mut r) = t.init_embeddings(4, 2);
        let cfg = CpuBoxTrainingConfig { learning_rate: 0.05, margin: 1.0, ..Default::default() };
        let mut last_loss = f32::MAX;
        for epoch in 0..50 {
            let loss = t.train_epoch(&mut e, &mut r, &triples, &cfg, &em, &rm);
            if epoch % 10 == 0 { eprintln!("Annular Epoch {epoch}: loss={loss:.4}"); }
            last_loss = loss;
        }
        eprintln!("Annular Final loss: {last_loss:.4}");
        let results = t.evaluate(&e, &r, &test, None);
        assert!(results.mrr > 0.3, "Annular MRR = {}", results.mrr);
        assert!(results.mean_rank < 3.0, "Annular Mean rank = {}", results.mean_rank);
    }
}
