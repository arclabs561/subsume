//! A closure-grounded complex-query benchmark: faithful EL++ boxes vs plain KGE.
//!
//! Evaluates many conjunctive queries whose CERTAIN answers come from the
//! ontology's deductive closure, comparing two atomic scorers on the identical
//! queries: a faithful EL++ geometric model (subsume box containment) and a
//! plain point-embedding KGE (tranz TransE trained on the same subsumption
//! triples). This is the faithful-vs-plain head-to-head the EL/CLQA literature
//! lacks: EL embedders score only atomic C ⊑ D, and complex-query benchmarks
//! (E-OMQA for DL-Lite, EFOk-CQA for plain KGs) are not over EL++ models.
//!
//! Task: for a leaf pair (A, B), "X such that A ⊑ X AND B ⊑ X" has certain
//! answers = the common ancestors from the transitive subsumption closure. Rank
//! every concept by the graded conjunction min(deg(A ⊑ X), deg(B ⊑ X)) and
//! measure how the least common ancestor (LCA) and the certain-answer set rank.
//!
//! The expected gap: box containment is transitive by geometry (A ⊆ B ⊆ C
//! implies A ⊆ C for free), so it answers deep-ancestor queries the model was
//! never directly trained on; TransE's single isa translation composes only
//! approximately over multiple hops. TransE training shuffles with a system RNG
//! (non-deterministic), so only the seeded box metrics are asserted; TransE is
//! reported for comparison.
//!
//! CAVEAT: this clean-tree advantage does NOT transfer to real GALEN (see
//! el_clqa_galen): on a messy multiple-inheritance DAG at a realistic budget
//! both models are weak and TransE narrowly wins. This benchmark demonstrates
//! the transitivity MECHANISM on a hierarchy where box containment is exact; it
//! is not a general claim that faithful boxes beat plain KGE in practice.
//!
//! Run: `cargo run --features burn-ndarray --example el_clqa_benchmark`

use std::collections::HashSet;
use subsume::el_training::{Axiom, Ontology};
use subsume::trainer::burn_el_trainer::{BurnElConfig, BurnElTrainer};
use tranz::burn_train::{train_kge, BurnModelType, BurnTrainConfig};
use tranz::dataset::TripleIds;

type Backend = burn::backend::Autodiff<burn_ndarray::NdArray>;

/// Faithful box containment `A ⊆ B` as a degree in `[0, 1]`.
fn box_degree(centers: &[f32], offsets: &[f32], a: usize, b: usize, dim: usize) -> f32 {
    let (ao, bo) = (a * dim, b * dim);
    let mut acc = 0.0f32;
    for i in 0..dim {
        let v = ((centers[ao + i] - centers[bo + i]).abs() + offsets[ao + i] - offsets[bo + i])
            .max(0.0);
        acc += v * v;
    }
    (-acc.sqrt()).exp()
}

/// Plain TransE score for `A ⊑ B` (relation isa): `exp(-‖emb[a] + rel - emb[b]‖)`.
fn transe_degree(emb: &[Vec<f32>], rel: &[f32], a: usize, b: usize) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..rel.len() {
        let v = emb[a][i] + rel[i] - emb[b][i];
        acc += v * v;
    }
    (-acc.sqrt()).exp()
}

/// Precomputed conjunctive query: two leaves, their certain common ancestors,
/// and the least common ancestor (the deepest, tightest certain answer).
struct CQuery {
    a: usize,
    b: usize,
    common: HashSet<usize>,
    lca: usize,
}

/// Score one model on all queries: (LCA MRR, LCA Hits@3, top-1-is-valid-CA).
fn score_model<F: Fn(usize, usize) -> f32>(
    queries: &[CQuery],
    deg: F,
    n: usize,
) -> (f64, f64, f64) {
    let (mut rr, mut hits3, mut top1) = (0.0f64, 0usize, 0usize);
    for q in queries {
        let mut scored: Vec<(usize, f32)> = (0..n)
            .filter(|&x| x != q.a && x != q.b)
            .map(|x| (x, deg(q.a, x).min(deg(q.b, x))))
            .collect();
        scored.sort_by(|p, r| r.1.partial_cmp(&p.1).unwrap());
        let rank = 1 + scored.iter().position(|&(x, _)| x == q.lca).unwrap();
        rr += 1.0 / rank as f64;
        if rank <= 3 {
            hits3 += 1;
        }
        if q.common.contains(&scored[0].0) {
            top1 += 1;
        }
    }
    let nq = queries.len() as f64;
    (rr / nq, hits3 as f64 / nq, top1 as f64 / nq)
}

fn main() {
    let device = Default::default();
    <Backend as burn::tensor::backend::Backend>::seed(&device, 3);
    let dim = 40;

    // Balanced 4-level taxonomy: root(1) > categories(3) > subcategories(9) >
    // leaves(18) = 31 concepts.
    let n = 31;
    let parent = |c: usize| -> Option<usize> {
        match c {
            0 => None,
            1..=3 => Some(0),
            4..=12 => Some(1 + (c - 4) / 3),
            _ => Some(4 + (c - 13) / 2),
        }
    };

    let mut ont = Ontology::new();
    for c in 0..n {
        ont.concept(&format!("C{c}"));
    }
    for c in 1..n {
        ont.axioms.push(Axiom::SubClassOf {
            sub: c,
            sup: parent(c).unwrap(),
        });
    }
    let mut by_parent: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for c in 1..n {
        by_parent.entry(parent(c).unwrap()).or_default().push(c);
    }
    for sibs in by_parent.values() {
        for i in 0..sibs.len() {
            for j in (i + 1)..sibs.len() {
                ont.axioms.push(Axiom::Disjoint {
                    a: sibs[i],
                    b: sibs[j],
                });
            }
        }
    }
    let closure: HashSet<(usize, usize)> = ont.subsumption_closure();

    // --- Faithful model: subsume EL++ box trainer ---
    let box_cfg = BurnElConfig {
        dim,
        epochs: 2000,
        lr: 0.05,
        negative_samples: 5,
        margin: 0.1,
        batch_size: 32,
        ..Default::default()
    };
    let trainer = BurnElTrainer::<Backend>::new();
    let mut model = BurnElTrainer::<Backend>::init_model(n, 0, dim, &device);
    trainer.fit(&mut model, &ont, &box_cfg, &device);
    let centers = BurnElTrainer::<Backend>::extract_centers(&model, &device);
    let offsets = BurnElTrainer::<Backend>::extract_offsets(&model, &device);

    // --- Plain KGE baseline: tranz TransE on the same direct subsumption triples ---
    let triples: Vec<TripleIds> = (1..n)
        .map(|c| TripleIds::new(c, 0, parent(c).unwrap()))
        .collect();
    let transe_cfg = BurnTrainConfig {
        dim,
        epochs: 2000,
        batch_size: 32,
        lr: 0.01,
        ..BurnTrainConfig::default()
    };
    let kge = train_kge::<Backend>(&triples, n, 1, BurnModelType::TransE, &transe_cfg, &device);
    let emb = &kge.entity_vecs;
    let rel = &kge.relation_vecs[0];

    // Build the conjunctive queries once (shared by both models).
    let ancestors =
        |c: usize| -> HashSet<usize> { (0..n).filter(|&x| closure.contains(&(c, x))).collect() };
    let depth = |c: usize| -> usize { (0..n).filter(|&x| closure.contains(&(c, x))).count() };
    let leaves: Vec<usize> = (13..n).collect();
    let mut queries: Vec<CQuery> = Vec::new();
    for (i, &a) in leaves.iter().enumerate() {
        for &b in &leaves[i + 1..] {
            let common: HashSet<usize> =
                ancestors(a).intersection(&ancestors(b)).copied().collect();
            if common.is_empty() {
                continue;
            }
            let lca = *common.iter().max_by_key(|&&x| depth(x)).unwrap();
            queries.push(CQuery { a, b, common, lca });
        }
    }

    let (box_mrr, box_h3, box_t1) = score_model(
        &queries,
        |a, x| box_degree(&centers, &offsets, a, x, dim),
        n,
    );
    let (te_mrr, te_h3, te_t1) = score_model(&queries, |a, x| transe_degree(emb, rel, a, x), n);

    println!(
        "{} conjunctive queries over a 31-concept taxonomy (certain answers from the closure)\n",
        queries.len()
    );
    println!(
        "{:<22} {:>8} {:>8} {:>8}",
        "model", "LCA MRR", "Hits@3", "top1CA"
    );
    println!("{}", "-".repeat(48));
    println!(
        "{:<22} {box_mrr:>8.3} {box_h3:>8.3} {box_t1:>8.3}",
        "faithful boxes"
    );
    println!(
        "{:<22} {te_mrr:>8.3} {te_h3:>8.3} {te_t1:>8.3}",
        "plain KGE (TransE)"
    );

    // The faithful boxes should recover closure answers well above chance
    // (random LCA MRR over ~29 candidates ~= 0.13). Only the seeded box metrics
    // are asserted; TransE shuffles non-deterministically and is reported only.
    assert!(
        box_mrr > 0.35,
        "box LCA MRR {box_mrr:.3} should beat chance (~0.13)"
    );
    assert!(
        box_t1 > 0.6,
        "box top-1 valid-CA {box_t1:.3} should be high"
    );
    let delta = box_mrr - te_mrr;
    if delta > 0.0 {
        println!(
            "\nFaithful boxes beat plain TransE on LCA MRR by {delta:+.3}. Box containment is\n\
             transitive by geometry (A ⊆ B ⊆ C implies A ⊆ C), so deep-ancestor queries the\n\
             model was never directly trained on are answered for free; TransE's single isa\n\
             translation composes only approximately over multiple hops."
        );
    } else {
        println!(
            "\nUnexpected: TransE matched or beat the boxes (delta {delta:+.3}); investigate."
        );
    }
}
