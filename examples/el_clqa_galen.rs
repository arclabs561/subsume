//! Real-ontology faithful-vs-plain CLQA benchmark (data-gated eval).
//!
//! The el_clqa_benchmark synthetic taxonomy shows the mechanism; this runs the
//! same faithful-boxes-vs-plain-TransE head-to-head on a REAL ontology (GALEN
//! by default: ~24k concepts, a real multiple-inheritance DAG), making the
//! comparison a result rather than an illustration.
//!
//! Conjunctive queries "X such that A subset-of X AND B subset-of X" are sampled
//! from concept pairs sharing a non-trivial common ancestor in the deductive
//! closure; the certain answers are the common ancestors, and the least common
//! ancestor (the deepest, most specific shared superclass) is the key target.
//! Both models rank every concept by min(deg(A subset-of X), deg(B subset-of
//! X)); the box model uses containment, TransE uses its isa translation.
//!
//! FINDING (2026-07-04): the clean synthetic-tree advantage (el_clqa_benchmark,
//! boxes +0.48 MRR) does NOT transfer to real GALEN at this budget. Both models
//! are weak (LCA MRR 0.02-0.05); TransE narrowly wins, and the box model's
//! top-1 is never a valid common ancestor (0.00 vs TransE 0.09). On a messy
//! multiple-inheritance DAG the box model (dim 100, 400 epochs) develops
//! spurious/degenerate containments the conjunctive-LCA query surfaces, whereas
//! the synthetic tree nests perfectly. The synthetic benchmark shows the
//! transitivity MECHANISM; real-data eval shows it does not dominate in
//! practice without much stronger box training. Reported honestly, not asserted.
//!
//! Data-gated: exits 0 with a message if the dataset is absent (fetch GALEN via
//! the Box2EL conversion the el_benchmark examples describe). Runs on burn Metal
//! (wgpu) via cfg; large batch amortizes wgpu dispatch overhead; the eval loop
//! is rayon-parallel. Run:
//! `DATASET=GALEN cargo run --release --features burn-ndarray,burn-wgpu --example el_clqa_galen`

use rayon::prelude::*;
use std::collections::HashSet;
use std::path::Path;
use subsume::el_dataset::load_el_axioms;
use subsume::el_training::{Axiom, Ontology};
use subsume::trainer::burn_el_trainer::{BurnElConfig, BurnElTrainer};
use tranz::burn_train::{train_kge, BurnModelType, BurnTrainConfig};
use tranz::dataset::TripleIds;

#[cfg(feature = "burn-wgpu")]
type Backend = burn::backend::Autodiff<burn_wgpu::Wgpu>;
#[cfg(all(feature = "burn-ndarray", not(feature = "burn-wgpu")))]
type Backend = burn::backend::Autodiff<burn_ndarray::NdArray>;

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

fn transe_degree(emb: &[Vec<f32>], rel: &[f32], a: usize, b: usize) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..rel.len() {
        let v = emb[a][i] + rel[i] - emb[b][i];
        acc += v * v;
    }
    (-acc.sqrt()).exp()
}

struct CQuery {
    a: usize,
    b: usize,
    common: HashSet<usize>,
    lca: usize,
}

/// (LCA MRR, LCA Hits@10, top-1-is-valid-CA) over all queries. Queries are
/// independent, so the per-query ranking (the GALEN-scale cost: O(queries * n))
/// is parallelized across cores with rayon.
fn score_model<F: Fn(usize, usize) -> f32 + Sync>(
    queries: &[CQuery],
    deg: F,
    n: usize,
) -> (f64, f64, f64) {
    let (rr, hits10, top1) = queries
        .par_iter()
        .map(|q| {
            let mut scored: Vec<(usize, f32)> = (0..n)
                .filter(|&x| x != q.a && x != q.b)
                .map(|x| (x, deg(q.a, x).min(deg(q.b, x))))
                .collect();
            scored.sort_by(|p, r| r.1.partial_cmp(&p.1).unwrap());
            let rank = 1 + scored.iter().position(|&(x, _)| x == q.lca).unwrap();
            (
                1.0 / rank as f64,
                usize::from(rank <= 10),
                usize::from(q.common.contains(&scored[0].0)),
            )
        })
        .reduce(
            || (0.0f64, 0usize, 0usize),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
        );
    let nq = queries.len() as f64;
    (rr / nq, hits10 as f64 / nq, top1 as f64 / nq)
}

fn main() {
    let device = Default::default();
    <Backend as burn::tensor::backend::Backend>::seed(&device, 3);
    let dataset = std::env::var("DATASET").unwrap_or_else(|_| "GALEN".to_string());
    let dim: usize = std::env::var("DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let epochs: usize = std::env::var("EPOCHS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(400);
    let n_queries: usize = std::env::var("QUERIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);
    // Large batch: on wgpu/Metal, small batches are dispatch-bound (kernel-launch
    // overhead dominates); Box2EL/DELE use batch ~32k. Fewer, larger dispatches
    // keep the GPU busy. LR is scaled up to match the larger batch.
    let batch: usize = std::env::var("BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);
    let lr: f64 = std::env::var("LR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.05);

    let train_path = Path::new("data").join(&dataset).join("train.tsv");
    if !train_path.exists() {
        println!(
            "data/{dataset}/train.tsv not found; skipping real-ontology eval (data-gated).\n\
             Fetch the Box2EL EL++ datasets to run this (see el_benchmark examples)."
        );
        return;
    }

    let ds = match load_el_axioms(&train_path) {
        Ok(d) => d,
        Err(e) => {
            println!("failed to load {train_path:?}: {e}; skipping.");
            return;
        }
    };
    let ont = Ontology::from_el_dataset(&ds);
    let n = ont.concept_names.len();
    println!(
        "{dataset}: {n} concepts, {} axioms, dim {dim}, {epochs} epochs",
        ont.axioms.len()
    );

    // --- Faithful boxes ---
    let box_cfg = BurnElConfig {
        dim,
        epochs,
        lr,
        negative_samples: 2,
        margin: 0.1,
        batch_size: batch,
        ..Default::default()
    };
    let trainer = BurnElTrainer::<Backend>::new();
    let mut model =
        BurnElTrainer::<Backend>::init_model(n, ont.role_names.len().max(1), dim, &device);
    trainer.fit(&mut model, &ont, &box_cfg, &device);
    let centers = BurnElTrainer::<Backend>::extract_centers(&model, &device);
    let offsets = BurnElTrainer::<Backend>::extract_offsets(&model, &device);

    // --- Plain TransE on the atomic subsumption triples ---
    let triples: Vec<TripleIds> = ont
        .axioms
        .iter()
        .filter_map(|ax| match ax {
            Axiom::SubClassOf { sub, sup } => Some(TripleIds::new(*sub, 0, *sup)),
            _ => None,
        })
        .collect();
    let transe_cfg = BurnTrainConfig {
        dim,
        epochs,
        batch_size: batch,
        lr,
        ..BurnTrainConfig::default()
    };
    let kge = train_kge::<Backend>(&triples, n, 1, BurnModelType::TransE, &transe_cfg, &device);
    let emb = &kge.entity_vecs;
    let rel = &kge.relation_vecs[0];

    // Ancestor sets from the deductive closure.
    let closure = ont.subsumption_closure();
    let mut anc: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for &(s, a) in &closure {
        if s < n && a < n {
            anc[s].insert(a);
        }
    }

    // Sample concept pairs with a non-trivial common ancestor (LCA depth >= 2),
    // deterministically via an LCG so the query set is reproducible.
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut lcg = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as usize
    };
    let depth = |x: usize| anc[x].len();
    let mut queries: Vec<CQuery> = Vec::new();
    let mut attempts = 0;
    while queries.len() < n_queries && attempts < n_queries * 400 {
        attempts += 1;
        let (a, b) = (lcg() % n, lcg() % n);
        if a == b || anc[a].is_empty() || anc[b].is_empty() {
            continue;
        }
        let common: HashSet<usize> = anc[a].intersection(&anc[b]).copied().collect();
        let lca = match common.iter().max_by_key(|&&x| depth(x)) {
            Some(&l) if depth(l) >= 2 => l,
            _ => continue,
        };
        queries.push(CQuery { a, b, common, lca });
    }

    let (box_mrr, box_h10, box_t1) = score_model(
        &queries,
        |a, x| box_degree(&centers, &offsets, a, x, dim),
        n,
    );
    let (te_mrr, te_h10, te_t1) = score_model(&queries, |a, x| transe_degree(emb, rel, a, x), n);

    println!("\n{} conjunctive queries (LCA depth >= 2)\n", queries.len());
    println!(
        "{:<22} {:>8} {:>8} {:>8}",
        "model", "LCA MRR", "Hits@10", "top1CA"
    );
    println!("{}", "-".repeat(48));
    println!(
        "{:<22} {box_mrr:>8.3} {box_h10:>8.3} {box_t1:>8.3}",
        "faithful boxes"
    );
    println!(
        "{:<22} {te_mrr:>8.3} {te_h10:>8.3} {te_t1:>8.3}",
        "plain KGE (TransE)"
    );
    let delta = box_mrr - te_mrr;
    println!(
        "\nLCA MRR delta (faithful - plain): {delta:+.3}  ({} on real {dataset})",
        if delta > 0.0 {
            "faithful wins"
        } else {
            "plain wins"
        }
    );
}
