//! A closure-grounded complex-query benchmark over trained EL++ boxes.
//!
//! el_clqa_trained hand-picks two queries; this evaluates *many* conjunctive
//! queries whose CERTAIN answers come from the ontology's deductive closure,
//! reporting a ranking metric. It is the missing EL++ complex-query benchmark:
//! EL embedders (Box2EL, DELE) score only atomic C ⊑ D, and the complex-query
//! benchmarks (E-OMQA for DL-Lite, EFOk-CQA for plain KGs) are not over EL++
//! geometric models.
//!
//! Task: for a leaf pair (A, B), the query "X such that A ⊑ X AND B ⊑ X" has
//! certain answers = the common ancestors of A and B (from the transitive
//! subsumption closure). We rank every concept by the graded conjunction
//! min(deg(A ⊑ X), deg(B ⊑ X)) and measure how the least common ancestor (LCA,
//! the tightest certain answer) and the full certain-answer set rank.
//!
//! Run: `cargo run --features burn-ndarray --example el_clqa_benchmark`

use std::collections::HashSet;
use subsume::el_training::{Axiom, Ontology};
use subsume::trainer::burn_el_trainer::{BurnElConfig, BurnElTrainer};

type Backend = burn::backend::Autodiff<burn_ndarray::NdArray>;

fn subsumption_degree(centers: &[f32], offsets: &[f32], a: usize, b: usize, dim: usize) -> f32 {
    let (ao, bo) = (a * dim, b * dim);
    let mut acc = 0.0f32;
    for i in 0..dim {
        let v = ((centers[ao + i] - centers[bo + i]).abs() + offsets[ao + i] - offsets[bo + i])
            .max(0.0);
        acc += v * v;
    }
    (-acc.sqrt()).exp()
}

fn main() {
    let device = Default::default();
    <Backend as burn::tensor::backend::Backend>::seed(&device, 3);
    let dim = 40;

    // Balanced 4-level taxonomy: root(1) > categories(3) > subcategories(9) >
    // leaves(18) = 31 concepts. parent() encodes the tree.
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
    // Disjointness between direct siblings (same parent), so nesting is
    // discriminative rather than collapsing to a point.
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

    // Deductive closure = certain (sub, ancestor) subsumptions.
    let closure: HashSet<(usize, usize)> = ont.subsumption_closure();

    let config = BurnElConfig {
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
    let losses = trainer.fit(&mut model, &ont, &config, &device);
    let centers = BurnElTrainer::<Backend>::extract_centers(&model, &device);
    let offsets = BurnElTrainer::<Backend>::extract_offsets(&model, &device);
    let deg = |a: usize, b: usize| subsumption_degree(&centers, &offsets, a, b, dim);
    println!(
        "Trained {n} concepts, dim {dim}, {} epochs: loss {:.3} -> {:.3}",
        config.epochs,
        losses.first().copied().unwrap_or(0.0),
        losses.last().copied().unwrap_or(0.0)
    );

    // Every leaf pair is a conjunctive query. Certain answers = common
    // ancestors from the closure; the LCA is the deepest (fewest descendants).
    let leaves: Vec<usize> = (13..n).collect();
    let ancestors =
        |c: usize| -> HashSet<usize> { (0..n).filter(|&x| closure.contains(&(c, x))).collect() };
    let depth = |c: usize| -> usize {
        // Number of proper ancestors = depth in the tree.
        (0..n).filter(|&x| closure.contains(&(c, x))).count()
    };

    let mut lca_rr = 0.0f64; // reciprocal rank of the LCA
    let mut lca_hits3 = 0usize;
    let mut top1_valid = 0usize; // top-ranked concept is a common ancestor
    let mut n_queries = 0usize;

    for (i, &a) in leaves.iter().enumerate() {
        for &b in &leaves[i + 1..] {
            let common: HashSet<usize> =
                ancestors(a).intersection(&ancestors(b)).copied().collect();
            if common.is_empty() {
                continue;
            }
            // LCA = the common ancestor with the greatest depth.
            let lca = *common.iter().max_by_key(|&&x| depth(x)).unwrap();

            // Rank all candidate concepts by the graded conjunction, excluding
            // the two query concepts themselves.
            let mut scored: Vec<(usize, f32)> = (0..n)
                .filter(|&x| x != a && x != b)
                .map(|x| (x, deg(a, x).min(deg(b, x))))
                .collect();
            scored.sort_by(|p, q| q.1.partial_cmp(&p.1).unwrap());

            let lca_rank = 1 + scored.iter().position(|&(x, _)| x == lca).unwrap();
            lca_rr += 1.0 / lca_rank as f64;
            if lca_rank <= 3 {
                lca_hits3 += 1;
            }
            if common.contains(&scored[0].0) {
                top1_valid += 1;
            }
            n_queries += 1;
        }
    }

    let nq = n_queries as f64;
    let lca_mrr = lca_rr / nq;
    let hits3 = lca_hits3 as f64 / nq;
    let top1 = top1_valid as f64 / nq;
    println!("\n{n_queries} conjunctive queries (all leaf pairs with a common ancestor)");
    println!("  LCA MRR              {lca_mrr:.3}");
    println!("  LCA Hits@3           {hits3:.3}");
    println!("  Top-1 is a valid CA  {top1:.3}");

    // Robust aggregate checks: the faithful boxes recover deductive-closure
    // answers well above chance (random LCA MRR over ~29 candidates ~= 0.13,
    // random top-1-valid ~= mean |common|/n ~= 0.1).
    assert!(
        lca_mrr > 0.35,
        "LCA MRR {lca_mrr:.3} should beat chance (~0.13)"
    );
    assert!(
        top1 > 0.6,
        "top-1 valid-common-ancestor rate {top1:.3} should be high"
    );
    println!(
        "\nAll assertions passed: faithful boxes answer closure-grounded conjunctive queries."
    );
}
