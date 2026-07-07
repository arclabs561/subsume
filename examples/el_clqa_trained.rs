//! Graded subsumption CLQA over ACTUALLY TRAINED EL++ boxes.
//!
//! The heyting `el_clqa` example proves the composition works on hand-built
//! boxes. This one closes the loop: it trains boxes with `BurnElTrainer` on a
//! known hierarchy, then answers graded complex queries over the *trained*
//! boxes, testing whether a faithful EL++ geometric model (boxes that satisfy
//! the trained axioms) yields correct complex-query answers.
//!
//! The load-bearing check is Query 2: `Dog` and `Sparrow` share only `Animal`
//! (Dog is a Mammal, Sparrow a Bird), so their common-superclass query must
//! rank `Animal` above `Mammal` and `Bird`. That holds only if training placed
//! `Dog` inside `Mammal` (not `Bird`) and `Sparrow` inside `Bird` (not
//! `Mammal`). A plain link predictor with no faithful model has no reason to.
//!
//! Composition uses `tnorms` for the standard Godel / Product / Lukasiewicz
//! t-norm formulas; the heyting engine composes identically (see heyting's
//! `el_clqa`). The eventual production path is a heyting `AtomicScorer` driven
//! by these extracted arrays.
//!
//! Run: `cargo run --features burn-ndarray --example el_clqa_trained`
//! or `cargo run --features burn-wgpu --example el_clqa_trained`

use subsume::el_training::{Axiom, Ontology};
use subsume::trainer::burn_el_trainer::{BurnElConfig, BurnElTrainer};

#[cfg(feature = "burn-wgpu")]
type Backend = burn::backend::Autodiff<burn_wgpu::Wgpu>;
#[cfg(all(feature = "burn-ndarray", not(feature = "burn-wgpu")))]
type Backend = burn::backend::Autodiff<burn_ndarray::NdArray>;

/// Graded box inclusion `A ⊆ B` in `[0, 1]` from flat center/offset arrays:
/// `exp(-‖relu(|cA - cB| + oA - oB)‖)`, 1 when `A` fully fits inside `B`.
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
    #[cfg(feature = "burn-wgpu")]
    let device = burn_wgpu::WgpuDevice::default();
    #[cfg(all(feature = "burn-ndarray", not(feature = "burn-wgpu")))]
    let device = burn_ndarray::NdArrayDevice::default();
    let dim = 50;

    // Known hierarchy: Animal > {Mammal, Bird, Fish} > leaves.
    let names = [
        "Animal", "Mammal", "Bird", "Fish", "Dog", "Cat", "Sparrow", "Eagle", "Salmon",
    ];
    let id = |n: &str| names.iter().position(|&x| x == n).unwrap();
    let mut ont = Ontology::new();
    for n in names {
        ont.concept(n);
    }
    let subs = [
        ("Mammal", "Animal"),
        ("Bird", "Animal"),
        ("Fish", "Animal"),
        ("Dog", "Mammal"),
        ("Cat", "Mammal"),
        ("Sparrow", "Bird"),
        ("Eagle", "Bird"),
        ("Salmon", "Fish"),
    ];
    for (c, d) in subs {
        ont.axioms.push(Axiom::SubClassOf {
            sub: id(c),
            sup: id(d),
        });
    }
    // Disjointness spreads sibling boxes apart, so nesting is discriminative
    // (without it the trainer can collapse every box onto one point).
    let disj = [
        ("Mammal", "Bird"),
        ("Mammal", "Fish"),
        ("Bird", "Fish"),
        ("Dog", "Cat"),
        ("Sparrow", "Eagle"),
    ];
    for (a, b) in disj {
        ont.axioms.push(Axiom::Disjoint { a: id(a), b: id(b) });
    }

    // Train.
    let config = BurnElConfig {
        dim,
        epochs: 1500,
        lr: 0.05,
        negative_samples: 5,
        margin: 0.1,
        batch_size: 16,
        ..Default::default()
    };
    let trainer = BurnElTrainer::<Backend>::new();
    let mut model = BurnElTrainer::<Backend>::init_model(names.len(), 0, dim, &device);
    let losses = trainer.fit(&mut model, &ont, &config, &device);
    println!(
        "Trained {} concepts, dim {dim}, {} epochs: loss {:.4} -> {:.4}",
        names.len(),
        config.epochs,
        losses.first().copied().unwrap_or(0.0),
        losses.last().copied().unwrap_or(0.0)
    );

    let centers = BurnElTrainer::<Backend>::extract_centers(&model, &device);
    let offsets = BurnElTrainer::<Backend>::extract_offsets(&model, &device);
    let sub = |a: &str, b: &str| subsumption_degree(&centers, &offsets, id(a), id(b), dim);

    // Sanity: the trained boxes must separate true from false subsumptions.
    println!("\nAtomic subsumption degrees (trained boxes):");
    println!(
        "  Dog ⊑ Mammal = {:.3}   Dog ⊑ Bird = {:.3}",
        sub("Dog", "Mammal"),
        sub("Dog", "Bird")
    );
    println!(
        "  Sparrow ⊑ Bird = {:.3}   Sparrow ⊑ Mammal = {:.3}",
        sub("Sparrow", "Bird"),
        sub("Sparrow", "Mammal")
    );
    assert!(
        sub("Dog", "Mammal") > sub("Dog", "Bird"),
        "Dog must fit Mammal better than Bird"
    );
    assert!(
        sub("Sparrow", "Bird") > sub("Sparrow", "Mammal"),
        "Sparrow must fit Bird better than Mammal"
    );

    // Common-superclass query: rank each candidate Z by the t-norm of
    // (X ⊑ Z) and (Y ⊑ Z), excluding X and Y themselves.
    let common_superclass = |x: &str, y: &str, logic: tnorms::LogicFamily| -> Vec<(String, f32)> {
        let (xi, yi) = (id(x), id(y));
        let mut scored: Vec<(String, f32)> = names
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != xi && i != yi)
            .map(|(i, &n)| {
                (
                    n.to_string(),
                    logic.tnorm_f32(sub(x, names[i]), sub(y, names[i])),
                )
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored
    };
    let show = |ranked: &[(String, f32)]| {
        ranked
            .iter()
            .take(4)
            .map(|(n, d)| format!("{n}={d:.3}"))
            .collect::<Vec<_>>()
            .join("  ")
    };

    // --- Query 1: common superclasses of Dog and Cat -> Mammal, Animal ---
    println!("\nQuery 1: X such that Dog ⊑ X AND Cat ⊑ X  (siblings, common = Mammal, Animal)");
    for (label, tn) in [
        ("Godel", tnorms::LogicFamily::Godel),
        ("Product", tnorms::LogicFamily::Product),
        ("Lukasiewicz", tnorms::LogicFamily::Lukasiewicz),
    ] {
        println!(
            "  {label:<12} {}",
            show(&common_superclass("Dog", "Cat", tn))
        );
    }
    let q1 = common_superclass("Dog", "Cat", tnorms::LogicFamily::Godel);
    let top2: Vec<&str> = q1.iter().take(2).map(|(n, _)| n.as_str()).collect();
    assert!(
        top2.contains(&"Mammal") && top2.contains(&"Animal"),
        "Dog+Cat common superclasses must be Mammal and Animal, got {top2:?}"
    );

    // --- Query 2 (discriminating): common superclass of Dog and Sparrow ---
    // They share only Animal. Animal must outrank Mammal and Bird.
    println!(
        "\nQuery 2: X such that Dog ⊑ X AND Sparrow ⊑ X  (cross-branch, common = Animal only)"
    );
    for (label, tn) in [
        ("Godel", tnorms::LogicFamily::Godel),
        ("Product", tnorms::LogicFamily::Product),
        ("Lukasiewicz", tnorms::LogicFamily::Lukasiewicz),
    ] {
        println!(
            "  {label:<12} {}",
            show(&common_superclass("Dog", "Sparrow", tn))
        );
    }
    let q2 = common_superclass("Dog", "Sparrow", tnorms::LogicFamily::Godel);
    assert_eq!(
        q2[0].0, "Animal",
        "Dog+Sparrow common superclass must be Animal, got {}",
        q2[0].0
    );
    let deg = |name: &str, ranked: &[(String, f32)]| {
        ranked
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, d)| *d)
            .unwrap_or(0.0)
    };
    assert!(
        deg("Animal", &q2) > deg("Mammal", &q2),
        "Animal must outrank Mammal for Dog+Sparrow"
    );
    assert!(
        deg("Animal", &q2) > deg("Bird", &q2),
        "Animal must outrank Bird for Dog+Sparrow"
    );

    println!("\nAll assertions passed: trained faithful boxes answer graded CLQA correctly.");
}
