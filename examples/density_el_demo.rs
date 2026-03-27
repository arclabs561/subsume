//! Density matrix EL++ training demo.
//!
//! Defines a small taxonomy, trains density matrix embeddings using fidelity-based
//! losses, and shows that parent-child fidelity increases while disjoint-pair
//! fidelity decreases.
//!
//! Run with: `cargo run --example density_el_demo`

use subsume::density_el::{train_density_el, DensityElConfig};
use subsume::el_training::{Axiom, Ontology};

fn main() {
    // Build a small ontology:
    //   Dog ⊑ Animal
    //   Cat ⊑ Animal
    //   Dog ⊑ Mammal
    //   Cat ⊑ Mammal
    //   Fish ⊑ Animal
    //   Disjoint(Dog, Cat)
    //   Disjoint(Dog, Fish)
    //   Disjoint(Cat, Fish)
    //   Disjoint(Mammal, Fish)
    //   ∃hasParent.Dog ⊑ Dog  (NF4)
    let mut ont = Ontology::new();
    let animal = ont.concept("Animal");
    let dog = ont.concept("Dog");
    let cat = ont.concept("Cat");
    let mammal = ont.concept("Mammal");
    let fish = ont.concept("Fish");
    let has_parent = ont.role("hasParent");

    // NF2 subsumption axioms
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
    ont.axioms.push(Axiom::SubClassOf {
        sub: fish,
        sup: animal,
    });

    // Disjointness
    ont.axioms.push(Axiom::Disjoint { a: dog, b: cat });
    ont.axioms.push(Axiom::Disjoint { a: dog, b: fish });
    ont.axioms.push(Axiom::Disjoint { a: cat, b: fish });
    ont.axioms.push(Axiom::Disjoint { a: mammal, b: fish });

    // NF4: ∃hasParent.Dog ⊑ Dog
    ont.axioms.push(Axiom::Existential {
        role: has_parent,
        filler: dog,
        target: dog,
    });

    println!(
        "Ontology: {} concepts, {} roles, {} axioms",
        ont.num_concepts(),
        ont.num_roles(),
        ont.axioms.len()
    );

    let config = DensityElConfig {
        dim: 16,
        margin: 0.3,
        lr: 0.02,
        epochs: 200,
        negative_samples: 2,
        neg_weight: 0.5,
        disj_weight: 1.0,
        existential_weight: 1.0,
        warmup_epochs: 10,
        seed: 42,
        log_interval: 50,
    };

    let result = train_density_el(&ont, &config);

    // Report loss trajectory
    let first = result.epoch_losses[0];
    let last = *result.epoch_losses.last().unwrap();
    println!(
        "\nLoss: epoch 1 = {first:.4}, epoch {} = {last:.4}",
        config.epochs
    );

    // Report fidelity for key pairs
    println!("\nFidelity (parent-child, higher = better):");
    for (name, a, b) in [
        ("Dog-Animal", dog, animal),
        ("Cat-Animal", cat, animal),
        ("Dog-Mammal", dog, mammal),
        ("Cat-Mammal", cat, mammal),
        ("Fish-Animal", fish, animal),
    ] {
        let f = result.concept_fidelity(a, b);
        println!("  F({name}) = {f:.4}");
    }

    println!("\nFidelity (disjoint pairs, lower = better):");
    for (name, a, b) in [
        ("Dog-Cat", dog, cat),
        ("Dog-Fish", dog, fish),
        ("Cat-Fish", cat, fish),
        ("Mammal-Fish", mammal, fish),
    ] {
        let f = result.concept_fidelity(a, b);
        println!("  F({name}) = {f:.4}");
    }
}
