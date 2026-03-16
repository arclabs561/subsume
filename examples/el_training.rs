//! Train EL++ box embeddings on a small biomedical-style ontology.
//!
//! Demonstrates end-to-end training: parsing axioms, training box embeddings
//! with the EL++ loss functions (Box2EL / TransBox), and evaluating subsumption
//! prediction.
//!
//! Ontology (15 concepts, 4 roles):
//!   LivingThing
//!     Animal
//!       Mammal: Dog, Cat, Whale
//!       Bird: Eagle, Sparrow
//!       Fish: Salmon
//!     Plant
//!       Tree, Flower
//!   Roles: partOf, hasHabitat, eats, locatedIn
//!
//! Run: cargo run -p subsume --example el_training

use subsume::{evaluate_subsumption, train_el_embeddings, ElTrainingConfig, Ontology};

const ONTOLOGY: &str = "\
# === Subsumption axioms (NF2: C ⊑ D) ===
SubClassOf Dog Mammal
SubClassOf Cat Mammal
SubClassOf Whale Mammal
SubClassOf Eagle Bird
SubClassOf Sparrow Bird
SubClassOf Salmon Fish
SubClassOf Mammal Animal
SubClassOf Bird Animal
SubClassOf Fish Animal
SubClassOf Animal LivingThing
SubClassOf Plant LivingThing
SubClassOf Tree Plant
SubClassOf Flower Plant

# === Disjointness ===
Disjoint Dog Cat
Disjoint Mammal Fish
Disjoint Animal Plant
Disjoint Bird Fish

# === Existential restrictions (NF4: ∃R.C ⊑ D) ===
# ∃hasHabitat.Water ⊑ Fish  (things with water habitat are fish-like)
Existential hasHabitat Water Fish
# ∃eats.Plant ⊑ Animal  (things that eat plants are animals)
Existential eats Plant Animal

# === Role inclusion (RI6: R ⊑ S) ===
RoleInclusion partOf locatedIn

# === Role composition (RI7: R ∘ S ⊑ T) ===
# partOf ∘ partOf ⊑ partOf  (transitivity of partOf)
RoleComposition partOf partOf partOf
";

fn main() {
    println!("=== EL++ Ontology Embedding Training ===\n");

    // Parse ontology
    let ontology = Ontology::parse(ONTOLOGY.as_bytes()).expect("failed to parse ontology");
    println!(
        "Ontology: {} concepts, {} roles, {} axioms",
        ontology.num_concepts(),
        ontology.num_roles(),
        ontology.axioms.len()
    );

    // Configure training
    let config = ElTrainingConfig {
        dim: 30,
        epochs: 500,
        learning_rate: 0.005,
        margin: 0.05,
        negative_samples: 3,
        warmup_epochs: 20,
        log_interval: 50,
        seed: 42,
        ..Default::default()
    };

    // Train
    println!(
        "\nTraining with dim={}, epochs={}...\n",
        config.dim, config.epochs
    );
    let result = train_el_embeddings(&ontology, &config);

    // Loss trajectory
    let first_loss = result.epoch_losses[0];
    let last_loss = *result.epoch_losses.last().unwrap();
    println!(
        "\nLoss: {first_loss:.4} (epoch 1) -> {last_loss:.4} (epoch {})",
        config.epochs
    );

    // Evaluate subsumption prediction
    let (hits1, hits10, mrr) = evaluate_subsumption(&result, &ontology.axioms);
    println!("\nSubsumption prediction (on training axioms):");
    println!("  Hits@1:  {hits1:.2}");
    println!("  Hits@10: {hits10:.2}");
    println!("  MRR:     {mrr:.4}");

    // Spot-check specific subsumptions
    println!("\n--- Spot checks (lower = better containment) ---");
    let pairs = [
        ("Dog", "Mammal", true),
        ("Dog", "Animal", true),
        ("Dog", "LivingThing", true),
        ("Dog", "Cat", false),
        ("Mammal", "Fish", false),
        ("Animal", "Plant", false),
        ("Eagle", "Bird", true),
        ("Salmon", "Fish", true),
    ];
    for (sub_name, sup_name, expected_low) in pairs {
        let sub = ontology.concept_index[sub_name];
        let sup = ontology.concept_index[sup_name];
        let score = result.subsumption_score(sub, sup);
        let label = if expected_low {
            "SHOULD be low"
        } else {
            "SHOULD be high"
        };
        println!("  {sub_name} ⊑ {sup_name}: {score:.4}  ({label})");
    }
}
