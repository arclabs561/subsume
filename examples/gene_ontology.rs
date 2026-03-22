//! Train EL++ box embeddings on a Gene Ontology subset.
//!
//! Demonstrates the full pipeline:
//! 1. Load EL++ normalized axioms via `el_dataset::load_el_axioms` (TSV format)
//! 2. Print dataset statistics (classes, roles, axiom counts per type)
//! 3. Build an `Ontology` (indexed representation) for the training API
//! 4. Train box embeddings with `train_el_embeddings` (handles all axiom types)
//! 5. Evaluate subsumption prediction and spot-check GO relationships
//!
//! The GO subset at `data/go_subset/go_normalized.tsv` contains ~70 axioms
//! covering the three GO root classes (biological_process, cellular_component,
//! molecular_function) with realistic roles (part_of, regulates, has_part).
//!
//! Run: cargo run -p subsume --example gene_ontology --release

use subsume::el_dataset::load_el_axioms;
use subsume::{
    evaluate_subsumption, train_el_embeddings, ElTrainingConfig, ElTrainingResult, Ontology,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "data/go_subset/go_normalized.tsv";
    if !std::path::Path::new(path).exists() {
        eprintln!("GO subset not found at {path}");
        eprintln!("Expected file: data/go_subset/go_normalized.tsv");
        std::process::exit(1);
    }

    println!("=== Gene Ontology Box Embedding Training ===\n");

    // ---------------------------------------------------------------
    // 1. Load EL++ axioms (new TSV loader)
    // ---------------------------------------------------------------
    let dataset = load_el_axioms(path)?;

    println!("Loaded {} axioms from {path}", dataset.len());
    println!("  NF1  (C1 ^ C2 c= D):    {:>3}", dataset.nf1.len());
    println!("  NF2  (C c= D):           {:>3}", dataset.nf2.len());
    println!("  NF3  (C c= Er.D):        {:>3}", dataset.nf3.len());
    println!("  NF4  (Er.C c= D):        {:>3}", dataset.nf4.len());
    println!("  RI6  (r c= s):           {:>3}", dataset.ri6.len());
    println!("  RI7  (r o s c= t):       {:>3}", dataset.ri7.len());
    println!("  DISJ (C ^ D c= bot):     {:>3}", dataset.disj.len());

    let classes = dataset.classes();
    let roles = dataset.roles();
    println!("\n  Classes: {}, Roles: {}", classes.len(), roles.len());

    // ---------------------------------------------------------------
    // 2. Build Ontology (indexed) from ElDataset
    // ---------------------------------------------------------------
    // from_el_dataset handles all axiom types: NF1 (Intersection),
    // NF2 (SubClassOf), NF3 (ExistentialRight), NF4 (Existential),
    // RI6 (RoleInclusion), RI7 (RoleComposition), DISJ (Disjoint).
    let ontology = Ontology::from_el_dataset(&dataset);

    println!(
        "\nOntology: {} concepts, {} roles, {} axioms (training)",
        ontology.num_concepts(),
        ontology.num_roles(),
        ontology.axioms.len()
    );

    // ---------------------------------------------------------------
    // 3. Train
    // ---------------------------------------------------------------
    let config = ElTrainingConfig {
        dim: 30,
        epochs: 500,
        learning_rate: 5e-3,
        margin: 0.05,
        negative_samples: 3,
        warmup_epochs: 20,
        log_interval: 50,
        seed: 42,
        ..Default::default()
    };

    println!(
        "\nTraining: dim={}, epochs={}, lr={}\n",
        config.dim, config.epochs, config.learning_rate
    );

    let result = train_el_embeddings(&ontology, &config);

    let first_loss = result.epoch_losses[0];
    let last_loss = *result.epoch_losses.last().unwrap();
    println!(
        "\nLoss: {first_loss:.4} (epoch 1) -> {last_loss:.4} (epoch {})",
        config.epochs
    );

    // ---------------------------------------------------------------
    // 4. Evaluate subsumption prediction (NF2 axioms)
    // ---------------------------------------------------------------
    let (hits1, hits10, mrr) = evaluate_subsumption(&result, &ontology.axioms);
    println!("\nSubsumption prediction (training axioms):");
    println!("  Hits@1:  {hits1:.2}");
    println!("  Hits@10: {hits10:.2}");
    println!("  MRR:     {mrr:.4}");

    // ---------------------------------------------------------------
    // 5. Spot-check GO relationships
    // ---------------------------------------------------------------
    println!("\n--- Subsumption spot checks (lower = better containment) ---");

    let checks = [
        // True positives: direct subsumptions from GO
        ("mitosis", "cell_division", true),
        ("cell_division", "cellular_process", true),
        ("cellular_process", "biological_process", true),
        ("kinase_activity", "catalytic_activity", true),
        ("catalytic_activity", "molecular_function", true),
        ("nucleus", "organelle", true),
        ("protein_binding", "binding", true),
        ("apoptosis", "cellular_process", true),
        // True negatives: disjoint or unrelated
        ("biological_process", "molecular_function", false),
        ("nucleus", "mitochondrion", false),
        ("apoptosis", "cell_division", false),
        ("kinase_activity", "phosphatase_activity", false),
    ];

    for (sub_name, sup_name, expected_low) in checks {
        if let (Some(&sub), Some(&sup)) = (
            ontology.concept_index.get(sub_name),
            ontology.concept_index.get(sup_name),
        ) {
            let score = result.subsumption_score(sub, sup);
            let label = if expected_low {
                "SHOULD be low"
            } else {
                "SHOULD be high"
            };
            println!("  {sub_name} c= {sup_name}: {score:.4}  ({label})");
        }
    }

    // ---------------------------------------------------------------
    // 6. Evaluate NF1 intersection axioms manually
    // ---------------------------------------------------------------
    println!("\n--- NF1 intersection checks (lower = better) ---");
    evaluate_nf1_axioms(&dataset.nf1, &ontology, &result);

    // ---------------------------------------------------------------
    // 7. Transitive subsumption (not directly trained, tests generalization)
    // ---------------------------------------------------------------
    println!("\n--- Transitive subsumption (inferred, not directly trained) ---");
    let transitive = [
        ("mitosis", "biological_process"),
        ("kinase_activity", "molecular_function"),
        ("ribosome", "cellular_component"),
        ("DNA_repair", "biological_process"),
    ];
    for (sub_name, sup_name) in transitive {
        if let (Some(&sub), Some(&sup)) = (
            ontology.concept_index.get(sub_name),
            ontology.concept_index.get(sup_name),
        ) {
            let score = result.subsumption_score(sub, sup);
            println!("  {sub_name} c= {sup_name}: {score:.4}");
        }
    }

    Ok(())
}

/// Evaluate NF1 axioms (C1 ^ C2 c= D) using el_intersection_loss.
///
/// These are not handled by the training loop's SubClassOf loss, so we
/// compute the geometric intersection of C1 and C2 and check containment
/// in D after training.
fn evaluate_nf1_axioms(
    nf1_axioms: &[(String, String, String)],
    ontology: &Ontology,
    result: &ElTrainingResult,
) {
    for (c1_name, c2_name, d_name) in nf1_axioms {
        let c1 = match ontology.concept_index.get(c1_name.as_str()) {
            Some(&idx) => idx,
            None => continue,
        };
        let c2 = match ontology.concept_index.get(c2_name.as_str()) {
            Some(&idx) => idx,
            None => continue,
        };
        let d = match ontology.concept_index.get(d_name.as_str()) {
            Some(&idx) => idx,
            None => continue,
        };

        let loss = subsume::el::el_intersection_loss(
            &result.concept_centers[c1],
            &result.concept_offsets[c1],
            &result.concept_centers[c2],
            &result.concept_offsets[c2],
            &result.concept_centers[d],
            &result.concept_offsets[d],
            0.0,
        );

        match loss {
            Ok(l) => println!("  {c1_name} ^ {c2_name} c= {d_name}: {l:.4}"),
            Err(e) => println!("  {c1_name} ^ {c2_name} c= {d_name}: error: {e}"),
        }
    }
}
