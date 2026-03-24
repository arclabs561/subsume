//! Benchmark EL++ box embeddings on Box2EL datasets (GALEN, GO, Anatomy).
//!
//! Run:
//!   cargo run --example el_benchmark --release -- data/GALEN
//!   cargo run --example el_benchmark --release -- data/GO
//!   cargo run --example el_benchmark --release -- data/ANATOMY
//!
//! Environment variables:
//!   DIM=50 EPOCHS=300 LR=0.005 MARGIN=0.1
//!
//! Expects train.tsv + test.tsv in the data directory (Box2EL TSV format).
//! Convert from Box2EL numpy with: uv run scripts/convert_box2el.py

use std::path::Path;
use std::time::Instant;
use subsume::el_dataset::load_el_axioms;
use subsume::el_training::{Axiom, ElTrainingConfig, Ontology};

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/GALEN".to_string());
    let data_path = Path::new(&data_dir);

    let dim: usize = env_or("DIM", 50);
    let epochs: usize = env_or("EPOCHS", 300);
    let lr: f32 = env_or("LR", 0.005);
    let margin: f32 = env_or("MARGIN", 0.1);

    println!("=== EL++ Box Embedding Benchmark ===");
    println!("Data: {data_dir}");
    println!("Config: dim={dim}, epochs={epochs}, lr={lr}, margin={margin}\n");

    // Load training axioms
    let train_path = data_path.join("train.tsv");
    let train_ds = load_el_axioms(&train_path)?;
    println!(
        "Train: {} axioms ({} NF1, {} NF2, {} NF3, {} NF4, {} RI6, {} RI7, {} DISJ)",
        train_ds.nf1.len()
            + train_ds.nf2.len()
            + train_ds.nf3.len()
            + train_ds.nf4.len()
            + train_ds.ri6.len()
            + train_ds.ri7.len()
            + train_ds.disj.len(),
        train_ds.nf1.len(),
        train_ds.nf2.len(),
        train_ds.nf3.len(),
        train_ds.nf4.len(),
        train_ds.ri6.len(),
        train_ds.ri7.len(),
        train_ds.disj.len(),
    );

    // Build ontology for training
    let ontology = Ontology::from_el_dataset(&train_ds);
    println!(
        "Ontology: {} concepts, {} roles\n",
        ontology.concept_names.len(),
        ontology.role_names.len()
    );

    // Train
    let config = ElTrainingConfig {
        dim,
        epochs,
        learning_rate: lr,
        margin,
        ..Default::default()
    };

    let start = Instant::now();
    let result = subsume::train_el_embeddings(&ontology, &config);
    let elapsed = start.elapsed();

    println!(
        "\nTraining: {:.1}s ({:.2}s/epoch)",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / epochs as f64
    );
    let final_loss = result.epoch_losses.last().copied().unwrap_or(0.0);
    println!("Final loss: {final_loss:.6}");

    // Load test axioms if available
    let test_path = data_path.join("test.tsv");
    if test_path.exists() {
        let test_ds = load_el_axioms(&test_path)?;

        // Convert test axioms to training Axiom format for evaluation.
        // We need to map test concept/role names to the TRAINING ontology's indices.
        let mut test_axioms: Vec<Axiom> = Vec::new();
        for (c, d) in &test_ds.nf2 {
            if let (Some(&sub), Some(&sup)) = (
                ontology.concept_index.get(c.as_str()),
                ontology.concept_index.get(d.as_str()),
            ) {
                test_axioms.push(Axiom::SubClassOf { sub, sup });
            }
        }

        let n_test = test_axioms.len();
        if n_test > 0 {
            let eval_size = n_test.min(1000); // cap for speed
            println!("\nEval: {eval_size}/{n_test} NF2 subsumption test axioms (concepts in training vocab)");

            let eval_start = Instant::now();
            let (hits1, hits10, mrr) =
                subsume::evaluate_subsumption(&result, &test_axioms[..eval_size]);
            let eval_time = eval_start.elapsed();

            println!("  Eval time: {:.1}s", eval_time.as_secs_f64());
            println!("  MRR: {mrr:.4}  H@1: {hits1:.4}  H@10: {hits10:.4}");
        } else {
            println!("\nNo NF2 test axioms with concepts in training vocabulary.");
        }
    }

    Ok(())
}
