//! Benchmark EL++ box embeddings on Box2EL datasets (GALEN, GO, Anatomy).
//!
//! Run (ndarray backend):
//!   cargo run --example el_benchmark --release -- data/GALEN
//!
//! Run (candle backend, recommended for large ontologies):
//!   BACKEND=candle cargo run --features candle-backend --example el_benchmark --release -- data/GALEN
//!
//! Environment variables:
//!   DIM=200 EPOCHS=300 LR=0.01 MARGIN=0.15 NEG_DIST=5 REG_FACTOR=0.4
//!   NEG_SAMPLES=1 BACKEND=ndarray|candle BATCH=512
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

    let dim: usize = env_or("DIM", 200);
    let epochs: usize = env_or("EPOCHS", 300);
    let lr: f32 = env_or("LR", 0.01);
    let margin: f32 = env_or("MARGIN", 0.15);
    let neg_dist: f32 = env_or("NEG_DIST", 5.0);
    let reg_factor: f32 = env_or("REG_FACTOR", 0.4);
    let neg_samples: usize = env_or("NEG_SAMPLES", 1);
    let batch_size: usize = env_or("BATCH", 512);
    let backend = std::env::var("BACKEND").unwrap_or_else(|_| "ndarray".to_string());

    println!("=== EL++ Box Embedding Benchmark ===");
    println!("Data: {data_dir}, Backend: {backend}");
    println!("Config: dim={dim}, epochs={epochs}, lr={lr}, margin={margin}, neg_dist={neg_dist}, reg={reg_factor}, neg_samples={neg_samples}, batch={batch_size}\n");

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

    // Map test concept names to ontology indices (needed for both backends)
    let test_path = data_path.join("test.tsv");
    let test_pairs: Vec<(usize, usize)> = if test_path.exists() {
        let test_ds = load_el_axioms(&test_path)?;
        test_ds
            .nf2
            .iter()
            .filter_map(|(c, d)| {
                let sub = ontology.concept_index.get(c.as_str())?;
                let sup = ontology.concept_index.get(d.as_str())?;
                Some((*sub, *sup))
            })
            .collect()
    } else {
        Vec::new()
    };

    #[cfg(feature = "candle-backend")]
    if backend == "candle" {
        use subsume::trainer::candle_el_trainer::CandleElTrainer;
        let device = candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu);
        println!("Device: {}", if device.is_cuda() { "CUDA" } else { "CPU" });

        let nc = ontology.concept_names.len();
        let nr = ontology.role_names.len();
        let trainer = CandleElTrainer::new(nc, nr, dim, margin, neg_dist, &device)?;

        let start = Instant::now();
        let losses = trainer.fit(
            &ontology,
            epochs,
            lr as f64,
            batch_size,
            neg_samples,
            reg_factor,
        )?;
        let elapsed = start.elapsed();

        let final_loss = losses.last().copied().unwrap_or(0.0);
        println!(
            "\nTraining: {:.1}s ({:.2}s/epoch)",
            elapsed.as_secs_f64(),
            elapsed.as_secs_f64() / epochs as f64
        );
        println!("Final loss: {final_loss:.6}");

        if !test_pairs.is_empty() {
            let eval_size = test_pairs.len().min(1000);
            println!("\nEval: {eval_size}/{} NF2 test axioms", test_pairs.len());
            let eval_start = Instant::now();
            let (h1, h10, mrr) = trainer.evaluate_subsumption(&test_pairs[..eval_size])?;
            println!("  Eval time: {:.1}s", eval_start.elapsed().as_secs_f64());
            println!("  MRR: {mrr:.4}  H@1: {h1:.4}  H@10: {h10:.4}");
        }

        return Ok(());
    }

    // Ndarray backend (default)
    let config = ElTrainingConfig {
        dim,
        epochs,
        learning_rate: lr,
        margin,
        neg_dist,
        reg_factor,
        negative_samples: neg_samples,
        ..Default::default()
    };

    let start = Instant::now();
    let result = subsume::train_el_embeddings(&ontology, &config);
    let elapsed = start.elapsed();

    let final_loss = result.epoch_losses.last().copied().unwrap_or(0.0);
    println!(
        "\nTraining: {:.1}s ({:.2}s/epoch)",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / epochs as f64
    );
    println!("Final loss: {final_loss:.6}");

    if !test_pairs.is_empty() {
        let eval_size = test_pairs.len().min(1000);
        let test_axioms: Vec<Axiom> = test_pairs[..eval_size]
            .iter()
            .map(|&(sub, sup)| Axiom::SubClassOf { sub, sup })
            .collect();
        println!("\nEval: {eval_size}/{} NF2 test axioms", test_pairs.len());
        let eval_start = Instant::now();
        let (h1, h10, mrr) = subsume::evaluate_subsumption(&result, &test_axioms);
        println!("  Eval time: {:.1}s", eval_start.elapsed().as_secs_f64());
        println!("  MRR: {mrr:.4}  H@1: {h1:.4}  H@10: {h10:.4}");
    }

    Ok(())
}
