//! Train cone embeddings on WN18RR using the candle backend (GPU-capable).
//!
//! ConE model (Zhang & Wang, NeurIPS 2021) with per-dimension angular
//! containment scoring and log-sigmoid loss.
//!
//! Run (CPU):  cargo run --features candle-backend --example wn18rr_cone --release
//! Run (GPU):  cargo run --features cuda --example wn18rr_cone --release
//!
//! Environment variables:
//!   DIM=200 EPOCHS=500 LR=0.001 NEG=64 BATCH=512 MARGIN=6.0 CEN=0.02

use candle_core::Device;
use std::path::Path;
use std::time::Instant;
use subsume::dataset::load_dataset;
use subsume::trainer::candle_cone_trainer::CandleConeTrainer;

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dim: usize = env_or("DIM", 200);
    let epochs: usize = env_or("EPOCHS", 500);
    let lr: f64 = env_or("LR", 0.001);
    let neg: usize = env_or("NEG", 64);
    let batch: usize = env_or("BATCH", 512);
    let margin: f32 = env_or("MARGIN", 6.0);
    let cen: f32 = env_or("CEN", 0.02);

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("=== WN18RR Candle Cone Training ===");
    println!(
        "Device: {}",
        if device.is_cuda() { "CUDA" } else { "CPU" }
    );
    println!("Config: dim={dim}, epochs={epochs}, lr={lr}, neg={neg}, batch={batch}, margin={margin}, cen={cen}\n");

    let data_path = Path::new("data/WN18RR");
    if !data_path.exists() {
        eprintln!("WN18RR data not found at data/WN18RR/");
        std::process::exit(1);
    }

    let dataset = load_dataset(data_path)?;
    let interned = dataset.into_interned();
    let num_entities = interned.entities.len();
    let num_relations = interned.relations.len();

    let train: Vec<(usize, usize, usize)> = interned
        .train
        .iter()
        .map(|t| (t.head, t.relation, t.tail))
        .collect();

    println!(
        "Dataset: {num_entities} entities, {num_relations} relations, {} train triples\n",
        train.len()
    );

    let trainer = CandleConeTrainer::new(num_entities, num_relations, dim, cen, &device)?;

    let start = Instant::now();
    let losses = trainer.fit(&train, epochs, lr, batch, margin, neg)?;
    let elapsed = start.elapsed();

    println!(
        "\nTraining: {:.1}s ({:.2}s/epoch)",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / epochs as f64,
    );
    println!(
        "Loss: {:.4} (epoch 1) -> {:.4} (epoch {epochs})",
        losses[0],
        losses[losses.len() - 1]
    );

    // Evaluate on test set
    let all_triples: Vec<(usize, usize, usize)> = interned
        .train
        .iter()
        .chain(interned.valid.iter())
        .chain(interned.test.iter())
        .map(|t| (t.head, t.relation, t.tail))
        .collect();

    let test_triples: Vec<(usize, usize, usize)> = interned
        .test
        .iter()
        .map(|t| (t.head, t.relation, t.tail))
        .collect();

    let eval_size = test_triples.len().min(500);
    println!(
        "\nFiltered eval (head+tail) on {eval_size}/{} test triples...",
        test_triples.len()
    );
    let eval_start = Instant::now();
    let (mrr, h1, h3, h10, mr) = trainer.evaluate(&test_triples[..eval_size], &all_triples)?;

    println!("  Eval time: {:.1}s", eval_start.elapsed().as_secs_f64());
    println!("  MRR: {mrr:.4}  H@1: {h1:.4}  H@3: {h3:.4}  H@10: {h10:.4}  MR: {mr:.1}");

    Ok(())
}
