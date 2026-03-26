//! Train box embeddings on WN18RR using the candle backend (GPU-capable).
//!
//! Uses per-dimension containment scoring with log-sigmoid loss and
//! self-adversarial negative sampling. Supports CPU, CUDA, and Metal.
//!
//! Run (CPU):  cargo run --features candle-backend --example wn18rr_candle --release
//! Run (GPU):  cargo run --features cuda --example wn18rr_candle --release
//!
//! Environment variables:
//!   DIM=200 EPOCHS=500 LR=0.001 NEG=128 BATCH=512 BETA=10.0 MARGIN=3.0 ADV_TEMP=2.0
//!   INSIDE_W=0.02 BOUNDS_EVERY=50 VOL_REG=0.0001

use candle_core::Device;
use std::path::Path;
use std::time::Instant;
use subsume::dataset::load_dataset;
use subsume::trainer::candle_trainer::CandleBoxTrainer;

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
    let neg: usize = env_or("NEG", 128);
    let batch: usize = env_or("BATCH", 512);
    let beta: f32 = env_or("BETA", 10.0);
    let margin: f32 = env_or("MARGIN", 3.0);
    let adv_temp: f32 = env_or("ADV_TEMP", 2.0);
    let inside_w: f32 = env_or("INSIDE_W", 0.0);
    let bounds_every: usize = env_or("BOUNDS_EVERY", 0);
    let vol_reg: f32 = env_or("VOL_REG", 0.0);

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("=== WN18RR Candle Box Training ===");
    println!("Device: {}", if device.is_cuda() { "CUDA" } else { "CPU" });
    println!("Config: dim={dim}, epochs={epochs}, lr={lr}, neg={neg}, batch={batch}, beta={beta}, margin={margin}, adv_temp={adv_temp}, inside_w={inside_w}, bounds_every={bounds_every}\n");

    let data_dir = std::env::var("DATA").unwrap_or_else(|_| "data/WN18RR".to_string());
    let data_path = Path::new(&data_dir);
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

    let trainer = CandleBoxTrainer::new(num_entities, num_relations, dim, beta, &device)?
        .with_inside_weight(inside_w)
        .with_vol_reg(vol_reg)
        .with_bounds_every(bounds_every);

    let start = Instant::now();
    let losses = trainer.fit(&train, epochs, lr, batch, margin, neg, adv_temp)?;
    let elapsed = start.elapsed();

    let triples_per_sec = (train.len() * epochs) as f64 / elapsed.as_secs_f64();
    println!(
        "\nTraining: {:.1}s ({:.2}s/epoch, {:.0} triples/sec)",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / epochs as f64,
        triples_per_sec,
    );
    println!(
        "Loss: {:.4} (epoch 1) -> {:.4} (epoch {epochs})",
        losses[0],
        losses[losses.len() - 1]
    );

    // Build all_triples for filtered evaluation
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

    // Evaluate on test set (filtered, both head + tail prediction)
    let eval_size = test_triples.len().min(1000);
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
