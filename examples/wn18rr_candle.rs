//! Train box embeddings on WN18RR using the candle backend (GPU-capable).
//!
//! Uses automatic differentiation instead of manual gradients.
//! Supports CUDA/Metal via candle when compiled with appropriate features.
//!
//! Run: cargo run -p subsume --features candle-backend --example wn18rr_candle --release
//!
//! Environment variables:
//!   DIM=200 EPOCHS=500 LR=0.001 NEG=10 BATCH=1024

use candle_core::Device;
use std::path::Path;
use std::time::Instant;
use subsume::dataset::load_dataset;
use subsume::trainer::candle_trainer::CandleBoxTrainer;
use subsume::trainer::FilteredTripleIndexIds;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dim: usize = std::env::var("DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let epochs: usize = std::env::var("EPOCHS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);
    let lr: f64 = std::env::var("LR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.001);
    let neg: usize = std::env::var("NEG")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let batch: usize = std::env::var("BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);

    // Try CUDA first, fall back to CPU
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("=== WN18RR Candle Box Training ===");
    println!("Device: {}", if device.is_cuda() { "CUDA" } else { "CPU" });
    println!("Config: dim={dim}, epochs={epochs}, lr={lr}, neg={neg}, batch={batch}\n");

    let data_path = Path::new("data/WN18RR");
    if !data_path.exists() {
        eprintln!("WN18RR data not found at data/WN18RR/");
        std::process::exit(1);
    }

    let dataset = load_dataset(data_path)?;
    let interned = dataset.into_interned();
    let num_entities = interned.entities.len();

    let train: Vec<(usize, usize, usize)> = interned
        .train
        .iter()
        .map(|t| (t.head, t.relation, t.tail))
        .collect();

    println!(
        "Dataset: {} entities, {} relations, {} train triples\n",
        num_entities,
        interned.relations.len(),
        train.len()
    );

    let trainer = CandleBoxTrainer::new(num_entities, dim, 10.0, &device)?;

    let start = Instant::now();
    let losses = trainer.fit(&train, epochs, lr, batch, 1.0, neg)?;
    let elapsed = start.elapsed();

    println!(
        "\nTraining: {:.1}s ({:.2}s/epoch)",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / epochs as f64
    );
    println!(
        "Loss: {:.4} (epoch 1) -> {:.4} (epoch {epochs})",
        losses[0],
        losses[losses.len() - 1]
    );

    // Evaluate on test set (CPU scoring for now)
    println!(
        "\nEvaluating on test set ({} triples)...",
        interned.test.len()
    );
    let eval_start = Instant::now();

    let mut ranks = Vec::new();
    let test_sample_size = 200.min(interned.test.len()); // Sample for speed
    for t in &interned.test[..test_sample_size] {
        // Score all entities as tails
        let head_ids = candle_core::Tensor::from_vec(
            vec![t.head as u32; num_entities],
            (num_entities,),
            &device,
        )?;
        let all_tails = candle_core::Tensor::from_vec(
            (0..num_entities as u32).collect::<Vec<_>>(),
            (num_entities,),
            &device,
        )?;
        let scores = trainer.score(&head_ids, &all_tails)?;
        let scores_vec: Vec<f32> = scores.to_vec1()?;

        // Rank the correct tail
        let correct_score = scores_vec[t.tail];
        let rank = scores_vec.iter().filter(|&&s| s < correct_score).count() + 1;
        ranks.push(rank);
    }

    let mrr: f32 = ranks.iter().map(|&r| 1.0 / r as f32).sum::<f32>() / ranks.len() as f32;
    let hits_10: f32 = ranks.iter().filter(|&&r| r <= 10).count() as f32 / ranks.len() as f32;
    let mean_rank: f32 = ranks.iter().sum::<usize>() as f32 / ranks.len() as f32;

    println!("  Eval time: {:.1}s", eval_start.elapsed().as_secs_f64());
    println!(
        "  MRR: {mrr:.4}, Hits@10: {hits_10:.4}, MR: {mean_rank:.1} (sample {test_sample_size})"
    );

    Ok(())
}
