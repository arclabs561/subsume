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

use candle_core::Device;
use std::collections::{HashMap, HashSet};
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

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("=== WN18RR Candle Box Training ===");
    println!("Device: {}", if device.is_cuda() { "CUDA" } else { "CPU" });
    println!("Config: dim={dim}, epochs={epochs}, lr={lr}, neg={neg}, batch={batch}, beta={beta}, margin={margin}, adv_temp={adv_temp}\n");

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

    let trainer = CandleBoxTrainer::new(num_entities, num_relations, dim, beta, &device)?;

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

    // Build filtered index: all known (head, rel) -> set of tails
    let mut known_tails: HashMap<(usize, usize), HashSet<usize>> = HashMap::new();
    for t in interned
        .train
        .iter()
        .chain(interned.valid.iter())
        .chain(interned.test.iter())
    {
        known_tails
            .entry((t.head, t.relation))
            .or_default()
            .insert(t.tail);
    }

    // Evaluate on test set (filtered ranking)
    let test_sample_size = interned.test.len().min(1000);
    println!(
        "\nFiltered eval on {test_sample_size}/{} test triples...",
        interned.test.len()
    );
    let eval_start = Instant::now();

    let mut ranks = Vec::new();
    let all_tails = candle_core::Tensor::from_vec(
        (0..num_entities as u32).collect::<Vec<_>>(),
        (num_entities,),
        &device,
    )?;

    for t in &interned.test[..test_sample_size] {
        let head_ids = candle_core::Tensor::from_vec(
            vec![t.head as u32; num_entities],
            (num_entities,),
            &device,
        )?;
        let rel_ids = candle_core::Tensor::from_vec(
            vec![t.relation as u32; num_entities],
            (num_entities,),
            &device,
        )?;
        let scores = trainer.score_with_rel(&head_ids, &all_tails, &rel_ids)?;
        let scores_vec: Vec<f32> = scores.to_vec1()?;

        let correct_score = scores_vec[t.tail];
        let filter_set = known_tails.get(&(t.head, t.relation));

        // Filtered rank: count entities scoring better, excluding known triples
        let rank = scores_vec
            .iter()
            .enumerate()
            .filter(|&(eid, &s)| {
                s < correct_score
                    && (eid == t.tail || !filter_set.is_some_and(|known| known.contains(&eid)))
            })
            .count()
            + 1;
        ranks.push(rank);
    }

    let mrr: f32 = ranks.iter().map(|&r| 1.0 / r as f32).sum::<f32>() / ranks.len() as f32;
    let hits_1: f32 = ranks.iter().filter(|&&r| r <= 1).count() as f32 / ranks.len() as f32;
    let hits_3: f32 = ranks.iter().filter(|&&r| r <= 3).count() as f32 / ranks.len() as f32;
    let hits_10: f32 = ranks.iter().filter(|&&r| r <= 10).count() as f32 / ranks.len() as f32;
    let mean_rank: f32 = ranks.iter().sum::<usize>() as f32 / ranks.len() as f32;

    println!("  Eval time: {:.1}s", eval_start.elapsed().as_secs_f64());
    println!(
        "  MRR: {mrr:.4}  H@1: {hits_1:.4}  H@3: {hits_3:.4}  H@10: {hits_10:.4}  MR: {mean_rank:.1}"
    );

    Ok(())
}
