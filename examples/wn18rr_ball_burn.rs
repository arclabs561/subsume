//! Batched ball embedding training on WN18RR using the Burn backend.
//!
//! Run (CPU / rayon multi-core):
//!   cargo run --features burn-ndarray --example wn18rr_ball_burn --release
//!
//! Run (Metal / GPU on Apple Silicon):
//!   cargo run --features burn-wgpu --example wn18rr_ball_burn --release
//!
//! Environment variables (all optional):
//!   DIM, EPOCHS, LR, MARGIN, NEG, BATCH, ADV_TEMP, K, REG, INFONCE,
//!   TRAIN_LIMIT, REPORT_EVERY
//!
//! Reference: SpherE (Li et al., SIGIR 2024): MRR 0.453 on WN18RR

use burn::backend::Autodiff;
use burn::optim::AdamConfig;
use burn::prelude::*;
use std::path::Path;
use subsume::dataset::load_dataset;
use subsume::trainer::burn_ball_trainer::BurnBallTrainer;
use subsume::trainer::{CpuBoxTrainingConfig, FilteredTripleIndexIds};

// ── Backend selection ────────────────────────────────────────────────────────

#[cfg(feature = "burn-wgpu")]
type Backend = Autodiff<burn_wgpu::Wgpu>;

#[cfg(all(feature = "burn-ndarray", not(feature = "burn-wgpu")))]
type Backend = Autodiff<burn_ndarray::NdArray>;

fn make_device() -> <Backend as burn::tensor::backend::Backend>::Device {
    #[cfg(feature = "burn-wgpu")]
    return burn_wgpu::WgpuDevice::default();
    #[cfg(not(feature = "burn-wgpu"))]
    Default::default()
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_bool(key: &str, default: bool) -> bool {
    match std::env::var(key).as_deref() {
        Ok("1") | Ok("true") | Ok("yes") => true,
        Ok("0") | Ok("false") | Ok("no") => false,
        _ => default,
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_path = Path::new("data/WN18RR");
    if !data_path.exists() {
        eprintln!("WN18RR data not found at data/WN18RR/");
        eprintln!("Run: python3 scripts/download_wn18rr.py");
        std::process::exit(1);
    }

    println!("=== WN18RR Ball Embedding Training (Burn) ===\n");

    let dataset = load_dataset(data_path)?;
    println!(
        "Dataset: {} train / {} valid / {} test triples",
        dataset.train.len(),
        dataset.valid.len(),
        dataset.test.len()
    );

    let interned = dataset.into_interned();
    let num_entities = interned.entities.len();
    let num_relations = interned.relations.len();
    println!("Entities: {num_entities}, Relations: {num_relations}");

    // Config.
    let dim: usize = env_parse("DIM", 64);
    let epochs: usize = env_parse("EPOCHS", 100);
    let lr: f32 = env_parse("LR", 0.01);
    let margin: f32 = env_parse("MARGIN", 1.0);
    let neg: usize = env_parse("NEG", 10);
    let batch: usize = env_parse("BATCH", 1024);
    let adv_temp: f32 = env_parse("ADV_TEMP", 0.0);
    let k: f32 = env_parse("K", 2.0);
    let reg: f32 = env_parse("REG", 1e-4);
    let use_infonce: bool = env_bool("INFONCE", false);
    let train_limit: usize = env_parse("TRAIN_LIMIT", interned.train.len());
    let report_every: usize = env_parse("REPORT_EVERY", 10);

    println!(
        "\nConfig: dim={dim}, epochs={epochs}, lr={lr}, margin={margin}, \
         neg={neg}, batch={batch}, adv_temp={adv_temp}, k={k}, reg={reg}, \
         infonce={use_infonce}, train_limit={train_limit}, report_every={report_every}"
    );

    let config = CpuBoxTrainingConfig {
        learning_rate: lr,
        margin,
        adversarial_temperature: adv_temp,
        self_adversarial: adv_temp > 0.0,
        epochs,
        negative_samples: neg,
        batch_size: batch,
        sigmoid_k: k,
        regularization: reg,
        use_infonce,
        ..Default::default()
    };

    // Train triples as pre-indexed TripleIds — no per-epoch string lookup.
    let train_triples: Vec<_> = interned.train.iter().copied().take(train_limit).collect();

    let device = make_device();
    let mut trainer = BurnBallTrainer::<Backend>::new();
    let mut model = trainer.init_model(num_entities, num_relations, dim, &device);
    let mut optim = AdamConfig::new().init::<Backend, _>();

    println!(
        "\nTraining {epochs} epochs ({} triples/epoch, batch={batch})...\n",
        train_triples.len()
    );

    let mut best_val_mrr = 0.0f32;
    let mut best_epoch = 0usize;

    // Fixed seed RNG for reproducible validation sampling.
    let mut val_rng = fastrand::Rng::with_seed(42);

    for epoch in 0..epochs {
        // Cosine LR decay: lr → lr * 0.01 over the full run.
        let t = epoch as f32 / epochs.max(1) as f32;
        let lr_min = lr * 0.01;
        let epoch_lr = lr_min + 0.5 * (lr - lr_min) * (1.0 + (std::f32::consts::PI * t).cos());

        let epoch_config = CpuBoxTrainingConfig {
            learning_rate: epoch_lr,
            ..config.clone()
        };
        let loss = trainer.train_epoch(
            &mut model,
            &mut optim,
            &train_triples,
            epoch,
            &epoch_config,
            &device,
        );

        if (epoch + 1) % report_every == 0 {
            println!("  epoch {epoch:>4}/{epochs}: loss = {loss:.6}, lr = {epoch_lr:.6}");

            // Random validation sample (not a fixed prefix).
            let val_n = 200.min(interned.valid.len());
            let mut val_indices: Vec<usize> = (0..interned.valid.len()).collect();
            for i in (1..val_indices.len()).rev() {
                val_indices.swap(i, val_rng.usize(0..=i));
            }
            let val_sample: Vec<_> = val_indices[..val_n]
                .iter()
                .map(|&i| interned.valid[i])
                .collect();

            let results = trainer.evaluate(&model, &val_sample, None);
            println!(
                "    val (sample {val_n}): MRR={:.4}, H@10={:.4}, MR={:.1}",
                results.mrr, results.hits_at_10, results.mean_rank
            );
            if results.mrr > best_val_mrr {
                best_val_mrr = results.mrr;
                best_epoch = epoch;
            }
        }
    }

    println!("\n  Best val MRR: {best_val_mrr:.4} at epoch {best_epoch}");

    // Final filtered test evaluation.
    println!("\n--- Test Set Evaluation (filtered) ---\n");
    let filter = FilteredTripleIndexIds::from_dataset(&interned);
    let results = trainer.evaluate(&model, &interned.test, Some(&filter));
    println!("  MRR:       {:.4}", results.mrr);
    println!("  Hits@1:    {:.4}", results.hits_at_1);
    println!("  Hits@3:    {:.4}", results.hits_at_3);
    println!("  Hits@10:   {:.4}", results.hits_at_10);
    println!("  Mean Rank: {:.1}", results.mean_rank);
    println!(
        "\n  ({} test triples, {num_entities} entities, filtered)",
        interned.test.len()
    );

    if !results.per_relation.is_empty() {
        println!("\n--- Per-Relation Breakdown ---");
        let mut per_rel = results.per_relation.clone();
        per_rel.sort_by(|a, b| b.mrr.total_cmp(&a.mrr));
        for r in &per_rel {
            println!(
                "  {:40}  MRR={:.4}  H@10={:.4}  n={}",
                r.relation, r.mrr, r.hits_at_10, r.count
            );
        }
    }

    Ok(())
}
