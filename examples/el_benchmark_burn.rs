//! Benchmark EL++ box embeddings on Box2EL datasets using the burn backend.
//!
//! This is the burn-based port of `el_benchmark.rs` (which uses the candle backend).
//! Supports multi-core CPU training via `burn-ndarray` and GPU via `burn-wgpu`.
//!
//! Run (ndarray backend, default):
//!   cargo run --features burn-ndarray --example el_benchmark_burn --release
//!
//! Run (wgpu backend):
//!   cargo run --features burn-wgpu --example el_benchmark_burn --release
//!
//! Environment variables:
//!   DATASET=GALEN|GO|ANATOMY  (default: GALEN)
//!   DIM=200 EPOCHS=1000 LR=0.01 BATCH=512 NEG=2
//!   MARGIN=0.1 NEG_DIST=2.0 REG=0.5
//!
//! Expects data/{DATASET}/train.tsv and data/{DATASET}/test.tsv in Box2EL TSV format.
//! Convert from Box2EL numpy with: uv run scripts/convert_box2el.py

use std::path::Path;
use std::time::Instant;
use subsume::el_dataset::load_el_axioms;
use subsume::el_training::Ontology;
use subsume::trainer::burn_el_trainer::{BurnElConfig, BurnElTrainer};

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

#[cfg(feature = "burn-wgpu")]
type TrainBackend = burn::backend::Autodiff<burn_wgpu::Wgpu>;
#[cfg(all(feature = "burn-ndarray", not(feature = "burn-wgpu")))]
type TrainBackend = burn::backend::Autodiff<burn_ndarray::NdArray>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = std::env::var("DATASET").unwrap_or_else(|_| "GALEN".to_string());
    let data_dir = format!("data/{dataset}");
    let data_path = Path::new(&data_dir);

    let dim: usize = env_or("DIM", 200);
    let epochs: usize = env_or("EPOCHS", 1000);
    let lr: f64 = env_or("LR", 0.01f64);
    let batch_size: usize = env_or("BATCH", 512);
    let neg_samples: usize = env_or("NEG", 2);
    let margin: f32 = env_or("MARGIN", 0.1f32);
    let neg_dist: f32 = env_or("NEG_DIST", 2.0f32);
    let reg_factor: f32 = env_or("REG", 0.5f32);
    let nf4_neg_weight: f32 = env_or("NF4_NEG_W", 0.0f32);

    println!("=== EL++ Burn Benchmark ===");
    println!("Dataset: {dataset}  dir: {data_dir}");
    println!("Config: dim={dim}, epochs={epochs}, lr={lr}, batch={batch_size}, neg={neg_samples}, margin={margin}, neg_dist={neg_dist}, reg={reg_factor}\n");

    // Load training axioms.
    let train_path = data_path.join("train.tsv");
    if !train_path.exists() {
        eprintln!(
            "ERROR: {train_path:?} not found.\n\
             Convert Box2EL data with: uv run scripts/convert_box2el.py"
        );
        std::process::exit(1);
    }
    let train_ds = load_el_axioms(&train_path)?;
    println!(
        "Train: {} axioms  (NF1={}, NF2={}, NF3={}, NF4={}, RI6={}, RI7={}, DISJ={})",
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

    // Build ontology for training.
    let train_ont = Ontology::from_el_dataset(&train_ds);
    println!(
        "Ontology: {} concepts, {} roles\n",
        train_ont.concept_names.len(),
        train_ont.role_names.len()
    );

    // Load test axioms and remap to training ontology indices.
    let test_path = data_path.join("test.tsv");
    let test_ont = if test_path.exists() {
        let test_ds = load_el_axioms(&test_path)?;
        let mut ont = Ontology::new();
        // Seed the test ontology with the same vocab so indices are consistent.
        for name in &train_ont.concept_names {
            ont.concept(name);
        }
        for name in &train_ont.role_names {
            ont.role(name);
        }
        // Re-resolve test axioms against training vocab; drop unknown concepts.
        for (c1, c2, d) in &test_ds.nf1 {
            if let (Some(&i1), Some(&i2), Some(&id)) = (
                train_ont.concept_index.get(c1.as_str()),
                train_ont.concept_index.get(c2.as_str()),
                train_ont.concept_index.get(d.as_str()),
            ) {
                ont.axioms.push(subsume::el_training::Axiom::Intersection {
                    c1: i1,
                    c2: i2,
                    target: id,
                });
            }
        }
        for (c, d) in &test_ds.nf2 {
            if let (Some(&sub), Some(&sup)) = (
                train_ont.concept_index.get(c.as_str()),
                train_ont.concept_index.get(d.as_str()),
            ) {
                ont.axioms
                    .push(subsume::el_training::Axiom::SubClassOf { sub, sup });
            }
        }
        for (c, r, d) in &test_ds.nf3 {
            if let (Some(&ic), Some(&ir), Some(&id)) = (
                train_ont.concept_index.get(c.as_str()),
                train_ont.role_index.get(r.as_str()),
                train_ont.concept_index.get(d.as_str()),
            ) {
                ont.axioms
                    .push(subsume::el_training::Axiom::ExistentialRight {
                        sub: ic,
                        role: ir,
                        filler: id,
                    });
            }
        }
        for (r, c, d) in &test_ds.nf4 {
            if let (Some(&ir), Some(&ic), Some(&id)) = (
                train_ont.role_index.get(r.as_str()),
                train_ont.concept_index.get(c.as_str()),
                train_ont.concept_index.get(d.as_str()),
            ) {
                ont.axioms.push(subsume::el_training::Axiom::Existential {
                    role: ir,
                    filler: ic,
                    target: id,
                });
            }
        }
        Some(ont)
    } else {
        eprintln!("WARNING: {test_path:?} not found — skipping evaluation.");
        None
    };

    // Count test axioms for reporting.
    let (n_test_nf1, n_test_nf2, n_test_nf3, n_test_nf4) = test_ont
        .as_ref()
        .map(|ont| {
            use subsume::el_training::Axiom;
            let (mut n1, mut n2, mut n3, mut n4) = (0usize, 0, 0, 0);
            for ax in &ont.axioms {
                match ax {
                    Axiom::Intersection { .. } => n1 += 1,
                    Axiom::SubClassOf { .. } => n2 += 1,
                    Axiom::ExistentialRight { .. } => n3 += 1,
                    Axiom::Existential { .. } => n4 += 1,
                    _ => {}
                }
            }
            (n1, n2, n3, n4)
        })
        .unwrap_or_default();
    println!("Test: NF1={n_test_nf1}, NF2={n_test_nf2}, NF3={n_test_nf3}, NF4={n_test_nf4}\n");

    // Select device and run.
    #[cfg(feature = "burn-wgpu")]
    let device = burn_wgpu::WgpuDevice::default();
    #[cfg(all(feature = "burn-ndarray", not(feature = "burn-wgpu")))]
    let device = burn_ndarray::NdArrayDevice::default();

    let config = BurnElConfig {
        dim,
        epochs,
        lr,
        batch_size,
        negative_samples: neg_samples,
        margin,
        neg_dist,
        reg_factor,
        nf4_neg_weight,
        ..Default::default()
    };

    let nc = train_ont.concept_names.len();
    let nr = train_ont.role_names.len();

    let trainer = BurnElTrainer::<TrainBackend>::new();
    let mut model = BurnElTrainer::<TrainBackend>::init_model(nc, nr, dim, &device);

    let start = Instant::now();
    let losses = trainer.fit(&mut model, &train_ont, &config, &device);
    let elapsed = start.elapsed();

    let final_loss = losses.last().copied().unwrap_or(0.0);
    println!(
        "Training: {:.1}s ({:.2}s/epoch)",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / epochs as f64,
    );
    println!("Final loss: {final_loss:.6}\n");

    // Evaluate on test set.
    if let Some(ref ont) = test_ont {
        println!("=== Evaluation (test set) ===");
        let eval_start = Instant::now();
        let (nf2, nf1, nf3, nf4) =
            BurnElTrainer::<TrainBackend>::evaluate(&model, ont, dim, &device);
        println!(
            "Evaluation time: {:.1}s\n",
            eval_start.elapsed().as_secs_f64()
        );

        println!("{:<12} {:>8} {:>8} {:>8}", "NF type", "H@1", "H@10", "MRR");
        println!("{}", "-".repeat(40));
        for (label, m) in [
            ("NF2 (C⊑D)", nf2),
            ("NF1 (C⊓C⊑D)", nf1),
            ("NF3 (C⊑∃r.D)", nf3),
            ("NF4 (∃r.C⊑D)", nf4),
        ] {
            println!("{:<12} {:>8.4} {:>8.4} {:>8.4}", label, m.0, m.1, m.2);
        }
    }

    Ok(())
}
