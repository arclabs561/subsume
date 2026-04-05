//! Random hyperparameter search for EL++ ontology embeddings.
//!
//! Run:
//!   BACKEND=candle N_TRIALS=20 EPOCHS=500 \
//!     cargo run --features candle-backend --example el_hyperparam_search --release -- data/GALEN

#[cfg(feature = "candle-backend")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::Path;
    use std::time::Instant;
    use subsume::el_dataset::load_el_axioms;
    use subsume::trainer::candle_el_trainer::CandleElTrainer;
    use subsume::Ontology;

    let args: Vec<String> = std::env::args().collect();
    let data_dir = args.get(1).map(|s| s.as_str()).unwrap_or("data/GALEN");
    let n_trials: usize = std::env::var("N_TRIALS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let epochs: usize = std::env::var("EPOCHS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);

    let data_path = Path::new(data_dir);
    println!("=== EL++ Hyperparameter Search ===");
    println!("Data: {data_dir}, Trials: {n_trials}, Epochs/trial: {epochs}\n");

    let train_ds = load_el_axioms(&data_path.join("train.tsv"))?;
    let ontology = Ontology::from_el_dataset(&train_ds);
    let nc = ontology.concept_names.len();
    let nr = ontology.role_names.len();

    // Load validation set and map to indices.
    let val_ds = load_el_axioms(&data_path.join("val.tsv"))?;
    let val_nf2: Vec<(usize, usize)> = val_ds
        .nf2
        .iter()
        .filter_map(|(c, d)| {
            Some((
                *ontology.concept_index.get(c.as_str())?,
                *ontology.concept_index.get(d.as_str())?,
            ))
        })
        .collect();
    let val_nf3: Vec<(usize, usize, usize)> = val_ds
        .nf3
        .iter()
        .filter_map(|(c, r, d)| {
            Some((
                *ontology.concept_index.get(c.as_str())?,
                *ontology.role_index.get(r.as_str())?,
                *ontology.concept_index.get(d.as_str())?,
            ))
        })
        .collect();

    let eval_n2 = val_nf2.len().min(500);
    let eval_n3 = val_nf3.len().min(500);
    let device = candle_core::Device::Cpu;
    let mut rng = fastrand::Rng::with_seed(42);

    let dims = [50usize, 100, 200];
    let mut best_score = f32::NEG_INFINITY;
    let mut best_params = String::new();
    let start = Instant::now();

    for trial in 0..n_trials {
        let dim = dims[rng.usize(0..dims.len())];
        let lr = 0.002 + rng.f64() * 0.028; // [0.002, 0.030]
        let margin = 0.02 + rng.f32() * 0.28; // [0.02, 0.30]
        let neg_dist = 2.0 + rng.f32() * 8.0; // [2, 10]
        let reg = 0.1 + rng.f32() * 0.9; // [0.1, 1.0]

        let trainer = CandleElTrainer::new(nc, nr, dim, margin, neg_dist, &device)?;
        let _ = trainer.fit(&ontology, epochs, lr, 512, 1, reg);

        let nf2_mrr = trainer
            .evaluate_subsumption(&val_nf2[..eval_n2])
            .map(|(_, _, mrr)| mrr)
            .unwrap_or(0.0);
        let nf3_mrr = trainer
            .evaluate_nf3(&val_nf3[..eval_n3])
            .map(|(_, _, mrr)| mrr)
            .unwrap_or(0.0);

        let score = nf2_mrr + nf3_mrr;
        let marker = if score > best_score { " *BEST*" } else { "" };
        println!(
            "Trial {trial:>3}/{n_trials}: dim={dim:>3} lr={lr:.4} margin={margin:.3} nd={neg_dist:.1} reg={reg:.2} => NF2={nf2_mrr:.4} NF3={nf3_mrr:.4}{marker}"
        );

        if score > best_score {
            best_score = score;
            best_params = format!(
                "dim={dim} lr={lr:.4} margin={margin:.3} neg_dist={neg_dist:.1} reg={reg:.2}"
            );
        }
    }

    println!("\n=== Best ===");
    println!("  Score (NF2+NF3 MRR): {best_score:.4}");
    println!("  Params: {best_params}");
    println!(
        "  Time: {:.0}s ({:.1}s/trial)",
        start.elapsed().as_secs_f64(),
        start.elapsed().as_secs_f64() / n_trials as f64
    );

    Ok(())
}

#[cfg(not(feature = "candle-backend"))]
fn main() {
    eprintln!("This example requires --features candle-backend");
    std::process::exit(1);
}
