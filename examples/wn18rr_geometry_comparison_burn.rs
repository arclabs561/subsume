//! Compare all relation-aware **Burn** region-embedding geometries on WN18RR.
//!
//! Trains every geometry that exposes the relation-aware triple API
//! (`init_model(n_e, n_r, dim)` + `train_epoch(&[TripleIds])` + `evaluate`) at
//! identical settings and prints a sorted MRR / Hits@10 / mean-rank table. This
//! is the cross-geometry counterpart to the per-geometry `wn18rr_*_burn`
//! examples.
//!
//! Run (Metal / GPU on Apple Silicon):
//!   cargo run --features "burn-ndarray,burn-wgpu,kge" --example wn18rr_geometry_comparison_burn --release
//!
//! Run (CPU / rayon):
//!   cargo run --features "burn-ndarray,kge" --example wn18rr_geometry_comparison_burn --release
//!
//! Environment variables (all optional):
//!   DIM (64), EPOCHS (30), LR (0.01), MARGIN (0.5), NEG (10), BATCH (1024),
//!   K (2), REG (1e-4), INFONCE (1), TRAIN_LIMIT (all)
//!
//! Note: WN18RR is a general link-prediction benchmark, not subsume's containment
//! niche, so absolute MRRs are modest. The comparison is what matters: distance-
//! based geometries (ball, cone) train well at high dim; the volume-based box
//! degenerates (its containment gradient vanishes when boxes are disjoint, which
//! is generic in high dim) unless its center-attraction path is enabled.

use burn::backend::Autodiff;
use burn::optim::AdamConfig;
use std::path::Path;
use std::time::Instant;

use subsume::dataset::load_dataset;
use subsume::trainer::{
    BurnBallTrainer, BurnBoxTrainer, BurnCapTrainer, BurnConeTrainer, BurnEllipsoidTrainer,
    BurnOctagonTrainer, BurnTransBoxTrainer, CpuBoxTrainingConfig, FilteredTripleIndexIds,
};

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // DATA points at any WN18RR-format dataset dir (default the full WN18RR).
    // e.g. DATA=data/WN18RR_hypernym for the hypernym-only containment subset.
    let data_dir = std::env::var("DATA").unwrap_or_else(|_| "data/WN18RR".to_string());
    let data_path = Path::new(&data_dir);
    if !data_path.exists() {
        eprintln!("Dataset not found at {data_dir}/ (set DATA=<path>)");
        std::process::exit(1);
    }
    println!("Dataset dir: {data_dir}");

    let dataset = load_dataset(data_path)?;
    let interned = dataset.into_interned();
    let num_entities = interned.entities.len();
    let num_relations = interned.relations.len();

    let dim: usize = env_parse("DIM", 64);
    let epochs: usize = env_parse("EPOCHS", 30);
    let lr: f32 = env_parse("LR", 0.01);
    let margin: f32 = env_parse("MARGIN", 0.5);
    let neg: usize = env_parse("NEG", 10);
    let batch: usize = env_parse("BATCH", 1024);
    let k: f32 = env_parse("K", 2.0);
    let reg: f32 = env_parse("REG", 1e-4);
    let use_infonce: bool = env_bool("INFONCE", true);
    let train_limit: usize = env_parse("TRAIN_LIMIT", interned.train.len());

    let config = CpuBoxTrainingConfig {
        learning_rate: lr,
        margin,
        epochs,
        negative_samples: neg,
        batch_size: batch,
        sigmoid_k: k,
        regularization: reg,
        use_infonce,
        ..Default::default()
    };

    let train_triples: Vec<_> = interned.train.iter().copied().take(train_limit).collect();
    let device = make_device();
    let filter = FilteredTripleIndexIds::from_dataset(&interned);

    println!("=== WN18RR Burn geometry comparison ===");
    println!("Entities: {num_entities}, Relations: {num_relations}");
    println!(
        "Config: dim={dim}, epochs={epochs}, lr={lr}, margin={margin}, neg={neg}, \
         batch={batch}, k={k}, reg={reg}, infonce={use_infonce}, \
         train={}\n",
        train_triples.len()
    );

    // Train one geometry with cosine LR decay and return its filtered test metrics.
    macro_rules! run_geom {
        ($label:expr, $Trainer:ty) => {{
            eprint!("  training {:<10} ...", $label);
            let start = Instant::now();
            let mut trainer = <$Trainer>::new();
            let mut model = trainer.init_model(num_entities, num_relations, dim, &device);
            let mut optim = AdamConfig::new().init::<Backend, _>();
            for epoch in 0..epochs {
                let t = epoch as f32 / epochs.max(1) as f32;
                let lr_min = lr * 0.01;
                let epoch_lr =
                    lr_min + 0.5 * (lr - lr_min) * (1.0 + (std::f32::consts::PI * t).cos());
                let ec = CpuBoxTrainingConfig {
                    learning_rate: epoch_lr,
                    ..config.clone()
                };
                trainer.train_epoch(&mut model, &mut optim, &train_triples, epoch, &ec, &device);
            }
            let r = trainer.evaluate(&model, &interned.test, Some(&filter));
            let secs = start.elapsed().as_secs_f32();
            eprintln!(" done ({secs:.1}s): MRR={:.4}", r.mrr);
            ($label, r.mrr, r.hits_at_1, r.hits_at_10, r.mean_rank, secs)
        }};
    }

    let mut rows: Vec<(&str, f32, f32, f32, f32, f32)> = vec![
        run_geom!("ball", BurnBallTrainer::<Backend>),
        run_geom!("box", BurnBoxTrainer::<Backend>),
        run_geom!("cone", BurnConeTrainer::<Backend>),
        run_geom!("transbox", BurnTransBoxTrainer::<Backend>),
        run_geom!("cap", BurnCapTrainer::<Backend>),
        run_geom!("ellipsoid", BurnEllipsoidTrainer::<Backend>),
        run_geom!("octagon", BurnOctagonTrainer::<Backend>),
    ];
    // Note: annular (2D angular, no `dim`) and subspace (relation-free) have
    // different init signatures and are compared separately, not here.
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));

    println!("\n--- Filtered test results (sorted by MRR) ---\n");
    println!(
        "  {:<12} {:>8} {:>8} {:>8} {:>10} {:>8}",
        "geometry", "MRR", "H@1", "H@10", "meanRank", "train_s"
    );
    for (label, mrr, h1, h10, mr, secs) in &rows {
        println!("  {label:<12} {mrr:>8.4} {h1:>8.4} {h10:>8.4} {mr:>10.1} {secs:>8.1}");
    }

    Ok(())
}
