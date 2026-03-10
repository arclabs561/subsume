//! TaxoBell end-to-end training: encoder + loss + evaluation.
//!
//! Trains a TaxoBell MLP encoder on a synthetic taxonomy with random
//! pre-computed embeddings. Uses candle autograd for exact gradients.
//!
//! Run: cargo run -p subsume --features candle-backend --example taxobell_training
//!
//! Reference: TaxoBell (WWW 2026, arXiv:2601.09633)

use std::collections::HashMap;
use subsume::taxobell::TaxoBellConfig;
use subsume::taxobell_encoder::{evaluate_taxobell, train_taxobell, TaxoBellTrainingConfig};

fn main() {
    println!("=== TaxoBell Training: MLP Encoder + Candle Autograd ===\n");

    // --- Build a synthetic taxonomy ---
    //
    //              entity (0)
    //             /      \
    //        animal (1)   vehicle (2)
    //       /  |   \      /    \
    //   dog(3) cat(4) bird(5) car(6) truck(7)
    //                 |
    //              eagle(8)

    let node_names = [
        "entity", "animal", "vehicle", "dog", "cat", "bird", "car", "truck", "eagle",
    ];
    let num_nodes = node_names.len();
    let node_ids: Vec<usize> = (0..num_nodes).collect();
    let node_index: HashMap<usize, usize> = node_ids.iter().map(|&id| (id, id)).collect();

    // Edges: (parent_id, child_id)
    let all_edges: Vec<(usize, usize)> = vec![
        (0, 1), // entity → animal
        (0, 2), // entity → vehicle
        (1, 3), // animal → dog
        (1, 4), // animal → cat
        (1, 5), // animal → bird
        (2, 6), // vehicle → car
        (2, 7), // vehicle → truck
        (5, 8), // bird → eagle
    ];

    // Split: first 6 for training, last 2 for test
    let train_edges = &all_edges[..6];
    let test_edges = &all_edges[6..];

    println!("Taxonomy: {} nodes, {} edges", num_nodes, all_edges.len());
    println!("  Train edges: {}", train_edges.len());
    println!("  Test edges:  {}", test_edges.len());

    // --- Generate random "pre-computed" text embeddings ---
    let embed_dim = 16;
    let embeddings = generate_random_embeddings(num_nodes, embed_dim, 12345);

    println!("  Embed dim:   {embed_dim}");

    // --- Train ---
    let config = TaxoBellTrainingConfig {
        learning_rate: 1e-3,
        epochs: 300,
        num_negatives: 5,
        hidden_dim: 32,
        box_dim: 8,
        seed: 42,
        warmup_epochs: 15,
        loss_config: TaxoBellConfig {
            alpha: 0.35,
            beta: 0.45,
            gamma: 0.10,
            delta: 0.10,
            ..TaxoBellConfig::default()
        },
    };

    println!("\nTraining config:");
    println!("  Hidden dim:  {}", config.hidden_dim);
    println!("  Box dim:     {}", config.box_dim);
    println!("  Epochs:      {}", config.epochs);
    println!("  LR:          {}", config.learning_rate);
    println!("  Negatives:   {}", config.num_negatives);

    println!("\nTraining...\n");
    println!(
        "{:>5}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>8}",
        "epoch", "total", "L_sym", "L_asym", "L_reg", "L_clip", "lr"
    );
    println!("{}", "-".repeat(75));

    let (encoder, snapshots) =
        train_taxobell(&embeddings, train_edges, &node_ids, &node_index, &config).unwrap();

    for snap in &snapshots {
        if snap.epoch % 30 == 0 || snap.epoch == config.epochs - 1 {
            println!(
                "{:>5}  {:>10.4}  {:>10.4}  {:>10.4}  {:>10.4}  {:>10.4}  {:>8.6}",
                snap.epoch,
                snap.loss.total,
                snap.loss.l_sym,
                snap.loss.l_asym,
                snap.loss.l_reg,
                snap.loss.l_clip,
                snap.lr,
            );
        }
    }

    let first_loss = snapshots[0].loss.total;
    let last_loss = snapshots.last().unwrap().loss.total;
    println!("\nLoss: {first_loss:.4} -> {last_loss:.4}");

    // --- Evaluate on held-out edges ---
    println!("\n--- Evaluation on test edges ---\n");

    let eval =
        evaluate_taxobell(&encoder, &embeddings, test_edges, &node_ids, &node_index).unwrap();
    println!("  MRR:      {:.4}", eval.mrr);
    println!("  Hits@1:   {:.4}", eval.hits_at_1);
    println!("  Hits@3:   {:.4}", eval.hits_at_3);
    println!("  Hits@10:  {:.4}", eval.hits_at_10);

    let eval_train =
        evaluate_taxobell(&encoder, &embeddings, train_edges, &node_ids, &node_index).unwrap();
    println!("\n--- Evaluation on train edges (sanity check) ---\n");
    println!("  MRR:      {:.4}", eval_train.mrr);
    println!("  Hits@1:   {:.4}", eval_train.hits_at_1);
    println!("  Hits@10:  {:.4}", eval_train.hits_at_10);

    // --- Inspect learned boxes ---
    println!("\n--- Learned Gaussian boxes ---\n");
    println!(
        "{:>10}  {:>12}  {:>12}  {:>12}",
        "concept", "log-volume", "sigma[0]", "mu[0]"
    );
    println!("{}", "-".repeat(50));
    for (i, name) in node_names.iter().enumerate() {
        let gb = encoder.encode_one(&embeddings[i]).unwrap();
        println!(
            "{:>10}  {:>12.4}  {:>12.4}  {:>12.4}",
            name,
            gb.log_volume(),
            gb.sigma()[0],
            gb.mu()[0],
        );
    }

    println!("\nKey observations:");
    println!("  - Parent nodes should have larger log-volume (wider distributions)");
    println!("  - KL(child || parent) should be small for true parent-child pairs");
}

/// Generate random embeddings using xorshift64.
fn generate_random_embeddings(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed.wrapping_add(1);
    let mut next_f32 = move || -> f32 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as f32 / u64::MAX as f32) * 2.0 - 1.0
    };

    (0..n)
        .map(|_| (0..dim).map(|_| next_f32()).collect())
        .collect()
}
