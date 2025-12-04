//! Example demonstrating training quality metrics and diagnostics for box embeddings.

use subsume_core::training::{
    metrics::{mean_reciprocal_rank, hits_at_k, mean_rank, ndcg},
    diagnostics::{TrainingStats, LossComponents},
};

fn main() -> Result<(), subsume_core::BoxError> {
    println!("=== Training Quality Metrics and Diagnostics ===\n");

    // 1. Rank-based evaluation metrics
    println!("1. Rank-Based Evaluation Metrics");
    println!("   Essential for evaluating box embeddings on knowledge graph tasks.\n");
    
    // Simulate ranking results for link prediction
    let ranks = vec![1, 3, 2, 5, 10, 1, 4, 2];
    let mrr = mean_reciprocal_rank(ranks.iter().copied());
    let hits_1 = hits_at_k(ranks.iter().copied(), 1);
    let hits_3 = hits_at_k(ranks.iter().copied(), 3);
    let hits_10 = hits_at_k(ranks.iter().copied(), 10);
    let mr = mean_rank(ranks.iter().copied());
    
    println!("   Query results (ranks of correct answers): {:?}", ranks);
    println!("   Mean Reciprocal Rank (MRR): {:.4}", mrr);
    println!("   Hits@1: {:.4}", hits_1);
    println!("   Hits@3: {:.4}", hits_3);
    println!("   Hits@10: {:.4}", hits_10);
    println!("   Mean Rank (MR): {:.2}", mr);
    println!();

    // 2. nDCG for ranking quality
    println!("2. Normalized Discounted Cumulative Gain (nDCG)");
    println!("   Measures ranking quality considering both relevance and position.\n");
    
    // Simulate relevance scores for a ranking
    let ranked_scores = vec![0.9, 0.5, 0.8, 0.2, 0.1];
    let ideal_scores = vec![0.9, 0.8, 0.5, 0.2, 0.1];
    let ndcg_score = ndcg(ranked_scores.iter().copied(), ideal_scores.iter().copied());
    
    println!("   Ranked relevance: {:?}", ranked_scores);
    println!("   Ideal relevance:  {:?}", ideal_scores);
    println!("   nDCG: {:.4}", ndcg_score);
    println!();

    // 3. Training statistics tracking
    println!("3. Training Statistics Tracking");
    println!("   Monitor training progress and detect convergence issues.\n");
    
    let mut stats = TrainingStats::new(10);
    
    // Simulate training progress
    for epoch in 0..15 {
        let loss = 1.0 - epoch as f32 * 0.05;
        let avg_volume = 0.5 + epoch as f32 * 0.01;
        let gradient_norm = 0.1 / (1.0 + epoch as f32 * 0.1);
        
        stats.record(loss, avg_volume, gradient_norm);
        
        if epoch % 5 == 0 {
            if let Some((mean, min, max)) = stats.loss_stats() {
                println!("   Epoch {}: loss = {:.4} (mean: {:.4}, range: [{:.4}, {:.4}])",
                         epoch, loss, mean, min, max);
            }
        }
    }
    
    // Check for convergence
    let converged = stats.is_converged(0.01, 5);
    let exploding = stats.is_gradient_exploding(100.0);
    let vanishing = stats.is_gradient_vanishing(0.001);
    let collapsed = stats.is_volume_collapsed(0.01);
    
    println!("\n   Diagnostics:");
    println!("     Converged: {}", converged);
    println!("     Gradient exploding: {}", exploding);
    println!("     Gradient vanishing: {}", vanishing);
    println!("     Volume collapsed: {}", collapsed);
    println!();

    // 4. Loss component analysis
    println!("4. Loss Component Analysis");
    println!("   Track and balance multiple loss components during training.\n");
    
    // Imbalanced case (containment loss dominates)
    let imbalanced = LossComponents::new(0.85, 0.1, 0.05);
    println!("   Imbalanced loss components:");
    println!("     Containment: {:.2}, Regularization: {:.2}, Constraint: {:.2}",
             imbalanced.containment_loss,
             imbalanced.regularization_loss,
             imbalanced.constraint_loss);
    println!("     Total: {:.2}", imbalanced.total());
    println!("     Is imbalanced: {}", imbalanced.is_imbalanced());
    if let Some(dominant) = imbalanced.dominant_component() {
        println!("     Dominant component: {}", dominant);
    }
    
    // Balanced case
    let balanced = LossComponents::new(0.4, 0.3, 0.3);
    println!("\n   Balanced loss components:");
    println!("     Containment: {:.2}, Regularization: {:.2}, Constraint: {:.2}",
             balanced.containment_loss,
             balanced.regularization_loss,
             balanced.constraint_loss);
    println!("     Total: {:.2}", balanced.total());
    println!("     Is imbalanced: {}", balanced.is_imbalanced());
    println!();

    // 5. Practical training loop example
    println!("5. Example Training Loop with Diagnostics");
    println!("   Combining metrics and diagnostics in a realistic scenario.\n");
    
    let mut training_stats = TrainingStats::new(20);
    let mut all_ranks = Vec::new();
    
    // Simulate training epochs
    for epoch in 0..10 {
        // Simulate training step
        let loss = 1.0 / (1.0 + epoch as f32);
        let avg_volume = 0.3 + epoch as f32 * 0.02;
        let gradient_norm = 0.5 / (1.0 + epoch as f32 * 0.2);
        
        training_stats.record(loss, avg_volume, gradient_norm);
        
        // Simulate evaluation (link prediction ranking)
        let epoch_ranks: Vec<usize> = (0..5)
            .map(|_| {
                // Simulate improving ranks over time
                (10.0 - epoch as f32 * 0.8).max(1.0) as usize
            })
            .collect();
        all_ranks.extend_from_slice(&epoch_ranks);
        
        if epoch % 3 == 0 {
            let mrr = mean_reciprocal_rank(all_ranks.iter().copied());
            let hits_10 = hits_at_k(all_ranks.iter().copied(), 10);
            
            println!("   Epoch {}: MRR = {:.4}, Hits@10 = {:.4}, Loss = {:.4}",
                     epoch, mrr, hits_10, loss);
            
            // Check for issues
            if training_stats.is_gradient_exploding(10.0) {
                println!("     ⚠️  Warning: Gradient explosion detected!");
            }
            if training_stats.is_volume_collapsed(0.1) {
                println!("     ⚠️  Warning: Volume collapse detected!");
            }
        }
    }
    
    println!("\n   Final evaluation:");
    let final_mrr = mean_reciprocal_rank(all_ranks.iter().copied());
    let final_hits_10 = hits_at_k(all_ranks.iter().copied(), 10);
    let final_mr = mean_rank(all_ranks.iter().copied());
    
    println!("     MRR: {:.4}", final_mrr);
    println!("     Hits@10: {:.4}", final_hits_10);
    println!("     Mean Rank: {:.2}", final_mr);
    
    if training_stats.is_converged(0.01, 5) {
        println!("     ✓ Training converged");
    }

    Ok(())
}

