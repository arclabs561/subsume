//! Example demonstrating advanced and nuanced diagnostic techniques for box embeddings.

use ndarray::Array1;
use subsume_core::{
    Box,
    training::{
        diagnostics::{TrainingStats, GradientFlowAnalysis},
        quality::{
            VolumeDistribution, ContainmentHierarchy, IntersectionTopology,
            kl_divergence,
        },
        calibration::{
            expected_calibration_error, adaptive_calibration_error,
            reliability_diagram,
        },
    },
};
use subsume_ndarray::NdarrayBox;

fn main() -> Result<(), subsume_core::BoxError> {
    println!("=== Advanced Diagnostic Techniques ===\n");

    // 1. Per-parameter gradient flow analysis
    println!("1. Per-Parameter Gradient Flow Analysis");
    println!("   Track gradients separately for center vs size, min vs max coordinates.\n");
    
    let mut gradient_flow = GradientFlowAnalysis::new(20);
    
    // Simulate training with imbalanced gradients
    for step in 0..15 {
        // Center coordinates get strong gradients
        let center_grad = 0.5 - step as f32 * 0.02;
        // Size parameters get weak gradients (potential problem)
        let size_grad = 0.05 - step as f32 * 0.001;
        // Min coordinates
        let min_grad = 0.3 - step as f32 * 0.01;
        // Max coordinates
        let max_grad = 0.25 - step as f32 * 0.01;
        
        gradient_flow.record(
            Some(center_grad.max(0.0)),
            Some(size_grad.max(0.0)),
            Some(min_grad.max(0.0)),
            Some(max_grad.max(0.0)),
        );
        
        if step % 5 == 0 {
            if let Some(ratio) = gradient_flow.check_imbalance(2.0) {
                println!("   Step {}: Gradient imbalance detected! Ratio: {:.2}", step, ratio);
            }
            let sparsity = gradient_flow.gradient_sparsity(0.01);
            println!("   Step {}: Gradient sparsity: {:.2}%", step, sparsity * 100.0);
        }
    }
    println!();

    // 2. Volume distribution entropy analysis
    println!("2. Volume Distribution Entropy Analysis");
    println!("   Shannon entropy reveals information content in volume distribution.\n");
    
    // Create boxes with varying volumes
    let mut volumes = Vec::new();
    for i in 0..30 {
        let side = 0.1 + (i as f32) * 0.03;
        let box_ = NdarrayBox::new(
            Array1::from(vec![0.0, 0.0]),
            Array1::from(vec![side, side]),
            1.0,
        )?;
        volumes.push(box_.volume(1.0)?);
    }
    
    let vol_dist = VolumeDistribution::from_volumes(volumes.iter().copied());
    
    println!("   Volume Statistics:");
    println!("     Mean: {:.6}, Std Dev: {:.6}", vol_dist.mean, vol_dist.std_dev);
    println!("     CV: {:.4}, Entropy: {:.4}", vol_dist.cv, vol_dist.entropy);
    println!("     Quantiles: Q25={:.4}, Q50={:.4}, Q75={:.4}, Q95={:.4}",
             vol_dist.quantiles.0, vol_dist.quantiles.1,
             vol_dist.quantiles.2, vol_dist.quantiles.3);
    println!("     Has hierarchy: {}", vol_dist.has_hierarchy(0.1));
    println!();

    // 3. KL divergence between volume distributions
    println!("3. KL Divergence Between Volume Distributions");
    println!("   Compare learned distribution to target distribution.\n");
    
    // Simulate learned volumes (skewed)
    let learned_volumes = vec![0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0];
    // Target: more uniform distribution
    let target_volumes = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
    
    let kl = kl_divergence(learned_volumes.iter().copied(), target_volumes.iter().copied());
    println!("   KL divergence (learned || target): {:.4}", kl);
    println!("   Higher values indicate greater divergence from target");
    println!();

    // 4. Containment hierarchy verification
    println!("4. Containment Hierarchy Verification");
    println!("   Verify transitive closure and detect cycles.\n");
    
    let mut hierarchy = ContainmentHierarchy::new();
    
    // Build a hierarchy: Animal -> Mammal -> Cat, Animal -> Bird -> Sparrow
    hierarchy.add_containment(0, 1); // Animal -> Mammal
    hierarchy.add_containment(1, 2); // Mammal -> Cat
    hierarchy.add_containment(0, 3); // Animal -> Bird
    hierarchy.add_containment(3, 4); // Bird -> Sparrow
    hierarchy.compute_transitive_closure();
    
    // Verify transitivity
    let (violations, total_checks) = hierarchy.verify_transitivity();
    println!("   Transitivity verification:");
    println!("     Violations: {}, Total checks: {}", violations, total_checks);
    println!("     Transitivity preserved: {}", violations == 0);
    
    // Check hierarchy depths
    let depths = hierarchy.hierarchy_depths();
    println!("   Hierarchy depths:");
    for (node, depth) in &depths {
        println!("     Node {}: depth {}", node, depth);
    }
    
    // Detect cycles (should be none)
    let cycles = hierarchy.detect_cycles();
    println!("   Cycles detected: {}", cycles.len());
    println!();

    // 5. Advanced calibration: ACE vs ECE
    println!("5. Advanced Calibration Metrics");
    println!("   Compare ECE (equal-width bins) vs ACE (equal-mass bins).\n");
    
    // Simulate predictions with varying density
    let mut preds = Vec::new();
    let mut actuals = Vec::new();
    
    // Dense region (many predictions around 0.5)
    for _ in 0..20 {
        preds.push(0.5);
        actuals.push(true);
    }
    // Sparse regions
    for _ in 0..5 {
        preds.push(0.1);
        actuals.push(false);
    }
    for _ in 0..5 {
        preds.push(0.9);
        actuals.push(true);
    }
    
    let ece = expected_calibration_error(preds.iter().copied(), actuals.iter().copied(), 10);
    let ace = adaptive_calibration_error(preds.iter().copied(), actuals.iter().copied(), 10);
    
    println!("   ECE (equal-width bins): {:.4}", ece);
    println!("   ACE (equal-mass bins): {:.4}", ace);
    println!("   ACE often more stable with uneven prediction distributions");
    println!();

    // 6. Reliability diagram
    println!("6. Reliability Diagram Data");
    println!("   For visualization of calibration quality.\n");
    
    let diagram = reliability_diagram(preds.iter().copied(), actuals.iter().copied(), 5);
    
    println!("   Reliability bins:");
    for i in 0..diagram.bin_centers.len() {
        println!("     Bin {}: predicted={:.3}, actual={:.3}, count={}",
                 i, diagram.bin_centers[i],
                 diagram.empirical_accuracies[i],
                 diagram.bin_counts[i]);
    }
    println!();

    // 7. Intersection topology with sibling/parent-child analysis
    println!("7. Intersection Topology Regularity");
    println!("   Analyze sibling and parent-child intersection patterns.\n");
    
    let mut topology = IntersectionTopology::new();
    
    // Record some relationships
    topology.record_intersection(true);
    topology.record_intersection(true);
    topology.record_intersection(false);
    topology.record_containment(true);
    
    // Record sibling intersection (should be low)
    topology.record_sibling_intersection(0.05, 1.0); // Small intersection relative to volume
    topology.record_sibling_intersection(0.03, 0.8);
    
    // Record parent-child intersection (should be high)
    topology.record_parent_child_intersection(0.9, 1.0); // Child mostly contained
    topology.record_parent_child_intersection(0.85, 0.9);
    
    println!("   Topology Statistics:");
    println!("     Intersection rate: {:.2}%", topology.intersection_rate() * 100.0);
    println!("     Containment rate: {:.2}%", topology.containment_rate() * 100.0);
    
    if let Some(sibling_ratio) = topology.sibling_intersection_ratio {
        println!("     Sibling intersection ratio: {:.4} (should be low)", sibling_ratio);
    }
    if let Some(pc_ratio) = topology.parent_child_intersection_ratio {
        println!("     Parent-child intersection ratio: {:.4} (should be high)", pc_ratio);
    }
    println!();

    // 8. Comprehensive diagnostic workflow
    println!("8. Comprehensive Diagnostic Workflow");
    println!("   Combining all advanced diagnostics.\n");
    
    let mut training_stats = TrainingStats::new(20);
    let mut gradient_flow = GradientFlowAnalysis::new(20);
    let mut volumes = Vec::new();
    
    // Simulate training progress
    for epoch in 0..25 {
        let loss = 1.0 / (1.0 + epoch as f32 * 0.1);
        let avg_volume = 0.3 + epoch as f32 * 0.02;
        let gradient_norm = 0.5 / (1.0 + epoch as f32 * 0.15);
        
        training_stats.record(loss, avg_volume, gradient_norm);
        
        // Simulate imbalanced gradients early, balanced later
        let center_grad = if epoch < 10 { 0.5 } else { 0.2 };
        let size_grad = if epoch < 10 { 0.05 } else { 0.15 };
        gradient_flow.record(Some(center_grad), Some(size_grad), None, None);
        
        volumes.push(avg_volume);
        
        if epoch % 8 == 0 {
            println!("   Epoch {}:", epoch);
            
            // Convergence check
            if training_stats.is_converged(0.01, 5) {
                println!("     ✓ Converged");
            }
            
            // Gradient imbalance
            if let Some(ratio) = gradient_flow.check_imbalance(2.0) {
                println!("     ⚠️  Gradient imbalance: {:.2}x", ratio);
            }
            
            // Volume distribution
            let vol_dist = VolumeDistribution::from_volumes(volumes.iter().copied());
            println!("     Volume entropy: {:.4}, CV: {:.4}",
                     vol_dist.entropy, vol_dist.cv);
        }
    }
    
    println!("\n   Final Diagnostics Summary:");
    let final_vol_dist = VolumeDistribution::from_volumes(volumes.iter().copied());
    println!("     Volume distribution: entropy={:.4}, has_hierarchy={}",
             final_vol_dist.entropy,
             final_vol_dist.has_hierarchy(0.1));
    
    let final_sparsity = gradient_flow.gradient_sparsity(0.01);
    println!("     Gradient sparsity: {:.2}%", final_sparsity * 100.0);
    
    if let Some(ratio) = gradient_flow.check_imbalance(2.0) {
        println!("     ⚠️  Final gradient imbalance: {:.2}x", ratio);
    } else {
        println!("     ✓ Gradient flow balanced");
    }

    Ok(())
}

