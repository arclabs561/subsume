//! Complete training loop example integrating all diagnostic techniques.
//!
//! This example demonstrates a realistic training scenario where we:
//! 1. Train box embeddings for a knowledge graph
//! 2. Monitor training with all available diagnostics
//! 3. Evaluate quality using comprehensive metrics
//! 4. Make training decisions based on diagnostic insights

use ndarray::Array1;
use subsume_core::{
    Box, BoxEmbedding,
    training::{
        metrics::{
            mean_reciprocal_rank, hits_at_k, mean_rank,
            StratifiedMetrics,
        },
        diagnostics::{
            TrainingStats, LossComponents, GradientFlowAnalysis,
            DepthStratifiedGradientFlow, PhaseDetector,
            RelationStratifiedTrainingStats,
        },
        quality::{
            VolumeDistribution, ContainmentAccuracy, IntersectionTopology,
            ContainmentHierarchy, VolumeConservation, DimensionalityUtilization,
            GeneralizationMetrics,
        },
        calibration::{expected_calibration_error, brier_score},
    },
};
use subsume_ndarray::NdarrayBox;
use subsume_core::BoxCollection;

fn main() -> Result<(), subsume_core::BoxError> {
    println!("=== Complete Training Loop with All Diagnostics ===\n");

    // Initialize all diagnostic trackers
    let mut training_stats = TrainingStats::new(20);
    let mut relation_stats = RelationStratifiedTrainingStats::new(20);
    let mut gradient_flow = GradientFlowAnalysis::new(20);
    let mut depth_flow = DepthStratifiedGradientFlow::new(20);
    let mut phase_detector = PhaseDetector::new(10);
    let mut loss_components = LossComponents::new(0.0, 0.0, 0.0);
    let mut containment_hierarchy = ContainmentHierarchy::new();
    let mut volume_conservation = VolumeConservation::new();
    let mut dimensionality_util = DimensionalityUtilization::new(3);
    let mut containment_accuracy = ContainmentAccuracy::new();
    let intersection_topology = IntersectionTopology::new();
    let generalization_metrics = GeneralizationMetrics::new();
    let mut stratified_metrics = StratifiedMetrics::new();

    // Simulate a knowledge graph: Animal -> Mammal -> Cat, Dog; Animal -> Bird -> Sparrow
    let mut boxes: BoxCollection<NdarrayBox> = BoxCollection::new();
    
    // Create entities
    let animal = NdarrayBox::new(
        Array1::from(vec![0.0, 0.0, 0.0]),
        Array1::from(vec![10.0, 10.0, 10.0]),
        1.0,
    )?;
    let mammal = NdarrayBox::new(
        Array1::from(vec![1.0, 1.0, 1.0]),
        Array1::from(vec![6.0, 6.0, 6.0]),
        1.0,
    )?;
    let bird = NdarrayBox::new(
        Array1::from(vec![7.0, 7.0, 7.0]),
        Array1::from(vec![9.0, 9.0, 9.0]),
        1.0,
    )?;
    let cat = NdarrayBox::new(
        Array1::from(vec![2.0, 2.0, 2.0]),
        Array1::from(vec![4.0, 4.0, 4.0]),
        1.0,
    )?;
    let dog = NdarrayBox::new(
        Array1::from(vec![4.5, 4.5, 4.5]),
        Array1::from(vec![5.5, 5.5, 5.5]),
        1.0,
    )?;
    let sparrow = NdarrayBox::new(
        Array1::from(vec![7.5, 7.5, 7.5]),
        Array1::from(vec![8.5, 8.5, 8.5]),
        1.0,
    )?;

    boxes.push(animal.clone());
    boxes.push(mammal.clone());
    boxes.push(bird.clone());
    boxes.push(cat.clone());
    boxes.push(dog.clone());
    boxes.push(sparrow.clone());

    // Build hierarchy
    containment_hierarchy.add_containment(0, 1); // Animal -> Mammal
    containment_hierarchy.add_containment(0, 2); // Animal -> Bird
    containment_hierarchy.add_containment(1, 3); // Mammal -> Cat
    containment_hierarchy.add_containment(1, 4); // Mammal -> Dog
    containment_hierarchy.add_containment(2, 5); // Bird -> Sparrow
    containment_hierarchy.compute_transitive_closure();

    // Record volume conservation
    let animal_vol = animal.volume(1.0)?;
    let mammal_vol = mammal.volume(1.0)?;
    let bird_vol = bird.volume(1.0)?;
    volume_conservation.record_parent_children(
        animal_vol,
        vec![mammal_vol, bird_vol].into_iter(),
        0.1,
    );

    let mammal_children_vol = cat.volume(1.0)? + dog.volume(1.0)?;
    volume_conservation.record_parent_children(
        mammal_vol,
        vec![cat.volume(1.0)?, dog.volume(1.0)?].into_iter(),
        0.1,
    );

    // Record dimensionality utilization
    for i in 0..boxes.len() {
        let b = boxes.get(i)?;
        // Convert Array1<f32> to Vec<f32> then to iterator for record_box
        let min_vec: Vec<f32> = b.min().to_vec();
        let max_vec: Vec<f32> = b.max().to_vec();
        dimensionality_util.record_box(
            min_vec.into_iter(),
            max_vec.into_iter(),
        );
    }

    // Simulate training loop
    println!("Training Progress:\n");
    let mut all_ranks = Vec::new();
    let mut all_predictions = Vec::new();
    let mut all_actuals = Vec::new();
    let mut volumes_history = Vec::new();

    for epoch in 0..50 {
        // Simulate training step
        let base_loss = 1.0 / (1.0 + epoch as f32 * 0.05);
        let containment_loss = base_loss * 0.6;
        let reg_loss = base_loss * 0.25;
        let constraint_loss = base_loss * 0.15;
        
        loss_components = LossComponents::new(containment_loss, reg_loss, constraint_loss);
        
        let avg_volume = 0.4 + epoch as f32 * 0.01;
        let gradient_norm = 0.6 / (1.0 + epoch as f32 * 0.1);
        
        // Record with intersection volumes
        let intersection_vol = if epoch > 5 {
            Some(0.2 + epoch as f32 * 0.005) // Improving containment
        } else {
            None
        };
        
        training_stats.record_with_intersection(
            base_loss,
            avg_volume,
            intersection_vol,
            gradient_norm,
        );
        
        // Relation-stratified stats
        relation_stats.record("is_a", base_loss, avg_volume, gradient_norm);
        relation_stats.record("has_part", base_loss * 0.9, avg_volume, gradient_norm);
        
        // Gradient flow (simulate imbalanced early, balanced later)
        let center_grad = if epoch < 20 { 0.5 - epoch as f32 * 0.01 } else { 0.2 };
        let size_grad = if epoch < 20 { 0.05 + epoch as f32 * 0.005 } else { 0.15 };
        gradient_flow.record(Some(center_grad), Some(size_grad), None, None);
        
        // Depth-stratified gradients
        depth_flow.record(0, center_grad); // Root level
        depth_flow.record(1, center_grad * 0.7); // Level 1
        depth_flow.record(2, center_grad * 0.4); // Level 2
        
        // Phase detection
        phase_detector.record(base_loss, gradient_norm);
        
        // Simulate evaluation (every 5 epochs)
        if epoch % 5 == 0 {
            // Simulate ranking results
            let rank = if epoch < 10 { 10 - epoch } else { 1 + (epoch % 3) };
            all_ranks.push(rank);
            
            // Simulate containment predictions
            let predicted_containment = epoch > 15;
            let actual_containment = true; // In real scenario, this comes from test data
            containment_accuracy.record(predicted_containment, actual_containment);
            
            // Calibration data
            let pred_prob = 0.5 + epoch as f32 * 0.01;
            all_predictions.push(pred_prob.min(0.95));
            all_actuals.push(actual_containment);
            
            // Stratified metrics
            stratified_metrics.add_relation_result("is_a".to_string(), rank);
            stratified_metrics.add_depth_result(0, rank, actual_containment);
            if epoch % 10 == 0 {
                stratified_metrics.add_frequency_result("high", rank);
            }
            
            volumes_history.push(avg_volume);
        }
        
        // Periodic diagnostic reports
        if epoch % 10 == 0 {
            println!("Epoch {}:", epoch);
            
            // Training phase
            let phase = phase_detector.detect_phase();
            println!("  Phase: {:?}", phase);
            
            // Convergence
            if training_stats.is_converged(0.01, 5) {
                println!("  ✓ Converged");
            }
            
            // Loss components
            if loss_components.is_imbalanced() {
                if let Some(dominant) = loss_components.dominant_component() {
                    println!("  ⚠️  Loss imbalance: {} dominates", dominant);
                }
            }
            
            // Gradient flow
            if let Some(ratio) = gradient_flow.check_imbalance(2.0) {
                println!("  ⚠️  Gradient imbalance: {:.2}x", ratio);
            }
            
            // Depth imbalance
            if let Some((min_d, max_d, ratio)) = depth_flow.check_depth_imbalance(2.0) {
                println!("  ⚠️  Depth gradient imbalance: {:.2}x (depth {} vs {})", ratio, max_d, min_d);
            }
            
            // Volume conservation
            let violation_rate = volume_conservation.violation_rate();
            if violation_rate > 0.0 {
                println!("  ⚠️  Volume conservation violations: {:.1}%", violation_rate * 100.0);
            }
            
            // Dimensionality utilization
            let effective_dim = dimensionality_util.effective_dimensionality(1.0);
            println!("  Effective dimensions: {}/3", effective_dim);
            
            println!();
        }
    }

    // Final evaluation
    println!("\n=== Final Evaluation ===\n");
    
    // Rank-based metrics
    let mrr = mean_reciprocal_rank(all_ranks.iter().copied());
    let hits_1 = hits_at_k(all_ranks.iter().copied(), 1);
    let hits_10 = hits_at_k(all_ranks.iter().copied(), 10);
    let mr = mean_rank(all_ranks.iter().copied());
    
    println!("Rank-Based Metrics:");
    println!("  MRR: {:.4}", mrr);
    println!("  Hits@1: {:.4}", hits_1);
    println!("  Hits@10: {:.4}", hits_10);
    println!("  Mean Rank: {:.2}", mr);
    println!();
    
    // Finalize stratified metrics
    stratified_metrics.finalize_relations();
    stratified_metrics.finalize_depths();
    stratified_metrics.finalize_frequency();
    
    println!("Stratified Metrics:");
    if let Some(rel_metrics) = stratified_metrics.by_relation.get("is_a") {
        println!("  is_a relation: MRR={:.4}, Hits@10={:.4}", rel_metrics.mrr, rel_metrics.hits_10);
    }
    if let Some(depth_metrics) = stratified_metrics.by_depth.get(&0) {
        println!("  Depth 0: MRR={:.4}, Containment Accuracy={:.4}", 
                 depth_metrics.mrr, depth_metrics.containment_accuracy);
    }
    println!();
    
    // Containment accuracy
    println!("Containment Accuracy:");
    println!("  Precision: {:.4}", containment_accuracy.precision());
    println!("  Recall: {:.4}", containment_accuracy.recall());
    println!("  F1: {:.4}", containment_accuracy.f1());
    println!("  Accuracy: {:.4}", containment_accuracy.accuracy());
    println!();
    
    // Calibration
    let ece = expected_calibration_error(
        all_predictions.iter().copied(),
        all_actuals.iter().copied(),
        10,
    );
    let brier = brier_score(
        all_predictions.iter().copied(),
        all_actuals.iter().copied(),
    );
    println!("Calibration Metrics:");
    println!("  ECE: {:.4}", ece);
    println!("  Brier Score: {:.4}", brier);
    println!();
    
    // Volume distribution
    let vol_dist = VolumeDistribution::from_volumes(volumes_history.iter().copied());
    println!("Volume Distribution:");
    println!("  Entropy: {:.4}", vol_dist.entropy);
    println!("  CV: {:.4}", vol_dist.cv);
    println!("  Has hierarchy: {}", vol_dist.has_hierarchy(0.1));
    println!();
    
    // Hierarchy verification
    let (violations, total) = containment_hierarchy.verify_transitivity();
    println!("Hierarchy Verification:");
    println!("  Transitivity violations: {}/{}", violations, total);
    let cycles = containment_hierarchy.detect_cycles();
    println!("  Cycles detected: {}", cycles.len());
    let depths = containment_hierarchy.hierarchy_depths();
    println!("  Max depth: {}", depths.values().max().copied().unwrap_or(0));
    println!();
    
    // Final diagnostics summary
    println!("=== Training Diagnostics Summary ===\n");
    
    let final_phase = phase_detector.detect_phase();
    println!("Final Training Phase: {:?}", final_phase);
    
    if let Some((mean, min, max)) = training_stats.intersection_volume_stats() {
        println!("Intersection Volume: mean={:.4}, range=[{:.4}, {:.4}]", mean, min, max);
        if let Some(trend) = training_stats.intersection_volume_trend(5) {
            println!("  Trend: {}", if trend { "increasing" } else { "decreasing" });
        }
    }
    
    let convergence_rate = relation_stats.convergence_rate(0.01, 5);
    println!("Relation Convergence Rate: {:.2}%", convergence_rate * 100.0);
    
    let effective_dim = dimensionality_util.effective_dimensionality(1.0);
    println!("Effective Dimensionality: {}/3", effective_dim);
    
    let mean_ratio = volume_conservation.mean_ratio();
    println!("Volume Conservation Ratio: {:.4} (should be <= 1.0)", mean_ratio);
    
    if let Some(gap) = generalization_metrics.generalization_gap() {
        println!("Generalization Gap: {:.4}", gap);
    }

    println!("\n✓ Complete training loop with all diagnostics demonstrated!");

    Ok(())
}

