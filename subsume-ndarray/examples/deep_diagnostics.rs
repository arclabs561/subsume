//! Example demonstrating the deepest and most nuanced diagnostic techniques for box embeddings.

use ndarray::Array1;
use subsume_core::{
    training::{
        diagnostics::{DepthStratifiedGradientFlow, PhaseDetector, TrainingStats},
        quality::{DimensionalityUtilization, GeneralizationMetrics, VolumeConservation},
    },
    Box,
};
use subsume_ndarray::NdarrayBox;

fn main() -> Result<(), subsume_core::BoxError> {
    println!("=== Deep Diagnostic Techniques ===\n");

    // 1. Gradient flow by hierarchy depth
    println!("1. Gradient Flow Analysis by Hierarchy Depth");
    println!("   Reveals whether training distributes effort uniformly across hierarchy levels.\n");

    let mut depth_flow = DepthStratifiedGradientFlow::new(20);

    // Simulate training where root level gets more gradients
    for step in 0..15 {
        depth_flow.record(0, 0.5 - step as f32 * 0.01); // Root
        depth_flow.record(1, 0.3 - step as f32 * 0.008); // Level 1
        depth_flow.record(2, 0.1 - step as f32 * 0.005); // Level 2 (deep)
    }

    let means = depth_flow.mean_gradients_by_depth();
    println!("   Mean gradients by depth:");
    for (depth, &mean_grad) in &means {
        println!("     Depth {}: {:.4}", depth, mean_grad);
    }

    if let Some((min_depth, max_depth, ratio)) = depth_flow.check_depth_imbalance(2.0) {
        println!("   ⚠️  Depth imbalance detected!");
        println!(
            "     Min depth {}: {:.4}, Max depth {}: {:.4}",
            min_depth, means[&min_depth], max_depth, means[&max_depth]
        );
        println!(
            "     Ratio: {:.2}x (higher levels getting more gradients)",
            ratio
        );
    }
    println!();

    // 2. Training phase detection
    println!("2. Training Phase Detection");
    println!("   Identifies exploration, exploitation, convergence, and instability phases.\n");

    let mut phase_detector = PhaseDetector::new(10);

    // Simulate training progression
    let phases = vec![
        (1.0, 0.8, "Early"),
        (0.7, 0.4, "Mid"),
        (0.5, 0.2, "Late"),
        (0.45, 0.05, "Very Late"),
    ];

    for (loss, grad, label) in phases {
        for _ in 0..3 {
            phase_detector.record(loss, grad);
        }
        let phase = phase_detector.detect_phase();
        println!(
            "   {} training: loss={:.2}, grad={:.2} -> Phase: {:?}",
            label, loss, grad, phase
        );
    }
    println!();

    // 3. Volume conservation analysis
    println!("3. Volume Conservation Analysis");
    println!("   Verifies that parent volumes properly contain sum of children volumes.\n");

    let mut conservation = VolumeConservation::new();

    // Create a hierarchy: Animal (parent) -> Mammal, Bird (children)
    let animal = NdarrayBox::new(
        Array1::from(vec![0.0, 0.0]),
        Array1::from(vec![10.0, 10.0]),
        1.0,
    )?;
    let mammal = NdarrayBox::new(
        Array1::from(vec![1.0, 1.0]),
        Array1::from(vec![5.0, 5.0]),
        1.0,
    )?;
    let bird = NdarrayBox::new(
        Array1::from(vec![6.0, 6.0]),
        Array1::from(vec![9.0, 9.0]),
        1.0,
    )?;

    let animal_vol = animal.volume(1.0)?;
    let mammal_vol = mammal.volume(1.0)?;
    let bird_vol = bird.volume(1.0)?;

    conservation.record_parent_children(animal_vol, [mammal_vol, bird_vol].into_iter(), 0.1);

    println!("   Parent (Animal) volume: {:.4}", animal_vol);
    println!(
        "   Children volumes: Mammal={:.4}, Bird={:.4}",
        mammal_vol, bird_vol
    );
    println!("   Sum of children: {:.4}", mammal_vol + bird_vol);

    let mean_ratio = conservation.mean_ratio();
    println!(
        "   Mean parent-child ratio: {:.4} (should be <= 1.0)",
        mean_ratio
    );
    println!(
        "   Violation rate: {:.2}%",
        conservation.violation_rate() * 100.0
    );

    if let Some((min, max, mean, median)) = conservation.ratio_statistics() {
        println!(
            "   Ratio statistics: min={:.4}, max={:.4}, mean={:.4}, median={:.4}",
            min, max, mean, median
        );
    }
    println!();

    // 4. Dimensionality utilization analysis
    println!("4. Dimensionality Utilization Analysis");
    println!("   Detects underutilized or redundant dimensions.\n");

    let mut util = DimensionalityUtilization::new(4);

    // Record boxes with varying dimension usage
    // Dim 0: well-used (range 0-10)
    // Dim 1: underutilized (range 0-0.5)
    // Dim 2: well-used (range 0-8)
    // Dim 3: well-used (range 0-12)
    for i in 0..5 {
        let offset = i as f32 * 2.0;
        util.record_box(
            [offset, 0.0, offset * 0.8, offset * 1.2].into_iter(),
            [
                offset + 2.0,
                0.1 + i as f32 * 0.05,
                offset * 0.8 + 1.6,
                offset * 1.2 + 2.4,
            ]
            .into_iter(),
        );
    }

    let effective_dim = util.effective_dimensionality(1.0);
    println!(
        "   Effective dimensionality (threshold=1.0): {}",
        effective_dim
    );

    let scores = util.utilization_scores(15.0);
    println!("   Dimension utilization scores:");
    for (dim, &score) in scores.iter().enumerate() {
        println!("     Dimension {}: {:.2}%", dim, score * 100.0);
    }

    let underutilized = util.underutilized_dimensions(15.0, 0.1);
    if !underutilized.is_empty() {
        println!("   ⚠️  Underutilized dimensions: {:?}", underutilized);
    }
    println!();

    // 5. Generalization vs memorization
    println!("5. Generalization vs Memorization Metrics");
    println!("   Distinguishes learning structure from memorizing facts.\n");

    let mut gen_metrics = GeneralizationMetrics::new();

    // Simulate evaluation results
    // Inference-requiring facts (multi-step reasoning)
    gen_metrics.record_inference(0.75); // MRR on inference facts
    gen_metrics.record_inference(0.72);
    gen_metrics.record_inference(0.78);

    // Direct facts (seen in training)
    gen_metrics.record_direct(0.85); // MRR on direct facts
    gen_metrics.record_direct(0.88);
    gen_metrics.record_direct(0.82);

    if let Some(gap) = gen_metrics.generalization_gap() {
        println!("   Generalization gap (inference - direct): {:.4}", gap);
        if gap < 0.0 {
            println!("     ⚠️  Negative gap suggests memorization (direct > inference)");
        } else {
            println!("     ✓ Positive gap suggests good generalization");
        }
    }

    if let Some(ratio) = gen_metrics.generalization_ratio() {
        println!("   Generalization ratio (inference / direct): {:.4}", ratio);
        if ratio < 1.0 {
            println!("     ⚠️  Ratio < 1.0 indicates memorization");
        } else if ratio > 1.0 {
            println!("     ✓ Ratio > 1.0 indicates excellent generalization");
        } else {
            println!("     → Ratio ≈ 1.0 indicates balanced learning");
        }
    }
    println!();

    // 6. Comprehensive deep diagnostic workflow
    println!("6. Comprehensive Deep Diagnostic Workflow");
    println!("   Combining all deep diagnostics for complete analysis.\n");

    let mut training_stats = TrainingStats::new(20);
    let mut depth_flow = DepthStratifiedGradientFlow::new(20);
    let mut phase_detector = PhaseDetector::new(10);
    let mut conservation = VolumeConservation::new();
    let mut util = DimensionalityUtilization::new(3);

    // Simulate full training with hierarchy
    for epoch in 0..30 {
        let loss = 1.0 / (1.0 + epoch as f32 * 0.08);
        let avg_volume = 0.4 + epoch as f32 * 0.015;
        let gradient_norm = 0.6 / (1.0 + epoch as f32 * 0.12);

        training_stats.record(loss, avg_volume, gradient_norm);
        phase_detector.record(loss, gradient_norm);

        // Gradients by depth (root gets more early on)
        let root_grad = if epoch < 15 { 0.5 } else { 0.2 };
        let depth1_grad = if epoch < 15 { 0.3 } else { 0.18 };
        let depth2_grad = if epoch < 15 { 0.1 } else { 0.15 };
        depth_flow.record(0, root_grad);
        depth_flow.record(1, depth1_grad);
        depth_flow.record(2, depth2_grad);

        // Volume conservation (simulate parent-child relationships)
        if epoch % 5 == 0 {
            let parent_vol = 10.0 + epoch as f32 * 0.1;
            let child1_vol = 3.0 + epoch as f32 * 0.05;
            let child2_vol = 2.0 + epoch as f32 * 0.03;
            conservation.record_parent_children(
                parent_vol,
                [child1_vol, child2_vol].into_iter(),
                0.1,
            );
        }

        // Dimensionality utilization
        if epoch % 3 == 0 {
            let offset = epoch as f32 * 0.1;
            util.record_box(
                [offset, offset * 0.1, offset * 0.8].into_iter(),
                [offset + 1.0, offset * 0.1 + 0.2, offset * 0.8 + 0.8].into_iter(),
            );
        }

        if epoch % 10 == 0 {
            println!("   Epoch {}:", epoch);

            let phase = phase_detector.detect_phase();
            println!("     Phase: {:?}", phase);

            if let Some((min_d, max_d, ratio)) = depth_flow.check_depth_imbalance(2.0) {
                println!(
                    "     ⚠️  Depth gradient imbalance: {:.2}x (depth {} vs {})",
                    ratio, max_d, min_d
                );
            }

            let effective_dim = util.effective_dimensionality(0.5);
            println!("     Effective dimensions: {}/3", effective_dim);

            let violation_rate = conservation.violation_rate();
            if violation_rate > 0.0 {
                println!(
                    "     ⚠️  Volume conservation violations: {:.1}%",
                    violation_rate * 100.0
                );
            }
        }
    }

    println!("\n   Final Deep Diagnostics:");
    let final_phase = phase_detector.detect_phase();
    println!("     Training phase: {:?}", final_phase);

    let means = depth_flow.mean_gradients_by_depth();
    println!("     Final gradient distribution by depth:");
    for (depth, &mean) in &means {
        println!("       Depth {}: {:.4}", depth, mean);
    }

    let effective_dim = util.effective_dimensionality(0.5);
    println!("     Effective dimensionality: {}/3", effective_dim);

    let mean_ratio = conservation.mean_ratio();
    println!("     Mean volume conservation ratio: {:.4}", mean_ratio);
    if mean_ratio > 1.0 {
        println!("       ⚠️  Violation: children volumes exceed parent");
    } else {
        println!("       ✓ Conservation maintained");
    }

    Ok(())
}
