//! Example demonstrating embedding quality assessment and calibration metrics.

use ndarray::Array1;
use subsume_core::{
    Box,
    training::{
        quality::{VolumeDistribution, ContainmentAccuracy},
        calibration::{expected_calibration_error, brier_score},
    },
};
use subsume_ndarray::NdarrayBox;

fn main() -> Result<(), subsume_core::BoxError> {
    println!("=== Embedding Quality Assessment and Calibration ===\n");

    // 1. Volume distribution analysis
    println!("1. Volume Distribution Analysis");
    println!("   Analyze the distribution of box volumes to assess embedding quality.\n");
    
    // Create a collection of boxes with varying volumes
    let mut boxes = Vec::new();
    let mut volumes = Vec::new();
    
    for i in 0..20 {
        let side = 0.1 + (i as f32) * 0.05;
        let box_ = NdarrayBox::new(
            Array1::from(vec![0.0, 0.0]),
            Array1::from(vec![side, side]),
            1.0,
        )?;
        let vol = box_.volume(1.0)?;
        boxes.push(box_);
        volumes.push(vol);
    }
    
    let vol_dist = VolumeDistribution::from_volumes(volumes.iter().copied());
    
    println!("   Volume Statistics:");
    println!("     Min: {:.6}, Max: {:.6}", vol_dist.min, vol_dist.max);
    println!("     Mean: {:.6}, Median: {:.6}", vol_dist.mean, vol_dist.median);
    println!("     Std Dev: {:.6}, CV: {:.4}", vol_dist.std_dev, vol_dist.cv);
    println!();
    
    println!("   Quality Checks:");
    println!("     Volume collapsed: {}", vol_dist.is_collapsed(0.001));
    println!("     Volume degenerate: {}", vol_dist.is_degenerate(0.0001));
    println!("     Has hierarchy: {}", vol_dist.has_hierarchy(0.1));
    println!();

    // 2. Containment accuracy verification
    println!("2. Containment Accuracy Verification");
    println!("   Measure precision, recall, and F1 for containment predictions.\n");
    
    let mut containment_acc = ContainmentAccuracy::new();
    
    // Simulate containment predictions vs ground truth
    let predictions = vec![
        (true, true),   // TP: correctly predicted containment
        (true, true),   // TP
        (true, false),  // FP: incorrectly predicted containment
        (false, true), // FN: missed containment
        (false, false), // TN: correctly predicted non-containment
        (false, false), // TN
        (true, true),   // TP
        (false, false), // TN
    ];
    
    for (predicted, actual) in predictions {
        containment_acc.record(predicted, actual);
    }
    
    println!("   Confusion Matrix:");
    println!("     True Positives: {}", containment_acc.true_positives);
    println!("     False Positives: {}", containment_acc.false_positives);
    println!("     True Negatives: {}", containment_acc.true_negatives);
    println!("     False Negatives: {}", containment_acc.false_negatives);
    println!();
    
    println!("   Metrics:");
    println!("     Precision: {:.4}", containment_acc.precision());
    println!("     Recall: {:.4}", containment_acc.recall());
    println!("     F1 Score: {:.4}", containment_acc.f1());
    println!("     Accuracy: {:.4}", containment_acc.accuracy());
    println!();

    // 3. Calibration metrics
    println!("3. Calibration Metrics");
    println!("   Assess how well predicted probabilities match empirical frequencies.\n");
    
    // Simulate well-calibrated predictions
    let well_calibrated_preds = vec![
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    ];
    let well_calibrated_actuals = vec![
        false, false, false, false, true, true, true, true, true,
    ];
    
    let ece_well = expected_calibration_error(
        well_calibrated_preds.iter().copied(),
        well_calibrated_actuals.iter().copied(),
        10,
    );
    let brier_well = brier_score(
        well_calibrated_preds.iter().copied(),
        well_calibrated_actuals.iter().copied(),
    );
    
    println!("   Well-Calibrated Predictions:");
    println!("     ECE: {:.4} (lower is better)", ece_well);
    println!("     Brier Score: {:.4} (lower is better)", brier_well);
    
    // Simulate poorly calibrated predictions (overconfident)
    let overconfident_preds = vec![
        0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.9, 0.9, 0.1,
    ];
    let overconfident_actuals = vec![
        true, false, false, false, true, true, true, false, false,
    ];
    
    let ece_poor = expected_calibration_error(
        overconfident_preds.iter().copied(),
        overconfident_actuals.iter().copied(),
        10,
    );
    let brier_poor = brier_score(
        overconfident_preds.iter().copied(),
        overconfident_actuals.iter().copied(),
    );
    
    println!("\n   Overconfident Predictions:");
    println!("     ECE: {:.4} (lower is better)", ece_poor);
    println!("     Brier Score: {:.4} (lower is better)", brier_poor);
    println!("     Note: Overconfident predictions have higher ECE");
    println!();

    // 4. Practical quality assessment workflow
    println!("4. Practical Quality Assessment Workflow");
    println!("   Combining all metrics for comprehensive evaluation.\n");
    
    // Simulate evaluating a trained model
    let mut eval_volumes = Vec::new();
    let mut eval_containment = ContainmentAccuracy::new();
    let mut eval_preds = Vec::new();
    let mut eval_actuals = Vec::new();
    
    // Simulate evaluation on test set
    for i in 0..50 {
        // Volume distribution
        let vol = 0.05 + (i as f32) * 0.02;
        eval_volumes.push(vol);
        
        // Containment predictions
        let predicted_containment = vol > 0.3;
        let actual_containment = vol > 0.35; // Slight mismatch to simulate errors
        eval_containment.record(predicted_containment, actual_containment);
        
        // Calibration data
        let prob = if predicted_containment { 0.7 } else { 0.3 };
        eval_preds.push(prob);
        eval_actuals.push(actual_containment);
    }
    
    let eval_vol_dist = VolumeDistribution::from_volumes(eval_volumes.iter().copied());
    let eval_ece = expected_calibration_error(
        eval_preds.iter().copied(),
        eval_actuals.iter().copied(),
        10,
    );
    let eval_brier = brier_score(
        eval_preds.iter().copied(),
        eval_actuals.iter().copied(),
    );
    
    println!("   Evaluation Results:");
    println!("     Volume Distribution:");
    println!("       Mean: {:.4}, CV: {:.4}", eval_vol_dist.mean, eval_vol_dist.cv);
    println!("       Has hierarchy: {}", eval_vol_dist.has_hierarchy(0.1));
    println!("     Containment Accuracy:");
    println!("       Precision: {:.4}, Recall: {:.4}, F1: {:.4}",
             eval_containment.precision(),
             eval_containment.recall(),
             eval_containment.f1());
    println!("     Calibration:");
    println!("       ECE: {:.4}, Brier: {:.4}", eval_ece, eval_brier);
    println!();
    
    // Quality assessment summary
    println!("   Quality Assessment Summary:");
    let mut issues = Vec::new();
    
    if eval_vol_dist.is_collapsed(0.01) {
        issues.push("⚠️  Volume collapse detected");
    }
    if eval_vol_dist.is_degenerate(0.0001) {
        issues.push("⚠️  Degenerate volume distribution");
    }
    if !eval_vol_dist.has_hierarchy(0.1) {
        issues.push("⚠️  Lack of volume hierarchy");
    }
    if eval_containment.precision() < 0.7 {
        issues.push("⚠️  Low containment precision");
    }
    if eval_containment.recall() < 0.7 {
        issues.push("⚠️  Low containment recall");
    }
    if eval_ece > 0.2 {
        issues.push("⚠️  Poor calibration (high ECE)");
    }
    
    if issues.is_empty() {
        println!("     ✓ All quality checks passed");
    } else {
        for issue in &issues {
            println!("     {}", issue);
        }
    }

    Ok(())
}

