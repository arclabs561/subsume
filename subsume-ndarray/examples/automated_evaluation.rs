//! Automated evaluation example with visualization.
//!
//! This example demonstrates:
//! 1. Running evaluations with different optimizers
//! 2. Collecting metrics
//! 3. Generating comparison plots
//! 4. Saving results to files

use ndarray::Array1;
use subsume_core::dataset::Triple;
use subsume_core::trainer::{evaluate_link_prediction, TrainingConfig};
use subsume_core::Box as CoreBox;
use subsume_ndarray::evaluation::{EvaluationConfig, EvaluationMetrics, OptimizerComparison};
#[cfg(feature = "plotting")]
use subsume_ndarray::evaluation::plotting;
use subsume_ndarray::{Adam, AdamW, NdarrayBox, SGD};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Automated Evaluation with Visualization");
    println!("========================================\n");

    // Create output directory
    let output_dir = Path::new("eval_results");
    std::fs::create_dir_all(output_dir)?;

    // Create synthetic dataset
    let train_triples = vec![
        Triple {
            head: "animal".to_string(),
            relation: "is_a".to_string(),
            tail: "mammal".to_string(),
        },
        Triple {
            head: "mammal".to_string(),
            relation: "is_a".to_string(),
            tail: "dog".to_string(),
        },
        Triple {
            head: "mammal".to_string(),
            relation: "is_a".to_string(),
            tail: "cat".to_string(),
        },
    ];

    let test_triples = vec![Triple {
        head: "animal".to_string(),
        relation: "is_a".to_string(),
        tail: "bird".to_string(),
    }];

    let mut entities = HashSet::new();
    for triple in train_triples.iter().chain(test_triples.iter()) {
        entities.insert(triple.head.clone());
        entities.insert(triple.tail.clone());
    }

    // Evaluation configuration
    let eval_config = EvaluationConfig::default();

    let mut comparison = OptimizerComparison::new();

    // Test different optimizers
    let optimizers = vec![
        ("Adam", run_with_optimizer::<Adam>),
        ("AdamW", run_with_optimizer::<AdamW>),
        ("SGD", run_with_optimizer::<SGD>),
    ];

    for (name, optimizer_fn) in optimizers {
        println!("Evaluating with {}...", name);
        let metrics = optimizer_fn(&train_triples, &test_triples, &entities, &eval_config)?;
        comparison.add_result(name.to_string(), metrics.clone());

        // Save individual results
        let result_path = output_dir.join(format!("{}_metrics.json", name));
        metrics.save_json(&result_path)?;
        println!("  Saved metrics to {:?}", result_path);

        // Generate plots
        if eval_config.generate_plots {
            let loss_plot = output_dir.join(format!("{}_loss.png", name));
            plotting::plot_loss_curve(&metrics, &loss_plot)?;
            println!("  Saved loss plot to {:?}", loss_plot);

            let mrr_plot = output_dir.join(format!("{}_mrr.png", name));
            plotting::plot_mrr_curve(&metrics, &mrr_plot)?;
            println!("  Saved MRR plot to {:?}", mrr_plot);
        }
    }

    // Save comparison
    let comparison_path = output_dir.join("optimizer_comparison.json");
    comparison.save_json(&comparison_path)?;
    println!("\nSaved comparison to {:?}", comparison_path);

    if eval_config.generate_plots {
        let comparison_plot = output_dir.join("optimizer_comparison.png");
        plotting::plot_optimizer_comparison(&comparison, &comparison_plot)?;
        println!("Saved comparison plot to {:?}", comparison_plot);
    }

    println!("\nEvaluation complete! Results saved to {:?}", output_dir);

    Ok(())
}

fn run_with_optimizer<O>(
    train_triples: &[Triple],
    test_triples: &[Triple],
    entities: &HashSet<String>,
    config: &EvaluationConfig,
) -> Result<EvaluationMetrics, Box<dyn std::error::Error>>
where
    O: OptimizerTrait,
{
    // Initialize embeddings
    let mut entity_boxes: HashMap<String, NdarrayBox> = HashMap::new();
    for (idx, entity) in entities.iter().enumerate() {
        let offset = idx as f32 * 2.0;
        let box_ = NdarrayBox::new(
            Array1::from_vec(vec![offset, offset]),
            Array1::from_vec(vec![offset + 1.5, offset + 1.5]),
            1.0,
        )?;
        entity_boxes.insert(entity.clone(), box_);
    }

    // Create optimizer
    let mut optimizer = O::new(config.learning_rate, config.weight_decay);

    let mut metrics = EvaluationMetrics::new();
    let total_start = Instant::now();

    // Training loop
    for epoch in 0..config.epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0;
        let mut batch_count = 0;

        for batch in train_triples.chunks(config.batch_size) {
            for triple in batch {
                if let (Some(head_box), Some(tail_box)) = (
                    entity_boxes.get(&triple.head),
                    entity_boxes.get(&triple.tail),
                ) {
                    let pos_score = head_box.containment_prob(tail_box, 1.0)?;
                    let loss = 1.0 - pos_score;
                    epoch_loss += loss;

                    // Simplified gradient update
                    let head_box_mut = entity_boxes.get_mut(&triple.head).unwrap();
                    let mut head_min = head_box_mut.min().to_owned();
                    let tail_min = tail_box.min();
                    let grad_min_vec: Vec<f32> = head_min
                        .iter()
                        .zip(tail_min.iter())
                        .map(|(h, t)| loss * (t - h))
                        .collect();
                    let grad_min = Array1::from_vec(grad_min_vec);
                    optimizer.update(&format!("{}_min", triple.head), &mut head_min, grad_min.view());
                    *head_box_mut = NdarrayBox::new(
                        head_min,
                        head_box_mut.max().to_owned(),
                        1.0,
                    )?;
                }
            }
            batch_count += 1;
        }

        let avg_loss = epoch_loss / batch_count as f32.max(1.0);
        let epoch_time = epoch_start.elapsed().as_secs_f64();

        // Evaluate
        let eval_results = evaluate_link_prediction::<NdarrayBox>(test_triples, &entity_boxes, None)?;
        let mrr = eval_results.mrr;

        metrics.record_epoch(avg_loss, mrr, epoch_time);

        if epoch % 10 == 0 {
            println!(
                "  Epoch {}: Loss = {:.4}, MRR = {:.4}",
                epoch + 1, avg_loss, mrr
            );
        }
    }

    // Final evaluation
    let final_results = evaluate_link_prediction::<NdarrayBox>(test_triples, &entity_boxes, None)?;
    metrics.final_mrr = final_results.mrr;
    metrics.final_hits_at_1 = final_results.hits_at_1;
    metrics.final_hits_at_10 = final_results.hits_at_10;
    metrics.total_training_time = total_start.elapsed().as_secs_f64();

    Ok(metrics)
}

trait OptimizerTrait {
    fn new(learning_rate: f32, weight_decay: f32) -> Self;
    fn update(&mut self, name: &str, param: &mut Array1<f32>, grad: ndarray::ArrayView1<f32>);
}

impl OptimizerTrait for Adam {
    fn new(learning_rate: f32, _weight_decay: f32) -> Self {
        Adam::new(learning_rate)
    }

    fn update(&mut self, name: &str, param: &mut Array1<f32>, grad: ndarray::ArrayView1<f32>) {
        self.update(name, param, grad);
    }
}

impl OptimizerTrait for AdamW {
    fn new(learning_rate: f32, weight_decay: f32) -> Self {
        AdamW::new(learning_rate, weight_decay)
    }

    fn update(&mut self, name: &str, param: &mut Array1<f32>, grad: ndarray::ArrayView1<f32>) {
        self.update(name, param, grad);
    }
}

impl OptimizerTrait for SGD {
    fn new(learning_rate: f32, _weight_decay: f32) -> Self {
        SGD::new(learning_rate)
    }

    fn update(&mut self, name: &str, param: &mut Array1<f32>, grad: ndarray::ArrayView1<f32>) {
        self.update(name, param, grad);
    }
}

